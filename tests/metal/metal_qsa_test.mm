// Metal twin for QSA normalization and RoPE (docs/mac/04 §5.6).
#include "metallib_sha256.inc"
#include "oracles/qsa.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/qsa_dispatch.hpp"
#include "twin.hpp"

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kMaxT = VERIFY_MAX_T;
constexpr std::size_t kRopeCells = 32u;

int g_failures = 0;

std::uint32_t heads_per_token(QsaNormRopeMode mode) {
    switch (mode) {
        case QsaNormRopeMode::key: return N_HEAD_KV;
        case QsaNormRopeMode::query: return N_HEAD_Q;
        case QsaNormRopeMode::indexer_query: return INDEXER_HEADS;
    }
    return 0u;
}

std::uint32_t width(QsaNormRopeMode mode) {
    return mode == QsaNormRopeMode::indexer_query
        ? INDEXER_HEAD_DIM
        : HEAD_DIM;
}

std::uint32_t input_stride(QsaNormRopeMode mode) {
    return mode == QsaNormRopeMode::query ? 2u * HEAD_DIM : width(mode);
}

const char* mode_name(QsaNormRopeMode mode) {
    switch (mode) {
        case QsaNormRopeMode::key: return "key";
        case QsaNormRopeMode::query: return "query";
        case QsaNormRopeMode::indexer_query: return "indexer-query";
    }
    return "invalid";
}

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

bool concurrent_dispatch_mode() {
    const char* const value = std::getenv("STRATA_METAL_DISPATCH");
    return value != nullptr && std::string_view(value) == "concurrent";
}

template <class Prepare, class Append>
bool expect_append_rejected_in_open_concurrent_group(
    Context& context, const std::string& label,
    std::string_view expected_error, Prepare&& prepare, Append&& append) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;

    bool saw_open_concurrent_group = false;
    bool append_succeeded = false;
    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& body_error) {
            if (!encoder.group_begin(body_error)) return false;
            saw_open_concurrent_group = encoder.concurrent_group_open();
            if (!saw_open_concurrent_group) {
                body_error = "test requires STRATA_METAL_DISPATCH=concurrent";
                return false;
            }
            if (!prepare(encoder, body_error)) {
                std::string group_error;
                encoder.group_end(group_error);
                return false;
            }

            std::string append_error;
            append_succeeded = append(encoder, append_error);
            std::string group_error;
            const bool group_closed = encoder.group_end(group_error);
            body_error = !group_closed
                ? group_error
                : append_succeeded
                    ? "KV append unexpectedly succeeded in an open concurrent group"
                    : append_error;
            return false;
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool rejected = saw_open_concurrent_group && !encoded &&
        !append_succeeded && error.find(expected_error) != std::string::npos &&
        discarded;
    check(rejected, label + " rejects open concurrent-group dependency", error);
    command->discard();
    return rejected;
}

template <class T>
std::span<T> typed(twin::GuardedBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
}

template <class T>
std::span<const T> typed(const twin::GuardedBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<const T*>(bytes.data()),
            bytes.size() / sizeof(T)};
}

BufferRef buffer_subview(BufferRef buffer, std::size_t offset,
                         std::size_t length) {
    return buffer.slice(offset, length);
}

float sample(twin::InputSet set, std::mt19937_64& rng,
             std::uint32_t index, float scale) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 12> edge{
            0.0f, -0.0f, 0.125f, -0.125f, 0.25f, -0.25f,
            0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f};
        return edge[index % edge.size()] * scale;
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t value =
            static_cast<std::int32_t>((index * 37u + 19u) % 1021u) - 510;
        return static_cast<float>(value) * (scale / 512.0f);
    }
    const std::int32_t value =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(value) * (scale / 32768.0f);
}

bool run_command(Context& context, const std::string& label,
                 const QsaNormRopeBuffers& buffers,
                 std::uint32_t token_rows, QsaNormRopeMode mode,
                 std::string& error, bool shared_encoder = false,
                 bool prefill_api = false) {
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared QSA norm/RoPE",
              [&](Encoder& encoder, std::string& encode_error) {
                  return prefill_api
                      ? encode_prefill_qsa_norm_rope(
                            context, encoder, token_rows, mode, buffers,
                            encode_error)
                      : encode_qsa_norm_rope(
                            context, encoder, token_rows, mode, buffers,
                            encode_error);
              },
              error)
        : (prefill_api
               ? encode_prefill_qsa_norm_rope(
                     context, *command, token_rows, mode, buffers, error)
               : encode_qsa_norm_rope(
                     context, *command, token_rows, mode, buffers, error));
    if (!encoded) {
        check(false, label + " encode", error);
        if (!shared_encoder) command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer",
              "command buffer became unavailable");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }
    error.clear();
    return true;
}

bool run_norm_case(Context& context, std::uint32_t token_rows,
                   QsaNormRopeMode mode, twin::InputSet set,
                   std::span<const float> input,
                   std::span<const float> gamma,
                   std::span<const StepRow> steps,
                   std::span<const float> rope,
                   const twin::Options& options,
                   std::string_view edge_label,
                   std::vector<float>& actual_output,
                   bool shared_encoder = false,
                   bool prefill_api = false) {
    const std::string label =
        "QSA norm_rope " + std::string(mode_name(mode)) +
        " T=" + std::to_string(token_rows) +
        " set=" + twin::input_set_name(set) + std::string(edge_label) +
        (shared_encoder ? " shared-encoder" : "") +
        (prefill_api ? " prefill" : "");
    const bool in_place = mode != QsaNormRopeMode::query;
    const std::size_t input_bytes = input.size_bytes();
    const std::size_t output_bytes =
        static_cast<std::size_t>(token_rows) * heads_per_token(mode) *
        width(mode) * sizeof(float);
    if (steps.size() != token_rows) {
        check(false, label + " host step-row count");
        return false;
    }

    std::vector<std::int32_t> positions(token_rows);
    for (std::uint32_t i = 0u; i < token_rows; ++i) {
        positions[i] = steps[i].pos;
    }
    std::vector<twin::tf> expected;
    oracle::qsa_norm_rope(input, gamma, positions, rope, token_rows, mode,
                          expected);

    std::string error;
    twin::GuardedBuffer input_buffer, gamma_buffer, steps_buffer;
    twin::GuardedBuffer rope_buffer, output_buffer;
    check(input_buffer.allocate(context, label + " input", input_bytes, error),
          label + " input allocation", error);
    check(gamma_buffer.allocate(context, label + " gamma",
                                gamma.size_bytes(), error),
          label + " gamma allocation", error);
    check(steps_buffer.allocate(context, label + " steps",
                                steps.size_bytes(), error),
          label + " steps allocation", error);
    check(rope_buffer.allocate(context, label + " rope", rope.size_bytes(),
                               error),
          label + " rope allocation", error);
    if (!in_place) {
        check(output_buffer.allocate(context, label + " output", output_bytes,
                                     error),
              label + " output allocation", error);
    }
    if (!input_buffer.view() || !gamma_buffer.view() ||
        !steps_buffer.view() || !rope_buffer.view() ||
        (!in_place && !output_buffer.view())) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), input.data(), input_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(),
                gamma.size_bytes());
    std::memcpy(steps_buffer.payload().data(), steps.data(),
                steps.size_bytes());
    std::memcpy(rope_buffer.payload().data(), rope.data(), rope.size_bytes());

    if (!in_place) {
        check(output_buffer.poison_floats_against(
                  expected, twin::Comparison::units, error),
              label + " output poison", error);
    }
    input_buffer.capture_before_dispatch();
    gamma_buffer.capture_before_dispatch();
    steps_buffer.capture_before_dispatch();
    rope_buffer.capture_before_dispatch();
    if (!in_place) output_buffer.capture_before_dispatch();

    const BufferRef output_ref =
        in_place ? input_buffer.view() : output_buffer.view();
    const QsaNormRopeBuffers buffers{
        input_buffer.view(), gamma_buffer.view(), steps_buffer.view(),
        rope_buffer.view(), output_ref};
    if (!run_command(context, label, buffers, token_rows, mode, error,
                     shared_encoder, prefill_api)) {
        return false;
    }

    const twin::GuardedBuffer& result_buffer =
        in_place ? input_buffer : output_buffer;
    const auto actual = typed<float>(result_buffer).first(expected.size());
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 64.0, 0.0, 0u});
    if (mismatch) {
        twin::report_failure(label.c_str(), options, token_rows, *mismatch);
    }
    bool passed = !mismatch;
    check(passed, label + " U(64)");
    actual_output.assign(actual.begin(), actual.end());

    const std::array<twin::ByteRange, 1> written{{
        {0u, output_bytes}}};
    const auto output_guard =
        result_buffer.first_unwritten_change(written);
    check(!output_guard, label + " output guards",
          output_guard ? output_guard->reason : "");
    passed &= !output_guard;
    if (!in_place) {
        const auto input_change = input_buffer.first_unwritten_change({});
        check(!input_change, label + " input remains unchanged",
              input_change ? input_change->reason : "");
        passed &= !input_change;
    }
    check(!gamma_buffer.first_unwritten_change({}),
          label + " gamma remains unchanged");
    check(!steps_buffer.first_unwritten_change({}),
          label + " steps remain unchanged");
    check(!rope_buffer.first_unwritten_change({}),
          label + " RoPE table remains unchanged");
    passed &= !gamma_buffer.first_unwritten_change({}) &&
              !steps_buffer.first_unwritten_change({}) &&
              !rope_buffer.first_unwritten_change({});
    return passed;
}

bool run_gate_command(Context& context, const std::string& label,
                      std::uint32_t token_rows,
                      const QsaGateBuffers& buffers,
                      std::string& error,
                      bool shared_encoder = false) {
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_qsa_gate(context, encoder, token_rows,
                                         buffers, encode_error);
              },
              error)
        : encode_qsa_gate(context, *command, token_rows, buffers, error);
    if (!encoded) {
        check(false, label + " encode", error);
        command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer",
              "command buffer became unavailable");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }
    error.clear();
    return true;
}

bool run_gate_case(Context& context, std::uint32_t token_rows,
                   twin::InputSet set, std::span<const float> attn,
                   std::span<const float> q_full,
                   const twin::Options& options, std::string_view suffix,
                   std::vector<float>* actual_output = nullptr,
                   bool shared_encoder = false) {
    constexpr std::size_t kAttnRow =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM;
    constexpr std::size_t kQFullRow =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM;
    const std::size_t output_count =
        static_cast<std::size_t>(token_rows) * kAttnRow;
    const std::string label =
        "QSA gate T=" + std::to_string(token_rows) +
        " set=" + twin::input_set_name(set) + std::string(suffix) +
        (shared_encoder ? " shared Encoder" : "");
    if (attn.size() != output_count ||
        q_full.size() != static_cast<std::size_t>(token_rows) * kQFullRow) {
        check(false, label + " host input extents");
        return false;
    }

    std::vector<twin::tf> expected;
    oracle::qsa_gate(attn, q_full, token_rows, expected);
    std::string error;
    twin::GuardedBuffer attn_buffer, q_full_buffer, output_buffer;
    const bool allocated =
        attn_buffer.allocate(context, label + " attn", attn.size_bytes(),
                             error) &&
        q_full_buffer.allocate(context, label + " q_full",
                               q_full.size_bytes(), error) &&
        output_buffer.allocate(context, label + " output",
                               output_count * sizeof(float), error);
    check(allocated, label + " allocations", error);
    if (!allocated || !attn_buffer.view() || !q_full_buffer.view() ||
        !output_buffer.view()) {
        return false;
    }
    std::memcpy(attn_buffer.payload().data(), attn.data(),
                attn.size_bytes());
    std::memcpy(q_full_buffer.payload().data(), q_full.data(),
                q_full.size_bytes());
    if (!output_buffer.poison_floats_against(
            expected, twin::Comparison::units, error)) {
        check(false, label + " output poison", error);
        return false;
    }

    attn_buffer.capture_before_dispatch();
    q_full_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    const QsaGateBuffers buffers{
        attn_buffer.view(), q_full_buffer.view(), output_buffer.view()};
    if (!run_gate_command(context, label, token_rows, buffers, error,
                          shared_encoder)) {
        return false;
    }

    const auto actual = typed<float>(output_buffer).first(output_count);
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 8.0, 0.0, 0u});
    if (mismatch) {
        twin::report_failure(label.c_str(), options, token_rows, *mismatch);
    }
    check(!mismatch, label + " U(8) against accurate sigmoid oracle");
    if (actual_output != nullptr) {
        actual_output->assign(actual.begin(), actual.end());
    }

    const std::array<twin::ByteRange, 1> written{{
        {0u, output_count * sizeof(float)}}};
    const auto output_guard = output_buffer.first_unwritten_change(written);
    const auto attn_change = attn_buffer.first_unwritten_change({});
    const auto q_full_change = q_full_buffer.first_unwritten_change({});
    check(!output_guard && !attn_change && !q_full_change,
          label + " input immutability and output guards",
          output_guard ? output_guard->reason
                       : attn_change ? attn_change->reason
                                     : q_full_change ? q_full_change->reason
                                                     : "");
    return !mismatch && !output_guard && !attn_change && !q_full_change;
}

bool run_gate_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    constexpr std::size_t kAttnRow =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM;
    constexpr std::size_t kQFullRow =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM;
    std::mt19937_64 rng(
        options.seed ^ 0x51534167617465ULL ^
        (static_cast<std::uint64_t>(set) << 32u));
    std::vector<float> all_attn(kMaxT * kAttnRow);
    std::vector<float> all_q_full(kMaxT * kQFullRow);
    for (std::uint32_t index = 0u; index < all_attn.size(); ++index) {
        all_attn[index] = sample(set, rng, index + 0x3701u, 0.75f);
    }
    for (std::uint32_t index = 0u; index < all_q_full.size(); ++index) {
        all_q_full[index] = sample(set, rng, index + 0x4b09u, 6.0f);
    }

    std::vector<float> batch_output;
    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= kMaxT; ++rows) {
        std::vector<float> output;
        passed &= run_gate_case(
            context, rows, set,
            std::span<const float>(all_attn).first(rows * kAttnRow),
            std::span<const float>(all_q_full).first(rows * kQFullRow),
            options, "", rows == kMaxT ? &batch_output : &output);
    }
    if (batch_output.size() != kMaxT * kAttnRow) return false;

    for (const std::uint32_t rows : {1u, kMaxT}) {
        passed &= run_gate_case(
            context, rows, set,
            std::span<const float>(all_attn).first(rows * kAttnRow),
            std::span<const float>(all_q_full).first(rows * kQFullRow),
            options, " shared Encoder", nullptr, true);
    }

    for (std::uint32_t row = 0u; row < kMaxT; ++row) {
        std::vector<float> row_output;
        passed &= run_gate_case(
            context, 1u, set,
            std::span<const float>(all_attn).subspan(row * kAttnRow,
                                                     kAttnRow),
            std::span<const float>(all_q_full).subspan(row * kQFullRow,
                                                       kQFullRow),
            options, " row independence", &row_output);
        const auto batch_row = std::span<const float>(batch_output).subspan(
            static_cast<std::size_t>(row) * kAttnRow, kAttnRow);
        std::vector<twin::tf> batch_row_expected;
        batch_row_expected.reserve(batch_row.size());
        for (float value : batch_row) {
            batch_row_expected.emplace_back(value);
        }
        const auto mismatch = twin::compare_floats(
            row_output, batch_row_expected,
            {twin::Comparison::units, 0.0, 0.0, 0u});
        check(!mismatch, "QSA gate row result is batch independent");
        passed &= !mismatch;
    }
    return passed;
}

bool run_gate_extreme_case(Context& context,
                           const twin::Options& options) {
    constexpr std::size_t kAttnRow =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM;
    constexpr std::size_t kQFullRow =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM;
    constexpr std::array<float, 12> gates{
        -30.0f, -20.0f, -8.0f, -2.0f, -1.0f, -0.0f,
        0.0f,    1.0f,   2.0f,  8.0f,  20.0f, 30.0f};
    constexpr std::array<float, 8> values{
        0.0f, -0.0f, 0.125f, -0.25f, 0.5f, -1.0f, 2.0f, -4.0f};
    std::vector<float> attn(kAttnRow);
    std::vector<float> q_full(kQFullRow);
    for (std::size_t index = 0u; index < attn.size(); ++index) {
        attn[index] = values[index % values.size()];
    }
    for (std::size_t index = 0u; index < q_full.size(); ++index) {
        q_full[index] =
            index % 512u >= HEAD_DIM
                ? gates[(index - HEAD_DIM) % gates.size()]
                : values[index % values.size()];
    }
    return run_gate_case(context, 1u, twin::InputSet::edge, attn, q_full,
                         options, " sigmoid range and saturation");
}

std::vector<float> make_rope_table() {
    std::vector<float> rope(kRopeCells * 32u * 2u);
    for (std::size_t position = 0u; position < kRopeCells; ++position) {
        for (std::uint32_t pair = 0u; pair < 32u; ++pair) {
            const float theta =
                0.013f * static_cast<float>((position + 1u) * (pair + 3u));
            const float magnitude =
                position == 0u ? 0.75f : 1.0f;
            const std::size_t index = (position * 32u + pair) * 2u;
            rope[index] = magnitude * std::cos(theta);
            rope[index + 1u] = magnitude * std::sin(theta);
        }
    }
    return rope;
}

void make_steps(std::vector<StepRow>& steps) {
    constexpr std::array<std::int32_t, kMaxT> positions{
        0, static_cast<std::int32_t>(kRopeCells - 1u), 3, 17, 5, 23, 9, 13};
    steps.resize(kMaxT);
    for (std::uint32_t token = 0u; token < kMaxT; ++token) {
        steps[token] = StepRow{
            positions[token], positions[token] + 1,
            static_cast<std::int32_t>(
                positions[token] / static_cast<std::int32_t>(
                                       INDEXER_BLOCK_CELLS)),
            static_cast<std::int32_t>(INDEXER_SELECTION_WIDTH_MAX)};
    }
}

bool run_prefill_norm_rope_case(Context& context,
                                const twin::Options& options) {
    constexpr std::uint32_t token_rows = 65u;
    const std::vector<float> rope = make_rope_table();
    bool passed = true;
    for (const auto mode : {QsaNormRopeMode::key, QsaNormRopeMode::query,
                            QsaNormRopeMode::indexer_query}) {
        const std::uint32_t heads = heads_per_token(mode);
        const std::uint32_t stride = input_stride(mode);
        const std::uint32_t value_width = width(mode);
        std::mt19937_64 rng(
            options.seed ^
            (std::uint64_t{static_cast<std::uint32_t>(mode)} << 40u) ^
            0x50524546494c4cull);
        std::vector<float> input(
            static_cast<std::size_t>(token_rows) * heads * stride);
        for (std::uint32_t index = 0u; index < input.size(); ++index) {
            input[index] =
                sample(twin::InputSet::fixture, rng, index, 0.4f);
        }
        std::vector<float> gamma(value_width);
        for (std::uint32_t index = 0u; index < gamma.size(); ++index) {
            gamma[index] =
                1.0f + sample(twin::InputSet::fixture, rng, index + 71u,
                              0.25f);
        }
        std::vector<StepRow> steps(token_rows);
        for (std::uint32_t token = 0u; token < token_rows; ++token) {
            const auto position = static_cast<std::int32_t>(
                token % static_cast<std::uint32_t>(kRopeCells));
            steps[token] = StepRow{
                position,
                position + 1,
                position / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS),
                static_cast<std::int32_t>(INDEXER_SELECTION_WIDTH_MAX)};
        }

        std::vector<float> standalone_output;
        passed &= run_norm_case(
            context, token_rows, mode, twin::InputSet::fixture, input,
            gamma, steps, rope, options, " 65-row Prefill chunk",
            standalone_output, false, true);
        std::vector<float> encoder_output;
        passed &= run_norm_case(
            context, token_rows, mode, twin::InputSet::fixture, input,
            gamma, steps, rope, options,
            " 65-row Prefill chunk",
            encoder_output, true, true);
        const auto mismatch = twin::compare_bytes(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    standalone_output.data()),
                standalone_output.size() * sizeof(float)),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(encoder_output.data()),
                encoder_output.size() * sizeof(float)));
        check(!mismatch,
              "QSA Prefill norm/RoPE " +
                  std::string(mode_name(mode)) +
                  " CommandBuffer/Encoder bit parity");
        passed &= !mismatch;
    }
    return passed;
}

bool run_mode_set(Context& context, QsaNormRopeMode mode,
                  twin::InputSet set, const twin::Options& options,
                  bool add_edge_profiles) {
    const std::uint32_t heads = heads_per_token(mode);
    const std::uint32_t stride = input_stride(mode);
    const std::uint32_t value_width = width(mode);
    const std::size_t max_input_count =
        static_cast<std::size_t>(kMaxT) * heads * stride;
    std::mt19937_64 rng(options.seed ^
                        (std::uint64_t{static_cast<std::uint32_t>(mode)} << 40u) ^
                        (std::uint64_t{static_cast<std::uint32_t>(set)} << 32u));
    std::vector<float> max_input(max_input_count);
    for (std::uint32_t i = 0u; i < max_input.size(); ++i) {
        max_input[i] = sample(set, rng, i, 0.4f);
    }
    std::vector<float> gamma(value_width);
    for (std::uint32_t i = 0u; i < gamma.size(); ++i) {
        gamma[i] = 1.0f + sample(set, rng, i + 71u, 0.25f);
    }
    const std::vector<float> rope = make_rope_table();
    std::vector<StepRow> max_steps;
    make_steps(max_steps);

    std::vector<float> rows_eight_output;
    for (std::uint32_t token_rows = 1u; token_rows <= kMaxT; ++token_rows) {
        const std::size_t input_count =
            static_cast<std::size_t>(token_rows) * heads * stride;
        std::vector<float> input(max_input.begin(),
                                 max_input.begin() + input_count);
        std::vector<float> actual;
        const bool passed = run_norm_case(
            context, token_rows, mode, set, input, gamma,
            std::span<const StepRow>(max_steps).first(token_rows), rope,
            options, "", actual);
        if (token_rows == kMaxT) rows_eight_output = std::move(actual);
        if (!passed) return false;
        if (set == twin::InputSet::fixture &&
            (token_rows == 1u || token_rows == kMaxT)) {
            std::vector<float> shared_output;
            if (!run_norm_case(
                    context, token_rows, mode, set, input, gamma,
                    std::span<const StepRow>(max_steps).first(token_rows),
                    rope, options, "", shared_output, true)) {
                return false;
            }
        }
    }

    bool independent = rows_eight_output.size() ==
        static_cast<std::size_t>(kMaxT) * heads * value_width;
    for (std::uint32_t token = 0u; token < kMaxT && independent; ++token) {
        const std::size_t row_input_count =
            static_cast<std::size_t>(heads) * stride;
        const auto first =
            max_input.begin() + static_cast<std::size_t>(token) *
                                    row_input_count;
        std::vector<float> input(first, first + row_input_count);
        std::vector<float> actual;
        if (!run_norm_case(context, 1u, mode, set, input, gamma,
                           std::span<const StepRow>(max_steps).subspan(token, 1u),
                           rope, options, " row-t", actual)) {
            independent = false;
            break;
        }
        const std::size_t output_offset =
            static_cast<std::size_t>(token) * heads * value_width;
        const auto expected_row =
            std::span<const float>(rows_eight_output)
                .subspan(output_offset,
                         static_cast<std::size_t>(heads) * value_width);
        const auto mismatch = twin::compare_bytes(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(actual.data()),
                actual.size() * sizeof(float)),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(expected_row.data()),
                expected_row.size_bytes()));
        check(!mismatch, "QSA norm_rope " + std::string(mode_name(mode)) +
                             " row_t_independent token=" +
                             std::to_string(token));
        independent &= !mismatch;
    }

    if (add_edge_profiles) {
        std::vector<float> zero_input(heads * stride, 0.0f);
        std::vector<float> actual;
        independent &= run_norm_case(
            context, 1u, mode, set, zero_input, gamma,
            std::span<const StepRow>(max_steps).first(1u), rope, options,
            " edge=all-zero", actual);

        constexpr float baseline = 1.0f / 1024.0f;
        std::vector<float> outlier_input(
            static_cast<std::size_t>(2u) * heads * stride, baseline);
        for (std::uint32_t head = 0u; head < heads; ++head) {
            const std::size_t row =
                static_cast<std::size_t>(heads + head) * stride;
            std::fill_n(outlier_input.begin() + row, value_width, 1.0f);
        }
        independent &= run_norm_case(
            context, 2u, mode, set, outlier_input, gamma,
            std::span<const StepRow>(max_steps).first(2u), rope, options,
            " edge=2^10-outlier", actual);
    }
    return independent;
}

bool test_gate_encoder_validation(Context& context,
                                 Context& foreign_context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t attn_bytes =
        static_cast<std::size_t>(rows) * N_HEAD_Q * HEAD_DIM *
        sizeof(float);
    constexpr std::size_t q_full_bytes =
        static_cast<std::size_t>(rows) * N_HEAD_Q * 2u * HEAD_DIM *
        sizeof(float);
    std::string error;
    twin::GuardedBuffer attn, q_full, output;
    if (!attn.allocate(context, "QSA gate validation attn", attn_bytes,
                       error) ||
        !q_full.allocate(context, "QSA gate validation q_full",
                         q_full_bytes, error) ||
        !output.allocate(context, "QSA gate validation output", attn_bytes,
                         error)) {
        check(false, "QSA gate validation allocations", error);
        return false;
    }
    const QsaGateBuffers valid{
        attn.view(), q_full.view(), output.view()};
    bool passed = true;
    auto reject = [&](const std::string& label,
                      const QsaGateBuffers& candidate,
                      std::uint32_t token_rows,
                      std::string_view message) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        error.clear();
        const bool rejected =
            !encode_qsa_gate(context, *command, token_rows, candidate, error) &&
            error.find(message) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    passed &= reject("QSA gate rejects zero rows", valid, 0u, "row count");
    passed &= reject("QSA gate rejects too many rows", valid, 9u,
                     "row count");

    QsaGateBuffers short_attn = valid;
    short_attn.attn.length -= sizeof(float);
    passed &= reject("QSA gate rejects short attention input", short_attn,
                     rows, "undersized or misaligned");
    QsaGateBuffers short_q_full = valid;
    short_q_full.q_full.length -= sizeof(float);
    passed &= reject("QSA gate rejects short q_full input", short_q_full,
                     rows, "undersized or misaligned");
    QsaGateBuffers short_output = valid;
    short_output.output.length -= sizeof(float);
    passed &= reject("QSA gate rejects short output", short_output, rows,
                     "undersized or misaligned");

    twin::GuardedBuffer extra_attn;
    if (!extra_attn.allocate(context, "QSA gate misaligned input",
                             attn_bytes + 16u, error)) {
        check(false, "QSA gate misaligned-input allocation", error);
        return false;
    }
    QsaGateBuffers misaligned = valid;
    misaligned.attn = buffer_subview(extra_attn.view(), 1u, attn_bytes);
    passed &= reject("QSA gate rejects misaligned attention input",
                     misaligned, rows, "undersized or misaligned");

    QsaGateBuffers output_attn_alias = valid;
    output_attn_alias.output = valid.attn;
    passed &= reject("QSA gate rejects output/attention overlap",
                     output_attn_alias, rows, "must be disjoint");
    QsaGateBuffers output_q_full_alias = valid;
    output_q_full_alias.output =
        buffer_subview(valid.q_full, 0u, attn_bytes);
    passed &= reject("QSA gate rejects output/q_full overlap",
                     output_q_full_alias, rows, "must be disjoint");

    QsaGateBuffers readonly_alias = valid;
    readonly_alias.attn = buffer_subview(valid.q_full, 0u, attn_bytes);
    auto readonly_command =
        context.make_command_buffer("QSA gate overlapping read-only inputs",
                                    error);
    if (!readonly_command) {
        check(false, "QSA gate overlapping-input command buffer", error);
        return false;
    }
    error.clear();
    const bool readonly_overlap_accepted =
        encode_qsa_gate(context, *readonly_command, rows, readonly_alias,
                        error);
    check(readonly_overlap_accepted,
          "QSA gate permits overlapping read-only inputs", error);
    passed &= readonly_overlap_accepted;
    readonly_command->discard();

    twin::GuardedBuffer foreign_buffer;
    if (!foreign_buffer.allocate(foreign_context, "foreign QSA gate input",
                                 attn_bytes, error)) {
        check(false, "QSA gate foreign-buffer allocation", error);
        return false;
    }
    QsaGateBuffers foreign_view = valid;
    foreign_view.attn = foreign_buffer.view();
    passed &= reject("QSA gate rejects a foreign Context buffer",
                     foreign_view, rows, "not owned");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA gate foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA gate foreign command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool foreign_command_rejected =
        !encode_qsa_gate(context, *foreign_command, rows, valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_command_rejected,
          "QSA gate rejects a foreign command buffer", error);
    passed &= foreign_command_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA gate submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA gate submitted command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_gate(context, *submitted, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA gate rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA gate submitted command-buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "QSA gate submitted command-buffer completion");
            passed = false;
        }
    }

    auto shared_command = context.make_command_buffer(
        "QSA shared gate validation and late failure", error);
    if (!shared_command) {
        check(false, "QSA shared gate command-buffer setup", error);
        return false;
    }
    const auto reject_shared =
        [&](Encoder& encoder, std::string& encode_error,
            const std::string& label, const QsaGateBuffers& candidate,
            std::uint32_t token_rows, std::string_view message) {
            const bool rejected =
                !encode_qsa_gate(context, encoder, token_rows, candidate,
                                 encode_error) &&
                encode_error.find(message) != std::string::npos;
            check(rejected, label, encode_error);
            encode_error.clear();
            return rejected;
        };
    bool shared_validation_passed = true;
    const bool shared_validation_encoded = Encoder::encode(
        context, *shared_command, "QSA shared gate validation",
        [&](Encoder& encoder, std::string& encode_error) {
            shared_validation_passed &= reject_shared(
                encoder, encode_error, "QSA shared gate rejects zero rows",
                valid, 0u, "row count");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects too many rows", valid, 9u,
                "row count");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects short attention input",
                short_attn, rows, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects short q_full input",
                short_q_full, rows, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error, "QSA shared gate rejects short output",
                short_output, rows, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects misaligned attention input",
                misaligned, rows, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects output/attention overlap",
                output_attn_alias, rows, "must be disjoint");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects output/q_full overlap",
                output_q_full_alias, rows, "must be disjoint");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects a foreign Context buffer",
                foreign_view, rows, "not owned");
            QsaGateBuffers forged_cpu = valid;
            forged_cpu.attn.contents =
                static_cast<char*>(forged_cpu.attn.contents) +
                sizeof(float);
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects forged CPU metadata", forged_cpu,
                rows, "not owned");
            QsaGateBuffers forged_gpu = valid;
            forged_gpu.q_full.gpu_address ^= 0x10u;
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared gate rejects forged GPU metadata", forged_gpu,
                rows, "not owned");
            return true;
        },
        error);
    check(shared_validation_encoded,
          "QSA shared gate validation body completes", error);
    passed &= shared_validation_encoded && shared_validation_passed;

    QsaGateBuffers shared_readonly_alias = readonly_alias;
    auto shared_readonly_command = context.make_command_buffer(
        "QSA shared gate overlapping read-only inputs", error);
    if (!shared_readonly_command) {
        check(false, "QSA shared gate read-only command-buffer setup", error);
        return false;
    }
    const bool shared_readonly_accepted = Encoder::encode(
        context, *shared_readonly_command, "QSA shared gate read-only inputs",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_qsa_gate(context, encoder, rows,
                                   shared_readonly_alias, encode_error);
        },
        error);
    check(shared_readonly_accepted,
          "QSA shared gate permits overlapping read-only inputs", error);
    passed &= shared_readonly_accepted;
    shared_readonly_command->discard();

    std::fill(output.payload().begin(), output.payload().end(), 0xa5u);
    const std::vector<std::uint8_t> output_before(
        output.payload().begin(), output.payload().end());
    error.clear();
    bool first_gate_encoded = false;
    const bool late_failure_propagated = !Encoder::encode(
        context, *shared_command, "QSA shared gate late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_gate(context, encoder, rows, valid, encode_error)) {
                return false;
            }
            first_gate_encoded = true;
            return encode_qsa_gate(context, encoder, rows, short_output,
                                   encode_error);
        },
        error) &&
        first_gate_encoded &&
        error.find("undersized or misaligned") != std::string::npos;
    check(late_failure_propagated,
          "QSA shared gate propagates a late failure", error);
    passed &= late_failure_propagated;
    const bool output_untouched = std::equal(
        output_before.begin(), output_before.end(), output.payload().begin());
    check(output_untouched,
          "QSA shared gate late failure leaves output untouched");
    passed &= output_untouched;
    error.clear();
    const bool late_failure_discarded =
        !shared_command->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_failure_discarded,
          "QSA shared gate late failure discards earlier work", error);
    passed &= late_failure_discarded;
    return passed;
}

bool test_encoder_validation(Context& context, Context& foreign_context) {
    constexpr std::uint32_t key_rows = 2u;
    constexpr std::size_t key_bytes =
        static_cast<std::size_t>(key_rows) * N_HEAD_KV * HEAD_DIM *
        sizeof(float);
    constexpr std::size_t query_input_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM * sizeof(float);
    constexpr std::size_t query_output_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM * sizeof(float);
    constexpr std::size_t gamma_bytes = HEAD_DIM * sizeof(float);
    constexpr std::size_t steps_bytes = kMaxT * sizeof(StepRow);
    constexpr std::size_t rope_bytes = kRopeCells * 32u * 2u * sizeof(float);
    std::string error;
    const auto alloc = [&](Context& owner, const char* label,
                           std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef input = alloc(context, "QSA validation input",
                            std::max(key_bytes, query_input_bytes));
    BufferRef gamma = alloc(context, "QSA validation gamma", gamma_bytes);
    BufferRef steps = alloc(context, "QSA validation steps", steps_bytes);
    BufferRef rope = alloc(context, "QSA validation rope", rope_bytes);
    BufferRef output = alloc(context, "QSA validation output",
                             std::max(key_bytes, query_output_bytes));
    BufferRef foreign_input =
        alloc(foreign_context, "QSA foreign input", query_input_bytes);
    if (!input || !gamma || !steps || !rope || !output || !foreign_input) {
        check(false, "QSA validation allocations", error);
        return false;
    }
    std::memset(input.contents, 0, input.length);
    std::memset(gamma.contents, 0, gamma.length);
    std::memset(rope.contents, 0, rope.length);
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    for (std::uint32_t i = 0u; i < kMaxT; ++i) {
        step_rows[i] = StepRow{0, 1, 0, 1};
    }
    const QsaNormRopeBuffers valid{input, gamma, steps, rope, output};
    auto command = context.make_command_buffer("QSA validation", error);
    if (!command) {
        check(false, "QSA validation command buffer", error);
        return false;
    }
    bool passed = true;
    const auto reject = [&](const std::string& label, auto&& call,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !call() && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    passed &= reject("QSA norm/RoPE rejects zero rows",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 0u, QsaNormRopeMode::key,
                               valid, error); },
                     "row count");
    passed &= reject("QSA norm/RoPE rejects rows above eight",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 9u, QsaNormRopeMode::key,
                               valid, error); },
                     "row count");
    passed &= reject("QSA norm/RoPE rejects invalid mode",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u,
                               static_cast<QsaNormRopeMode>(9u), valid,
                               error); },
                     "mode");
    QsaNormRopeBuffers short_input = valid;
    short_input.input.length =
        N_HEAD_KV * HEAD_DIM * sizeof(float) - sizeof(float);
    passed &= reject("QSA norm/RoPE rejects a short input view",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               short_input, error); },
                     "undersized");
    QsaNormRopeBuffers out_of_allocation = valid;
    id<MTLBuffer> native_input =
        (__bridge id<MTLBuffer>)out_of_allocation.input.handle;
    const std::size_t key_input_bytes =
        N_HEAD_KV * HEAD_DIM * sizeof(float);
    const std::size_t overflowing_offset =
        native_input.length - key_input_bytes / 2u;
    out_of_allocation.input.offset = overflowing_offset;
    out_of_allocation.input.length = key_input_bytes;
    out_of_allocation.input.contents =
        static_cast<char*>(native_input.contents) + overflowing_offset;
    out_of_allocation.input.gpu_address = native_input.gpuAddress == 0u
        ? 0u
        : native_input.gpuAddress + overflowing_offset;
    passed &= reject("QSA norm/RoPE rejects a view past its Metal allocation",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               out_of_allocation, error); },
                     "not owned by this Context");
    QsaNormRopeBuffers forged_gpu = valid;
    forged_gpu.input.gpu_address ^= 0x10u;
    passed &= reject("QSA norm/RoPE rejects forged GPU metadata",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               forged_gpu, error); },
                     "not owned by this Context");
    QsaNormRopeBuffers misaligned_gamma = valid;
    misaligned_gamma.gamma =
        gamma.slice(1u, gamma.length - 1u);
    passed &= reject("QSA norm/RoPE rejects a misaligned gamma",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               misaligned_gamma, error); },
                     "misaligned");
    QsaNormRopeBuffers incomplete_rope = valid;
    incomplete_rope.rope.length -= sizeof(float);
    passed &= reject("QSA norm/RoPE rejects an incomplete RoPE row",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               incomplete_rope, error); },
                     "complete 32-pair rows");
    QsaNormRopeBuffers short_steps = valid;
    short_steps.steps.length = sizeof(StepRow) - sizeof(std::int32_t);
    passed &= reject("QSA norm/RoPE rejects short step rows",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               short_steps, error); },
                     "undersized");
    step_rows[0].pos = -1;
    passed &= reject("QSA norm/RoPE rejects a negative step position",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               valid, error); },
                     "outside the RoPE table");
    step_rows[0].pos = static_cast<std::int32_t>(kRopeCells);
    passed &= reject("QSA norm/RoPE rejects a position past the table",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               valid, error); },
                     "outside the RoPE table");
    step_rows[0].pos = 0;

    QsaNormRopeBuffers partial_in_place = valid;
    partial_in_place.output =
        input.slice(sizeof(float), input.length - sizeof(float));
    passed &= reject("QSA norm/RoPE rejects partial in-place aliases",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               partial_in_place, error); },
                     "exact same view");
    QsaNormRopeBuffers output_gamma_alias = valid;
    output_gamma_alias.gamma = output;
    passed &= reject("QSA norm/RoPE rejects output/gamma overlap",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               output_gamma_alias, error); },
                     "overlap gamma");
    QsaNormRopeBuffers query_alias{input, gamma, steps, rope, input};
    passed &= reject("QSA query rejects input/output overlap",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::query,
                               query_alias, error); },
                     "must be disjoint");
    QsaNormRopeBuffers foreign_buffer = valid;
    foreign_buffer.input = foreign_input;
    passed &= reject("QSA norm/RoPE rejects foreign Context buffers",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               foreign_buffer, error); },
                     "not owned");
    QsaNormRopeBuffers forged_cpu = valid;
    forged_cpu.input.contents =
        static_cast<char*>(forged_cpu.input.contents) + sizeof(float);
    passed &= reject("QSA norm/RoPE rejects forged CPU metadata",
                     [&] { return encode_qsa_norm_rope(
                               context, *command, 1u, QsaNormRopeMode::key,
                               forged_cpu, error); },
                     "not owned by this Context");
    passed &= reject("QSA FWHT256 rejects zero rows",
                     [&] { return encode_qsa_fwht256(
                               context, *command, 0u, input, error); },
                     "row count must be positive");
    BufferRef short_fwht = input;
    short_fwht.length = 256u * sizeof(float) - sizeof(float);
    passed &= reject("QSA FWHT256 rejects a short input view",
                     [&] { return encode_qsa_fwht256(
                               context, *command, 1u, short_fwht, error); },
                     "undersized");
    BufferRef misaligned_fwht = input;
    misaligned_fwht = input.slice(1u, input.length - 1u);
    passed &= reject("QSA FWHT256 rejects a misaligned input",
                     [&] { return encode_qsa_fwht256(
                               context, *command, 1u, misaligned_fwht,
                               error); },
                     "misaligned");
    passed &= reject("QSA FWHT256 rejects a view past its Metal allocation",
                     [&] { return encode_qsa_fwht256(
                               context, *command, 1u,
                               out_of_allocation.input, error); },
                     "not owned by this Context");
    BufferRef forged_fwht_gpu = input;
    forged_fwht_gpu.gpu_address ^= 0x10u;
    passed &= reject("QSA FWHT256 rejects forged GPU metadata",
                     [&] { return encode_qsa_fwht256(
                               context, *command, 1u, forged_fwht_gpu,
                               error); },
                     "not owned by this Context");
    BufferRef forged_fwht_cpu = input;
    forged_fwht_cpu.contents =
        static_cast<char*>(forged_fwht_cpu.contents) + sizeof(float);
    passed &= reject("QSA FWHT256 rejects forged CPU metadata",
                     [&] { return encode_qsa_fwht256(
                               context, *command, 1u, forged_fwht_cpu,
                               error); },
                     "not owned by this Context");
    passed &= reject("QSA FWHT256 rejects foreign Context buffers",
                     [&] { return encode_qsa_fwht256(
                               context, *command, 1u, foreign_input, error); },
                     "not owned");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA foreign command buffer allocation", error);
        command->discard();
        return false;
    }
    passed &= reject("QSA norm/RoPE rejects a foreign command buffer",
                     [&] { return encode_qsa_norm_rope(
                               context, *foreign_command, 1u,
                               QsaNormRopeMode::key, valid, error); },
                     "different Context");
    passed &= reject("QSA FWHT256 rejects a foreign command buffer",
                     [&] { return encode_qsa_fwht256(
                               context, *foreign_command, 1u, input, error); },
                     "different Context");
    foreign_command->discard();

    auto submitted_command = context.make_command_buffer(
        "QSA submitted command buffer", error);
    if (!submitted_command) {
        check(false, "QSA submitted command buffer allocation", error);
        command->discard();
        return false;
    }
    if (!submitted_command->commit(error)) {
        check(false, "QSA submitted command buffer commit", error);
        command->discard();
        return false;
    }
    passed &= reject("QSA FWHT256 rejects a submitted command buffer",
                     [&] { return encode_qsa_fwht256(
                               context, *submitted_command, 1u, input,
                               error); },
                     "before command-buffer submission");
    void* submitted_handle = submitted_command->unsafe_native_handle();
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted_handle;
    if (submitted_native == nil) {
        check(false, "QSA submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "QSA submitted command buffer completion");
            passed = false;
        }
    }
    command->discard();
    return passed;
}

bool test_norm_rope_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open QSA norm/RoPE shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    constexpr std::size_t float_bytes = sizeof(float);
    constexpr std::size_t key_input_bytes =
        static_cast<std::size_t>(kMaxT) * N_HEAD_KV * HEAD_DIM * float_bytes;
    constexpr std::size_t query_input_bytes =
        static_cast<std::size_t>(kMaxT) * N_HEAD_Q * 2u * HEAD_DIM *
        float_bytes;
    constexpr std::size_t indexer_input_bytes =
        static_cast<std::size_t>(kMaxT) * INDEXER_HEADS *
        INDEXER_HEAD_DIM * float_bytes;
    constexpr std::size_t max_input_bytes =
        std::max({key_input_bytes, query_input_bytes, indexer_input_bytes});
    constexpr std::size_t key_output_bytes =
        static_cast<std::size_t>(kMaxT) * N_HEAD_KV * HEAD_DIM * float_bytes;
    constexpr std::size_t query_output_bytes =
        static_cast<std::size_t>(kMaxT) * N_HEAD_Q * HEAD_DIM * float_bytes;
    constexpr std::size_t indexer_output_bytes =
        static_cast<std::size_t>(kMaxT) * INDEXER_HEADS *
        INDEXER_HEAD_DIM * float_bytes;
    constexpr std::size_t max_output_bytes =
        std::max({key_output_bytes, query_output_bytes, indexer_output_bytes});
    constexpr std::size_t gamma_bytes =
        std::max(HEAD_DIM, INDEXER_HEAD_DIM) * float_bytes;
    constexpr std::size_t steps_bytes =
        static_cast<std::size_t>(kMaxT) * sizeof(StepRow);
    constexpr std::size_t rope_cell_bytes = 32u * 2u * float_bytes;
    constexpr std::size_t rope_bytes =
        kRopeCells * 32u * 2u * float_bytes;
    const auto allocate = [&](Context& owner, const std::string& label,
                              std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };

    BufferRef input_storage = allocate(
        context, "QSA norm/RoPE shared input",
        max_input_bytes + 2u * float_bytes);
    BufferRef gamma_storage = allocate(
        context, "QSA norm/RoPE shared gamma",
        gamma_bytes + 2u * float_bytes);
    BufferRef steps_storage = allocate(
        context, "QSA norm/RoPE shared steps",
        steps_bytes + 2u * sizeof(StepRow));
    BufferRef rope_storage = allocate(
        context, "QSA norm/RoPE shared table",
        rope_bytes + 2u * rope_cell_bytes);
    BufferRef output_storage = allocate(
        context, "QSA norm/RoPE shared output",
        max_output_bytes + 2u * float_bytes);
    BufferRef foreign_input = allocate(
        foreign_context, "QSA norm/RoPE shared foreign input",
        max_input_bytes);
    if (!input_storage || !gamma_storage || !steps_storage ||
        !rope_storage || !output_storage || !foreign_input) {
        check(false, "QSA norm/RoPE shared-validation allocations", error);
        return false;
    }

    const BufferRef input = input_storage.slice(0u, max_input_bytes);
    const BufferRef gamma = gamma_storage.slice(0u, gamma_bytes);
    const BufferRef steps = steps_storage.slice(0u, steps_bytes);
    const BufferRef rope = rope_storage.slice(0u, rope_bytes);
    const BufferRef output = output_storage.slice(0u, max_output_bytes);
    std::memset(input.contents, 0, input.length);
    std::memset(gamma.contents, 0, gamma.length);
    std::memset(rope.contents, 0, rope.length);
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    for (std::uint32_t row = 0u; row < kMaxT; ++row) {
        step_rows[row] = StepRow{0, 1, 0, 1};
    }

    const QsaNormRopeBuffers valid{input, gamma, steps, rope, output};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            QsaNormRopeMode mode,
                            const QsaNormRopeBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_qsa_norm_rope(
                        context, encoder, rows, mode, buffers, encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    const auto reject_prefill_rows =
        [&](const std::string& label, std::uint32_t rows,
            const std::string& expected_error) {
            error.clear();
            auto command = context.make_command_buffer(label, error);
            if (!command) {
                check(false, label + " command buffer", error);
                return false;
            }
            const bool failed = !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_prefill_qsa_norm_rope(
                        context, encoder, rows, QsaNormRopeMode::key, {},
                        encode_error);
                },
                error);
            const bool rejected =
                failed &&
                command->unsafe_native_handle() == nullptr &&
                error.find(expected_error) != std::string::npos;
            check(rejected, label, error);
            return rejected;
        };
    const auto reject_prefill_command_buffer =
        [&](const std::string& label, std::uint32_t rows,
            const std::string& expected_error) {
            error.clear();
            auto command = context.make_command_buffer(label, error);
            if (!command) {
                check(false, label + " command buffer", error);
                return false;
            }
            const bool failed = !encode_prefill_qsa_norm_rope(
                context, *command, rows, QsaNormRopeMode::key, {}, error);
            const bool rejected =
                failed && command->unsafe_native_handle() == nullptr &&
                error.find(expected_error) != std::string::npos;
            check(rejected, label, error);
            return rejected;
        };
    const auto accept_prefill_row_limit =
        [&] {
            constexpr std::string_view kExpectedBufferError =
                "norm input buffer is not owned by this Context";
            error.clear();
            auto command = context.make_command_buffer(
                "Prefill QSA norm/RoPE accepts shader-row limit", error);
            if (!command) {
                check(false, "Prefill QSA row-limit command buffer", error);
                return false;
            }
            const bool reached_buffer_validation =
                !encode_prefill_qsa_norm_rope(
                    context, *command, kPrefillMaxShaderRows,
                    QsaNormRopeMode::key, {}, error) &&
                command->unsafe_native_handle() == nullptr &&
                error.find(kExpectedBufferError) != std::string::npos;
            check(reached_buffer_validation,
                  "Prefill QSA accepts inclusive shader-row limit", error);
            return reached_buffer_validation;
        };

    bool passed = true;
    passed &= reject("QSA shared norm/RoPE rejects zero rows", 0u,
                     QsaNormRopeMode::key, {}, "row count");
    passed &= reject("QSA shared norm/RoPE rejects rows above eight", 9u,
                     QsaNormRopeMode::key, {}, "row count");
    passed &= reject_prefill_rows(
        "Prefill QSA norm/RoPE rejects zero rows", 0u,
        "must be positive");
    passed &= reject_prefill_rows(
        "Prefill QSA norm/RoPE rejects rows beyond shader limit",
        kPrefillMaxShaderRows + 1u, "shader-row limit");
    passed &= accept_prefill_row_limit();
    passed &= reject_prefill_command_buffer(
        "Prefill QSA standalone overload rolls back above row limit",
        kPrefillMaxShaderRows + 1u, "shader-row limit");
    passed &= reject("QSA shared norm/RoPE rejects invalid mode", 1u,
                     static_cast<QsaNormRopeMode>(9u), {}, "mode");

    QsaNormRopeBuffers short_view = valid;
    short_view.input.length = float_bytes;
    passed &= reject("QSA shared norm/RoPE rejects short input", 1u,
                     QsaNormRopeMode::key, short_view, "undersized");
    short_view = valid;
    short_view.gamma.length = float_bytes;
    passed &= reject("QSA shared norm/RoPE rejects short gamma", 1u,
                     QsaNormRopeMode::key, short_view, "undersized");
    short_view = valid;
    short_view.steps.length = sizeof(std::int32_t);
    passed &= reject("QSA shared norm/RoPE rejects short steps", 1u,
                     QsaNormRopeMode::key, short_view, "undersized");
    short_view = valid;
    short_view.output.length = float_bytes;
    passed &= reject("QSA shared norm/RoPE rejects short output", 1u,
                     QsaNormRopeMode::key, short_view, "undersized");
    short_view = valid;
    short_view.rope.length = 0u;
    passed &= reject("QSA shared norm/RoPE rejects a short RoPE table", 1u,
                     QsaNormRopeMode::key, short_view, "undersized");
    short_view = valid;
    short_view.rope.length -= sizeof(float);
    passed &= reject("QSA shared norm/RoPE rejects an incomplete RoPE row",
                     1u, QsaNormRopeMode::key, short_view,
                     "complete 32-pair rows");

    QsaNormRopeBuffers one_row_for_two = valid;
    one_row_for_two.input.length =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM * float_bytes;
    passed &= reject("QSA shared norm/RoPE rejects one-row query input for T=2",
                     2u, QsaNormRopeMode::query, one_row_for_two,
                     "undersized");
    one_row_for_two = valid;
    one_row_for_two.output.length =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM * float_bytes;
    passed &= reject(
        "QSA shared norm/RoPE rejects one-row query output for T=2", 2u,
        QsaNormRopeMode::query, one_row_for_two, "undersized");

    QsaNormRopeBuffers misaligned = valid;
    misaligned.input = input_storage.slice(1u, max_input_bytes);
    passed &= reject("QSA shared norm/RoPE rejects misaligned input", 1u,
                     QsaNormRopeMode::key, misaligned, "misaligned");
    misaligned = valid;
    misaligned.gamma =
        gamma_storage.slice(1u, gamma_bytes);
    passed &= reject("QSA shared norm/RoPE rejects misaligned gamma", 1u,
                     QsaNormRopeMode::key, misaligned, "misaligned");
    misaligned = valid;
    misaligned.steps = steps_storage.slice(1u, steps_bytes);
    passed &= reject("QSA shared norm/RoPE rejects misaligned steps", 1u,
                     QsaNormRopeMode::key, misaligned, "misaligned");
    misaligned = valid;
    misaligned.rope =
        rope_storage.slice(float_bytes, rope_bytes);
    passed &= reject("QSA shared norm/RoPE rejects misaligned RoPE table", 1u,
                     QsaNormRopeMode::key, misaligned, "misaligned");
    misaligned = valid;
    misaligned.output = output_storage.slice(1u, max_output_bytes);
    passed &= reject("QSA shared norm/RoPE rejects misaligned output", 1u,
                     QsaNormRopeMode::key, misaligned, "misaligned");

    QsaNormRopeBuffers invalid_position = valid;
    step_rows[0].pos = -1;
    passed &= reject("QSA shared norm/RoPE rejects negative positions", 1u,
                     QsaNormRopeMode::key, invalid_position,
                     "outside the RoPE table");
    step_rows[0].pos = static_cast<std::int32_t>(kRopeCells);
    passed &= reject("QSA shared norm/RoPE rejects positions past the table",
                     1u, QsaNormRopeMode::key, invalid_position,
                     "outside the RoPE table");
    step_rows[0].pos = 0;

    QsaNormRopeBuffers partial_in_place = valid;
    partial_in_place.output =
        input_storage.slice(float_bytes, key_input_bytes);
    passed &= reject("QSA shared norm/RoPE rejects partial in-place aliases",
                     1u, QsaNormRopeMode::key, partial_in_place,
                     "exact same view");
    QsaNormRopeBuffers query_in_place = valid;
    query_in_place.output = input;
    passed &= reject("QSA shared norm/RoPE rejects query in-place aliases",
                     1u, QsaNormRopeMode::query, query_in_place,
                     "must be disjoint");
    QsaNormRopeBuffers output_gamma_alias = valid;
    output_gamma_alias.gamma = output;
    passed &= reject("QSA shared norm/RoPE rejects output/gamma aliases", 1u,
                     QsaNormRopeMode::key, output_gamma_alias,
                     "overlap gamma");
    QsaNormRopeBuffers output_steps_alias = valid;
    output_steps_alias.steps = output;
    passed &= reject("QSA shared norm/RoPE rejects output/steps aliases", 1u,
                     QsaNormRopeMode::key, output_steps_alias,
                     "overlap gamma");
    QsaNormRopeBuffers output_rope_alias = valid;
    output_rope_alias.rope = output;
    passed &= reject("QSA shared norm/RoPE rejects output/RoPE aliases", 1u,
                     QsaNormRopeMode::key, output_rope_alias,
                     "overlap gamma");

    QsaNormRopeBuffers foreign = valid;
    foreign.input = foreign_input;
    passed &= reject("QSA shared norm/RoPE rejects foreign Context views", 1u,
                     QsaNormRopeMode::key, foreign, "not owned");
    QsaNormRopeBuffers out_of_range = valid;
    id<MTLBuffer> native_input =
        (__bridge id<MTLBuffer>)out_of_range.input.handle;
    const std::size_t overflowing_offset =
        native_input.length - key_input_bytes / 2u;
    out_of_range.input.offset = overflowing_offset;
    out_of_range.input.length = key_input_bytes;
    out_of_range.input.contents =
        static_cast<char*>(native_input.contents) + overflowing_offset;
    out_of_range.input.gpu_address = native_input.gpuAddress == 0u
        ? 0u
        : native_input.gpuAddress + overflowing_offset;
    passed &= reject("QSA shared norm/RoPE rejects out-of-range views", 1u,
                     QsaNormRopeMode::key, out_of_range, "not owned");
    QsaNormRopeBuffers forged_cpu = valid;
    forged_cpu.input.contents =
        static_cast<char*>(forged_cpu.input.contents) + float_bytes;
    passed &= reject("QSA shared norm/RoPE rejects forged CPU metadata", 1u,
                     QsaNormRopeMode::key, forged_cpu, "not owned");
    QsaNormRopeBuffers forged_gpu = valid;
    forged_gpu.input.gpu_address ^= 0x10u;
    passed &= reject("QSA shared norm/RoPE rejects forged GPU metadata", 1u,
                     QsaNormRopeMode::key, forged_gpu, "not owned");
    return passed;
}

bool test_fwht_shared_encoder_validation(Context& context,
                                        Context& foreign_context) {
    constexpr std::size_t row_bytes = 256u * sizeof(float);
    std::string error;
    BufferRef storage = context.buffers().allocate(
        "QSA FWHT shared validation", 2u * row_bytes + sizeof(float), error);
    BufferRef foreign = foreign_context.buffers().allocate(
        "QSA FWHT shared foreign view", row_bytes, error);
    if (!storage || !foreign) {
        check(false, "QSA FWHT shared-validation allocations", error);
        return false;
    }
    const BufferRef valid = storage.slice(0u, 2u * row_bytes);
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            BufferRef values,
                            const std::string& expected_error) {
        error.clear();
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool failed = !Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_qsa_fwht256(
                    context, encoder, rows, values, encode_error);
            },
            error);
        const bool rejected =
            failed && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    passed &= reject("QSA shared FWHT rejects zero rows", 0u, valid,
                     "row count must be positive");
    passed &= reject("QSA shared FWHT rejects a huge undersized view",
                     UINT32_MAX, valid, "undersized");
    BufferRef short_view = valid;
    short_view.length = row_bytes - sizeof(float);
    passed &= reject("QSA shared FWHT rejects a short row", 1u, short_view,
                     "undersized");
    const BufferRef misaligned = storage.slice(1u, row_bytes);
    passed &= reject("QSA shared FWHT rejects a misaligned view", 1u,
                     misaligned, "misaligned");
    passed &= reject("QSA shared FWHT rejects a foreign Context view", 1u,
                     foreign, "not owned");
    BufferRef forged_cpu = valid;
    forged_cpu.contents =
        static_cast<char*>(forged_cpu.contents) + sizeof(float);
    passed &= reject("QSA shared FWHT rejects forged CPU metadata", 1u,
                     forged_cpu, "not owned");
    BufferRef forged_gpu = valid;
    forged_gpu.gpu_address ^= 0x10u;
    passed &= reject("QSA shared FWHT rejects forged GPU metadata", 1u,
                     forged_gpu, "not owned");

    auto late_failure = context.make_command_buffer(
        "QSA shared FWHT late failure", error);
    if (!late_failure) {
        check(false, "QSA shared FWHT late-failure command buffer", error);
        return false;
    }
    const bool late_failure_propagated = !Encoder::encode(
        context, *late_failure, "QSA shared FWHT late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_fwht256(
                    context, encoder, 1u, valid, encode_error)) {
                return false;
            }
            return encode_qsa_fwht256(
                context, encoder, 1u, short_view, encode_error);
        },
        error) &&
        error.find("undersized") != std::string::npos;
    check(late_failure_propagated,
          "QSA shared FWHT propagates a late segment failure", error);
    passed &= late_failure_propagated;

    error.clear();
    const bool late_failure_discarded =
        !late_failure->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_failure_discarded,
          "QSA shared FWHT late failure discards earlier encoded work", error);
    passed &= late_failure_discarded;
    return passed;
}

bool test_kv_encoder_validation(Context& context, Context& foreign_context) {
    constexpr std::uint32_t kCellCapacity = 8u;
    constexpr std::size_t kCurrentBytes =
        N_HEAD_KV * HEAD_DIM * sizeof(float);
    constexpr std::size_t kPoolBytes =
        kCellCapacity * N_HEAD_KV * HEAD_DIM * sizeof(std::uint16_t);
    std::string error;
    const auto allocate = [&](Context& owner, const char* label,
                              std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef keys =
        allocate(context, "QSA fp16 KV validation keys", kPoolBytes);
    BufferRef values =
        allocate(context, "QSA fp16 KV validation values", kPoolBytes);
    BufferRef steps = allocate(context, "QSA fp16 KV validation steps",
                               2u * sizeof(StepRow));
    BufferRef key_pool =
        allocate(context, "QSA fp16 KV validation key pool", kPoolBytes);
    BufferRef value_pool =
        allocate(context, "QSA fp16 KV validation value pool", kPoolBytes);
    BufferRef foreign =
        allocate(foreign_context, "QSA fp16 KV foreign buffer", kPoolBytes);
    if (!keys || !values || !steps || !key_pool || !value_pool || !foreign) {
        check(false, "QSA fp16 KV validation allocations", error);
        return false;
    }
    std::memset(keys.contents, 0, keys.length);
    std::memset(values.contents, 0, values.length);
    std::memset(key_pool.contents, 0, key_pool.length);
    std::memset(value_pool.contents, 0, value_pool.length);
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    step_rows[0] = StepRow{0, 1, 0, 1};
    step_rows[1] = StepRow{1, 2, 0, 2};

    const QsaKvAppendF16Buffers valid{
        keys, values, steps, key_pool, value_pool};
    auto command = context.make_command_buffer("QSA fp16 KV validation",
                                               error);
    if (!command) {
        check(false, "QSA fp16 KV validation command buffer", error);
        return false;
    }
    bool passed = true;
    const auto reject = [&](const std::string& label, auto&& call,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !call() && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    const auto encode = [&](std::uint32_t rows, std::uint32_t capacity,
                            const QsaKvAppendF16Buffers& buffers) {
        return encode_qsa_kv_append_f16(
            context, *command, rows, capacity, buffers, error);
    };

    passed &= reject("QSA fp16 KV rejects zero rows",
                     [&] { return encode(0u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA fp16 KV rejects rows above eight",
                     [&] { return encode(9u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA fp16 KV rejects zero cell capacity",
                     [&] { return encode(1u, 0u, valid); }, "capacity");
    passed &= reject("QSA fp16 KV rejects a non-four-cell capacity",
                     [&] { return encode(1u, 7u, valid); }, "capacity");
    passed &= reject("QSA fp16 KV rejects an unaddressable capacity",
                     [&] { return encode(1u, 0x80000004u, valid); },
                     "capacity");

    QsaKvAppendF16Buffers short_keys = valid;
    short_keys.keys.length = kCurrentBytes - sizeof(float);
    passed &= reject("QSA fp16 KV rejects a short K input",
                     [&] { return encode(1u, kCellCapacity, short_keys); },
                     "undersized");
    QsaKvAppendF16Buffers short_pool = valid;
    short_pool.key_pool.length = kPoolBytes - sizeof(std::uint16_t);
    passed &= reject("QSA fp16 KV rejects a short pool",
                     [&] { return encode(1u, kCellCapacity, short_pool); },
                     "undersized");
    QsaKvAppendF16Buffers out_of_allocation = valid;
    id<MTLBuffer> native_pool =
        (__bridge id<MTLBuffer>)out_of_allocation.key_pool.handle;
    const std::size_t overflowing_offset =
        native_pool.length - kPoolBytes / 2u;
    out_of_allocation.key_pool.offset = overflowing_offset;
    out_of_allocation.key_pool.length = kPoolBytes;
    out_of_allocation.key_pool.contents =
        static_cast<char*>(native_pool.contents) + overflowing_offset;
    out_of_allocation.key_pool.gpu_address = native_pool.gpuAddress == 0u
        ? 0u
        : native_pool.gpuAddress + overflowing_offset;
    passed &= reject("QSA fp16 KV rejects a pool past its allocation",
                     [&] { return encode(1u, kCellCapacity,
                                         out_of_allocation); },
                     "not owned by this Context");

    step_rows[0].pos = -1;
    passed &= reject("QSA fp16 KV rejects a negative cell position",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = static_cast<std::int32_t>(kCellCapacity);
    passed &= reject("QSA fp16 KV rejects a cell past capacity",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = 0;
    step_rows[1].pos = 0;
    passed &= reject("QSA fp16 KV rejects duplicate cell positions",
                     [&] { return encode(2u, kCellCapacity, valid); },
                     "positions must be unique");
    step_rows[1].pos = 1;

    QsaKvAppendF16Buffers misaligned_pool = valid;
    misaligned_pool.key_pool =
        valid.key_pool.slice(1u, valid.key_pool.length - 1u);
    passed &= reject("QSA fp16 KV rejects a misaligned pool",
                     [&] { return encode(1u, kCellCapacity, misaligned_pool); },
                     "misaligned");

    QsaKvAppendF16Buffers pool_alias = valid;
    pool_alias.value_pool = pool_alias.key_pool;
    passed &= reject("QSA fp16 KV rejects aliased output pools",
                     [&] { return encode(1u, kCellCapacity, pool_alias); },
                     "must be disjoint");
    BufferRef overlapping_pool_backing = allocate(
        context, "QSA fp16 KV partial pool alias backing", 2u * kPoolBytes);
    if (!overlapping_pool_backing) {
        check(false, "QSA fp16 KV partial pool alias allocation", error);
        command->discard();
        return false;
    }
    QsaKvAppendF16Buffers partial_pool_alias = valid;
    partial_pool_alias.key_pool =
        overlapping_pool_backing.slice(0u, kPoolBytes);
    partial_pool_alias.value_pool = overlapping_pool_backing.slice(
        kPoolBytes - sizeof(std::uint16_t), kPoolBytes);
    passed &= reject("QSA fp16 KV rejects partial pool overlap",
                     [&] { return encode(1u, kCellCapacity,
                                         partial_pool_alias); },
                     "must be disjoint");

    QsaKvAppendF16Buffers input_alias = valid;
    input_alias.key_pool = input_alias.keys;
    passed &= reject("QSA fp16 KV rejects pool/input overlap",
                     [&] { return encode(1u, kCellCapacity, input_alias); },
                     "must not overlap");
    BufferRef partial_input_backing = allocate(
        context, "QSA fp16 KV partial input alias backing",
        kPoolBytes + 2u * kCurrentBytes);
    if (!partial_input_backing) {
        check(false, "QSA fp16 KV partial input alias allocation", error);
        command->discard();
        return false;
    }
    QsaKvAppendF16Buffers partial_input_alias = valid;
    partial_input_alias.key_pool =
        partial_input_backing.slice(0u, kPoolBytes);
    partial_input_alias.keys = partial_input_backing.slice(
        kPoolBytes - sizeof(float), kCurrentBytes);
    passed &= reject("QSA fp16 KV rejects partial pool/input overlap",
                     [&] { return encode(1u, kCellCapacity,
                                         partial_input_alias); },
                     "must not overlap");

    const long page_size_raw = getpagesize();
    if (page_size_raw <= 0) {
        check(false, "QSA fp16 KV query page size");
        command->discard();
        return false;
    }
    const std::size_t page_size =
        static_cast<std::size_t>(page_size_raw);
    const std::size_t no_copy_required =
        2u * kPoolBytes + 2u * kCurrentBytes;
    const std::size_t no_copy_bytes =
        ((no_copy_required + page_size - 1u) / page_size) * page_size;
    void* const mapped_backing = mmap(
        nullptr, no_copy_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
    if (mapped_backing == MAP_FAILED) {
        check(false, "QSA fp16 KV map cross-handle alias backing");
        command->discard();
        return false;
    }
    auto backing_lifetime = std::shared_ptr<void>(
        mapped_backing, [no_copy_bytes](void* address) {
            if (address != nullptr) munmap(address, no_copy_bytes);
        });
    BufferRef shared_pool_handle = context.buffers().wrap_no_copy(
        "QSA fp16 KV cross-handle pool", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    BufferRef shared_input_handle = context.buffers().wrap_no_copy(
        "QSA fp16 KV cross-handle input", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    if (!shared_pool_handle || !shared_input_handle) {
        check(false, "QSA fp16 KV wrap cross-handle alias backing", error);
        command->discard();
        return false;
    }
    QsaKvAppendF16Buffers cross_handle_alias = valid;
    cross_handle_alias.key_pool =
        shared_pool_handle.slice(0u, kPoolBytes);
    cross_handle_alias.keys = shared_input_handle.slice(
        kPoolBytes - sizeof(float), kCurrentBytes);
    passed &= reject("QSA fp16 KV rejects cross-handle shared backing overlap",
                     [&] { return encode(1u, kCellCapacity,
                                         cross_handle_alias); },
                     "must not overlap");

    QsaKvAppendF16Buffers forged_cpu = valid;
    forged_cpu.keys.contents =
        static_cast<char*>(forged_cpu.keys.contents) + sizeof(float);
    passed &= reject("QSA fp16 KV rejects forged CPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_cpu); },
                     "not owned by this Context");
    QsaKvAppendF16Buffers forged_gpu = valid;
    forged_gpu.keys.gpu_address ^= 0x10u;
    passed &= reject("QSA fp16 KV rejects forged GPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_gpu); },
                     "not owned by this Context");
    QsaKvAppendF16Buffers foreign_buffer = valid;
    foreign_buffer.keys = foreign;
    passed &= reject("QSA fp16 KV rejects foreign Context buffers",
                     [&] { return encode(1u, kCellCapacity, foreign_buffer); },
                     "not owned");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA fp16 KV foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA fp16 KV foreign command buffer allocation", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool foreign_rejected =
        !encode_qsa_kv_append_f16(context, *foreign_command, 1u,
                                  kCellCapacity, valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_rejected, "QSA fp16 KV rejects a foreign command buffer",
          error);
    passed &= foreign_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA fp16 KV submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA fp16 KV submitted command buffer setup", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_kv_append_f16(context, *submitted, 1u, kCellCapacity,
                                  valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA fp16 KV rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA fp16 KV submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "QSA fp16 KV submitted command buffer completion");
            passed = false;
        }
    }
    command->discard();
    return passed;
}

bool test_kv_i8_encoder_validation(Context& context,
                                   Context& foreign_context) {
    constexpr std::uint32_t kCellCapacity = 8u;
    constexpr std::size_t kCurrentBytes =
        N_HEAD_KV * HEAD_DIM * sizeof(float);
    constexpr std::size_t kCodeBytes =
        kCellCapacity * N_HEAD_KV * HEAD_DIM * sizeof(std::int8_t);
    constexpr std::size_t kScaleBytes =
        kCellCapacity * N_HEAD_KV * 4u * sizeof(std::uint16_t);
    std::string error;
    const auto allocate = [&](Context& owner, const char* label,
                              std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef keys =
        allocate(context, "QSA int8 KV validation keys", 2u * kCodeBytes);
    BufferRef values =
        allocate(context, "QSA int8 KV validation values", 2u * kCodeBytes);
    BufferRef steps = allocate(context, "QSA int8 KV validation steps",
                               2u * sizeof(StepRow));
    BufferRef key_codes =
        allocate(context, "QSA int8 KV validation key codes",
                 2u * kCodeBytes);
    BufferRef key_scales =
        allocate(context, "QSA int8 KV validation key scales", kScaleBytes);
    BufferRef value_codes =
        allocate(context, "QSA int8 KV validation value codes",
                 2u * kCodeBytes);
    BufferRef value_scales =
        allocate(context, "QSA int8 KV validation value scales",
                 kScaleBytes);
    BufferRef foreign =
        allocate(foreign_context, "QSA int8 KV foreign buffer", 2u * kCodeBytes);
    if (!keys || !values || !steps || !key_codes || !key_scales ||
        !value_codes || !value_scales || !foreign) {
        check(false, "QSA int8 KV validation allocations", error);
        return false;
    }
    std::memset(keys.contents, 0, keys.length);
    std::memset(values.contents, 0, values.length);
    std::memset(key_codes.contents, 0, key_codes.length);
    std::memset(value_codes.contents, 0, value_codes.length);
    std::memset(key_scales.contents, 0, key_scales.length);
    std::memset(value_scales.contents, 0, value_scales.length);
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    step_rows[0] = StepRow{0, 1, 0, 1};
    step_rows[1] = StepRow{1, 2, 0, 2};

    const QsaKvAppendI8Buffers valid{
        keys, values, steps, key_codes, key_scales, value_codes, value_scales};
    auto command = context.make_command_buffer("QSA int8 KV validation",
                                               error);
    if (!command) {
        check(false, "QSA int8 KV validation command buffer", error);
        return false;
    }
    bool passed = true;
    const auto reject = [&](const std::string& label, auto&& call,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !call() && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    const auto encode = [&](std::uint32_t rows, std::uint32_t capacity,
                            const QsaKvAppendI8Buffers& buffers) {
        return encode_qsa_kv_append_i8(
            context, *command, rows, capacity, buffers, error);
    };

    passed &= reject("QSA int8 KV rejects zero rows",
                     [&] { return encode(0u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA int8 KV rejects rows above eight",
                     [&] { return encode(9u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA int8 KV rejects zero capacity",
                     [&] { return encode(1u, 0u, valid); }, "capacity");
    passed &= reject("QSA int8 KV rejects a non-four-cell capacity",
                     [&] { return encode(1u, 7u, valid); }, "capacity");
    passed &= reject("QSA int8 KV rejects an unaddressable capacity",
                     [&] { return encode(1u, 0x80000004u, valid); },
                     "capacity");

    QsaKvAppendI8Buffers short_values = valid;
    short_values.values.length = kCurrentBytes - sizeof(float);
    passed &= reject("QSA int8 KV rejects a short V input",
                     [&] { return encode(1u, kCellCapacity, short_values); },
                     "undersized");
    QsaKvAppendI8Buffers short_scales = valid;
    short_scales.value_scales.length =
        kScaleBytes - sizeof(std::uint16_t);
    passed &= reject("QSA int8 KV rejects short scale storage",
                     [&] { return encode(1u, kCellCapacity, short_scales); },
                     "undersized");
    QsaKvAppendI8Buffers out_of_allocation = valid;
    id<MTLBuffer> native_scales =
        (__bridge id<MTLBuffer>)out_of_allocation.key_scales.handle;
    const std::size_t overflowing_offset =
        native_scales.length - kScaleBytes / 2u;
    out_of_allocation.key_scales.offset = overflowing_offset;
    out_of_allocation.key_scales.length = kScaleBytes;
    out_of_allocation.key_scales.contents =
        static_cast<char*>(native_scales.contents) + overflowing_offset;
    out_of_allocation.key_scales.gpu_address =
        native_scales.gpuAddress == 0u
            ? 0u
            : native_scales.gpuAddress + overflowing_offset;
    passed &= reject("QSA int8 KV rejects scales past their allocation",
                     [&] { return encode(1u, kCellCapacity,
                                         out_of_allocation); },
                     "not owned by this Context");

    QsaKvAppendI8Buffers misaligned_scales = valid;
    misaligned_scales.key_scales =
        valid.key_scales.slice(1u, valid.key_scales.length - 1u);
    passed &= reject("QSA int8 KV rejects misaligned scales",
                     [&] { return encode(1u, kCellCapacity,
                                         misaligned_scales); },
                     "misaligned");

    step_rows[0].pos = -1;
    passed &= reject("QSA int8 KV rejects a negative position",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = static_cast<std::int32_t>(kCellCapacity);
    passed &= reject("QSA int8 KV rejects a position past capacity",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = 0;
    step_rows[1].pos = 0;
    passed &= reject("QSA int8 KV rejects duplicate positions",
                     [&] { return encode(2u, kCellCapacity, valid); },
                     "positions must be unique");
    step_rows[1].pos = 1;

    QsaKvAppendI8Buffers output_alias = valid;
    output_alias.value_scales = output_alias.key_scales;
    passed &= reject("QSA int8 KV rejects aliased output arrays",
                     [&] { return encode(1u, kCellCapacity, output_alias); },
                     "output arrays must be disjoint");
    BufferRef shared_codes = allocate(
        context, "QSA int8 KV partial output alias backing",
        2u * kCodeBytes);
    if (!shared_codes) {
        check(false, "QSA int8 KV partial alias allocation", error);
        command->discard();
        return false;
    }
    QsaKvAppendI8Buffers partial_output_alias = valid;
    partial_output_alias.key_codes =
        shared_codes.slice(0u, kCodeBytes);
    partial_output_alias.value_codes =
        shared_codes.slice(kCodeBytes - 1u, kCodeBytes);
    passed &= reject("QSA int8 KV rejects partial output overlap",
                     [&] { return encode(1u, kCellCapacity,
                                         partial_output_alias); },
                     "output arrays must be disjoint");
    QsaKvAppendI8Buffers input_alias = valid;
    input_alias.key_codes = input_alias.keys;
    passed &= reject("QSA int8 KV rejects output/input overlap",
                     [&] { return encode(1u, kCellCapacity, input_alias); },
                     "must not overlap");

    QsaKvAppendI8Buffers forged_cpu = valid;
    forged_cpu.keys.contents =
        static_cast<char*>(forged_cpu.keys.contents) + sizeof(float);
    passed &= reject("QSA int8 KV rejects forged CPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_cpu); },
                     "not owned by this Context");
    QsaKvAppendI8Buffers forged_gpu = valid;
    forged_gpu.values.gpu_address ^= 0x10u;
    passed &= reject("QSA int8 KV rejects forged GPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_gpu); },
                     "not owned by this Context");
    QsaKvAppendI8Buffers foreign_buffer = valid;
    foreign_buffer.keys = foreign;
    passed &= reject("QSA int8 KV rejects foreign Context buffers",
                     [&] { return encode(1u, kCellCapacity, foreign_buffer); },
                     "not owned");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA int8 KV foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA int8 KV foreign command buffer allocation", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool foreign_rejected =
        !encode_qsa_kv_append_i8(context, *foreign_command, 1u,
                                 kCellCapacity, valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_rejected, "QSA int8 KV rejects a foreign command buffer",
          error);
    passed &= foreign_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA int8 KV submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA int8 KV submitted command buffer setup", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_kv_append_i8(context, *submitted, 1u, kCellCapacity,
                                 valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA int8 KV rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA int8 KV submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "QSA int8 KV submitted command buffer completion");
            passed = false;
        }
    }
    command->discard();
    return passed;
}

bool test_kv_q4_encoder_validation(Context& context,
                                   Context& foreign_context) {
    constexpr std::uint32_t kCellCapacity = 8u;
    constexpr std::size_t kCurrentBytes =
        N_HEAD_KV * HEAD_DIM * sizeof(float);
    constexpr std::size_t kPoolBytes =
        kCellCapacity * N_HEAD_KV * (HEAD_DIM / 32u) *
        sizeof(block_q4_0);
    std::string error;
    const auto allocate = [&](Context& owner, const char* label,
                              std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef keys =
        allocate(context, "QSA q4 KV validation keys", 2u * kPoolBytes);
    BufferRef values =
        allocate(context, "QSA q4 KV validation values", 2u * kPoolBytes);
    BufferRef steps = allocate(context, "QSA q4 KV validation steps",
                               2u * sizeof(StepRow));
    BufferRef key_pool =
        allocate(context, "QSA q4 KV validation key pool", 2u * kPoolBytes);
    BufferRef value_pool =
        allocate(context, "QSA q4 KV validation value pool", 2u * kPoolBytes);
    BufferRef foreign =
        allocate(foreign_context, "QSA q4 KV foreign buffer",
                 2u * kPoolBytes);
    if (!keys || !values || !steps || !key_pool || !value_pool || !foreign) {
        check(false, "QSA q4 KV validation allocations", error);
        return false;
    }
    std::memset(keys.contents, 0, keys.length);
    std::memset(values.contents, 0, values.length);
    std::memset(key_pool.contents, 0, key_pool.length);
    std::memset(value_pool.contents, 0, value_pool.length);
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    step_rows[0] = StepRow{0, 1, 0, 1};
    step_rows[1] = StepRow{1, 2, 0, 2};

    const QsaKvAppendQ4Buffers valid{
        keys, values, steps, key_pool, value_pool};
    auto command =
        context.make_command_buffer("QSA q4 KV validation", error);
    if (!command) {
        check(false, "QSA q4 KV validation command buffer", error);
        return false;
    }
    bool passed = true;
    const auto reject = [&](const std::string& label, auto&& call,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !call() && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    const auto encode = [&](std::uint32_t rows, std::uint32_t capacity,
                            const QsaKvAppendQ4Buffers& buffers) {
        return encode_qsa_kv_append_q4(
            context, *command, rows, capacity, buffers, error);
    };

    passed &= reject("QSA q4 KV rejects zero rows",
                     [&] { return encode(0u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA q4 KV rejects rows above eight",
                     [&] { return encode(9u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA q4 KV rejects zero capacity",
                     [&] { return encode(1u, 0u, valid); }, "capacity");
    passed &= reject("QSA q4 KV rejects a non-four-cell capacity",
                     [&] { return encode(1u, 7u, valid); }, "capacity");
    passed &= reject("QSA q4 KV rejects an unaddressable capacity",
                     [&] { return encode(1u, 0x80000004u, valid); },
                     "capacity");

    QsaKvAppendQ4Buffers short_values = valid;
    short_values.values.length = kCurrentBytes - sizeof(float);
    passed &= reject("QSA q4 KV rejects a short V input",
                     [&] { return encode(1u, kCellCapacity, short_values); },
                     "undersized");
    QsaKvAppendQ4Buffers short_pool = valid;
    short_pool.value_pool.length = kPoolBytes - sizeof(block_q4_0);
    passed &= reject("QSA q4 KV rejects a short pool",
                     [&] { return encode(1u, kCellCapacity, short_pool); },
                     "undersized");
    QsaKvAppendQ4Buffers out_of_allocation = valid;
    id<MTLBuffer> native_pool =
        (__bridge id<MTLBuffer>)out_of_allocation.key_pool.handle;
    const std::size_t overflowing_offset =
        native_pool.length - kPoolBytes / 2u;
    out_of_allocation.key_pool.offset = overflowing_offset;
    out_of_allocation.key_pool.length = kPoolBytes;
    out_of_allocation.key_pool.contents =
        static_cast<char*>(native_pool.contents) + overflowing_offset;
    out_of_allocation.key_pool.gpu_address =
        native_pool.gpuAddress == 0u
            ? 0u
            : native_pool.gpuAddress + overflowing_offset;
    passed &= reject("QSA q4 KV rejects a pool past its allocation",
                     [&] { return encode(1u, kCellCapacity,
                                         out_of_allocation); },
                     "not owned by this Context");
    QsaKvAppendQ4Buffers misaligned_pool = valid;
    misaligned_pool.key_pool =
        valid.key_pool.slice(1u, valid.key_pool.length - 1u);
    passed &= reject("QSA q4 KV rejects a misaligned pool",
                     [&] { return encode(1u, kCellCapacity, misaligned_pool); },
                     "misaligned");

    step_rows[0].pos = -1;
    passed &= reject("QSA q4 KV rejects a negative position",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = static_cast<std::int32_t>(kCellCapacity);
    passed &= reject("QSA q4 KV rejects a position past capacity",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = 0;
    step_rows[1].pos = 0;
    passed &= reject("QSA q4 KV rejects duplicate positions",
                     [&] { return encode(2u, kCellCapacity, valid); },
                     "positions must be unique");
    step_rows[1].pos = 1;

    QsaKvAppendQ4Buffers output_alias = valid;
    output_alias.value_pool = output_alias.key_pool;
    passed &= reject("QSA q4 KV rejects aliased output pools",
                     [&] { return encode(1u, kCellCapacity, output_alias); },
                     "must be disjoint");
    BufferRef shared_pool =
        allocate(context, "QSA q4 KV partial output alias backing",
                 2u * kPoolBytes);
    if (!shared_pool) {
        check(false, "QSA q4 KV partial alias allocation", error);
        command->discard();
        return false;
    }
    QsaKvAppendQ4Buffers partial_output_alias = valid;
    partial_output_alias.key_pool =
        buffer_subview(shared_pool, 0u, kPoolBytes);
    partial_output_alias.value_pool =
        buffer_subview(shared_pool, kPoolBytes - alignof(block_q4_0),
                       kPoolBytes);
    passed &= reject("QSA q4 KV rejects partial pool overlap",
                     [&] { return encode(1u, kCellCapacity,
                                         partial_output_alias); },
                     "must be disjoint");
    QsaKvAppendQ4Buffers input_alias = valid;
    input_alias.key_pool = input_alias.keys;
    passed &= reject("QSA q4 KV rejects output/input overlap",
                     [&] { return encode(1u, kCellCapacity, input_alias); },
                     "must not overlap");
    QsaKvAppendQ4Buffers partial_input_alias = valid;
    partial_input_alias.key_pool =
        buffer_subview(partial_input_alias.keys,
                       kCurrentBytes - alignof(block_q4_0), kPoolBytes);
    passed &= reject("QSA q4 KV rejects partial output/input overlap",
                     [&] { return encode(1u, kCellCapacity,
                                         partial_input_alias); },
                     "must not overlap");

    const long page_size_raw = getpagesize();
    if (page_size_raw <= 0) {
        check(false, "QSA q4 KV query page size");
        command->discard();
        return false;
    }
    const std::size_t page_size =
        static_cast<std::size_t>(page_size_raw);
    const std::size_t no_copy_required =
        2u * kPoolBytes + 2u * kCurrentBytes;
    const std::size_t no_copy_bytes =
        ((no_copy_required + page_size - 1u) / page_size) * page_size;
    void* const mapped_backing = mmap(
        nullptr, no_copy_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
    if (mapped_backing == MAP_FAILED) {
        check(false, "QSA q4 KV map cross-handle alias backing");
        command->discard();
        return false;
    }
    auto backing_lifetime = std::shared_ptr<void>(
        mapped_backing, [no_copy_bytes](void* address) {
            if (address != nullptr) munmap(address, no_copy_bytes);
        });
    BufferRef shared_pool_handle = context.buffers().wrap_no_copy(
        "QSA q4 KV cross-handle pool", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    BufferRef shared_value_handle = context.buffers().wrap_no_copy(
        "QSA q4 KV cross-handle value", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    BufferRef shared_input_handle = context.buffers().wrap_no_copy(
        "QSA q4 KV cross-handle input", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    if (!shared_pool_handle || !shared_value_handle ||
        !shared_input_handle) {
        check(false, "QSA q4 KV wrap cross-handle alias backing", error);
        command->discard();
        return false;
    }
    const bool distinct_no_copy_handles =
        shared_pool_handle.handle != shared_value_handle.handle &&
        shared_pool_handle.handle != shared_input_handle.handle &&
        shared_value_handle.handle != shared_input_handle.handle;
    check(distinct_no_copy_handles,
          "QSA q4 KV no-copy aliases use distinct Metal handles");
    passed &= distinct_no_copy_handles;

    QsaKvAppendQ4Buffers cross_handle_output_alias = valid;
    cross_handle_output_alias.key_pool =
        buffer_subview(shared_pool_handle, 0u, kPoolBytes);
    cross_handle_output_alias.value_pool =
        buffer_subview(shared_value_handle,
                       kPoolBytes - alignof(block_q4_0), kPoolBytes);
    passed &= reject(
        "QSA q4 KV rejects cross-handle shared backing output overlap",
        [&] { return encode(1u, kCellCapacity, cross_handle_output_alias); },
        "must be disjoint");

    QsaKvAppendQ4Buffers cross_handle_input_alias = valid;
    cross_handle_input_alias.key_pool =
        buffer_subview(shared_pool_handle, 0u, kPoolBytes);
    cross_handle_input_alias.keys =
        buffer_subview(shared_input_handle,
                       kPoolBytes - sizeof(float), kCurrentBytes);
    passed &= reject(
        "QSA q4 KV rejects cross-handle shared backing input overlap",
        [&] { return encode(1u, kCellCapacity, cross_handle_input_alias); },
        "must not overlap");

    QsaKvAppendQ4Buffers forged_cpu = valid;
    forged_cpu.keys.contents =
        static_cast<char*>(forged_cpu.keys.contents) + sizeof(float);
    passed &= reject("QSA q4 KV rejects forged CPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_cpu); },
                     "not owned by this Context");
    QsaKvAppendQ4Buffers forged_gpu = valid;
    forged_gpu.values.gpu_address ^= 0x10u;
    passed &= reject("QSA q4 KV rejects forged GPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_gpu); },
                     "not owned by this Context");
    QsaKvAppendQ4Buffers foreign_buffer = valid;
    foreign_buffer.keys = foreign;
    passed &= reject("QSA q4 KV rejects foreign Context buffers",
                     [&] { return encode(1u, kCellCapacity, foreign_buffer); },
                     "not owned");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA q4 KV foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA q4 KV foreign command buffer allocation", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool foreign_rejected =
        !encode_qsa_kv_append_q4(context, *foreign_command, 1u,
                                 kCellCapacity, valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_rejected, "QSA q4 KV rejects a foreign command buffer",
          error);
    passed &= foreign_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA q4 KV submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA q4 KV submitted command buffer setup", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_kv_append_q4(context, *submitted, 1u, kCellCapacity,
                                 valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA q4 KV rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA q4 KV submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "QSA q4 KV submitted command buffer completion");
            passed = false;
        }
    }
    command->discard();
    return passed;
}

bool test_kv_k8v4_encoder_validation(Context& context,
                                     Context& foreign_context) {
    constexpr std::uint32_t kCellCapacity = 8u;
    constexpr std::size_t kCurrentBytes =
        N_HEAD_KV * HEAD_DIM * sizeof(float);
    constexpr std::size_t kCodeBytes =
        kCellCapacity * N_HEAD_KV * HEAD_DIM * sizeof(std::int8_t);
    constexpr std::size_t kScaleBytes =
        kCellCapacity * N_HEAD_KV * 4u * sizeof(std::uint16_t);
    constexpr std::size_t kValuePoolBytes =
        kCellCapacity * N_HEAD_KV * (HEAD_DIM / 32u) *
        sizeof(block_q4_0);
    constexpr std::size_t kOutputBytes =
        kCodeBytes + kScaleBytes + kValuePoolBytes;
    std::string error;
    const auto allocate = [&](Context& owner, const char* label,
                              std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef keys =
        allocate(context, "QSA k8v4 validation keys", 2u * kCurrentBytes);
    BufferRef values =
        allocate(context, "QSA k8v4 validation values", 2u * kCurrentBytes);
    BufferRef steps = allocate(context, "QSA k8v4 validation steps",
                               2u * sizeof(StepRow));
    BufferRef output =
        allocate(context, "QSA k8v4 validation output", 2u * kOutputBytes);
    BufferRef foreign =
        allocate(foreign_context, "QSA k8v4 foreign buffer",
                 2u * kCurrentBytes);
    if (!keys || !values || !steps || !output || !foreign) {
        check(false, "QSA k8v4 validation allocations", error);
        return false;
    }
    std::memset(keys.contents, 0, keys.length);
    std::memset(values.contents, 0, values.length);
    std::memset(output.contents, 0, output.length);
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    step_rows[0] = StepRow{0, 1, 0, 1};
    step_rows[1] = StepRow{1, 2, 0, 2};

    const QsaKvAppendK8V4Buffers valid{
        keys,
        values,
        steps,
        buffer_subview(output, 0u, kCodeBytes),
        buffer_subview(output, kCodeBytes, kScaleBytes),
        buffer_subview(output, kCodeBytes + kScaleBytes,
                       kValuePoolBytes),
    };
    auto command =
        context.make_command_buffer("QSA k8v4 KV validation", error);
    if (!command) {
        check(false, "QSA k8v4 KV validation command buffer", error);
        return false;
    }
    bool passed = true;
    const auto reject = [&](const std::string& label, auto&& call,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !call() && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    const auto encode = [&](std::uint32_t rows, std::uint32_t capacity,
                            const QsaKvAppendK8V4Buffers& buffers) {
        return encode_qsa_kv_append_k8v4(
            context, *command, rows, capacity, buffers, error);
    };

    passed &= reject("QSA k8v4 rejects zero rows",
                     [&] { return encode(0u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA k8v4 rejects rows above eight",
                     [&] { return encode(9u, kCellCapacity, valid); },
                     "row count");
    passed &= reject("QSA k8v4 rejects zero capacity",
                     [&] { return encode(1u, 0u, valid); }, "capacity");
    passed &= reject("QSA k8v4 rejects a non-four-cell capacity",
                     [&] { return encode(1u, 7u, valid); }, "capacity");
    passed &= reject("QSA k8v4 rejects an unaddressable capacity",
                     [&] { return encode(1u, 0x80000004u, valid); },
                     "capacity");

    QsaKvAppendK8V4Buffers short_values = valid;
    short_values.values.length = kCurrentBytes - sizeof(float);
    passed &= reject("QSA k8v4 rejects a short V input",
                     [&] { return encode(1u, kCellCapacity, short_values); },
                     "undersized");
    QsaKvAppendK8V4Buffers short_scales = valid;
    short_scales.key_scales.length = kScaleBytes - sizeof(std::uint16_t);
    passed &= reject("QSA k8v4 rejects short K scale storage",
                     [&] { return encode(1u, kCellCapacity, short_scales); },
                     "undersized");
    QsaKvAppendK8V4Buffers short_pool = valid;
    short_pool.value_pool.length =
        kValuePoolBytes - sizeof(block_q4_0);
    passed &= reject("QSA k8v4 rejects short V q4 storage",
                     [&] { return encode(1u, kCellCapacity, short_pool); },
                     "undersized");
    QsaKvAppendK8V4Buffers out_of_allocation = valid;
    id<MTLBuffer> native_pool =
        (__bridge id<MTLBuffer>)out_of_allocation.value_pool.handle;
    const std::size_t overflowing_offset =
        native_pool.length - kValuePoolBytes / 2u;
    out_of_allocation.value_pool.offset = overflowing_offset;
    out_of_allocation.value_pool.length = kValuePoolBytes;
    out_of_allocation.value_pool.contents =
        static_cast<char*>(native_pool.contents) + overflowing_offset;
    out_of_allocation.value_pool.gpu_address =
        native_pool.gpuAddress == 0u
            ? 0u
            : native_pool.gpuAddress + overflowing_offset;
    passed &= reject("QSA k8v4 rejects V q4 storage past its allocation",
                     [&] { return encode(1u, kCellCapacity,
                                         out_of_allocation); },
                     "not owned by this Context");
    QsaKvAppendK8V4Buffers misaligned_scales = valid;
    misaligned_scales.key_scales =
        valid.key_scales.slice(1u, valid.key_scales.length - 1u);
    passed &= reject("QSA k8v4 rejects misaligned K scales",
                     [&] { return encode(1u, kCellCapacity,
                                         misaligned_scales); },
                     "misaligned");

    step_rows[0].pos = -1;
    passed &= reject("QSA k8v4 rejects a negative position",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = static_cast<std::int32_t>(kCellCapacity);
    passed &= reject("QSA k8v4 rejects a position past capacity",
                     [&] { return encode(1u, kCellCapacity, valid); },
                     "outside the cell capacity");
    step_rows[0].pos = 0;
    step_rows[1].pos = 0;
    passed &= reject("QSA k8v4 rejects duplicate positions",
                     [&] { return encode(2u, kCellCapacity, valid); },
                     "positions must be unique");
    step_rows[1].pos = 1;

    QsaKvAppendK8V4Buffers output_alias = valid;
    output_alias.key_scales = output_alias.key_codes;
    passed &= reject("QSA k8v4 rejects aliased output arrays",
                     [&] { return encode(1u, kCellCapacity, output_alias); },
                     "output arrays must be disjoint");
    BufferRef partial_backing = allocate(
        context, "QSA k8v4 partial output overlap", kOutputBytes);
    if (!partial_backing) {
        check(false, "QSA k8v4 partial alias allocation", error);
        command->discard();
        return false;
    }
    QsaKvAppendK8V4Buffers partial_output_alias = valid;
    partial_output_alias.key_codes =
        buffer_subview(partial_backing, 0u, kCodeBytes);
    partial_output_alias.value_pool = buffer_subview(
        partial_backing, kCodeBytes - sizeof(std::uint16_t),
        kValuePoolBytes);
    passed &= reject("QSA k8v4 rejects partial output overlap",
                     [&] { return encode(1u, kCellCapacity,
                                         partial_output_alias); },
                     "output arrays must be disjoint");
    QsaKvAppendK8V4Buffers input_alias = valid;
    input_alias.key_codes = input_alias.keys;
    passed &= reject("QSA k8v4 rejects output/input overlap",
                     [&] { return encode(1u, kCellCapacity, input_alias); },
                     "must not overlap");
    QsaKvAppendK8V4Buffers key_value_alias = valid;
    key_value_alias.values = key_value_alias.keys;
    passed &= reject("QSA k8v4 rejects K/V input overlap before V FWHT",
                     [&] { return encode(1u, kCellCapacity, key_value_alias); },
                     "input arrays must be disjoint");

    QsaKvAppendK8V4Buffers forged_cpu = valid;
    forged_cpu.keys.contents =
        static_cast<char*>(forged_cpu.keys.contents) + sizeof(float);
    passed &= reject("QSA k8v4 rejects forged CPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_cpu); },
                     "not owned by this Context");
    QsaKvAppendK8V4Buffers forged_gpu = valid;
    forged_gpu.values.gpu_address ^= 0x10u;
    passed &= reject("QSA k8v4 rejects forged GPU metadata",
                     [&] { return encode(1u, kCellCapacity, forged_gpu); },
                     "not owned by this Context");
    QsaKvAppendK8V4Buffers foreign_buffer = valid;
    foreign_buffer.keys = foreign;
    passed &= reject("QSA k8v4 rejects foreign Context buffers",
                     [&] { return encode(1u, kCellCapacity, foreign_buffer); },
                     "not owned");

    const long page_size_raw = getpagesize();
    if (page_size_raw <= 0) {
        check(false, "QSA k8v4 query page size");
        command->discard();
        return false;
    }
    const std::size_t page_size =
        static_cast<std::size_t>(page_size_raw);
    const std::size_t mapped_bytes =
        ((kOutputBytes + kCurrentBytes + page_size - 1u) / page_size) *
        page_size;
    void* const mapped_backing = mmap(
        nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
    if (mapped_backing == MAP_FAILED) {
        check(false, "QSA k8v4 map cross-handle alias backing");
        command->discard();
        return false;
    }
    auto backing_lifetime = std::shared_ptr<void>(
        mapped_backing, [mapped_bytes](void* address) {
            if (address != nullptr) munmap(address, mapped_bytes);
        });
    BufferRef shared_output_handle = context.buffers().wrap_no_copy(
        "QSA k8v4 cross-handle output", mapped_backing, mapped_bytes,
        backing_lifetime, error);
    BufferRef shared_input_handle = context.buffers().wrap_no_copy(
        "QSA k8v4 cross-handle input", mapped_backing, mapped_bytes,
        backing_lifetime, error);
    if (!shared_output_handle || !shared_input_handle) {
        check(false, "QSA k8v4 wrap cross-handle aliases", error);
        command->discard();
        return false;
    }
    const bool distinct_handles =
        shared_output_handle.handle != shared_input_handle.handle;
    check(distinct_handles,
          "QSA k8v4 shared backing uses distinct Metal handles");
    passed &= distinct_handles;

    QsaKvAppendK8V4Buffers cross_handle_output_alias = valid;
    cross_handle_output_alias.key_codes =
        buffer_subview(shared_output_handle, 0u, kCodeBytes);
    cross_handle_output_alias.value_pool = buffer_subview(
        shared_input_handle, kCodeBytes - sizeof(std::uint16_t),
        kValuePoolBytes);
    passed &= reject(
        "QSA k8v4 rejects cross-handle output overlap",
        [&] { return encode(1u, kCellCapacity, cross_handle_output_alias); },
        "output arrays must be disjoint");
    QsaKvAppendK8V4Buffers cross_handle_input_alias = valid;
    cross_handle_input_alias.key_codes =
        buffer_subview(shared_output_handle, 0u, kCodeBytes);
    cross_handle_input_alias.keys =
        buffer_subview(shared_input_handle, 0u, kCurrentBytes);
    passed &= reject(
        "QSA k8v4 rejects cross-handle input/output overlap",
        [&] { return encode(1u, kCellCapacity, cross_handle_input_alias); },
        "must not overlap");
    QsaKvAppendK8V4Buffers cross_handle_key_value_alias = valid;
    cross_handle_key_value_alias.keys =
        buffer_subview(shared_output_handle, 0u, kCurrentBytes);
    cross_handle_key_value_alias.values =
        buffer_subview(shared_input_handle, 0u, kCurrentBytes);
    passed &= reject(
        "QSA k8v4 rejects cross-handle K/V alias before V FWHT",
        [&] { return encode(1u, kCellCapacity,
                            cross_handle_key_value_alias); },
        "input arrays must be disjoint");
    QsaKvAppendK8V4Buffers cross_handle_value_steps_alias = valid;
    cross_handle_value_steps_alias.values =
        buffer_subview(shared_input_handle, 0u, kCurrentBytes);
    cross_handle_value_steps_alias.steps =
        buffer_subview(shared_output_handle, 0u, sizeof(StepRow));
    passed &= reject(
        "QSA k8v4 rejects cross-handle V/steps alias before V FWHT",
        [&] { return encode(1u, kCellCapacity,
                            cross_handle_value_steps_alias); },
        "input arrays must be disjoint");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA k8v4 foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA k8v4 foreign command buffer allocation", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool foreign_rejected =
        !encode_qsa_kv_append_k8v4(context, *foreign_command, 1u,
                                   kCellCapacity, valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_rejected, "QSA k8v4 rejects a foreign command buffer",
          error);
    passed &= foreign_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA k8v4 submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA k8v4 submitted command buffer setup", error);
        command->discard();
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_kv_append_k8v4(context, *submitted, 1u, kCellCapacity,
                                   valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA k8v4 rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA k8v4 submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "QSA k8v4 submitted command buffer completion");
            passed = false;
        }
    }
    command->discard();
    return passed;
}

bool test_prefill_kv_append_row_bounds(Context& context) {
    bool passed = true;
    const auto check_format =
        [&](const std::string& format, std::string_view input_error,
            auto&& command_call, auto&& encoder_call) {
            const auto check_limit =
                [&](std::uint32_t rows, std::string_view expected_error,
                    bool shared) {
                    const std::string label =
                        "Prefill QSA " + format + " KV append " +
                        (rows > kPrefillMaxShaderRows
                             ? "rejects shader-row overflow"
                             : "accepts inclusive shader-row limit") +
                        (shared ? " through Encoder" : " standalone");
                    std::string error;
                    auto command = context.make_command_buffer(label, error);
                    if (!command) {
                        check(false, label + " command buffer", error);
                        return false;
                    }
                    bool rejected = false;
                    if (shared) {
                        rejected = !Encoder::encode(
                            context, *command, label,
                            [&](Encoder& encoder,
                                std::string& encode_error) {
                                return encoder_call(
                                    encoder, rows, encode_error);
                            },
                            error);
                    } else {
                        rejected = !command_call(*command, rows, error);
                    }
                    rejected = rejected &&
                        (!shared ||
                         command->unsafe_native_handle() == nullptr) &&
                        error.find(expected_error) != std::string::npos;
                    check(rejected, label, error);
                    if (command->unsafe_native_handle() != nullptr) {
                        command->discard();
                    }
                    return rejected;
                };
            passed &= check_limit(
                kPrefillMaxShaderRows + 1u, "shader-row limit", false);
            passed &= check_limit(
                kPrefillMaxShaderRows + 1u, "shader-row limit", true);
            passed &= check_limit(kPrefillMaxShaderRows, input_error, false);
            passed &= check_limit(kPrefillMaxShaderRows, input_error, true);
        };

    check_format(
        "fp16", "fp16 KV key input buffer is not owned by this Context",
        [&](CommandBuffer& command, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_f16(
                context, command, rows, 8u, QsaKvAppendF16Buffers{}, error);
        },
        [&](Encoder& encoder, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_f16(
                context, encoder, rows, 8u, QsaKvAppendF16Buffers{}, error);
        });
    check_format(
        "int8", "int8 KV key input buffer is not owned by this Context",
        [&](CommandBuffer& command, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_i8(
                context, command, rows, 8u, QsaKvAppendI8Buffers{}, error);
        },
        [&](Encoder& encoder, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_i8(
                context, encoder, rows, 8u, QsaKvAppendI8Buffers{}, error);
        });
    check_format(
        "q4", "q4 KV key input buffer is not owned by this Context",
        [&](CommandBuffer& command, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_q4(
                context, command, rows, 8u, QsaKvAppendQ4Buffers{}, error);
        },
        [&](Encoder& encoder, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_q4(
                context, encoder, rows, 8u, QsaKvAppendQ4Buffers{}, error);
        });
    check_format(
        "k8v4", "k8v4 KV key input buffer is not owned by this Context",
        [&](CommandBuffer& command, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_k8v4(
                context, command, rows, 8u, QsaKvAppendK8V4Buffers{}, error);
        },
        [&](Encoder& encoder, std::uint32_t rows, std::string& error) {
            return encode_prefill_qsa_kv_append_k8v4(
                context, encoder, rows, 8u, QsaKvAppendK8V4Buffers{}, error);
        });

    constexpr std::uint32_t kRows = VERIFY_MAX_T + 1u;
    constexpr std::uint32_t kCellCapacity = 32u;
    constexpr std::size_t kCurrentBytes =
        static_cast<std::size_t>(kRows) * N_HEAD_KV * HEAD_DIM *
        sizeof(float);
    constexpr std::size_t kF16PoolBytes =
        kCellCapacity * N_HEAD_KV * HEAD_DIM * sizeof(std::uint16_t);
    constexpr std::size_t kI8CodeBytes =
        kCellCapacity * N_HEAD_KV * HEAD_DIM * sizeof(std::int8_t);
    constexpr std::size_t kI8ScaleBytes =
        kCellCapacity * N_HEAD_KV * 4u * sizeof(std::uint16_t);
    constexpr std::size_t kQ4PoolBytes =
        kCellCapacity * N_HEAD_KV * (HEAD_DIM / 32u) *
        sizeof(block_q4_0);
    std::string error;
    const auto allocate = [&](const char* name, std::size_t bytes) {
        BufferRef buffer = context.buffers().allocate(name, bytes, error);
        if (!buffer) {
            check(false, std::string(name) + " allocation", error);
        } else {
            std::memset(buffer.contents, 0, buffer.length);
        }
        return buffer;
    };
    BufferRef keys = allocate("Prefill QSA duplicate-test K", kCurrentBytes);
    BufferRef values = allocate("Prefill QSA duplicate-test V", kCurrentBytes);
    BufferRef steps = allocate(
        "Prefill QSA duplicate-test steps", kRows * sizeof(StepRow));
    BufferRef f16_key_pool =
        allocate("Prefill QSA duplicate-test fp16 K", kF16PoolBytes);
    BufferRef f16_value_pool =
        allocate("Prefill QSA duplicate-test fp16 V", kF16PoolBytes);
    BufferRef i8_key_codes =
        allocate("Prefill QSA duplicate-test int8 K codes", kI8CodeBytes);
    BufferRef i8_key_scales =
        allocate("Prefill QSA duplicate-test int8 K scales", kI8ScaleBytes);
    BufferRef i8_value_codes =
        allocate("Prefill QSA duplicate-test int8 V codes", kI8CodeBytes);
    BufferRef i8_value_scales =
        allocate("Prefill QSA duplicate-test int8 V scales", kI8ScaleBytes);
    BufferRef q4_key_pool =
        allocate("Prefill QSA duplicate-test q4 K", kQ4PoolBytes);
    BufferRef q4_value_pool =
        allocate("Prefill QSA duplicate-test q4 V", kQ4PoolBytes);
    BufferRef k8v4_key_codes =
        allocate("Prefill QSA duplicate-test k8v4 K codes", kI8CodeBytes);
    BufferRef k8v4_key_scales =
        allocate("Prefill QSA duplicate-test k8v4 K scales", kI8ScaleBytes);
    BufferRef k8v4_value_pool =
        allocate("Prefill QSA duplicate-test k8v4 V", kQ4PoolBytes);
    const std::array required_buffers{
        keys, values, steps, f16_key_pool, f16_value_pool, i8_key_codes,
        i8_key_scales, i8_value_codes, i8_value_scales, q4_key_pool,
        q4_value_pool, k8v4_key_codes, k8v4_key_scales, k8v4_value_pool};
    if (std::any_of(required_buffers.begin(), required_buffers.end(),
                    [](BufferRef buffer) { return !buffer; })) {
        return false;
    }
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    for (std::uint32_t row = 0u; row < kRows; ++row) {
        const std::int32_t position =
            static_cast<std::int32_t>(row == kRows - 1u ? row - 1u : row);
        step_rows[row] = StepRow{position, position + 1,
                                 (position + 1) / 4, position + 1};
    }

    const auto check_duplicate = [&](const std::string& format,
                                     auto&& command_call,
                                     auto&& encoder_call) {
        const auto check_form = [&](bool shared) {
            const std::string label =
                "Prefill QSA " + format +
                " rejects duplicate positions above eight rows" +
                (shared ? " through Encoder" : " standalone");
            error.clear();
            auto command = context.make_command_buffer(label, error);
            if (!command) {
                check(false, label + " command buffer", error);
                return false;
            }
            const bool failed = shared
                ? !Encoder::encode(
                      context, *command, label,
                      [&](Encoder& encoder, std::string& encode_error) {
                          return encoder_call(encoder, encode_error);
                      },
                      error)
                : !command_call(*command, error);
            const bool rejected =
                failed &&
                (!shared ||
                 command->unsafe_native_handle() == nullptr) &&
                error.find("positions must be unique") != std::string::npos;
            check(rejected, label, error);
            if (command->unsafe_native_handle() != nullptr) {
                command->discard();
            }
            return rejected;
        };
        passed &= check_form(false);
        passed &= check_form(true);
    };

    const QsaKvAppendF16Buffers f16_buffers{
        keys, values, steps, f16_key_pool, f16_value_pool};
    const QsaKvAppendI8Buffers i8_buffers{
        keys, values, steps, i8_key_codes, i8_key_scales,
        i8_value_codes, i8_value_scales};
    const QsaKvAppendQ4Buffers q4_buffers{
        keys, values, steps, q4_key_pool, q4_value_pool};
    const QsaKvAppendK8V4Buffers k8v4_buffers{
        keys, values, steps, k8v4_key_codes, k8v4_key_scales,
        k8v4_value_pool};
    check_duplicate(
        "fp16",
        [&](CommandBuffer& command, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_f16(
                context, command, kRows, kCellCapacity, f16_buffers,
                encode_error);
        },
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_f16(
                context, encoder, kRows, kCellCapacity, f16_buffers,
                encode_error);
        });
    check_duplicate(
        "int8",
        [&](CommandBuffer& command, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_i8(
                context, command, kRows, kCellCapacity, i8_buffers,
                encode_error);
        },
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_i8(
                context, encoder, kRows, kCellCapacity, i8_buffers,
                encode_error);
        });
    check_duplicate(
        "q4",
        [&](CommandBuffer& command, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_q4(
                context, command, kRows, kCellCapacity, q4_buffers,
                encode_error);
        },
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_q4(
                context, encoder, kRows, kCellCapacity, q4_buffers,
                encode_error);
        });
    check_duplicate(
        "k8v4",
        [&](CommandBuffer& command, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_k8v4(
                context, command, kRows, kCellCapacity, k8v4_buffers,
                encode_error);
        },
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_kv_append_k8v4(
                context, encoder, kRows, kCellCapacity, k8v4_buffers,
                encode_error);
        });
    return passed;
}

bool run_set(Context& context, twin::InputSet set,
             const twin::Options& options) {
    bool passed = true;
    constexpr std::array modes{
        QsaNormRopeMode::key, QsaNormRopeMode::query,
        QsaNormRopeMode::indexer_query};
    for (const auto mode : modes) {
        passed &= run_mode_set(context, mode, set, options,
                               set == twin::InputSet::edge);
    }
    return passed;
}

float kv_sample(twin::InputSet set, std::mt19937_64& rng,
                std::uint32_t index) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 16> edge{
            0.0f, -0.0f, 0x1p-24f, 0x1p-25f, 3.0f * 0x1p-25f,
            1.00048828125f, -1.00048828125f, 0x1p-14f,
            -0x1p-14f, 65519.0f, -65519.0f, 65520.0f,
            -65520.0f, 0.333251953125f, -0.333251953125f,
            0x1p-10f};
        const std::uint32_t row = index / HEAD_DIM;
        return edge[(index + row * 5u) % edge.size()];
    }
    return sample(set, rng, index, 2.75f);
}

bool run_kv_append_f16_case(Context& context, std::uint32_t token_rows,
                            std::uint32_t pos0, twin::InputSet set,
                            const twin::Options& options,
                            bool shared_encoder = false,
                            bool prefill_api = false) {
    constexpr std::uint32_t kCellCapacity = 32u;
    constexpr std::size_t kPoolElements =
        kCellCapacity * N_HEAD_KV * HEAD_DIM;
    const std::string label =
        "QSA fp16 KV append T=" + std::to_string(token_rows) +
        " pos0=" + std::to_string(pos0) +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared Encoder" : " standalone") +
        (prefill_api ? " Prefill" : "");
    if (token_rows == 0u ||
        token_rows > (prefill_api ? kPrefillMaxShaderRows : VERIFY_MAX_T) ||
        pos0 > kCellCapacity || token_rows > kCellCapacity - pos0) {
        check(false, label + " test parameters");
        return false;
    }

    const std::size_t current_elements =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV * HEAD_DIM;
    std::vector<float> keys(current_elements);
    std::vector<float> values(current_elements);
    std::vector<StepRow> steps(token_rows);
    std::vector<std::int32_t> positions(token_rows);
    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(token_rows) << 32u) ^
                        (static_cast<std::uint64_t>(pos0) << 16u) ^
                        static_cast<std::uint64_t>(set));
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::int32_t position =
            static_cast<std::int32_t>(pos0 + row);
        positions[row] = position;
        const std::int32_t n_kv = position + 1;
        steps[row] = StepRow{
            position, n_kv, n_kv / 4,
            std::min<std::int32_t>(n_kv, 2051)};
    }
    for (std::uint32_t i = 0u; i < current_elements; ++i) {
        keys[i] = kv_sample(set, rng, i);
        values[i] = kv_sample(set, rng, i + 4099u);
    }

    std::vector<std::uint16_t> expected_keys(kPoolElements);
    std::vector<std::uint16_t> expected_values(kPoolElements);
    for (std::size_t i = 0u; i < kPoolElements; ++i) {
        expected_keys[i] = static_cast<std::uint16_t>(
            0x3555u ^ static_cast<std::uint16_t>(i * 37u));
        expected_values[i] = static_cast<std::uint16_t>(
            0x4a39u ^ static_cast<std::uint16_t>(i * 53u));
    }
    oracle::qsa_kv_append_f16(
        keys, values, positions, expected_keys, expected_values);

    const std::size_t current_bytes = current_elements * sizeof(float);
    const std::size_t step_bytes = steps.size() * sizeof(StepRow);
    const std::size_t pool_bytes =
        kPoolElements * sizeof(std::uint16_t);
    std::string error;
    twin::GuardedBuffer key_buffer, value_buffer, steps_buffer;
    twin::GuardedBuffer key_pool_buffer, value_pool_buffer;
    const bool allocated =
        key_buffer.allocate(context, label + " K input", current_bytes,
                            error) &&
        value_buffer.allocate(context, label + " V input", current_bytes,
                              error) &&
        steps_buffer.allocate(context, label + " steps", step_bytes, error) &&
        key_pool_buffer.allocate(context, label + " K pool", pool_bytes,
                                 error) &&
        value_pool_buffer.allocate(context, label + " V pool", pool_bytes,
                                   error);
    check(allocated, label + " allocations", error);
    if (!allocated || !key_buffer.view() || !value_buffer.view() ||
        !steps_buffer.view() || !key_pool_buffer.view() ||
        !value_pool_buffer.view()) {
        return false;
    }
    std::memcpy(key_buffer.payload().data(), keys.data(), current_bytes);
    std::memcpy(value_buffer.payload().data(), values.data(), current_bytes);
    std::memcpy(steps_buffer.payload().data(), steps.data(), step_bytes);
    std::memcpy(key_pool_buffer.payload().data(), expected_keys.data(),
                pool_bytes);
    std::memcpy(value_pool_buffer.payload().data(), expected_values.data(),
                pool_bytes);
    key_buffer.capture_before_dispatch();
    value_buffer.capture_before_dispatch();
    steps_buffer.capture_before_dispatch();
    key_pool_buffer.capture_before_dispatch();
    value_pool_buffer.capture_before_dispatch();

    const QsaKvAppendF16Buffers buffers{
        key_buffer.view(), value_buffer.view(), steps_buffer.view(),
        key_pool_buffer.view(), value_pool_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return prefill_api
                      ? encode_prefill_qsa_kv_append_f16(
                            context, encoder, token_rows, kCellCapacity,
                            buffers, encode_error)
                      : encode_qsa_kv_append_f16(
                            context, encoder, token_rows, kCellCapacity,
                            buffers, encode_error);
              },
              error)
        : (prefill_api
               ? encode_prefill_qsa_kv_append_f16(
                     context, *command, token_rows, kCellCapacity, buffers,
                     error)
               : encode_qsa_kv_append_f16(
                     context, *command, token_rows, kCellCapacity, buffers,
                     error));
    if (!encoded) {
        check(false, label + " encode", error);
        command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }

    const auto actual_keys = typed<std::uint16_t>(key_pool_buffer);
    const auto actual_values = typed<std::uint16_t>(value_pool_buffer);
    const bool keys_match = std::equal(
        actual_keys.begin(), actual_keys.end(), expected_keys.begin());
    const bool values_match = std::equal(
        actual_values.begin(), actual_values.end(), expected_values.begin());
    check(keys_match, label + " K pool exact half bits");
    check(values_match, label + " V pool exact half bits");

    std::vector<twin::ByteRange> written;
    written.reserve(static_cast<std::size_t>(token_rows) * N_HEAD_KV);
    for (const std::int32_t position : positions) {
        const std::uint32_t cell = static_cast<std::uint32_t>(position);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::size_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
            written.push_back(twin::ByteRange{
                pool_row * HEAD_DIM * sizeof(std::uint16_t),
                HEAD_DIM * sizeof(std::uint16_t)});
        }
    }
    const auto key_guards = key_pool_buffer.first_unwritten_change(written);
    const auto value_guards =
        value_pool_buffer.first_unwritten_change(written);
    check(!key_guards, label + " K pool guards and unwritten rows",
          key_guards ? key_guards->reason : "");
    check(!value_guards, label + " V pool guards and unwritten rows",
          value_guards ? value_guards->reason : "");
    const bool inputs_unchanged =
        !key_buffer.first_unwritten_change({}) &&
        !value_buffer.first_unwritten_change({}) &&
        !steps_buffer.first_unwritten_change({});
    check(inputs_unchanged, label + " read-only inputs unchanged");
    return keys_match && values_match && !key_guards && !value_guards &&
           inputs_unchanged;
}

bool run_kv_append_f16_set(Context& context, twin::InputSet set,
                           const twin::Options& options) {
    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        passed &= run_kv_append_f16_case(
            context, rows, 32u - rows, set, options);
    }
    passed &= run_kv_append_f16_case(context, 1u, 0u, set, options);
    passed &= run_kv_append_f16_case(
        context, 1u, 0u, set, options, true);
    passed &= run_kv_append_f16_case(
        context, VERIFY_MAX_T, 0u, set, options);
    passed &= run_kv_append_f16_case(
        context, VERIFY_MAX_T, 0u, set, options, true);
    passed &= run_kv_append_f16_case(
        context, VERIFY_MAX_T, 4u, set, options);
    if (set == twin::InputSet::fixture) {
        passed &= run_kv_append_f16_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, false, true);
        passed &= run_kv_append_f16_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, true, true);
    }
    return passed;
}

bool run_kv_append_i8_case(Context& context, std::uint32_t token_rows,
                           std::uint32_t pos0, twin::InputSet set,
                           const twin::Options& options,
                           bool all_zero = false, bool tie_case = false,
                           bool shared_encoder = false,
                           bool prefill_api = false) {
    constexpr std::uint32_t kCellCapacity = 32u;
    constexpr std::size_t kPoolRows =
        kCellCapacity * N_HEAD_KV;
    constexpr std::size_t kCurrentElementsPerRow = HEAD_DIM;
    constexpr std::size_t kCodeBytes =
        kPoolRows * HEAD_DIM * sizeof(std::int8_t);
    constexpr std::size_t kScaleBytes =
        kPoolRows * 4u * sizeof(std::uint16_t);
    constexpr std::size_t kKeyCodesOffset = 0u;
    constexpr std::size_t kKeyScalesOffset = kCodeBytes;
    constexpr std::size_t kValueCodesOffset = kCodeBytes + kScaleBytes;
    constexpr std::size_t kValueScalesOffset =
        2u * kCodeBytes + kScaleBytes;
    constexpr std::size_t kOutputBytes =
        2u * kCodeBytes + 2u * kScaleBytes;
    const std::string pattern_label =
        all_zero ? " all-zero" : (tie_case ? " ties-to-even" : "");
    const std::string label =
        "QSA int8 KV append T=" + std::to_string(token_rows) +
        " pos0=" + std::to_string(pos0) +
        " set=" + twin::input_set_name(set) + pattern_label +
        (shared_encoder ? " shared Encoder" : " standalone") +
        (prefill_api ? " Prefill" : "");
    if (token_rows == 0u ||
        token_rows > (prefill_api ? kPrefillMaxShaderRows : VERIFY_MAX_T) ||
        pos0 > kCellCapacity || token_rows > kCellCapacity - pos0 ||
        (all_zero && tie_case)) {
        check(false, label + " test parameters");
        return false;
    }

    const std::size_t current_elements =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV * HEAD_DIM;
    std::vector<float> keys(current_elements);
    std::vector<float> values(current_elements);
    std::vector<StepRow> steps(token_rows);
    std::vector<std::int32_t> positions(token_rows);
    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(token_rows) << 32u) ^
                        (static_cast<std::uint64_t>(pos0) << 16u) ^
                        static_cast<std::uint64_t>(set));
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::int32_t position =
            static_cast<std::int32_t>(pos0 + row);
        positions[row] = position;
        const std::int32_t n_kv = position + 1;
        steps[row] = StepRow{
            position, n_kv, n_kv / 4,
            std::min<std::int32_t>(n_kv, 2051)};
    }
    for (std::uint32_t i = 0u; i < current_elements; ++i) {
        keys[i] = kv_sample(set, rng, i);
        values[i] = kv_sample(set, rng, i + 4099u);
    }
    if (all_zero) {
        std::fill(keys.begin(), keys.end(), 0.0f);
        std::fill(values.begin(), values.end(), 0.0f);
    } else if (tie_case) {
        std::fill(keys.begin(), keys.end(), 0.0f);
        std::fill(values.begin(), values.end(), 0.0f);
        for (std::size_t row = 0u;
             row < static_cast<std::size_t>(token_rows) * N_HEAD_KV;
             ++row) {
            const std::size_t base = row * kCurrentElementsPerRow;
            constexpr std::array<float, 9> rounding_values{
                127.0f, 2.5f, 3.5f, -2.5f, -3.5f,
                1.5f, 0.5f, -1.5f, -0.5f};
            std::copy(rounding_values.begin(), rounding_values.end(),
                      keys.begin() + base);
            std::copy(rounding_values.begin(), rounding_values.end(),
                      values.begin() + base);
        }
    }

    std::vector<std::int8_t> expected_key_codes(kPoolRows * HEAD_DIM);
    std::vector<std::int8_t> expected_value_codes(kPoolRows * HEAD_DIM);
    std::vector<std::uint16_t> expected_key_scales(kPoolRows * 4u);
    std::vector<std::uint16_t> expected_value_scales(kPoolRows * 4u);
    for (std::size_t i = 0u; i < expected_key_codes.size(); ++i) {
        const auto poison =
            static_cast<std::int32_t>((i * 17u) % 255u) - 127;
        expected_key_codes[i] = static_cast<std::int8_t>(poison);
        expected_value_codes[i] = static_cast<std::int8_t>(-poison);
    }
    for (std::size_t i = 0u; i < expected_key_scales.size(); ++i) {
        expected_key_scales[i] = static_cast<std::uint16_t>(
            0x3555u ^ static_cast<std::uint16_t>(i * 37u));
        expected_value_scales[i] = static_cast<std::uint16_t>(
            0x4a39u ^ static_cast<std::uint16_t>(i * 53u));
    }
    oracle::qsa_kv_append_i8(
        keys, values, positions, expected_key_codes, expected_key_scales,
        expected_value_codes, expected_value_scales);
    auto initial_key_codes = expected_key_codes;
    auto initial_key_scales = expected_key_scales;
    auto initial_value_codes = expected_value_codes;
    auto initial_value_scales = expected_value_scales;
    bool poison_distinct = true;
    for (const std::int32_t position : positions) {
        const std::uint32_t cell = static_cast<std::uint32_t>(position);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::size_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
            for (std::uint32_t dimension = 0u; dimension < HEAD_DIM;
                 ++dimension) {
                const std::size_t index = pool_row * HEAD_DIM + dimension;
                initial_key_codes[index] =
                    expected_key_codes[index] == -127 ? 127 : -127;
                initial_value_codes[index] =
                    expected_value_codes[index] == -127 ? 127 : -127;
                poison_distinct &=
                    initial_key_codes[index] != expected_key_codes[index] &&
                    initial_value_codes[index] !=
                        expected_value_codes[index];
            }
            for (std::uint32_t group = 0u; group < 4u; ++group) {
                const std::size_t index = pool_row * 4u + group;
                initial_key_scales[index] =
                    static_cast<std::uint16_t>(
                        expected_key_scales[index] ^ 1u);
                initial_value_scales[index] =
                    static_cast<std::uint16_t>(
                        expected_value_scales[index] ^ 1u);
                poison_distinct &=
                    initial_key_scales[index] != expected_key_scales[index] &&
                    initial_value_scales[index] !=
                        expected_value_scales[index];
            }
        }
    }
    check(poison_distinct,
          label + " output poison rejects a no-op kernel");

    const std::size_t current_bytes = current_elements * sizeof(float);
    const std::size_t step_bytes = steps.size() * sizeof(StepRow);
    std::string error;
    twin::GuardedBuffer key_buffer, value_buffer, steps_buffer, pool_buffer;
    const bool allocated =
        key_buffer.allocate(context, label + " K input", current_bytes,
                            error) &&
        value_buffer.allocate(context, label + " V input", current_bytes,
                              error) &&
        steps_buffer.allocate(context, label + " steps", step_bytes, error) &&
        pool_buffer.allocate(context, label + " shared KV output",
                             kOutputBytes, error);
    check(allocated, label + " allocations", error);
    if (!allocated || !key_buffer.view() || !value_buffer.view() ||
        !steps_buffer.view() || !pool_buffer.view()) {
        return false;
    }
    std::memcpy(key_buffer.payload().data(), keys.data(), current_bytes);
    std::memcpy(value_buffer.payload().data(), values.data(), current_bytes);
    std::memcpy(steps_buffer.payload().data(), steps.data(), step_bytes);
    auto output_payload = pool_buffer.payload();
    std::memcpy(output_payload.data() + kKeyCodesOffset,
                initial_key_codes.data(), kCodeBytes);
    std::memcpy(output_payload.data() + kKeyScalesOffset,
                initial_key_scales.data(), kScaleBytes);
    std::memcpy(output_payload.data() + kValueCodesOffset,
                initial_value_codes.data(), kCodeBytes);
    std::memcpy(output_payload.data() + kValueScalesOffset,
                initial_value_scales.data(), kScaleBytes);
    key_buffer.capture_before_dispatch();
    value_buffer.capture_before_dispatch();
    steps_buffer.capture_before_dispatch();
    pool_buffer.capture_before_dispatch();

    const BufferRef shared = pool_buffer.view();
    const QsaKvAppendI8Buffers buffers{
        key_buffer.view(),
        value_buffer.view(),
        steps_buffer.view(),
        buffer_subview(shared, kKeyCodesOffset, kCodeBytes),
        buffer_subview(shared, kKeyScalesOffset, kScaleBytes),
        buffer_subview(shared, kValueCodesOffset, kCodeBytes),
        buffer_subview(shared, kValueScalesOffset, kScaleBytes),
    };
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return prefill_api
                      ? encode_prefill_qsa_kv_append_i8(
                            context, encoder, token_rows, kCellCapacity,
                            buffers, encode_error)
                      : encode_qsa_kv_append_i8(
                            context, encoder, token_rows, kCellCapacity,
                            buffers, encode_error);
              },
              error)
        : (prefill_api
               ? encode_prefill_qsa_kv_append_i8(
                     context, *command, token_rows, kCellCapacity, buffers,
                     error)
               : encode_qsa_kv_append_i8(
                     context, *command, token_rows, kCellCapacity, buffers,
                     error));
    if (!encoded) {
        check(false, label + " encode", error);
        command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }

    output_payload = pool_buffer.payload();
    const auto actual_key_codes = std::span<const std::int8_t>(
        reinterpret_cast<const std::int8_t*>(
            output_payload.data() + kKeyCodesOffset),
        expected_key_codes.size());
    const auto actual_key_scales = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(
            output_payload.data() + kKeyScalesOffset),
        expected_key_scales.size());
    const auto actual_value_codes = std::span<const std::int8_t>(
        reinterpret_cast<const std::int8_t*>(
            output_payload.data() + kValueCodesOffset),
        expected_value_codes.size());
    const auto actual_value_scales = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(
            output_payload.data() + kValueScalesOffset),
        expected_value_scales.size());
    const bool key_codes_match = std::equal(
        actual_key_codes.begin(), actual_key_codes.end(),
        expected_key_codes.begin());
    const bool key_scales_match = std::equal(
        actual_key_scales.begin(), actual_key_scales.end(),
        expected_key_scales.begin());
    const bool value_codes_match = std::equal(
        actual_value_codes.begin(), actual_value_codes.end(),
        expected_value_codes.begin());
    const bool value_scales_match = std::equal(
        actual_value_scales.begin(), actual_value_scales.end(),
        expected_value_scales.begin());
    check(key_codes_match, label + " K code bytes exact");
    check(key_scales_match, label + " K scale half bits exact");
    check(value_codes_match, label + " V code bytes exact");
    check(value_scales_match, label + " V scale half bits exact");

    std::vector<twin::ByteRange> written;
    written.reserve(static_cast<std::size_t>(token_rows) * N_HEAD_KV * 2u);
    for (const std::int32_t position : positions) {
        const std::uint32_t cell = static_cast<std::uint32_t>(position);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::size_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
            written.push_back(twin::ByteRange{
                kKeyCodesOffset + pool_row * HEAD_DIM,
                HEAD_DIM * sizeof(std::int8_t)});
            written.push_back(twin::ByteRange{
                kKeyScalesOffset +
                    pool_row * 4u * sizeof(std::uint16_t),
                4u * sizeof(std::uint16_t)});
            written.push_back(twin::ByteRange{
                kValueCodesOffset + pool_row * HEAD_DIM,
                HEAD_DIM * sizeof(std::int8_t)});
            written.push_back(twin::ByteRange{
                kValueScalesOffset +
                    pool_row * 4u * sizeof(std::uint16_t),
                4u * sizeof(std::uint16_t)});
        }
    }
    const auto output_guards = pool_buffer.first_unwritten_change(written);
    check(!output_guards, label + " shared pool guards and untouched rows",
          output_guards ? output_guards->reason : "");
    const bool inputs_unchanged =
        !key_buffer.first_unwritten_change({}) &&
        !value_buffer.first_unwritten_change({}) &&
        !steps_buffer.first_unwritten_change({});
    check(inputs_unchanged, label + " read-only inputs unchanged");

    bool special_match = true;
    if (all_zero) {
        for (const std::int32_t position : positions) {
            const std::uint32_t cell = static_cast<std::uint32_t>(position);
            for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
                const std::size_t pool_row =
                    ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
                for (std::uint32_t dimension = 0u; dimension < HEAD_DIM;
                     ++dimension) {
                    special_match &=
                        actual_key_codes[pool_row * HEAD_DIM + dimension] ==
                            0 &&
                        actual_value_codes[pool_row * HEAD_DIM + dimension] ==
                            0;
                }
                for (std::uint32_t group = 0u; group < 4u; ++group) {
                    special_match &=
                        actual_key_scales[pool_row * 4u + group] == 0u &&
                        actual_value_scales[pool_row * 4u + group] == 0u;
                }
            }
        }
        check(special_match, label + " zero scales produce zero codes");
    } else if (tie_case) {
        constexpr std::array<std::int8_t, 8> expected_rounding{
            2, 4, -2, -4, 2, 0, -2, 0};
        const bool key_rounding_match = std::equal(
            expected_rounding.begin(), expected_rounding.end(),
            actual_key_codes.begin() + 1);
        const bool value_rounding_match = std::equal(
            expected_rounding.begin(), expected_rounding.end(),
            actual_value_codes.begin() + 1);
        special_match = key_rounding_match && value_rounding_match &&
            actual_key_codes[0] == 127 &&
            actual_value_codes[0] == 127 &&
            actual_key_scales[0] == 0x3c00u &&
            actual_value_scales[0] == 0x3c00u;
        std::string detail;
        if (!special_match) {
            detail = "K scale=" + std::to_string(actual_key_scales[0]) +
                " V scale=" + std::to_string(actual_value_scales[0]) +
                " K codes=" + std::to_string(actual_key_codes[0]);
            for (std::size_t i = 1u; i <= expected_rounding.size(); ++i) {
                detail += "," + std::to_string(actual_key_codes[i]);
            }
            detail += " V codes=" + std::to_string(actual_value_codes[0]);
            for (std::size_t i = 1u; i <= expected_rounding.size(); ++i) {
                detail += "," + std::to_string(actual_value_codes[i]);
            }
        }
        check(special_match,
              label + " explicit round-to-nearest-even and unit scale",
              detail);
    }
    return key_codes_match && key_scales_match && value_codes_match &&
           value_scales_match && poison_distinct && !output_guards &&
           inputs_unchanged && special_match;
}

bool run_kv_append_i8_set(Context& context, twin::InputSet set,
                          const twin::Options& options) {
    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        passed &= run_kv_append_i8_case(
            context, rows, 32u - rows, set, options);
    }
    passed &= run_kv_append_i8_case(context, 1u, 0u, set, options);
    passed &= run_kv_append_i8_case(
        context, 1u, 0u, set, options, false, false, true);
    passed &= run_kv_append_i8_case(
        context, VERIFY_MAX_T, 0u, set, options);
    passed &= run_kv_append_i8_case(
        context, VERIFY_MAX_T, 0u, set, options, false, false, true);
    passed &= run_kv_append_i8_case(
        context, VERIFY_MAX_T, 4u, set, options);
    if (set == twin::InputSet::fixture) {
        passed &= run_kv_append_i8_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, false, false,
            false, true);
        passed &= run_kv_append_i8_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, false, false,
            true, true);
    }
    return passed;
}

bool run_kv_append_q4_case(Context& context, std::uint32_t token_rows,
                           std::uint32_t pos0, twin::InputSet set,
                           const twin::Options& options,
                           bool all_zero = false, bool tie_case = false,
                           bool shared_encoder = false,
                           bool prefill_api = false) {
    constexpr std::uint32_t kCellCapacity = 32u;
    constexpr std::size_t kBlocksPerRow = HEAD_DIM / 32u;
    constexpr std::size_t kBlockBytes = sizeof(block_q4_0);
    constexpr std::size_t kPoolRows =
        kCellCapacity * N_HEAD_KV;
    constexpr std::size_t kPoolRowBytes =
        kBlocksPerRow * kBlockBytes;
    constexpr std::size_t kPoolBytes = kPoolRows * kPoolRowBytes;
    constexpr std::size_t kKeyPoolOffset = 0u;
    constexpr std::size_t kValuePoolOffset = kPoolBytes;
    constexpr std::size_t kOutputBytes = 2u * kPoolBytes;
    const std::string pattern_label =
        all_zero ? " all-zero" : (tie_case ? " max-tie packing" : "");
    const std::string label =
        "QSA q4 KV append T=" + std::to_string(token_rows) +
        " pos0=" + std::to_string(pos0) +
        " set=" + twin::input_set_name(set) + pattern_label +
        (shared_encoder ? " shared Encoder" : " standalone") +
        (prefill_api ? " Prefill" : "");
    if (token_rows == 0u ||
        token_rows > (prefill_api ? kPrefillMaxShaderRows : VERIFY_MAX_T) ||
        pos0 > kCellCapacity || token_rows > kCellCapacity - pos0 ||
        (all_zero && tie_case)) {
        check(false, label + " test parameters");
        return false;
    }

    const std::size_t current_elements =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV * HEAD_DIM;
    const std::uint32_t current_rows = token_rows * N_HEAD_KV;
    std::vector<float> keys(current_elements);
    std::vector<float> values(current_elements);
    std::vector<StepRow> steps(token_rows);
    std::vector<std::int32_t> positions(token_rows);
    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(token_rows) << 32u) ^
                        (static_cast<std::uint64_t>(pos0) << 16u) ^
                        static_cast<std::uint64_t>(set));
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::int32_t position =
            static_cast<std::int32_t>(pos0 + row);
        positions[row] = position;
        const std::int32_t n_kv = position + 1;
        steps[row] = StepRow{
            position, n_kv, n_kv / 4,
            std::min<std::int32_t>(n_kv, 2051)};
    }
    for (std::uint32_t i = 0u; i < current_elements; ++i) {
        keys[i] = kv_sample(set, rng, i);
        values[i] = kv_sample(set, rng, i + 4099u);
    }
    if (all_zero) {
        std::fill(keys.begin(), keys.end(), 0.0f);
        std::fill(values.begin(), values.end(), 0.0f);
    } else if (tie_case) {
        std::fill(keys.begin(), keys.end(), 0.0f);
        std::fill(values.begin(), values.end(), 0.0f);
        for (std::size_t row = 0u;
             row < static_cast<std::size_t>(current_rows); ++row) {
            const std::size_t base = row * HEAD_DIM;
            keys[base + 1u] = 128.0f;
            values[base + 2u] = -64.0f;
        }
    }

    std::vector<twin::tf> rotated_key_tf;
    std::vector<twin::tf> rotated_value_tf;
    oracle::qsa_fwht256(keys, current_rows, rotated_key_tf);
    oracle::qsa_fwht256(values, current_rows, rotated_value_tf);
    std::vector<float> rotated_keys(current_elements);
    std::vector<float> rotated_values(current_elements);
    for (std::size_t i = 0u; i < current_elements; ++i) {
        rotated_keys[i] = rotated_key_tf[i].value;
        rotated_values[i] = rotated_value_tf[i].value;
    }

    std::vector<std::uint8_t> expected_key_pool(kPoolBytes);
    std::vector<std::uint8_t> expected_value_pool(kPoolBytes);
    for (std::size_t i = 0u; i < kPoolBytes; ++i) {
        expected_key_pool[i] =
            static_cast<std::uint8_t>((i * 37u + 0x5au) & 0xffu);
        expected_value_pool[i] =
            static_cast<std::uint8_t>((i * 53u + 0xa7u) & 0xffu);
    }
    oracle::qsa_kv_append_q4(
        rotated_keys, rotated_values, positions,
        expected_key_pool, expected_value_pool);
    auto initial_key_pool = expected_key_pool;
    auto initial_value_pool = expected_value_pool;
    bool poison_distinct = true;
    for (const std::int32_t position : positions) {
        const std::uint32_t cell = static_cast<std::uint32_t>(position);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::size_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
            const std::size_t row_base = pool_row * kPoolRowBytes;
            for (std::size_t byte = 0u; byte < kPoolRowBytes; ++byte) {
                initial_key_pool[row_base + byte] =
                    static_cast<std::uint8_t>(
                        expected_key_pool[row_base + byte] ^ 0xffu);
                initial_value_pool[row_base + byte] =
                    static_cast<std::uint8_t>(
                        expected_value_pool[row_base + byte] ^ 0xffu);
                poison_distinct &=
                    initial_key_pool[row_base + byte] !=
                        expected_key_pool[row_base + byte] &&
                    initial_value_pool[row_base + byte] !=
                        expected_value_pool[row_base + byte];
            }
        }
    }
    check(poison_distinct,
          label + " output poison rejects a no-op kernel");

    const std::size_t current_bytes = current_elements * sizeof(float);
    const std::size_t step_bytes = steps.size() * sizeof(StepRow);
    std::string error;
    twin::GuardedBuffer key_buffer, value_buffer, steps_buffer, pool_buffer;
    const bool allocated =
        key_buffer.allocate(context, label + " K input", current_bytes,
                            error) &&
        value_buffer.allocate(context, label + " V input", current_bytes,
                              error) &&
        steps_buffer.allocate(context, label + " steps", step_bytes, error) &&
        pool_buffer.allocate(context, label + " shared q4 output",
                             kOutputBytes, error);
    check(allocated, label + " allocations", error);
    if (!allocated || !key_buffer.view() || !value_buffer.view() ||
        !steps_buffer.view() || !pool_buffer.view()) {
        return false;
    }
    std::memcpy(key_buffer.payload().data(), keys.data(), current_bytes);
    std::memcpy(value_buffer.payload().data(), values.data(), current_bytes);
    std::memcpy(steps_buffer.payload().data(), steps.data(), step_bytes);
    auto output_payload = pool_buffer.payload();
    std::memcpy(output_payload.data() + kKeyPoolOffset,
                initial_key_pool.data(), kPoolBytes);
    std::memcpy(output_payload.data() + kValuePoolOffset,
                initial_value_pool.data(), kPoolBytes);
    key_buffer.capture_before_dispatch();
    value_buffer.capture_before_dispatch();
    steps_buffer.capture_before_dispatch();
    pool_buffer.capture_before_dispatch();

    const BufferRef shared = pool_buffer.view();
    const QsaKvAppendQ4Buffers buffers{
        key_buffer.view(),
        value_buffer.view(),
        steps_buffer.view(),
        buffer_subview(shared, kKeyPoolOffset, kPoolBytes),
        buffer_subview(shared, kValuePoolOffset, kPoolBytes),
    };
    if (shared_encoder && token_rows == 1u && pos0 == 0u &&
        set == twin::InputSet::random && !all_zero && !tie_case &&
        concurrent_dispatch_mode()) {
        const bool rejected = expect_append_rejected_in_open_concurrent_group(
            context, label + " Q4 group-boundary preflight",
            "must follow FWHT outside an open concurrent dispatch group",
            [&](Encoder& encoder, std::string& prepare_error) {
                return encode_qsa_fwht256(
                           context, encoder, current_rows,
                           key_buffer.view(), prepare_error) &&
                       encode_qsa_fwht256(
                           context, encoder, current_rows,
                           value_buffer.view(), prepare_error);
            },
            [&](Encoder& encoder, std::string& append_error) {
                return encode_qsa_kv_append_q4(
                    context, encoder, token_rows, kCellCapacity, buffers,
                    append_error);
            });
        if (!rejected) return false;
    }
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_qsa_fwht256(
                             context, encoder, current_rows,
                             key_buffer.view(), encode_error) &&
                         encode_qsa_fwht256(
                             context, encoder, current_rows,
                             value_buffer.view(), encode_error) &&
                         (prefill_api
                              ? encode_prefill_qsa_kv_append_q4(
                                    context, encoder, token_rows,
                                    kCellCapacity, buffers, encode_error)
                              : encode_qsa_kv_append_q4(
                                    context, encoder, token_rows,
                                    kCellCapacity, buffers, encode_error));
              },
              error)
        : encode_qsa_fwht256(context, *command, current_rows,
                             key_buffer.view(), error) &&
              encode_qsa_fwht256(context, *command, current_rows,
                                 value_buffer.view(), error) &&
              (prefill_api
                   ? encode_prefill_qsa_kv_append_q4(
                         context, *command, token_rows, kCellCapacity,
                         buffers, error)
                   : encode_qsa_kv_append_q4(
                         context, *command, token_rows, kCellCapacity,
                         buffers, error));
    if (!encoded) {
        check(false, label + " FWHT/KV append encode", error);
        command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }

    output_payload = pool_buffer.payload();
    const auto actual_key_pool = std::span<const std::uint8_t>(
        output_payload.data() + kKeyPoolOffset, kPoolBytes);
    const auto actual_value_pool = std::span<const std::uint8_t>(
        output_payload.data() + kValuePoolOffset, kPoolBytes);
    const bool keys_match = std::equal(
        actual_key_pool.begin(), actual_key_pool.end(),
        expected_key_pool.begin());
    const bool values_match = std::equal(
        actual_value_pool.begin(), actual_value_pool.end(),
        expected_value_pool.begin());
    const auto first_difference = [](std::span<const std::uint8_t> actual,
                                     const std::vector<std::uint8_t>& expected) {
        for (std::size_t i = 0u; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) {
                return "byte " + std::to_string(i) +
                    " actual=" + std::to_string(actual[i]) +
                    " expected=" + std::to_string(expected[i]);
            }
        }
        return std::string{};
    };
    check(keys_match, label + " K q4_0 bytes exact",
          keys_match ? "" : first_difference(actual_key_pool,
                                               expected_key_pool));
    check(values_match, label + " V q4_0 bytes exact",
          values_match ? "" : first_difference(actual_value_pool,
                                                 expected_value_pool));
    const bool rotated_inputs_match =
        std::memcmp(key_buffer.payload().data(), rotated_keys.data(),
                    current_bytes) == 0 &&
        std::memcmp(value_buffer.payload().data(), rotated_values.data(),
                    current_bytes) == 0;
    check(rotated_inputs_match,
          label + " chained FWHT leaves exact rotated rows");

    std::vector<twin::ByteRange> written;
    written.reserve(static_cast<std::size_t>(token_rows) * N_HEAD_KV * 2u);
    for (const std::int32_t position : positions) {
        const std::uint32_t cell = static_cast<std::uint32_t>(position);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::size_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
            written.push_back(twin::ByteRange{
                kKeyPoolOffset + pool_row * kPoolRowBytes,
                kPoolRowBytes});
            written.push_back(twin::ByteRange{
                kValuePoolOffset + pool_row * kPoolRowBytes,
                kPoolRowBytes});
        }
    }
    const auto output_guards = pool_buffer.first_unwritten_change(written);
    check(!output_guards, label + " q4 pool guards and untouched rows",
          output_guards ? output_guards->reason : "");
    const std::array<twin::ByteRange, 1> complete_input{{
        twin::ByteRange{0u, current_bytes}}};
    const bool input_guards =
        !key_buffer.first_unwritten_change(complete_input) &&
        !value_buffer.first_unwritten_change(complete_input);
    check(input_guards, label + " FWHT input guards");
    const bool steps_unchanged = !steps_buffer.first_unwritten_change({});
    check(steps_unchanged, label + " read-only step rows unchanged");

    bool special_match = true;
    if (all_zero) {
        for (const std::int32_t position : positions) {
            const std::uint32_t cell = static_cast<std::uint32_t>(position);
            for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
                const std::size_t pool_row =
                    ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
                for (std::size_t block = 0u; block < kBlocksPerRow; ++block) {
                    const std::size_t base =
                        pool_row * kPoolRowBytes + block * kBlockBytes;
                    special_match &=
                        actual_key_pool[base] == 0u &&
                        actual_key_pool[base + 1u] == 0x80u &&
                        actual_value_pool[base] == 0u &&
                        actual_value_pool[base + 1u] == 0x80u;
                    for (std::size_t byte = 0u; byte < 16u; ++byte) {
                        special_match &=
                            actual_key_pool[base + 2u + byte] == 0x88u &&
                            actual_value_pool[base + 2u + byte] == 0x88u;
                    }
                }
            }
        }
        check(special_match, label + " zero scale and code packing");
    } else if (tie_case) {
        for (std::size_t block = 0u; block < kBlocksPerRow; ++block) {
            const std::size_t key_base = block * kBlockBytes;
            const std::size_t value_base = block * kBlockBytes;
            const std::uint16_t key_scale = static_cast<std::uint16_t>(
                actual_key_pool[key_base] |
                (static_cast<std::uint16_t>(
                     actual_key_pool[key_base + 1u]) << 8u));
            const std::uint16_t value_scale = static_cast<std::uint16_t>(
                actual_value_pool[key_base] |
                (static_cast<std::uint16_t>(
                     actual_value_pool[key_base + 1u]) << 8u));
            special_match &= key_scale == 0xbc00u &&
                             value_scale == 0xb800u;
            for (std::size_t lane = 0u; lane < 16u; ++lane) {
                const std::uint8_t expected_key_byte =
                    lane % 2u == 0u ? 0x00u : 0xffu;
                const std::uint8_t expected_value_byte =
                    lane % 4u < 2u ? 0xffu : 0x00u;
                special_match &=
                    actual_key_pool[key_base + 2u + lane] ==
                        expected_key_byte &&
                    actual_value_pool[value_base + 2u + lane] ==
                        expected_value_byte;
            }
        }
        check(special_match,
              label + " larger-value max ties and q4 nibble order");
    }
    return keys_match && values_match && rotated_inputs_match &&
           poison_distinct && !output_guards && input_guards &&
           steps_unchanged && special_match;
}

bool run_kv_append_q4_set(Context& context, twin::InputSet set,
                          const twin::Options& options) {
    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        passed &= run_kv_append_q4_case(
            context, rows, 32u - rows, set, options);
    }
    passed &= run_kv_append_q4_case(context, 1u, 0u, set, options);
    passed &= run_kv_append_q4_case(
        context, 1u, 0u, set, options, false, false, true);
    passed &= run_kv_append_q4_case(
        context, VERIFY_MAX_T, 0u, set, options);
    passed &= run_kv_append_q4_case(
        context, VERIFY_MAX_T, 0u, set, options, false, false, true);
    passed &= run_kv_append_q4_case(
        context, VERIFY_MAX_T, 4u, set, options);
    if (set == twin::InputSet::fixture) {
        passed &= run_kv_append_q4_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, false, false,
            false, true);
        passed &= run_kv_append_q4_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, false, false,
            true, true);
    }
    return passed;
}

bool run_kv_append_k8v4_case(Context& context, std::uint32_t token_rows,
                             std::uint32_t pos0, twin::InputSet set,
                             const twin::Options& options,
                             bool all_zero = false, bool tie_case = false,
                             bool shared_encoder = false,
                             bool prefill_api = false) {
    constexpr std::uint32_t kCellCapacity = 32u;
    constexpr std::size_t kPoolRows =
        static_cast<std::size_t>(kCellCapacity) * N_HEAD_KV;
    constexpr std::size_t kCodeBytes =
        kPoolRows * HEAD_DIM * sizeof(std::int8_t);
    constexpr std::size_t kScaleBytes =
        kPoolRows * 4u * sizeof(std::uint16_t);
    constexpr std::size_t kValuePoolBytes =
        kPoolRows * (HEAD_DIM / 32u) * sizeof(block_q4_0);
    constexpr std::size_t kKeyCodesOffset = 0u;
    constexpr std::size_t kKeyScalesOffset = kCodeBytes;
    constexpr std::size_t kValuePoolOffset = kCodeBytes + kScaleBytes;
    constexpr std::size_t kOutputBytes =
        kCodeBytes + kScaleBytes + kValuePoolBytes;
    const std::string pattern_label =
        all_zero ? " all-zero" : (tie_case ? " q4 sign ties and i8 RNE" : "");
    const std::string label =
        "QSA k8v4 KV append T=" + std::to_string(token_rows) +
        " pos0=" + std::to_string(pos0) +
        " set=" + twin::input_set_name(set) + pattern_label +
        (shared_encoder ? " shared Encoder" : " standalone") +
        (prefill_api ? " Prefill" : "");
    if (token_rows == 0u ||
        token_rows > (prefill_api ? kPrefillMaxShaderRows : VERIFY_MAX_T) ||
        pos0 > kCellCapacity || token_rows > kCellCapacity - pos0 ||
        (all_zero && tie_case)) {
        check(false, label + " test parameters");
        return false;
    }

    const std::size_t current_elements =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV * HEAD_DIM;
    const std::uint32_t current_rows = token_rows * N_HEAD_KV;
    std::vector<float> keys(current_elements);
    std::vector<float> values(current_elements);
    std::vector<StepRow> steps(token_rows);
    std::vector<std::int32_t> positions(token_rows);
    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(token_rows) << 32u) ^
                        (static_cast<std::uint64_t>(pos0) << 16u) ^
                        static_cast<std::uint64_t>(set));
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::int32_t position =
            static_cast<std::int32_t>(pos0 + row);
        positions[row] = position;
        const std::int32_t n_kv = position + 1;
        steps[row] = StepRow{
            position, n_kv, n_kv / 4,
            std::min<std::int32_t>(n_kv, 2051)};
    }
    for (std::uint32_t i = 0u; i < current_elements; ++i) {
        keys[i] = kv_sample(set, rng, i);
        values[i] = kv_sample(set, rng, i + 4099u);
    }
    if (all_zero) {
        std::fill(keys.begin(), keys.end(), 0.0f);
        std::fill(values.begin(), values.end(), 0.0f);
    } else if (tie_case) {
        std::fill(keys.begin(), keys.end(), 0.0f);
        std::fill(values.begin(), values.end(), 0.0f);
        constexpr std::array<float, 9> rounding_values{
            127.0f, 2.5f, 3.5f, -2.5f, -3.5f,
            1.5f, 0.5f, -1.5f, -0.5f};
        for (std::size_t row = 0u;
             row < static_cast<std::size_t>(current_rows); ++row) {
            const std::size_t base = row * HEAD_DIM;
            std::transform(
                rounding_values.begin(), rounding_values.end(),
                keys.begin() + base,
                [](float value) { return value * 65504.0f; });
            values[base + 1u] = 16.0f * 8.0f * 65504.0f;
        }
    }

    std::vector<twin::tf> rotated_value_tf;
    oracle::qsa_fwht256(values, current_rows, rotated_value_tf);
    std::vector<float> rotated_values(current_elements);
    for (std::size_t i = 0u; i < current_elements; ++i) {
        rotated_values[i] = rotated_value_tf[i].value;
    }

    std::vector<std::int8_t> expected_key_codes(kPoolRows * HEAD_DIM);
    std::vector<std::uint16_t> expected_key_scales(kPoolRows * 4u);
    std::vector<std::int8_t> ignored_value_codes(kPoolRows * HEAD_DIM);
    std::vector<std::uint16_t> ignored_value_scales(kPoolRows * 4u);
    for (std::size_t i = 0u; i < expected_key_codes.size(); ++i) {
        const int value = static_cast<int>(i % 254u) + 1;
        expected_key_codes[i] = static_cast<std::int8_t>(
            value <= 127 ? value : value - 255);
    }
    std::fill(expected_key_scales.begin(), expected_key_scales.end(),
              0x3555u);
    std::vector<float> zero_values(current_elements, 0.0f);
    oracle::qsa_kv_append_i8(
        keys, zero_values, positions, expected_key_codes,
        expected_key_scales, ignored_value_codes, ignored_value_scales);

    std::vector<std::uint8_t> ignored_key_pool(kValuePoolBytes);
    std::vector<std::uint8_t> expected_value_pool(kValuePoolBytes);
    for (std::size_t i = 0u; i < expected_value_pool.size(); ++i) {
        expected_value_pool[i] =
            static_cast<std::uint8_t>((i * 37u) % 255u + 1u);
    }
    oracle::qsa_kv_append_q4(
        zero_values, rotated_values, positions,
        ignored_key_pool, expected_value_pool);

    std::vector<std::uint8_t> initial_key_codes(kCodeBytes);
    std::vector<std::uint8_t> initial_key_scales(kScaleBytes);
    for (std::size_t i = 0u; i < initial_key_codes.size(); ++i) {
        initial_key_codes[i] =
            static_cast<std::uint8_t>(expected_key_codes[i]);
    }
    for (std::size_t i = 0u; i < initial_key_scales.size(); ++i) {
        initial_key_scales[i] = static_cast<std::uint8_t>(
            expected_key_scales[i / 2u] >> ((i & 1u) * 8u));
    }
    auto initial_value_pool = expected_value_pool;
    bool poison_distinct = true;
    for (const std::int32_t position : positions) {
        const std::uint32_t cell = static_cast<std::uint32_t>(position);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::size_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
            for (std::uint32_t dimension = 0u; dimension < HEAD_DIM;
                 ++dimension) {
                const std::size_t index = pool_row * HEAD_DIM + dimension;
                initial_key_codes[index] = static_cast<std::uint8_t>(
                    expected_key_codes[index] == -127 ? 127 : -127);
                poison_distinct &=
                    initial_key_codes[index] !=
                    static_cast<std::uint8_t>(expected_key_codes[index]);
            }
            for (std::uint32_t group = 0u; group < 4u; ++group) {
                const std::size_t index = pool_row * 4u + group;
                const std::uint16_t poison_bits =
                    expected_key_scales[index] ^ 0x0101u;
                const std::size_t byte = index * sizeof(std::uint16_t);
                initial_key_scales[byte] =
                    static_cast<std::uint8_t>(poison_bits & 0xffu);
                initial_key_scales[byte + 1u] =
                    static_cast<std::uint8_t>(poison_bits >> 8u);
                poison_distinct &=
                    poison_bits != expected_key_scales[index];
            }
            const std::size_t value_base =
                pool_row * (HEAD_DIM / 32u) * sizeof(block_q4_0);
            const std::size_t value_bytes =
                (HEAD_DIM / 32u) * sizeof(block_q4_0);
            for (std::size_t byte = 0u; byte < value_bytes; ++byte) {
                initial_value_pool[value_base + byte] =
                    expected_value_pool[value_base + byte] ^ 0xffu;
                poison_distinct &=
                    initial_value_pool[value_base + byte] !=
                    expected_value_pool[value_base + byte];
            }
        }
    }
    check(poison_distinct,
          label + " output poison rejects a no-op kernel");

    const std::size_t current_bytes = current_elements * sizeof(float);
    const std::size_t step_bytes = steps.size() * sizeof(StepRow);
    std::string error;
    twin::GuardedBuffer key_buffer, value_buffer, steps_buffer, output_buffer;
    const bool allocated =
        key_buffer.allocate(context, label + " K input", current_bytes,
                            error) &&
        value_buffer.allocate(context, label + " V input", current_bytes,
                              error) &&
        steps_buffer.allocate(context, label + " steps", step_bytes, error) &&
        output_buffer.allocate(context, label + " shared k8v4 output",
                               kOutputBytes, error);
    check(allocated, label + " allocations", error);
    if (!allocated || !key_buffer.view() || !value_buffer.view() ||
        !steps_buffer.view() || !output_buffer.view()) {
        return false;
    }
    std::memcpy(key_buffer.payload().data(), keys.data(), current_bytes);
    std::memcpy(value_buffer.payload().data(), values.data(), current_bytes);
    std::memcpy(steps_buffer.payload().data(), steps.data(), step_bytes);
    auto output_payload = output_buffer.payload();
    std::memcpy(output_payload.data() + kKeyCodesOffset,
                initial_key_codes.data(), kCodeBytes);
    std::memcpy(output_payload.data() + kKeyScalesOffset,
                initial_key_scales.data(), kScaleBytes);
    std::memcpy(output_payload.data() + kValuePoolOffset,
                initial_value_pool.data(), kValuePoolBytes);
    key_buffer.capture_before_dispatch();
    value_buffer.capture_before_dispatch();
    steps_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    const BufferRef shared = output_buffer.view();
    const QsaKvAppendK8V4Buffers buffers{
        key_buffer.view(),
        value_buffer.view(),
        steps_buffer.view(),
        buffer_subview(shared, kKeyCodesOffset, kCodeBytes),
        buffer_subview(shared, kKeyScalesOffset, kScaleBytes),
        buffer_subview(shared, kValuePoolOffset, kValuePoolBytes),
    };
    if (shared_encoder && token_rows == 1u && pos0 == 0u &&
        set == twin::InputSet::random && !all_zero && !tie_case &&
        concurrent_dispatch_mode()) {
        const bool rejected = expect_append_rejected_in_open_concurrent_group(
            context, label + " K8V4 group-boundary preflight",
            "must follow V FWHT outside an open concurrent dispatch group",
            [&](Encoder& encoder, std::string& prepare_error) {
                return encode_qsa_fwht256(
                    context, encoder, current_rows, value_buffer.view(),
                    prepare_error);
            },
            [&](Encoder& encoder, std::string& append_error) {
                return encode_qsa_kv_append_k8v4(
                    context, encoder, token_rows, kCellCapacity, buffers,
                    append_error);
            });
        if (!rejected) return false;
    }
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_qsa_fwht256(
                             context, encoder, current_rows,
                             value_buffer.view(), encode_error) &&
                         (prefill_api
                              ? encode_prefill_qsa_kv_append_k8v4(
                                    context, encoder, token_rows,
                                    kCellCapacity, buffers, encode_error)
                              : encode_qsa_kv_append_k8v4(
                                    context, encoder, token_rows,
                                    kCellCapacity, buffers, encode_error));
              },
              error)
        : encode_qsa_fwht256(context, *command, current_rows,
                             value_buffer.view(), error) &&
              (prefill_api
                   ? encode_prefill_qsa_kv_append_k8v4(
                         context, *command, token_rows, kCellCapacity,
                         buffers, error)
                   : encode_qsa_kv_append_k8v4(
                         context, *command, token_rows, kCellCapacity,
                         buffers, error));
    if (!encoded) {
        check(false, label + " FWHT/KV append encode", error);
        command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }

    output_payload = output_buffer.payload();
    const auto actual_key_codes = std::span<const std::int8_t>(
        reinterpret_cast<const std::int8_t*>(
            output_payload.data() + kKeyCodesOffset),
        expected_key_codes.size());
    const auto actual_key_scales = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(
            output_payload.data() + kKeyScalesOffset),
        expected_key_scales.size());
    const auto actual_value_pool = std::span<const std::uint8_t>(
        output_payload.data() + kValuePoolOffset, kValuePoolBytes);
    const bool key_codes_match = std::equal(
        actual_key_codes.begin(), actual_key_codes.end(),
        expected_key_codes.begin());
    const bool key_scales_match = std::equal(
        actual_key_scales.begin(), actual_key_scales.end(),
        expected_key_scales.begin());
    const bool value_pool_match = std::equal(
        actual_value_pool.begin(), actual_value_pool.end(),
        expected_value_pool.begin());
    check(key_codes_match, label + " unrotated K int8 code bytes exact");
    check(key_scales_match, label + " K int8 scale half bits exact");
    check(value_pool_match, label + " FWHT V q4_0 block bytes exact");
    const bool key_unrotated =
        std::memcmp(key_buffer.payload().data(), keys.data(),
                    current_bytes) == 0;
    const bool value_rotated =
        std::memcmp(value_buffer.payload().data(), rotated_values.data(),
                    current_bytes) == 0;
    check(key_unrotated, label + " K remains unrotated and unchanged");
    check(value_rotated, label + " V-only FWHT result is exact");

    std::vector<twin::ByteRange> written;
    written.reserve(static_cast<std::size_t>(token_rows) * N_HEAD_KV * 3u);
    for (const std::int32_t position : positions) {
        const std::uint32_t cell = static_cast<std::uint32_t>(position);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::size_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
            written.push_back(twin::ByteRange{
                kKeyCodesOffset + pool_row * HEAD_DIM, HEAD_DIM});
            written.push_back(twin::ByteRange{
                kKeyScalesOffset +
                    pool_row * 4u * sizeof(std::uint16_t),
                4u * sizeof(std::uint16_t)});
            written.push_back(twin::ByteRange{
                kValuePoolOffset +
                    pool_row * (HEAD_DIM / 32u) * sizeof(block_q4_0),
                (HEAD_DIM / 32u) * sizeof(block_q4_0)});
        }
    }
    const auto output_guards = output_buffer.first_unwritten_change(written);
    check(!output_guards, label + " shared pool guards and untouched rows",
          output_guards ? output_guards->reason : "");
    const std::array<twin::ByteRange, 1> rotated_value_range{{
        twin::ByteRange{0u, current_bytes}}};
    const bool input_guards =
        !key_buffer.first_unwritten_change({}) &&
        !value_buffer.first_unwritten_change(rotated_value_range) &&
        !steps_buffer.first_unwritten_change({});
    check(input_guards, label + " input guards and read-only steps");

    bool special_match = true;
    if (all_zero) {
        for (const std::int32_t position : positions) {
            const std::uint32_t cell = static_cast<std::uint32_t>(position);
            for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
                const std::size_t pool_row =
                    ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u;
                for (std::uint32_t dimension = 0u; dimension < HEAD_DIM;
                     ++dimension) {
                    special_match &=
                        actual_key_codes[pool_row * HEAD_DIM + dimension] ==
                            0;
                }
                for (std::uint32_t group = 0u; group < 4u; ++group) {
                    special_match &=
                        actual_key_scales[pool_row * 4u + group] == 0u;
                }
                const std::size_t value_base =
                    pool_row * (HEAD_DIM / 32u) * sizeof(block_q4_0);
                for (std::size_t block = 0u;
                     block < HEAD_DIM / 32u; ++block) {
                    const std::size_t base =
                        value_base + block * sizeof(block_q4_0);
                    special_match &= actual_value_pool[base] == 0u &&
                                     actual_value_pool[base + 1u] == 0x80u;
                    for (std::size_t byte = 0u; byte < 16u; ++byte) {
                        special_match &=
                            actual_value_pool[base + 2u + byte] == 0x88u;
                    }
                }
            }
        }
        check(special_match,
              label + " zero K scales and negative-zero V q4 scales");
    } else if (tie_case) {
        constexpr std::array<std::int8_t, 8> expected_rounding{
            2, 4, -2, -4, 2, 0, -2, 0};
        special_match = std::equal(
            expected_rounding.begin(), expected_rounding.end(),
            actual_key_codes.begin() + 1) &&
            actual_key_codes[0] == 127 &&
            actual_key_scales[0] == 0x7bffu;
        for (std::size_t block = 0u; block < HEAD_DIM / 32u; ++block) {
            const std::size_t base = block * sizeof(block_q4_0);
            const std::uint16_t scale = static_cast<std::uint16_t>(
                actual_value_pool[base] |
                (static_cast<std::uint16_t>(
                     actual_value_pool[base + 1u]) << 8u));
            special_match &= scale == 0xfbffu;
        }
        std::string detail;
        if (!special_match) {
            const std::size_t first_value_scale = kValuePoolOffset;
            const std::uint16_t value_scale = static_cast<std::uint16_t>(
                output_payload[first_value_scale] |
                (static_cast<std::uint16_t>(
                     output_payload[first_value_scale + 1u]) << 8u));
            detail = "K scale=" + std::to_string(actual_key_scales[0]) +
                " K codes=" + std::to_string(actual_key_codes[0]);
            for (std::size_t i = 1u; i <= expected_rounding.size(); ++i) {
                detail += "," + std::to_string(actual_key_codes[i]);
            }
            detail += " V scales=" + std::to_string(value_scale);
            for (std::size_t block = 1u; block < HEAD_DIM / 32u; ++block) {
                const std::size_t block_base =
                    kValuePoolOffset + block * sizeof(block_q4_0);
                const std::uint16_t block_scale =
                    static_cast<std::uint16_t>(
                        output_payload[block_base] |
                        (static_cast<std::uint16_t>(
                             output_payload[block_base + 1u]) << 8u));
                detail += "," + std::to_string(block_scale);
            }
        }
        check(special_match,
              label + " K RNE and V max tie at largest finite fp16 scales",
              detail);
    }
    return key_codes_match && key_scales_match && value_pool_match &&
           key_unrotated && value_rotated && poison_distinct &&
           !output_guards && input_guards && special_match;
}

bool run_kv_append_k8v4_set(Context& context, twin::InputSet set,
                            const twin::Options& options) {
    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        passed &= run_kv_append_k8v4_case(
            context, rows, 32u - rows, set, options);
    }
    passed &= run_kv_append_k8v4_case(context, 1u, 0u, set, options);
    passed &= run_kv_append_k8v4_case(
        context, 1u, 0u, set, options, false, false, true);
    passed &= run_kv_append_k8v4_case(
        context, VERIFY_MAX_T, 0u, set, options);
    passed &= run_kv_append_k8v4_case(
        context, VERIFY_MAX_T, 0u, set, options, false, false, true);
    passed &= run_kv_append_k8v4_case(
        context, VERIFY_MAX_T, 4u, set, options);
    if (set == twin::InputSet::fixture) {
        passed &= run_kv_append_k8v4_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, false, false,
            false, true);
        passed &= run_kv_append_k8v4_case(
            context, VERIFY_MAX_T + 1u, 0u, set, options, false, false,
            true, true);
    }
    return passed;
}

bool run_fwht_command(Context& context, const std::string& label,
                      BufferRef values, std::uint32_t rows,
                      std::string& error, bool shared_encoder = false) {
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared QSA FWHT256",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_qsa_fwht256(
                      context, encoder, rows, values, encode_error);
              },
              error)
        : encode_qsa_fwht256(context, *command, rows, values, error);
    if (!encoded) {
        check(false, label + " encode", error);
        if (!shared_encoder) command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer",
              "command buffer became unavailable");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }
    error.clear();
    return true;
}

bool run_fwht_case(Context& context, const std::string& label,
                   std::span<const float> input,
                   const twin::Options& options,
                   std::vector<float>* actual_output = nullptr,
                   bool shared_encoder = false) {
    if (input.empty() || input.size() % 256u != 0u) {
        check(false, label + " input row shape");
        return false;
    }
    const auto rows = static_cast<std::uint32_t>(input.size() / 256u);
    const std::size_t input_bytes = input.size_bytes();
    std::vector<twin::tf> expected;
    oracle::qsa_fwht256(input, rows, expected);
    std::vector<float> expected_values;
    expected_values.reserve(expected.size());
    for (const twin::tf value : expected) {
        expected_values.push_back(value.value);
    }

    std::string error;
    twin::GuardedBuffer values;
    if (!values.allocate(context, label + " in-place values", input_bytes,
                         error) ||
        !values.view()) {
        check(false, label + " allocation", error);
        return false;
    }
    std::memcpy(values.payload().data(), input.data(), input_bytes);
    values.capture_before_dispatch();
    if (!run_fwht_command(context, label, values.view(), rows, error,
                          shared_encoder)) {
        return false;
    }

    const auto actual = typed<float>(values).first(expected_values.size());
    const auto mismatch = twin::compare_bytes(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(actual.data()),
            actual.size_bytes()),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_values.data()),
            expected_values.size() * sizeof(float)));
    if (mismatch) {
        twin::report_failure(label.c_str(), options, rows, *mismatch);
    }
    check(!mismatch, label + " bitwise scalar oracle");
    const std::array<twin::ByteRange, 1> written{{
        {0u, input_bytes}}};
    const auto guard = values.first_unwritten_change(written);
    check(!guard, label + " in-place output guards",
          guard ? guard->reason : "");
    if (actual_output != nullptr) {
        actual_output->assign(actual.begin(), actual.end());
    }
    return !mismatch && !guard;
}

bool run_fwht_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    constexpr std::array row_counts{
        1u, 2u, 7u, 8u, 9u, 15u, 16u, 17u, 191u, 192u};
    std::mt19937_64 rng(options.seed ^
                        0x4657485432353600ull ^
                        (std::uint64_t{static_cast<std::uint32_t>(set)}
                         << 32u));
    bool passed = true;
    for (const std::uint32_t rows : row_counts) {
        std::vector<float> input(static_cast<std::size_t>(rows) * 256u);
        for (std::uint32_t i = 0u; i < input.size(); ++i) {
            input[i] = sample(set, rng, i, 4.0f);
        }
        const std::string label =
            "QSA FWHT256 rows=" + std::to_string(rows) +
            " set=" + twin::input_set_name(set);
        const bool shared_encoder =
            set == twin::InputSet::fixture &&
            (rows == 1u || rows == 8u || rows == 9u || rows == 192u);
        passed &= run_fwht_case(context, label, input, options);
        if (shared_encoder) {
            passed &= run_fwht_case(context, label + " shared-encoder",
                                    input, options, nullptr, true);
        }
    }

    std::mt19937_64 independent_rng(
        options.seed ^ 0x4657485432353600ull ^
        (std::uint64_t{static_cast<std::uint32_t>(set)} << 32u) ^
        0x9e3779b97f4a7c15ull);
    std::vector<float> independent_input(17u * 256u);
    for (std::uint32_t i = 0u; i < independent_input.size(); ++i) {
        independent_input[i] = sample(set, independent_rng, i, 4.0f);
    }
    std::vector<float> rows_nine_output;
    std::vector<float> rows_seventeen_output;
    const bool nine_passed = run_fwht_case(
        context, "QSA FWHT256 independence T=9",
        std::span<const float>(independent_input).first(9u * 256u),
        options, &rows_nine_output);
    const bool seventeen_passed = run_fwht_case(
        context, "QSA FWHT256 independence T=17", independent_input, options,
        &rows_seventeen_output);
    constexpr std::size_t independent_bytes = 9u * 256u * sizeof(float);
    const bool independent_shape =
        rows_nine_output.size() >= 9u * 256u &&
        rows_seventeen_output.size() >= 9u * 256u;
    bool independent_match = false;
    if (independent_shape) {
        const auto mismatch = twin::compare_bytes(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    rows_nine_output.data()),
                independent_bytes),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    rows_seventeen_output.data()),
                independent_bytes));
        independent_match = !mismatch;
    }
    const bool independent_passed =
        nine_passed && seventeen_passed && independent_shape &&
        independent_match;
    check(independent_passed,
          "QSA FWHT256 row_t_independent across group tails");
    passed &= independent_passed;

    if (set == twin::InputSet::edge) {
        std::vector<float> outlier(9u * 256u, 1.0f / 1024.0f);
        outlier[8u * 256u + 255u] = 1.0f;
        passed &= run_fwht_case(
            context, "QSA FWHT256 edge=2^10-outlier", outlier, options);
    }
    return passed;
}

bool test_fwht_impulse_golden(Context& context,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 8> impulses{
        0u, 31u, 32u, 63u, 64u, 127u, 128u, 255u};
    std::vector<float> input(impulses.size() * 256u, 0.0f);
    for (std::uint32_t row = 0u; row < impulses.size(); ++row) {
        input[static_cast<std::size_t>(row) * 256u + impulses[row]] = 16.0f;
    }
    std::vector<float> actual;
    if (!run_fwht_case(context, "QSA FWHT256 impulse", input, options,
                       &actual)) {
        return false;
    }
    std::vector<float> expected(input.size());
    for (std::uint32_t row = 0u; row < impulses.size(); ++row) {
        for (std::uint32_t index = 0u; index < 256u; ++index) {
            expected[static_cast<std::size_t>(row) * 256u + index] =
                (std::popcount(index & impulses[row]) & 1) == 0
                    ? 1.0f
                    : -1.0f;
        }
    }
    const auto mismatch = twin::compare_bytes(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(actual.data()),
            actual.size() * sizeof(float)),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float)));
    check(!mismatch, "QSA FWHT256 analytic impulse golden");
    return !mismatch;
}

std::uint64_t fwht_output_hash(std::span<const float> values) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const float value : values) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
            hash ^= (bits >> shift) & 0xffu;
            hash *= 0x100000001b3ull;
        }
    }
    return hash;
}

bool test_fwht_dense_golden(Context& context,
                            const twin::Options& options) {
    // Fixed float32 input and expected output digest from the independent
    // contiguous FWHT spec oracle. A CUDA-captured golden remains under
    // REV-025 because the reference assets are unavailable on this host.
    std::vector<float> input(256u);
    std::uint32_t state = 0x4d595df4u;
    for (std::uint32_t index = 0u; index < input.size(); ++index) {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        const std::uint32_t exponent = 120u + state % 16u;
        const std::uint32_t bits =
            (state & 0x80000000u) | (exponent << 23u) |
            (state & 0x007fffffu);
        input[index] = std::bit_cast<float>(bits);
    }

    std::vector<float> actual;
    if (!run_fwht_case(context, "QSA FWHT256 dense golden", input, options,
                       &actual)) {
        return false;
    }
    constexpr std::uint64_t kExpectedHash = 0x683d78b4d4fa46c2ull;
    const bool matches = fwht_output_hash(actual) == kExpectedHash;
    check(matches, "QSA FWHT256 dense rounding-sensitive golden",
          matches ? "" : "fixed scalar-spec hash mismatch");
    return matches;
}

bool run_indexer_snapshot_case(Context& context,
                               std::uint64_t tail_stride_floats,
                               std::uint64_t block_pos_stride_ints,
                               std::string_view layout_name,
                               bool shared_encoder) {
    constexpr std::size_t kLayers = INDEXER_SNAPSHOT_LAYERS;
    constexpr std::size_t kTailFloats = 3u * INDEXER_HEAD_DIM;
    constexpr std::size_t kTailBytes = kLayers * kTailFloats * sizeof(float);
    constexpr std::size_t kBlockPosBytes = kLayers * sizeof(std::int32_t);
    const std::size_t source_tail_floats =
        (kLayers - 1u) * static_cast<std::size_t>(tail_stride_floats) +
        kTailFloats;
    const std::size_t source_block_pos_ints =
        (kLayers - 1u) * static_cast<std::size_t>(block_pos_stride_ints) + 1u;
    const std::size_t block_pos_offset = kTailBytes + 16u;
    constexpr std::size_t kOutputPaddingBytes = 32u;
    const std::size_t output_bytes =
        block_pos_offset + kBlockPosBytes + kOutputPaddingBytes;
    const std::string label =
        "QSA indexer snapshot " + std::string(layout_name) +
        (shared_encoder ? " shared Encoder" : " standalone");

    twin::GuardedBuffer tails_buffer, block_positions_buffer, output_buffer;
    std::string error;
    check(tails_buffer.allocate(context, label + " tails",
                                source_tail_floats * sizeof(float), error),
          label + " tails allocation", error);
    check(block_positions_buffer.allocate(
              context, label + " block positions",
              std::max(source_block_pos_ints * sizeof(std::int32_t),
                       kTailBytes),
              error),
          label + " block positions allocation", error);
    check(output_buffer.allocate(context, label + " output", output_bytes,
                                 error),
          label + " output allocation", error);
    if (!tails_buffer.view() || !block_positions_buffer.view() ||
        !output_buffer.view()) {
        return false;
    }

    std::fill(tails_buffer.payload().begin(), tails_buffer.payload().end(),
              0x6du);
    std::fill(block_positions_buffer.payload().begin(),
              block_positions_buffer.payload().end(), 0xa7u);
    std::fill(output_buffer.payload().begin(), output_buffer.payload().end(),
              0xa5u);
    std::vector<std::uint32_t> expected_tails(kLayers * kTailFloats);
    std::vector<std::int32_t> expected_block_positions(kLayers);
    auto source_tail_words = typed<std::uint32_t>(tails_buffer);
    auto source_block_pos_words = typed<std::int32_t>(block_positions_buffer);
    for (std::size_t layer = 0u; layer < kLayers; ++layer) {
        for (std::size_t element = 0u; element < kTailFloats; ++element) {
            const std::uint32_t ordinal =
                static_cast<std::uint32_t>(layer * kTailFloats + element);
            std::uint32_t bits = 0x3f000000u ^
                (static_cast<std::uint32_t>(layer) << 20u) ^
                (ordinal * 0x45d9f3bu);
            if (ordinal % 11u == 0u) bits = 0x7f800001u;  // signaling NaN
            if (ordinal % 11u == 1u) bits = 0x00000001u;  // minimum subnormal
            if (ordinal % 11u == 2u) bits = 0x80000000u;  // negative zero
            const std::size_t source =
                layer * static_cast<std::size_t>(tail_stride_floats) +
                element;
            source_tail_words[source] = bits;
            expected_tails[layer * kTailFloats + element] = bits;
        }
        const std::int32_t block_pos =
            layer == 0u ? std::numeric_limits<std::int32_t>::min()
            : layer == 1u ? -1
            : layer == 2u ? 0
            : layer == 3u ? std::numeric_limits<std::int32_t>::max()
                          : static_cast<std::int32_t>(layer * 0x10203u - 17u);
        source_block_pos_words[
            layer * static_cast<std::size_t>(block_pos_stride_ints)] =
            block_pos;
        expected_block_positions[layer] = block_pos;
    }
    const BufferRef tail_snapshot = buffer_subview(
        output_buffer.view(), 0u, kTailBytes);
    const BufferRef block_pos_snapshot = buffer_subview(
        output_buffer.view(), block_pos_offset, kBlockPosBytes);
    const QsaIndexerSnapshotBuffers buffers{
        tails_buffer.view(), block_positions_buffer.view(), tail_snapshot,
        block_pos_snapshot};
    const bool poison_is_distinct =
        std::memcmp(output_buffer.payload().data(), expected_tails.data(),
                    kTailBytes) != 0;
    check(poison_is_distinct, label + " poison detects a no-op kernel");

    tails_buffer.capture_before_dispatch();
    block_positions_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_qsa_indexer_snapshot(
                      context, encoder, tail_stride_floats,
                      block_pos_stride_ints, buffers, encode_error);
              },
              error)
        : encode_qsa_indexer_snapshot(
              context, *command, tail_stride_floats,
              block_pos_stride_ints, buffers, error);
    if (!encoded) {
        check(false, label + " encode", error);
        if (!shared_encoder) command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer",
              "command buffer became unavailable");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }

    const auto output_bytes_view = output_buffer.payload();
    const bool tails_match =
        std::memcmp(output_bytes_view.data(), expected_tails.data(),
                    kTailBytes) == 0;
    check(tails_match, label + " all tail bits match in QSA-layer order");
    const bool block_positions_match =
        std::memcmp(output_bytes_view.data() + block_pos_offset,
                    expected_block_positions.data(), kBlockPosBytes) == 0;
    check(block_positions_match,
          label + " block-position int32 values match bitwise");
    const std::array<twin::ByteRange, 2> written{{
        {0u, kTailBytes}, {block_pos_offset, kBlockPosBytes}}};
    const auto output_guard = output_buffer.first_unwritten_change(written);
    check(!output_guard, label + " output padding and guards unchanged",
          output_guard ? output_guard->reason : "");
    const bool inputs_unchanged =
        !tails_buffer.first_unwritten_change({}) &&
        !block_positions_buffer.first_unwritten_change({});
    check(inputs_unchanged, label + " source views remain unchanged");
    return poison_is_distinct && tails_match && block_positions_match &&
           !output_guard && inputs_unchanged;
}

bool run_indexer_append_case(Context& context, std::uint32_t token_rows,
                             std::uint32_t pos0, std::uint32_t cell_capacity,
                             twin::InputSet set,
                             const twin::Options& options,
                             bool commit, std::uint32_t n_keep = 0u,
                             std::uint32_t qsa_layer_index =
                                 INDEXER_SNAPSHOT_LAYERS - 1u,
                             bool shared_encoder = false) {
    constexpr std::size_t kTailFloats = 3u * INDEXER_HEAD_DIM;
    constexpr std::size_t kDeadFloats = INDEXER_HEAD_DIM;
    constexpr std::size_t kPoolRowFloats = INDEXER_HEAD_DIM;
    constexpr std::size_t kLayers = INDEXER_SNAPSHOT_LAYERS;
    constexpr float kRopeZeroCos = 1.125f;
    constexpr float kRopeZeroSin = -0.0f;
    const std::size_t pool_rows =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::string label =
        "QSA indexer append " +
        std::string(commit ? "commit " : "window ") +
        "T=" + std::to_string(token_rows) +
        " pos0=" + std::to_string(pos0) +
        (commit ? " keep=" + std::to_string(n_keep) : "") +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared Encoder" : " standalone");
    if (token_rows == 0u || token_rows > VERIFY_MAX_T ||
        pos0 > cell_capacity || token_rows > cell_capacity - pos0 ||
        (commit && (n_keep == 0u || n_keep > token_rows ||
                    qsa_layer_index >= INDEXER_SNAPSHOT_LAYERS))) {
        check(false, label + " test parameters");
        return false;
    }

    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(token_rows) << 40u) ^
                        (static_cast<std::uint64_t>(pos0) << 24u) ^
                        (static_cast<std::uint64_t>(set) << 16u) ^
                        (commit ? 0xc01117u : 0x71d0u));
    std::vector<float> gamma(INDEXER_HEAD_DIM);
    for (std::uint32_t dimension = 0u; dimension < INDEXER_HEAD_DIM;
         ++dimension) {
        gamma[dimension] =
            1.0f + sample(set, rng, dimension, 0.375f);
    }
    std::vector<float> rope(
        static_cast<std::size_t>(cell_capacity) * 32u * 2u);
    for (std::uint32_t cell = 0u; cell < cell_capacity; ++cell) {
        for (std::uint32_t pair = 0u; pair < 32u; ++pair) {
            const float angle =
                static_cast<float>((cell * 13u + pair * 7u) % 257u) * 0.013f;
            const std::size_t index =
                (static_cast<std::size_t>(cell) * 32u + pair) * 2u;
            rope[index] = std::cos(angle) * 1.0625f;
            rope[index + 1u] = std::sin(angle) * 1.0625f;
        }
    }

    std::vector<float> raw_keys(
        static_cast<std::size_t>(token_rows) * INDEXER_HEAD_DIM);
    std::vector<std::int32_t> positions(token_rows);
    std::vector<StepRow> steps(token_rows);
    constexpr std::array<float, 12> indexer_edge_raw{
        0.0f, -0.0f, 0x1p-24f, -0x1p-24f, 0x1.8p-24f, -0x1.8p-24f,
        1.0f + 0x1p-11f, 1.0f + 0x1.8p-10f,
        0x1.ffcp+15f, -0x1.ffcp+15f, 0.125f, -0.125f};
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        positions[row] = static_cast<std::int32_t>(pos0 + row);
        const std::int32_t n_kv = positions[row] + 1;
        steps[row] = StepRow{
            positions[row], n_kv,
            static_cast<std::int32_t>(n_kv / INDEXER_BLOCK_CELLS),
            std::min<std::int32_t>(n_kv, 2051)};
        for (std::uint32_t dimension = 0u;
             dimension < INDEXER_HEAD_DIM; ++dimension) {
            const std::uint32_t index =
                row * INDEXER_HEAD_DIM + dimension;
            raw_keys[index] = set == twin::InputSet::edge
                ? indexer_edge_raw[(index + row * 5u) %
                                   indexer_edge_raw.size()]
                : sample(set, rng, index + 809u, 2.75f);
        }
    }

    oracle::QsaIndexerState initial;
    initial.tail.assign(kTailFloats, -13.25f);
    initial.dead.assign(kDeadFloats, -17.5f);
    initial.pooled.assign(pool_rows * kPoolRowFloats, -19.75f);
    initial.block_position = -23;
    std::vector<std::int32_t> prefix_positions(pos0);
    std::vector<float> prefix_keys(
        static_cast<std::size_t>(pos0) * INDEXER_HEAD_DIM);
    for (std::uint32_t row = 0u; row < pos0; ++row) {
        prefix_positions[row] = static_cast<std::int32_t>(row);
        for (std::uint32_t dimension = 0u;
             dimension < INDEXER_HEAD_DIM; ++dimension) {
            const std::uint32_t index = row * INDEXER_HEAD_DIM + dimension;
            prefix_keys[index] =
                sample(set, rng, index + 0x1103u, 2.75f);
        }
    }
    oracle::qsa_indexer_append(
        prefix_keys, prefix_positions, gamma, rope, kRopeZeroCos,
        kRopeZeroSin, initial);

    oracle::QsaIndexerState after_window = initial;
    oracle::qsa_indexer_append(raw_keys, positions, gamma, rope,
                               kRopeZeroCos, kRopeZeroSin, after_window);
    oracle::QsaIndexerState expected = after_window;
    if (commit) {
        expected.tail = initial.tail;
        expected.block_position = initial.block_position;
        oracle::qsa_indexer_append(
            std::span<const float>(raw_keys).first(
                static_cast<std::size_t>(n_keep) * INDEXER_HEAD_DIM),
            std::span<const std::int32_t>(positions).first(n_keep), gamma,
            rope, kRopeZeroCos, kRopeZeroSin, expected);
    } else {
        expected = after_window;
    }

    constexpr std::size_t kSnapshotTailFloats = kLayers * kTailFloats;
    constexpr std::size_t kSnapshotBlockPositions = kLayers;
    const std::size_t target_layer =
        commit ? qsa_layer_index : 0u;
    std::vector<float> all_tails(kSnapshotTailFloats);
    std::vector<std::int32_t> all_block_positions(
        kSnapshotBlockPositions);
    for (std::size_t layer = 0u; layer < kLayers; ++layer) {
        std::fill_n(all_tails.begin() + layer * kTailFloats,
                    kTailFloats, 0.03125f * static_cast<float>(layer + 1u));
        all_block_positions[layer] =
            static_cast<std::int32_t>(100u + layer * 4u);
    }
    std::copy(initial.tail.begin(), initial.tail.end(),
              all_tails.begin() + target_layer * kTailFloats);
    all_block_positions[target_layer] = initial.block_position;

    const std::size_t raw_bytes = raw_keys.size() * sizeof(float);
    const std::size_t steps_bytes = steps.size() * sizeof(StepRow);
    const std::size_t gamma_bytes = gamma.size() * sizeof(float);
    const std::size_t rope_bytes = rope.size() * sizeof(float);
    const std::size_t tails_bytes = all_tails.size() * sizeof(float);
    const std::size_t block_positions_bytes =
        all_block_positions.size() * sizeof(std::int32_t);
    const std::size_t tail_bytes = kTailFloats * sizeof(float);
    const std::size_t dead_bytes = kDeadFloats * sizeof(float);
    const std::size_t pooled_bytes =
        expected.pooled.size() * sizeof(float);
    const std::size_t snapshot_tail_bytes =
        kSnapshotTailFloats * sizeof(float);
    const std::size_t snapshot_block_pos_bytes =
        kSnapshotBlockPositions * sizeof(std::int32_t);
    std::string error;
    twin::GuardedBuffer raw_buffer, steps_buffer, gamma_buffer, rope_buffer;
    twin::GuardedBuffer tails_buffer, block_positions_buffer;
    twin::GuardedBuffer dead_buffer, pooled_buffer;
    twin::GuardedBuffer tail_snapshot_buffer, block_pos_snapshot_buffer;
    const bool allocated =
        raw_buffer.allocate(context, label + " raw keys", raw_bytes, error) &&
        steps_buffer.allocate(context, label + " steps", steps_bytes, error) &&
        gamma_buffer.allocate(context, label + " gamma", gamma_bytes, error) &&
        rope_buffer.allocate(context, label + " RoPE table", rope_bytes,
                             error) &&
        tails_buffer.allocate(context, label + " layer tails", tails_bytes,
                              error) &&
        block_positions_buffer.allocate(
            context, label + " layer block positions",
            block_positions_bytes, error) &&
        dead_buffer.allocate(context, label + " dead key", dead_bytes,
                             error) &&
        pooled_buffer.allocate(context, label + " pooled keys", pooled_bytes,
                               error) &&
        tail_snapshot_buffer.allocate(context, label + " tail snapshot",
                                      snapshot_tail_bytes, error) &&
        block_pos_snapshot_buffer.allocate(
            context, label + " block-position snapshot",
            snapshot_block_pos_bytes, error);
    check(allocated, label + " allocations", error);
    if (!allocated || !raw_buffer.view() || !steps_buffer.view() ||
        !gamma_buffer.view() || !rope_buffer.view() || !tails_buffer.view() ||
        !block_positions_buffer.view() || !dead_buffer.view() ||
        !pooled_buffer.view() || !tail_snapshot_buffer.view() ||
        !block_pos_snapshot_buffer.view()) {
        return false;
    }

    std::memcpy(raw_buffer.payload().data(), raw_keys.data(), raw_bytes);
    std::memcpy(steps_buffer.payload().data(), steps.data(), steps_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(), gamma_bytes);
    std::memcpy(rope_buffer.payload().data(), rope.data(), rope_bytes);
    std::memcpy(tails_buffer.payload().data(), all_tails.data(), tails_bytes);
    std::memcpy(block_positions_buffer.payload().data(),
                all_block_positions.data(), block_positions_bytes);
    std::memcpy(dead_buffer.payload().data(), initial.dead.data(), dead_bytes);
    std::memcpy(pooled_buffer.payload().data(), initial.pooled.data(),
                pooled_bytes);
    std::fill(tail_snapshot_buffer.payload().begin(),
              tail_snapshot_buffer.payload().end(), 0xa5u);
    std::fill(block_pos_snapshot_buffer.payload().begin(),
              block_pos_snapshot_buffer.payload().end(), 0x5au);

    raw_buffer.capture_before_dispatch();
    steps_buffer.capture_before_dispatch();
    gamma_buffer.capture_before_dispatch();
    rope_buffer.capture_before_dispatch();
    tails_buffer.capture_before_dispatch();
    block_positions_buffer.capture_before_dispatch();
    dead_buffer.capture_before_dispatch();
    pooled_buffer.capture_before_dispatch();
    tail_snapshot_buffer.capture_before_dispatch();
    block_pos_snapshot_buffer.capture_before_dispatch();

    const BufferRef target_tail = buffer_subview(
        tails_buffer.view(), target_layer * tail_bytes, tail_bytes);
    const BufferRef target_block_position = buffer_subview(
        block_positions_buffer.view(),
        target_layer * sizeof(std::int32_t), sizeof(std::int32_t));
    const QsaIndexerSnapshotBuffers snapshot_buffers{
        tails_buffer.view(), block_positions_buffer.view(),
        tail_snapshot_buffer.view(), block_pos_snapshot_buffer.view()};
    const QsaIndexerAppendBuffers append_buffers{
        raw_buffer.view(),
        steps_buffer.view(),
        gamma_buffer.view(),
        rope_buffer.view(),
        target_tail,
        dead_buffer.view(),
        pooled_buffer.view(),
        target_block_position,
        tail_snapshot_buffer.view(),
        block_pos_snapshot_buffer.view(),
    };
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    if (shared_encoder) {
        const bool encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                if (commit &&
                    !encode_qsa_indexer_snapshot(
                        context, encoder, kTailFloats, 1u, snapshot_buffers,
                        encode_error)) {
                    return false;
                }
                if (!encode_qsa_indexer_append(
                        context, encoder, token_rows, cell_capacity,
                        QsaIndexerAppendMode::window, 0u, kRopeZeroCos,
                        kRopeZeroSin, append_buffers, encode_error)) {
                    return false;
                }
                return !commit || encode_qsa_indexer_append(
                    context, encoder, n_keep, cell_capacity,
                    QsaIndexerAppendMode::commit, qsa_layer_index,
                    kRopeZeroCos, kRopeZeroSin, append_buffers, encode_error);
            },
            error);
        if (!encoded) {
            check(false, label + " shared encode", error);
            return false;
        }
    } else {
        if (commit &&
            !encode_qsa_indexer_snapshot(
                context, *command, kTailFloats, 1u, snapshot_buffers, error)) {
            check(false, label + " snapshot encode", error);
            command->discard();
            return false;
        }
        if (!encode_qsa_indexer_append(
                context, *command, token_rows, cell_capacity,
                QsaIndexerAppendMode::window, 0u, kRopeZeroCos, kRopeZeroSin,
                append_buffers, error)) {
            check(false, label + " window encode", error);
            command->discard();
            return false;
        }
        if (commit &&
            !encode_qsa_indexer_append(
                context, *command, n_keep, cell_capacity,
                QsaIndexerAppendMode::commit, qsa_layer_index, kRopeZeroCos,
                kRopeZeroSin, append_buffers, error)) {
            check(false, label + " commit encode", error);
            command->discard();
            return false;
        }
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer",
              "command buffer became unavailable");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }

    std::vector<float> expected_tails = all_tails;
    std::copy(expected.tail.begin(), expected.tail.end(),
              expected_tails.begin() + target_layer * kTailFloats);
    std::vector<std::int32_t> expected_block_positions =
        all_block_positions;
    expected_block_positions[target_layer] = expected.block_position;
    const auto actual_tails = typed<float>(tails_buffer);
    const auto tail_bytes_match = twin::compare_bytes(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(actual_tails.data()),
            tails_bytes),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_tails.data()),
            tails_bytes));
    check(!tail_bytes_match, label + " all layer tails match exact fp16 values");
    const auto actual_block_positions =
        typed<std::int32_t>(block_positions_buffer);
    const auto block_pos_bytes_match = twin::compare_bytes(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(
                actual_block_positions.data()),
            block_positions_bytes),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(
                expected_block_positions.data()),
            block_positions_bytes));
    check(!block_pos_bytes_match,
          label + " all layer block positions match exactly");

    std::vector<twin::tf> expected_dead(expected.dead.size());
    std::transform(expected.dead.begin(), expected.dead.end(),
                   expected_dead.begin(),
                   [](float value) { return twin::tf{value}; });
    const auto dead_mismatch = twin::compare_floats(
        typed<float>(dead_buffer), expected_dead,
        {twin::Comparison::units, 64.0});
    check(!dead_mismatch, label + " spare key matches U(64)");
    std::vector<twin::tf> expected_pooled(expected.pooled.size());
    std::transform(expected.pooled.begin(), expected.pooled.end(),
                   expected_pooled.begin(),
                   [](float value) { return twin::tf{value}; });
    const auto pooled_mismatch = twin::compare_floats(
        typed<float>(pooled_buffer), expected_pooled,
        {twin::Comparison::units, 64.0});
    check(!pooled_mismatch, label + " pooled rows match U(64)");

    bool snapshot_match = true;
    if (commit) {
        const auto snapshot_tails = typed<float>(tail_snapshot_buffer);
        const auto mismatch = twin::compare_bytes(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(snapshot_tails.data()),
                snapshot_tail_bytes),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(all_tails.data()),
                snapshot_tail_bytes));
        const auto snapshot_positions =
            typed<std::int32_t>(block_pos_snapshot_buffer);
        const auto position_mismatch = twin::compare_bytes(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    snapshot_positions.data()),
                snapshot_block_pos_bytes),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    all_block_positions.data()),
                snapshot_block_pos_bytes));
        snapshot_match = !mismatch && !position_mismatch;
        check(snapshot_match,
              label + " commit restores the pre-window packed snapshot");
    }

    const bool inputs_unchanged =
        !raw_buffer.first_unwritten_change({}) &&
        !steps_buffer.first_unwritten_change({}) &&
        !gamma_buffer.first_unwritten_change({}) &&
        !rope_buffer.first_unwritten_change({});
    check(inputs_unchanged, label + " raw, steps, gamma and RoPE remain read-only");
    const std::array<twin::ByteRange, 1> tails_written{{{0u, tails_bytes}}};
    const std::array<twin::ByteRange, 1> block_positions_written{{
        {0u, block_positions_bytes}}};
    const std::array<twin::ByteRange, 1> dead_written{{{0u, dead_bytes}}};
    const std::array<twin::ByteRange, 1> pooled_written{{{0u, pooled_bytes}}};
    const std::array<twin::ByteRange, 1> tail_snapshot_written{{
        {0u, commit ? snapshot_tail_bytes : 0u}}};
    const std::array<twin::ByteRange, 1> block_pos_snapshot_written{{
        {0u, commit ? snapshot_block_pos_bytes : 0u}}};
    const bool outer_guards =
        !tails_buffer.first_unwritten_change(tails_written) &&
        !block_positions_buffer.first_unwritten_change(
            block_positions_written) &&
        !dead_buffer.first_unwritten_change(dead_written) &&
        !pooled_buffer.first_unwritten_change(pooled_written) &&
        !tail_snapshot_buffer.first_unwritten_change(tail_snapshot_written) &&
        !block_pos_snapshot_buffer.first_unwritten_change(
            block_pos_snapshot_written);
    check(outer_guards, label + " allocation guards remain unchanged");
    return !tail_bytes_match && !block_pos_bytes_match && !dead_mismatch &&
           !pooled_mismatch && snapshot_match && inputs_unchanged &&
           outer_guards;
}

bool test_indexer_commit_batch(Context& context) {
    constexpr std::uint32_t kRows = 8u;
    constexpr std::uint32_t kAcceptedRows = 3u;
    constexpr std::uint32_t kCellCapacity = 16u;
    constexpr std::size_t kLayers = INDEXER_SNAPSHOT_LAYERS;
    constexpr std::size_t kTailFloats =
        3u * static_cast<std::size_t>(INDEXER_HEAD_DIM);
    constexpr std::size_t kDeadFloats = INDEXER_HEAD_DIM;
    constexpr std::size_t kPoolRows =
        kCellCapacity / INDEXER_BLOCK_CELLS + 2u;
    constexpr std::size_t kPooledFloats =
        kPoolRows * static_cast<std::size_t>(INDEXER_HEAD_DIM);
    constexpr std::size_t kTailBytes = kTailFloats * sizeof(float);
    constexpr std::size_t kDeadBytes = kDeadFloats * sizeof(float);
    constexpr std::size_t kPooledBytes = kPooledFloats * sizeof(float);
    constexpr std::size_t kRawLayerBytes =
        static_cast<std::size_t>(kRows) * INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t kGammaLayerBytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t kRopeBytes =
        static_cast<std::size_t>(kCellCapacity) * 32u * 2u * sizeof(float);
    constexpr float kRopeZeroCos = 1.0625f;
    constexpr float kRopeZeroSin = 0.0f;
    constexpr std::array<std::int32_t, kRows> kPositions{
        1, 2, 3, 4, 5, 6, 7, 8};

    std::string error;
    auto allocate = [&](const std::string& name, std::size_t bytes) {
        BufferRef view = context.buffers().allocate(name, bytes, error);
        check(static_cast<bool>(view), "allocate " + name, error);
        return view;
    };
    const BufferRef raw_storage =
        allocate("QSA batch raw replay", kLayers * kRawLayerBytes);
    const BufferRef gamma_storage =
        allocate("QSA batch gamma", kLayers * kGammaLayerBytes);
    const BufferRef tails =
        allocate("QSA batch indexer tails", kLayers * kTailBytes);
    const BufferRef dead =
        allocate("QSA batch indexer dead keys", kLayers * kDeadBytes);
    const BufferRef pooled =
        allocate("QSA batch indexer pooled keys", kLayers * kPooledBytes);
    const BufferRef block_positions =
        allocate("QSA batch indexer block positions",
                 kLayers * sizeof(std::int32_t));
    const BufferRef steps =
        allocate("QSA batch steps",
                 kRows * sizeof(StepRow) + kTailBytes);
    const BufferRef rope =
        allocate("QSA batch RoPE with inactive suffix",
                 kRopeBytes + kTailBytes);
    const BufferRef tail_snapshot =
        allocate("QSA batch tail snapshot", kLayers * kTailBytes);
    const BufferRef block_pos_snapshot =
        allocate("QSA batch block-position snapshot",
                 kLayers * sizeof(std::int32_t));
    const BufferRef alternate_steps =
        allocate("QSA batch alternate steps", kRows * sizeof(StepRow));
    const BufferRef alternate_rope =
        allocate("QSA batch alternate RoPE", kRopeBytes);
    const BufferRef alternate_tail_snapshot =
        allocate("QSA batch alternate tail snapshot",
                 kLayers * kTailBytes);
    const BufferRef alternate_block_pos_snapshot =
        allocate("QSA batch alternate block-position snapshot",
                 kLayers * sizeof(std::int32_t));
    if (!raw_storage || !gamma_storage || !tails || !dead || !pooled ||
        !block_positions || !steps || !rope || !tail_snapshot ||
        !block_pos_snapshot || !alternate_steps || !alternate_rope ||
        !alternate_tail_snapshot || !alternate_block_pos_snapshot) {
        return false;
    }

    std::array<StepRow, kRows> expected_steps{};
    for (std::uint32_t row = 0u; row < kRows; ++row) {
        const std::int32_t position = kPositions[row];
        const std::int32_t n_kv = position + 1;
        expected_steps[row] = StepRow{
            position, n_kv,
            static_cast<std::int32_t>(n_kv / INDEXER_BLOCK_CELLS),
            std::min<std::int32_t>(n_kv, 2051)};
    }
    std::memset(steps.contents, 0x6d, steps.length);
    std::memcpy(steps.contents, expected_steps.data(),
                kRows * sizeof(StepRow));
    std::vector<std::byte> initial_step_storage(steps.length);
    std::memcpy(initial_step_storage.data(), steps.contents, steps.length);
    std::vector<float> expected_rope(kRopeBytes / sizeof(float));
    auto* rope_values = static_cast<float*>(rope.contents);
    for (std::uint32_t cell = 0u; cell < kCellCapacity; ++cell) {
        for (std::uint32_t pair = 0u; pair < 32u; ++pair) {
            const float angle =
                static_cast<float>((cell * 13u + pair * 7u) % 257u) *
                0.013f;
            const std::size_t index =
                (static_cast<std::size_t>(cell) * 32u + pair) * 2u;
            expected_rope[index] = std::cos(angle) * 1.0625f;
            expected_rope[index + 1u] = std::sin(angle) * 1.0625f;
        }
    }
    std::memcpy(rope.contents, expected_rope.data(), kRopeBytes);
    std::memcpy(alternate_steps.contents, expected_steps.data(),
                kRows * sizeof(StepRow));
    std::memcpy(alternate_rope.contents, expected_rope.data(), kRopeBytes);

    std::array<oracle::QsaIndexerState, kLayers> initial{};
    std::array<oracle::QsaIndexerState, kLayers> expected{};
    std::array<std::vector<float>, kLayers> raw_keys{};
    std::array<std::vector<float>, kLayers> gammas{};
    std::array<QsaIndexerAppendBuffers, kLayers> layer_buffers{};
    std::vector<float> initial_tails(kLayers * kTailFloats);
    std::vector<float> initial_dead(kLayers * kDeadFloats);
    std::vector<float> initial_pooled(kLayers * kPooledFloats);
    std::vector<std::int32_t> initial_block_positions(kLayers);
    std::memset(tail_snapshot.contents, 0xa5, tail_snapshot.length);
    std::memset(block_pos_snapshot.contents, 0x5a,
                block_pos_snapshot.length);

    for (std::uint32_t layer = 0u; layer < kLayers; ++layer) {
        auto& start = initial[layer];
        start.tail.resize(kTailFloats);
        start.dead.resize(kDeadFloats);
        start.pooled.resize(kPooledFloats);
        start.block_position = 0;
        for (std::size_t index = 0u; index < start.tail.size(); ++index) {
            start.tail[index] = 0.015625f * static_cast<float>(layer + 1u) +
                                static_cast<float>(index % 23u) * 0.0005f;
        }
        for (std::size_t index = 0u; index < start.dead.size(); ++index) {
            start.dead[index] = -0.375f +
                                static_cast<float>(layer) * 0.0078125f +
                                static_cast<float>(index % 19u) * 0.00025f;
        }
        for (std::size_t index = 0u; index < start.pooled.size(); ++index) {
            start.pooled[index] = -0.625f +
                                  static_cast<float>(layer) * 0.00390625f +
                                  static_cast<float>(index % 31u) * 0.000125f;
        }
        std::copy(start.tail.begin(), start.tail.end(),
                  initial_tails.begin() + layer * kTailFloats);
        std::copy(start.dead.begin(), start.dead.end(),
                  initial_dead.begin() + layer * kDeadFloats);
        std::copy(start.pooled.begin(), start.pooled.end(),
                  initial_pooled.begin() + layer * kPooledFloats);
        initial_block_positions[layer] = start.block_position;
        std::memcpy(static_cast<std::byte*>(tails.contents) +
                        static_cast<std::size_t>(layer) * kTailBytes,
                    start.tail.data(), kTailBytes);
        std::memcpy(static_cast<std::byte*>(dead.contents) +
                        static_cast<std::size_t>(layer) * kDeadBytes,
                    start.dead.data(), kDeadBytes);
        std::memcpy(static_cast<std::byte*>(pooled.contents) +
                        static_cast<std::size_t>(layer) * kPooledBytes,
                    start.pooled.data(), kPooledBytes);
        std::memcpy(static_cast<std::byte*>(block_positions.contents) +
                        static_cast<std::size_t>(layer) *
                            sizeof(std::int32_t),
                    &start.block_position, sizeof(std::int32_t));

        raw_keys[layer].resize(kRows * INDEXER_HEAD_DIM);
        gammas[layer].resize(INDEXER_HEAD_DIM);
        for (std::uint32_t row = 0u; row < kRows; ++row) {
            for (std::uint32_t dimension = 0u;
                 dimension < INDEXER_HEAD_DIM; ++dimension) {
                const std::size_t index =
                    static_cast<std::size_t>(row) * INDEXER_HEAD_DIM +
                    dimension;
                const int pattern = static_cast<int>(
                    (dimension * 11u + row * 17u + layer * 29u) % 101u) -
                    50;
                raw_keys[layer][index] =
                    static_cast<float>(pattern) * 0.00390625f;
            }
        }
        for (std::uint32_t dimension = 0u;
             dimension < INDEXER_HEAD_DIM; ++dimension) {
            gammas[layer][dimension] =
                0.75f + static_cast<float>(layer) * 0.015625f +
                static_cast<float>(dimension % 17u) * 0.0009765625f;
        }
        std::memcpy(static_cast<std::byte*>(raw_storage.contents) +
                        static_cast<std::size_t>(layer) * kRawLayerBytes,
                    raw_keys[layer].data(), kRawLayerBytes);
        std::memcpy(static_cast<std::byte*>(gamma_storage.contents) +
                        static_cast<std::size_t>(layer) *
                            kGammaLayerBytes,
                    gammas[layer].data(), kGammaLayerBytes);

        layer_buffers[layer] = QsaIndexerAppendBuffers{
            raw_storage.slice(
                static_cast<std::size_t>(layer) * kRawLayerBytes,
                kRawLayerBytes),
            steps,
            gamma_storage.slice(
                static_cast<std::size_t>(layer) * kGammaLayerBytes,
                kGammaLayerBytes),
            rope,
            tails.slice(static_cast<std::size_t>(layer) * kTailBytes,
                        kTailBytes),
            dead.slice(static_cast<std::size_t>(layer) * kDeadBytes,
                       kDeadBytes),
            pooled.slice(static_cast<std::size_t>(layer) * kPooledBytes,
                         kPooledBytes),
            block_positions.slice(
                static_cast<std::size_t>(layer) * sizeof(std::int32_t),
                sizeof(std::int32_t)),
            tail_snapshot,
            block_pos_snapshot,
        };
        check(static_cast<bool>(layer_buffers[layer].raw_keys) &&
                  static_cast<bool>(layer_buffers[layer].gamma) &&
                  static_cast<bool>(layer_buffers[layer].tail) &&
                  static_cast<bool>(layer_buffers[layer].dead) &&
                  static_cast<bool>(layer_buffers[layer].pooled) &&
                  static_cast<bool>(layer_buffers[layer].block_position),
              "slice QSA indexer batch layer " + std::to_string(layer));

        oracle::QsaIndexerState speculative = start;
        oracle::qsa_indexer_append(
            raw_keys[layer], kPositions, gammas[layer],
            std::span<const float>(rope_values,
                                   kRopeBytes / sizeof(float)),
            kRopeZeroCos, kRopeZeroSin, speculative);
        expected[layer] = speculative;
        expected[layer].tail = start.tail;
        expected[layer].block_position = start.block_position;
        oracle::qsa_indexer_append(
            std::span<const float>(raw_keys[layer]).first(
                kAcceptedRows * INDEXER_HEAD_DIM),
            std::span<const std::int32_t>(kPositions).first(kAcceptedRows),
            gammas[layer],
            std::span<const float>(rope_values,
                                   kRopeBytes / sizeof(float)),
            kRopeZeroCos, kRopeZeroSin, expected[layer]);
    }

    auto state_unchanged = [&] {
        return std::memcmp(tails.contents, initial_tails.data(),
                           initial_tails.size() * sizeof(float)) == 0 &&
               std::memcmp(dead.contents, initial_dead.data(),
                           initial_dead.size() * sizeof(float)) == 0 &&
               std::memcmp(pooled.contents, initial_pooled.data(),
                           initial_pooled.size() * sizeof(float)) == 0 &&
               std::memcmp(block_positions.contents,
                           initial_block_positions.data(),
                           initial_block_positions.size() *
                               sizeof(std::int32_t)) == 0;
    };
    {
        auto command = context.make_command_buffer(
            "QSA indexer batch rejects layer count", error);
        check(command != nullptr, "create QSA batch layer-count command",
              error);
        if (!command) return false;
        const bool accepted = encode_qsa_indexer_commit_batch(
            context, *command, kAcceptedRows, kCellCapacity, kRopeZeroCos,
            kRopeZeroSin,
            std::span<const QsaIndexerAppendBuffers>(layer_buffers)
                .first(kLayers - 1u),
            error);
        const bool rejected = !accepted &&
            command->unsafe_native_handle() == nullptr &&
            error.find("exactly 12 layers") != std::string::npos &&
            state_unchanged();
        check(rejected, "QSA indexer batch rejects a short layer list", error);
        if (!rejected) return false;
    }
    auto rejects_shared_view_mismatch = [&](
                                            const std::string& label,
                                            const std::array<
                                                QsaIndexerAppendBuffers,
                                                kLayers>& invalid_layers) {
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command", error);
        if (!command) return false;
        const bool accepted = encode_qsa_indexer_commit_batch(
            context, *command, kAcceptedRows, kCellCapacity, kRopeZeroCos,
            kRopeZeroSin, invalid_layers, error);
        const bool rejected = !accepted &&
            command->unsafe_native_handle() == nullptr &&
            error.find("must share steps, RoPE and snapshot views") !=
                std::string::npos &&
            state_unchanged();
        check(rejected, label, error);
        return rejected;
    };
    {
        auto invalid_layers = layer_buffers;
        invalid_layers.back().steps = alternate_steps;
        if (!rejects_shared_view_mismatch(
                "QSA indexer batch rejects a distinct valid StepRow view",
                invalid_layers)) {
            return false;
        }
    }
    {
        auto invalid_layers = layer_buffers;
        invalid_layers.back().rope = alternate_rope;
        if (!rejects_shared_view_mismatch(
                "QSA indexer batch rejects a distinct valid RoPE view",
                invalid_layers)) {
            return false;
        }
    }
    {
        auto invalid_layers = layer_buffers;
        invalid_layers.back().tail_snapshot = alternate_tail_snapshot;
        if (!rejects_shared_view_mismatch(
                "QSA indexer batch rejects a distinct tail snapshot view",
                invalid_layers)) {
            return false;
        }
    }
    {
        auto invalid_layers = layer_buffers;
        invalid_layers.back().block_pos_snapshot =
            alternate_block_pos_snapshot;
        if (!rejects_shared_view_mismatch(
                "QSA indexer batch rejects a distinct block-position snapshot",
                invalid_layers)) {
            return false;
        }
    }
    {
        auto invalid_layers = layer_buffers;
        invalid_layers.back().tail = invalid_layers.front().raw_keys;
        auto command = context.make_command_buffer(
            "QSA indexer batch rejects cross-layer alias", error);
        check(command != nullptr, "create QSA batch alias command", error);
        if (!command) return false;
        const bool accepted = encode_qsa_indexer_commit_batch(
            context, *command, kAcceptedRows, kCellCapacity, kRopeZeroCos,
            kRopeZeroSin, invalid_layers, error);
        const bool rejected = !accepted &&
            command->unsafe_native_handle() == nullptr &&
            error.find("overlaps layer 0 raw keys") != std::string::npos &&
            state_unchanged();
        check(rejected,
              "QSA indexer batch rejects cross-layer output/input alias",
              error);
        if (!rejected) return false;
    }
    {
        auto reusable_layers = layer_buffers;
        reusable_layers.back().tail = steps.slice(
            kAcceptedRows * sizeof(StepRow), kTailBytes);
        auto command = context.make_command_buffer(
            "QSA indexer batch permits inactive StepRow suffix reuse", error);
        check(command != nullptr, "create QSA inactive-suffix command", error);
        if (!command) return false;
        const bool accepted = encode_qsa_indexer_commit_batch(
            context, *command, kAcceptedRows, kCellCapacity, kRopeZeroCos,
            kRopeZeroSin, reusable_layers, error);
        const bool allowed = accepted &&
            reusable_layers.back().tail.length == kTailBytes &&
            state_unchanged();
        check(allowed,
              "QSA indexer batch permits output reuse beyond active StepRows",
              error);
        command->discard();
        if (!allowed) return false;
    }
    {
        auto reusable_layers = layer_buffers;
        reusable_layers.back().tail =
            rope.slice(kRopeBytes, kTailBytes);
        auto command = context.make_command_buffer(
            "QSA indexer batch permits inactive RoPE suffix reuse", error);
        check(command != nullptr, "create QSA RoPE-suffix command", error);
        if (!command) return false;
        const bool accepted = encode_qsa_indexer_commit_batch(
            context, *command, kAcceptedRows, kCellCapacity, kRopeZeroCos,
            kRopeZeroSin, reusable_layers, error);
        const bool allowed = accepted &&
            reusable_layers.back().tail.length == kTailBytes &&
            state_unchanged();
        check(allowed,
              "QSA indexer batch permits output reuse beyond active RoPE",
              error);
        command->discard();
        if (!allowed) return false;
    }
    {
        auto invalid_layers = layer_buffers;
        invalid_layers.back().steps =
            steps.slice(0u, sizeof(StepRow) - 1u);
        auto command = context.make_command_buffer(
            "QSA indexer batch caller-owned rollback", error);
        check(command != nullptr, "create QSA batch rollback command", error);
        if (!command) return false;
        const bool accepted = Encoder::encode(
            context, *command, "QSA indexer batch enclosing transaction",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encode_qsa_indexer_append(
                        context, encoder, kAcceptedRows, kCellCapacity,
                        QsaIndexerAppendMode::commit, 0u, kRopeZeroCos,
                        kRopeZeroSin, layer_buffers[0], encode_error)) {
                    return false;
                }
                return encode_qsa_indexer_commit_batch(
                    context, encoder, kAcceptedRows, kCellCapacity,
                    kRopeZeroCos, kRopeZeroSin, invalid_layers,
                    encode_error);
            },
            error);
        const bool rejected = !accepted &&
            command->unsafe_native_handle() == nullptr &&
            error.find("step rows buffer is undersized or misaligned") !=
                std::string::npos &&
            state_unchanged();
        check(rejected,
              "caller-owned QSA batch failure discards earlier replay",
              error);
        if (!rejected) return false;
    }

    auto command = context.make_command_buffer(
        "QSA indexer all-layer accepted-prefix commit", error);
    check(command != nullptr, "create QSA indexer commit batch command",
          error);
    if (!command) return false;
    const QsaIndexerSnapshotBuffers snapshot_buffers{
        tails, block_positions, tail_snapshot, block_pos_snapshot};
    const bool encoded = Encoder::encode(
        context, *command, "QSA indexer speculative and accepted replay",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_indexer_snapshot(
                    context, encoder, kTailFloats, 1u, snapshot_buffers,
                    encode_error)) {
                return false;
            }
            for (std::uint32_t layer = 0u; layer < kLayers; ++layer) {
                if (!encode_qsa_indexer_append(
                        context, encoder, kRows, kCellCapacity,
                        QsaIndexerAppendMode::window, layer, kRopeZeroCos,
                        kRopeZeroSin, layer_buffers[layer], encode_error)) {
                    return false;
                }
            }
            return encode_qsa_indexer_commit_batch(
                context, encoder, kAcceptedRows, kCellCapacity,
                kRopeZeroCos, kRopeZeroSin, layer_buffers, encode_error);
        },
        error);
    check(encoded, "encode all-layer QSA indexer commit batch", error);
    if (!encoded || !command->commit(error)) {
        check(false, "submit all-layer QSA indexer commit batch", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, "QSA indexer commit batch native command buffer",
              "command buffer became unavailable");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, "QSA indexer commit batch GPU completion", error);
        return false;
    }

    bool tails_match = true;
    bool block_positions_match = true;
    bool dead_match = true;
    bool pooled_match = true;
    bool inputs_unchanged =
        std::memcmp(steps.contents, initial_step_storage.data(),
                    steps.length) == 0 &&
        std::memcmp(rope.contents, expected_rope.data(), kRopeBytes) == 0;
    for (std::uint32_t layer = 0u; layer < kLayers; ++layer) {
        const BufferRef tail_view = layer_buffers[layer].tail;
        const BufferRef dead_view = layer_buffers[layer].dead;
        const BufferRef pooled_view = layer_buffers[layer].pooled;
        const auto* actual_tail =
            static_cast<const float*>(tail_view.contents);
        const auto* actual_dead =
            static_cast<const float*>(dead_view.contents);
        const auto* actual_pooled =
            static_cast<const float*>(pooled_view.contents);
        tails_match &= !twin::compare_bytes(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(actual_tail),
                kTailBytes),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    expected[layer].tail.data()),
                kTailBytes));
        const std::int32_t actual_block_position =
            *static_cast<const std::int32_t*>(
                layer_buffers[layer].block_position.contents);
        block_positions_match &=
            actual_block_position == expected[layer].block_position;
        std::vector<twin::tf> expected_dead(kDeadFloats);
        std::transform(expected[layer].dead.begin(),
                       expected[layer].dead.end(), expected_dead.begin(),
                       [](float value) { return twin::tf{value}; });
        dead_match &= !twin::compare_floats(
            std::span<const float>(actual_dead, kDeadFloats), expected_dead,
            {twin::Comparison::units, 64.0});
        std::vector<twin::tf> expected_pooled(kPooledFloats);
        std::transform(expected[layer].pooled.begin(),
                       expected[layer].pooled.end(), expected_pooled.begin(),
                       [](float value) { return twin::tf{value}; });
        pooled_match &= !twin::compare_floats(
            std::span<const float>(actual_pooled, kPooledFloats),
            expected_pooled, {twin::Comparison::units, 64.0});
        inputs_unchanged &=
            std::memcmp(layer_buffers[layer].raw_keys.contents,
                        raw_keys[layer].data(), kRawLayerBytes) == 0 &&
            std::memcmp(layer_buffers[layer].gamma.contents,
                        gammas[layer].data(), kGammaLayerBytes) == 0;
    }
    check(tails_match,
          "all 12 QSA tails match accepted-prefix replay exactly");
    check(block_positions_match,
          "all 12 QSA block positions match accepted-prefix replay");
    check(dead_match, "all 12 QSA spare keys match U(64)");
    check(pooled_match,
          "all 12 QSA pooled rows preserve and replay the right blocks");
    check(inputs_unchanged,
          "QSA indexer batch preserves per-layer replay inputs");

    std::vector<std::uint8_t> expected_snapshot(kLayers * kTailBytes);
    std::vector<std::int32_t> expected_snapshot_positions(
        initial_block_positions);
    std::memcpy(expected_snapshot.data(), initial_tails.data(),
                expected_snapshot.size());
    const bool snapshots_match =
        std::memcmp(tail_snapshot.contents, expected_snapshot.data(),
                    expected_snapshot.size()) == 0 &&
        std::memcmp(block_pos_snapshot.contents,
                    expected_snapshot_positions.data(),
                    expected_snapshot_positions.size() *
                        sizeof(std::int32_t)) == 0;
    check(snapshots_match, "QSA batch uses the packed all-layer snapshot");
    return tails_match && block_positions_match && dead_match &&
           pooled_match && inputs_unchanged && snapshots_match;
}

bool test_indexer_append_validation(Context& context,
                                   Context& foreign_context) {
    constexpr std::uint32_t kRows = 2u;
    constexpr std::uint32_t kCellCapacity = 16u;
    constexpr std::size_t kRawBytes =
        kRows * INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t kStepsBytes = kRows * sizeof(StepRow);
    constexpr std::size_t kGammaBytes =
        INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t kRopeBytes =
        kCellCapacity * 32u * 2u * sizeof(float);
    constexpr std::size_t kTailBytes =
        3u * INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t kDeadBytes =
        INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t kPoolBytes =
        (kCellCapacity / INDEXER_BLOCK_CELLS + 2u) *
        INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t kTailSnapshotBytes =
        INDEXER_SNAPSHOT_LAYERS * kTailBytes;
    constexpr std::size_t kBlockPosSnapshotBytes =
        INDEXER_SNAPSHOT_LAYERS * sizeof(std::int32_t);
    std::string error;
    const auto allocate = [&](Context& owner, const char* name,
                              std::size_t bytes) {
        return owner.buffers().allocate(name, bytes, error);
    };
    BufferRef raw_storage = allocate(
        context, "QSA append validation raw",
        kRawBytes + kTailBytes);
    BufferRef raw = raw_storage
        ? buffer_subview(raw_storage, 0u, kRawBytes)
        : BufferRef{};
    BufferRef steps =
        allocate(context, "QSA append validation steps", kStepsBytes);
    BufferRef gamma =
        allocate(context, "QSA append validation gamma", kGammaBytes);
    BufferRef rope =
        allocate(context, "QSA append validation rope", kRopeBytes);
    BufferRef tails =
        allocate(context, "QSA append validation tails", kTailBytes);
    BufferRef dead =
        allocate(context, "QSA append validation dead", kDeadBytes);
    BufferRef pooled =
        allocate(context, "QSA append validation pooled", kPoolBytes);
    BufferRef block_position =
        allocate(context, "QSA append validation block position",
                 sizeof(std::int32_t));
    BufferRef tail_snapshot =
        allocate(context, "QSA append validation tail snapshot",
                 kTailSnapshotBytes);
    BufferRef block_pos_snapshot =
        allocate(context, "QSA append validation block snapshot",
                 kBlockPosSnapshotBytes);
    BufferRef foreign =
        allocate(foreign_context, "QSA append foreign buffer", kRawBytes);
    BufferRef combined_state = allocate(
        context, "QSA append validation shared state",
        kTailBytes + kDeadBytes);
    BufferRef combined_snapshots = allocate(
        context, "QSA append validation shared snapshots",
        kTailSnapshotBytes + kBlockPosSnapshotBytes);
    if (!raw_storage || !raw || !steps || !gamma || !rope || !tails || !dead || !pooled ||
        !block_position || !tail_snapshot || !block_pos_snapshot || !foreign ||
        !combined_state || !combined_snapshots) {
        check(false, "QSA indexer append validation allocations", error);
        return false;
    }
    std::memset(raw_storage.contents, 0, raw_storage.length);
    std::memset(gamma.contents, 0, gamma.length);
    std::memset(rope.contents, 0, rope.length);
    std::memset(tails.contents, 0, tails.length);
    std::memset(dead.contents, 0, dead.length);
    std::memset(pooled.contents, 0, pooled.length);
    auto* step_rows = static_cast<StepRow*>(steps.contents);
    step_rows[0] = StepRow{1, 2, 0, 2};
    step_rows[1] = StepRow{2, 3, 0, 3};
    const QsaIndexerAppendBuffers valid{
        raw, steps, gamma, rope, tails, dead, pooled, block_position,
        tail_snapshot, block_pos_snapshot};
    auto command =
        context.make_command_buffer("QSA indexer append validation", error);
    if (!command) {
        check(false, "QSA indexer append validation command buffer", error);
        return false;
    }
    bool passed = true;
    const auto reject = [&](const std::string& label, auto&& call,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !call() && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    const auto encode = [&](std::uint32_t rows, std::uint32_t capacity,
                            QsaIndexerAppendMode mode, std::uint32_t layer,
                            const QsaIndexerAppendBuffers& buffers) {
        return encode_qsa_indexer_append(
            context, *command, rows, capacity, mode, layer, 1.125f, 0.0f,
            buffers, error);
    };
    const auto reject_shared = [&](Encoder& encoder,
                                   std::string& encode_error,
                                   const std::string& label,
                                   std::uint32_t rows,
                                   std::uint32_t capacity,
                                   QsaIndexerAppendMode mode,
                                   std::uint32_t layer,
                                   const QsaIndexerAppendBuffers& buffers,
                                   std::string_view error_fragment) {
        const bool rejected =
            !encode_qsa_indexer_append(
                context, encoder, rows, capacity, mode, layer, 1.125f, 0.0f,
                buffers, encode_error) &&
            encode_error.find(error_fragment) != std::string::npos;
        check(rejected, label, encode_error);
        encode_error.clear();
        return rejected;
    };

    passed &= reject(
        "QSA indexer append rejects zero rows",
        [&] {
            return encode(0u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, valid);
        },
        "row count");
    passed &= reject(
        "QSA indexer append rejects rows above eight",
        [&] {
            return encode(9u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, valid);
        },
        "row count");
    passed &= reject(
        "QSA indexer append rejects unknown mode",
        [&] {
            return encode(1u, kCellCapacity,
                          static_cast<QsaIndexerAppendMode>(7u), 0u, valid);
        },
        "mode");
    passed &= reject(
        "QSA indexer append rejects a non-four-cell capacity",
        [&] {
            return encode(1u, 15u, QsaIndexerAppendMode::window, 0u, valid);
        },
        "capacity");
    passed &= reject(
        "QSA indexer append rejects a layer outside the 12 QSA layers",
        [&] {
            return encode(1u, kCellCapacity, QsaIndexerAppendMode::window,
                          INDEXER_SNAPSHOT_LAYERS, valid);
        },
        "layer index");

    step_rows[1].pos = 3;
    passed &= reject(
        "QSA indexer append rejects noncontiguous steps",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, valid);
        },
        "contiguous");
    step_rows[1].pos = 2;
    step_rows[0].pos = static_cast<std::int32_t>(kCellCapacity - 1u);
    step_rows[1].pos = static_cast<std::int32_t>(kCellCapacity);
    passed &= reject(
        "QSA indexer append rejects a batch beyond cell capacity",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, valid);
        },
        "cell capacity");
    step_rows[0].pos = 1;
    step_rows[1].pos = 2;

    QsaIndexerAppendBuffers short_view = valid;
    short_view.raw_keys.length -= sizeof(float);
    passed &= reject(
        "QSA indexer append rejects a short raw-key view",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, short_view);
        },
        "undersized");
    short_view = valid;
    short_view.steps.length -= sizeof(StepRow);
    passed &= reject(
        "QSA indexer append rejects a short StepRow view",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, short_view);
        },
        "undersized");
    short_view = valid;
    short_view.gamma.length -= sizeof(float);
    passed &= reject(
        "QSA indexer append rejects a short gamma view",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, short_view);
        },
        "undersized");
    short_view = valid;
    short_view.rope = buffer_subview(rope, 0u, 12u * 32u * 2u *
                                             sizeof(float));
    passed &= reject(
        "QSA indexer append rejects a short RoPE table",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, short_view);
        },
        "undersized");
    short_view = valid;
    short_view.tail.length -= sizeof(float);
    passed &= reject(
        "QSA indexer append rejects a short tail view",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, short_view);
        },
        "undersized");
    short_view = valid;
    short_view.dead.length -= sizeof(float);
    passed &= reject(
        "QSA indexer append rejects a short dead-key view",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, short_view);
        },
        "undersized");
    short_view = valid;
    short_view.pooled.length -= sizeof(float);
    passed &= reject(
        "QSA indexer append rejects a short pooled-key view",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, short_view);
        },
        "undersized");

    QsaIndexerAppendBuffers missing_commit_snapshot = valid;
    missing_commit_snapshot.tail_snapshot = {};
    passed &= reject(
        "QSA indexer commit rejects a missing tail snapshot",
        [&] {
            return encode(1u, kCellCapacity, QsaIndexerAppendMode::commit,
                          INDEXER_SNAPSHOT_LAYERS - 1u,
                          missing_commit_snapshot);
        },
        "tail snapshot");
    QsaIndexerAppendBuffers exact_snapshot_alias = valid;
    exact_snapshot_alias.tail_snapshot = combined_snapshots;
    exact_snapshot_alias.block_pos_snapshot = combined_snapshots;
    passed &= reject(
        "QSA indexer commit rejects exact snapshot overlap",
        [&] {
            return encode(1u, kCellCapacity, QsaIndexerAppendMode::commit,
                          0u, exact_snapshot_alias);
        },
        "snapshots must be disjoint");
    QsaIndexerAppendBuffers partial_snapshot_alias = valid;
    partial_snapshot_alias.tail_snapshot = buffer_subview(
        combined_snapshots, 0u, kTailSnapshotBytes);
    partial_snapshot_alias.block_pos_snapshot = buffer_subview(
        combined_snapshots, kTailSnapshotBytes - sizeof(std::int32_t),
        kBlockPosSnapshotBytes);
    passed &= reject(
        "QSA indexer commit rejects partial snapshot overlap",
        [&] {
            return encode(1u, kCellCapacity, QsaIndexerAppendMode::commit,
                          0u, partial_snapshot_alias);
        },
        "snapshots must be disjoint");
    QsaIndexerAppendBuffers disjoint_shared_snapshots = valid;
    disjoint_shared_snapshots.tail_snapshot = buffer_subview(
        combined_snapshots, 0u, kTailSnapshotBytes);
    disjoint_shared_snapshots.block_pos_snapshot = buffer_subview(
        combined_snapshots, kTailSnapshotBytes,
        kBlockPosSnapshotBytes);
    error.clear();
    const bool shared_snapshot_views_accepted =
        encode(1u, kCellCapacity, QsaIndexerAppendMode::commit, 0u,
               disjoint_shared_snapshots);
    check(shared_snapshot_views_accepted,
          "QSA indexer commit accepts disjoint snapshot views in one allocation",
          error);
    passed &= shared_snapshot_views_accepted;

    QsaIndexerAppendBuffers alias = valid;
    alias.tail = raw_storage;
    passed &= reject(
        "QSA indexer append rejects output/input alias",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, alias);
        },
        "overlap inputs");
    alias = valid;
    alias.tail = buffer_subview(combined_state, 0u, kTailBytes);
    alias.dead = buffer_subview(combined_state, 0u, kDeadBytes);
    passed &= reject(
        "QSA indexer append rejects overlapping writable views",
        [&] {
            return encode(2u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, alias);
        },
        "outputs must be disjoint");
    QsaIndexerAppendBuffers foreign_buffer = valid;
    foreign_buffer.raw_keys = foreign;
    passed &= reject(
        "QSA indexer append rejects foreign Context buffers",
        [&] {
            return encode(1u, kCellCapacity, QsaIndexerAppendMode::window,
                          0u, foreign_buffer);
        },
        "not owned");

    // The standalone validations above reuse this same unsubmitted buffer.
    bool shared_validation_passed = true;
    const bool shared_validation_encoded = Encoder::encode(
        context, *command,
        "QSA shared indexer append validation",
        [&](Encoder& encoder, std::string& encode_error) {
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects zero rows", 0u,
                kCellCapacity, QsaIndexerAppendMode::window, 0u, valid,
                "row count");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects invalid mode", 1u,
                kCellCapacity, static_cast<QsaIndexerAppendMode>(7u), 0u,
                valid, "mode");
            step_rows[1].pos = 3;
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects noncontiguous steps", 2u,
                kCellCapacity, QsaIndexerAppendMode::window, 0u, valid,
                "contiguous");
            step_rows[1].pos = 2;
            short_view = valid;
            short_view.raw_keys.length -= sizeof(float);
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects a short raw-key view", 2u,
                kCellCapacity, QsaIndexerAppendMode::window, 0u, short_view,
                "undersized");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects missing commit snapshots",
                1u, kCellCapacity, QsaIndexerAppendMode::commit,
                INDEXER_SNAPSHOT_LAYERS - 1u, missing_commit_snapshot,
                "tail snapshot");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects overlapping writable views",
                2u, kCellCapacity, QsaIndexerAppendMode::window, 0u, alias,
                "outputs must be disjoint");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects foreign Context buffers",
                1u, kCellCapacity, QsaIndexerAppendMode::window, 0u,
                foreign_buffer, "not owned");
            QsaIndexerAppendBuffers forged_cpu = valid;
            forged_cpu.steps.contents =
                static_cast<char*>(forged_cpu.steps.contents) +
                sizeof(StepRow);
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects forged CPU metadata", 1u,
                kCellCapacity, QsaIndexerAppendMode::window, 0u, forged_cpu,
                "not owned");
            QsaIndexerAppendBuffers forged_gpu = valid;
            forged_gpu.raw_keys.gpu_address ^= 0x10u;
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared indexer append rejects forged GPU metadata", 1u,
                kCellCapacity, QsaIndexerAppendMode::window, 0u, forged_gpu,
                "not owned");
            encode_error.clear();
            return true;
        },
        error);
    check(shared_validation_encoded,
          "QSA shared indexer append validation body completes", error);
    passed &= shared_validation_encoded && shared_validation_passed;

    std::memset(tails.contents, 0xa5, tails.length);
    std::memset(dead.contents, 0xa5, dead.length);
    std::memset(pooled.contents, 0xa5, pooled.length);
    std::memset(block_position.contents, 0xa5, block_position.length);
    QsaIndexerAppendBuffers late_short_raw = valid;
    late_short_raw.raw_keys.length -= sizeof(float);
    error.clear();
    bool first_append_encoded = false;
    const bool late_failure_propagated = !Encoder::encode(
        context, *command, "QSA shared indexer append late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_indexer_append(
                    context, encoder, 2u, kCellCapacity,
                    QsaIndexerAppendMode::window, 0u, 1.125f, 0.0f, valid,
                    encode_error)) {
                return false;
            }
            first_append_encoded = true;
            return encode_qsa_indexer_append(
                context, encoder, 2u, kCellCapacity,
                QsaIndexerAppendMode::window, 0u, 1.125f, 0.0f,
                late_short_raw, encode_error);
        },
        error) &&
        first_append_encoded && error.find("undersized") != std::string::npos;
    check(late_failure_propagated,
          "QSA shared indexer append propagates a late segment failure",
          error);
    passed &= late_failure_propagated;
    const auto* tail_output_bytes =
        static_cast<const unsigned char*>(tails.contents);
    const auto* dead_output_bytes =
        static_cast<const unsigned char*>(dead.contents);
    const auto* pooled_output_bytes =
        static_cast<const unsigned char*>(pooled.contents);
    const auto* block_pos_output_bytes =
        static_cast<const unsigned char*>(block_position.contents);
    const bool late_outputs_untouched =
        std::all_of(tail_output_bytes, tail_output_bytes + tails.length,
                    [](unsigned char value) { return value == 0xa5u; }) &&
        std::all_of(dead_output_bytes, dead_output_bytes + dead.length,
                    [](unsigned char value) { return value == 0xa5u; }) &&
        std::all_of(pooled_output_bytes,
                    pooled_output_bytes + pooled.length,
                    [](unsigned char value) { return value == 0xa5u; }) &&
        std::all_of(block_pos_output_bytes,
                    block_pos_output_bytes + block_position.length,
                    [](unsigned char value) { return value == 0xa5u; });
    check(late_outputs_untouched,
          "QSA shared indexer append late failure leaves outputs untouched");
    passed &= late_outputs_untouched;
    error.clear();
    const bool late_failure_discarded =
        !command->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_failure_discarded,
          "QSA shared indexer append late failure discards earlier work",
          error);
    passed &= late_failure_discarded;

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA indexer append foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA indexer append foreign command buffer", error);
        return false;
    }
    error.clear();
    const bool foreign_command_rejected =
        !encode_qsa_indexer_append(
            context, *foreign_command, 1u, kCellCapacity,
            QsaIndexerAppendMode::window, 0u, 1.125f, 0.0f, valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_command_rejected,
          "QSA indexer append rejects a foreign command buffer", error);
    passed &= foreign_command_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA indexer append submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA indexer append submitted command buffer setup",
              error);
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_indexer_append(
            context, *submitted, 1u, kCellCapacity,
            QsaIndexerAppendMode::window, 0u, 1.125f, 0.0f, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA indexer append rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    return passed;
}

bool test_indexer_snapshot_validation(Context& context,
                                     Context& foreign_context) {
    constexpr std::size_t kLayers = INDEXER_SNAPSHOT_LAYERS;
    constexpr std::size_t kTailFloats = 3u * INDEXER_HEAD_DIM;
    constexpr std::uint64_t kTailStride = kTailFloats + 17u;
    constexpr std::uint64_t kBlockPosStride = 3u;
    constexpr std::size_t kTailBytes =
        kLayers * kTailFloats * sizeof(float);
    constexpr std::size_t kBlockPosBytes =
        kLayers * sizeof(std::int32_t);
    const std::size_t tail_source_bytes =
        ((kLayers - 1u) * static_cast<std::size_t>(kTailStride) +
         kTailFloats) *
        sizeof(float);
    const std::size_t block_pos_source_bytes = std::max(
        ((kLayers - 1u) * static_cast<std::size_t>(kBlockPosStride) + 1u) *
            sizeof(std::int32_t),
        kTailBytes);
    twin::GuardedBuffer tails, block_positions, tail_snapshot,
        block_pos_snapshot, foreign_buffer;
    std::string error;
    check(tails.allocate(context, "QSA indexer validation tails",
                         tail_source_bytes + kTailBytes, error),
          "QSA indexer validation tails allocation", error);
    check(block_positions.allocate(
              context, "QSA indexer validation block positions",
              block_pos_source_bytes, error),
          "QSA indexer validation block positions allocation", error);
    check(tail_snapshot.allocate(context, "QSA indexer validation tail output",
                                 kTailBytes, error),
          "QSA indexer validation tail output allocation", error);
    check(block_pos_snapshot.allocate(
              context, "QSA indexer validation block-position output",
              kBlockPosBytes, error),
          "QSA indexer validation block-position output allocation", error);
    check(foreign_buffer.allocate(
              foreign_context, "QSA indexer foreign buffer",
              std::max(tail_source_bytes, kTailBytes), error),
          "QSA indexer foreign buffer allocation", error);
    if (!tails.view() || !block_positions.view() || !tail_snapshot.view() ||
        !block_pos_snapshot.view() || !foreign_buffer.view()) {
        return false;
    }

    const QsaIndexerSnapshotBuffers valid{
        tails.view(), block_positions.view(), tail_snapshot.view(),
        block_pos_snapshot.view()};
    bool passed = true;
    const auto reject = [&](const std::string& label,
                            const QsaIndexerSnapshotBuffers& buffers,
                            std::uint64_t tail_stride,
                            std::uint64_t block_pos_stride,
                            std::string_view error_fragment) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        error.clear();
        const bool rejected =
            !encode_qsa_indexer_snapshot(
                context, *command, tail_stride, block_pos_stride, buffers,
                error) &&
            error.find(error_fragment) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };
    const auto reject_shared = [&](const std::string& label,
                                   const QsaIndexerSnapshotBuffers& buffers,
                                   std::uint64_t tail_stride,
                                   std::uint64_t block_pos_stride,
                                   std::string_view error_fragment) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        error.clear();
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_qsa_indexer_snapshot(
                        context, encoder, tail_stride, block_pos_stride,
                        buffers, encode_error);
                },
                error) &&
            error.find(error_fragment) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    passed &= reject("QSA indexer rejects tail stride below 384 floats",
                     valid, kTailFloats - 1u, kBlockPosStride,
                     "tail stride is too small");
    passed &= reject("QSA indexer rejects zero block-position stride",
                     valid, kTailStride, 0u,
                     "block-position stride is too small");
    passed &= reject("QSA indexer rejects overflowing source extent",
                     valid, std::numeric_limits<std::uint64_t>::max(),
                     kBlockPosStride, "extent overflows");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects undersized tail stride", valid,
        kTailFloats - 1u, kBlockPosStride, "tail stride is too small");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects zero block-position stride",
        valid, kTailStride, 0u,
        "block-position stride is too small");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects overflowing source extent", valid,
        std::numeric_limits<std::uint64_t>::max(), kBlockPosStride,
        "extent overflows");

    QsaIndexerSnapshotBuffers short_tail = valid;
    short_tail.tails.length = tail_source_bytes - sizeof(float);
    passed &= reject("QSA indexer rejects undersized tail view", short_tail,
                     kTailStride, kBlockPosStride,
                     "undersized or misaligned");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects undersized tail view", short_tail,
        kTailStride, kBlockPosStride, "undersized or misaligned");
    QsaIndexerSnapshotBuffers short_block_pos = valid;
    short_block_pos.block_positions.length =
        ((kLayers - 1u) * static_cast<std::size_t>(kBlockPosStride) + 1u) *
            sizeof(std::int32_t) -
        sizeof(std::int32_t);
    passed &= reject("QSA indexer rejects undersized block-position view",
                     short_block_pos, kTailStride, kBlockPosStride,
                     "undersized or misaligned");
    QsaIndexerSnapshotBuffers short_tail_output = valid;
    short_tail_output.tail_snapshot.length = kTailBytes - sizeof(float);
    passed &= reject("QSA indexer rejects undersized tail snapshot view",
                     short_tail_output, kTailStride, kBlockPosStride,
                     "undersized or misaligned");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects undersized tail output",
        short_tail_output, kTailStride, kBlockPosStride,
        "undersized or misaligned");
    QsaIndexerSnapshotBuffers short_block_pos_output = valid;
    short_block_pos_output.block_pos_snapshot.length =
        kBlockPosBytes - sizeof(std::int32_t);
    passed &= reject(
        "QSA indexer rejects undersized block-position snapshot view",
        short_block_pos_output, kTailStride, kBlockPosStride,
        "undersized or misaligned");

    const auto misaligned = [](BufferRef buffer) {
        return buffer.slice(1u, buffer.length - 1u);
    };
    QsaIndexerSnapshotBuffers misaligned_tail = valid;
    misaligned_tail.tails = misaligned(misaligned_tail.tails);
    passed &= reject("QSA indexer rejects misaligned tail view",
                     misaligned_tail, kTailStride, kBlockPosStride,
                     "undersized or misaligned");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects misaligned tail view",
        misaligned_tail, kTailStride, kBlockPosStride,
        "undersized or misaligned");
    QsaIndexerSnapshotBuffers misaligned_block_pos = valid;
    misaligned_block_pos.block_positions =
        misaligned(misaligned_block_pos.block_positions);
    passed &= reject("QSA indexer rejects misaligned block-position view",
                     misaligned_block_pos, kTailStride, kBlockPosStride,
                     "undersized or misaligned");
    QsaIndexerSnapshotBuffers misaligned_tail_output = valid;
    misaligned_tail_output.tail_snapshot =
        misaligned(misaligned_tail_output.tail_snapshot);
    passed &= reject("QSA indexer rejects misaligned tail snapshot view",
                     misaligned_tail_output, kTailStride, kBlockPosStride,
                     "undersized or misaligned");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects misaligned tail output",
        misaligned_tail_output, kTailStride, kBlockPosStride,
        "undersized or misaligned");
    QsaIndexerSnapshotBuffers misaligned_block_pos_output = valid;
    misaligned_block_pos_output.block_pos_snapshot =
        misaligned(misaligned_block_pos_output.block_pos_snapshot);
    passed &= reject(
        "QSA indexer rejects misaligned block-position snapshot view",
        misaligned_block_pos_output, kTailStride, kBlockPosStride,
        "undersized or misaligned");

    const auto beyond_allocation = [](BufferRef buffer,
                                      std::size_t required_bytes) {
        return buffer.slice(buffer.length, required_bytes);
    };
    QsaIndexerSnapshotBuffers tail_outside_allocation = valid;
    tail_outside_allocation.tails =
        beyond_allocation(tail_outside_allocation.tails, tail_source_bytes);
    passed &= reject("QSA indexer rejects tail view beyond its allocation",
                     tail_outside_allocation, kTailStride, kBlockPosStride,
                     "not owned by this Context");
    QsaIndexerSnapshotBuffers block_pos_outside_allocation = valid;
    block_pos_outside_allocation.block_positions = beyond_allocation(
        block_pos_outside_allocation.block_positions,
        ((kLayers - 1u) * static_cast<std::size_t>(kBlockPosStride) + 1u) *
            sizeof(std::int32_t));
    passed &= reject(
        "QSA indexer rejects block-position view beyond its allocation",
        block_pos_outside_allocation, kTailStride, kBlockPosStride,
        "not owned by this Context");
    QsaIndexerSnapshotBuffers tail_output_outside_allocation = valid;
    tail_output_outside_allocation.tail_snapshot = beyond_allocation(
        tail_output_outside_allocation.tail_snapshot, kTailBytes);
    passed &= reject(
        "QSA indexer rejects tail snapshot beyond its allocation",
        tail_output_outside_allocation, kTailStride, kBlockPosStride,
        "not owned by this Context");
    QsaIndexerSnapshotBuffers block_pos_output_outside_allocation = valid;
    block_pos_output_outside_allocation.block_pos_snapshot =
        beyond_allocation(block_pos_output_outside_allocation.block_pos_snapshot,
                          kBlockPosBytes);
    passed &= reject(
        "QSA indexer rejects block-position snapshot beyond its allocation",
        block_pos_output_outside_allocation, kTailStride, kBlockPosStride,
        "not owned by this Context");

    QsaIndexerSnapshotBuffers exact_alias = valid;
    exact_alias.tail_snapshot =
        buffer_subview(tails.view(), 0u, kTailBytes);
    passed &= reject("QSA indexer rejects exact input/output alias",
                     exact_alias, kTailStride, kBlockPosStride,
                     "overlap each other or an input");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects input/output alias",
        exact_alias, kTailStride, kBlockPosStride,
        "overlap each other or an input");
    QsaIndexerSnapshotBuffers partial_alias = valid;
    partial_alias.tail_snapshot =
        buffer_subview(tails.view(), sizeof(float), kTailBytes);
    passed &= reject("QSA indexer rejects partial input/output alias",
                     partial_alias, kTailStride, kBlockPosStride,
                     "overlap each other or an input");
    const std::size_t last_tail_layer_offset =
        (kLayers - 1u) * static_cast<std::size_t>(kTailStride) *
        sizeof(float);
    QsaIndexerSnapshotBuffers last_layer_exact_alias = valid;
    last_layer_exact_alias.tail_snapshot = buffer_subview(
        tails.view(), last_tail_layer_offset, kTailBytes);
    passed &= reject(
        "QSA indexer rejects exact alias in layer 11",
        last_layer_exact_alias, kTailStride, kBlockPosStride,
        "overlap each other or an input");
    QsaIndexerSnapshotBuffers last_layer_partial_alias = valid;
    last_layer_partial_alias.tail_snapshot = buffer_subview(
        tails.view(), tail_source_bytes - sizeof(float), kTailBytes);
    passed &= reject(
        "QSA indexer rejects partial alias in layer 11",
        last_layer_partial_alias, kTailStride, kBlockPosStride,
        "overlap each other or an input");

    const long page_size_raw = getpagesize();
    if (page_size_raw <= 0) {
        check(false, "QSA indexer query page size");
        return false;
    }
    const std::size_t page_size =
        static_cast<std::size_t>(page_size_raw);
    const std::size_t no_copy_required =
        tail_source_bytes + kTailBytes;
    const std::size_t no_copy_bytes =
        ((no_copy_required + page_size - 1u) / page_size) * page_size;
    void* const mapped_backing = mmap(
        nullptr, no_copy_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
    if (mapped_backing == MAP_FAILED) {
        check(false, "QSA indexer map cross-handle alias backing");
        return false;
    }
    auto backing_lifetime = std::shared_ptr<void>(
        mapped_backing, [no_copy_bytes](void* address) {
            if (address != nullptr) munmap(address, no_copy_bytes);
        });
    BufferRef shared_source_handle = context.buffers().wrap_no_copy(
        "QSA indexer cross-handle source", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    BufferRef shared_output_handle = context.buffers().wrap_no_copy(
        "QSA indexer cross-handle output", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    if (!shared_source_handle || !shared_output_handle) {
        check(false, "QSA indexer wrap cross-handle alias backing", error);
        return false;
    }
    const auto make_cross_handle_alias = [&](std::size_t output_offset) {
        QsaIndexerSnapshotBuffers aliases = valid;
        aliases.tails =
            shared_source_handle.slice(0u, tail_source_bytes);
        aliases.tail_snapshot =
            shared_output_handle.slice(output_offset, kTailBytes);
        return aliases;
    };
    const std::size_t last_layer_start =
        (kLayers - 1u) * static_cast<std::size_t>(kTailStride) *
        sizeof(float);
    const QsaIndexerSnapshotBuffers cross_handle_exact_alias =
        make_cross_handle_alias(last_layer_start);
    passed &= reject(
        "QSA indexer rejects exact layer-11 alias across no-copy handles",
        cross_handle_exact_alias, kTailStride, kBlockPosStride,
        "overlap each other or an input");
    const QsaIndexerSnapshotBuffers cross_handle_partial_alias =
        make_cross_handle_alias(tail_source_bytes - sizeof(float));
    passed &= reject(
        "QSA indexer rejects partial layer-11 alias across no-copy handles",
        cross_handle_partial_alias, kTailStride, kBlockPosStride,
        "overlap each other or an input");

    QsaIndexerSnapshotBuffers block_pos_source_alias = valid;
    block_pos_source_alias.tail_snapshot =
        buffer_subview(block_positions.view(), 0u, kTailBytes);
    passed &= reject("QSA indexer rejects block-position input/output alias",
                     block_pos_source_alias, kTailStride, kBlockPosStride,
                     "overlap each other or an input");
    QsaIndexerSnapshotBuffers tail_source_alias = valid;
    tail_source_alias.block_pos_snapshot =
        buffer_subview(tails.view(), 0u, kBlockPosBytes);
    passed &= reject("QSA indexer rejects tail input/output alias",
                     tail_source_alias, kTailStride, kBlockPosStride,
                     "overlap each other or an input");

    QsaIndexerSnapshotBuffers output_alias = valid;
    output_alias.block_pos_snapshot =
        buffer_subview(tail_snapshot.view(), 0u, kBlockPosBytes);
    passed &= reject("QSA indexer rejects overlapping outputs", output_alias,
                     kTailStride, kBlockPosStride,
                     "overlap each other or an input");

    QsaIndexerSnapshotBuffers forged_cpu = valid;
    forged_cpu.tails.contents =
        static_cast<char*>(forged_cpu.tails.contents) + sizeof(float);
    passed &= reject("QSA indexer rejects forged CPU metadata", forged_cpu,
                     kTailStride, kBlockPosStride,
                     "not owned by this Context");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects forged CPU metadata", forged_cpu,
        kTailStride, kBlockPosStride, "not owned by this Context");
    QsaIndexerSnapshotBuffers forged_gpu = valid;
    forged_gpu.block_positions.gpu_address ^= 0x10u;
    passed &= reject("QSA indexer rejects forged GPU metadata", forged_gpu,
                     kTailStride, kBlockPosStride,
                     "not owned by this Context");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects forged GPU metadata", forged_gpu,
        kTailStride, kBlockPosStride, "not owned by this Context");
    QsaIndexerSnapshotBuffers foreign_view = valid;
    foreign_view.tail_snapshot = foreign_buffer.view();
    passed &= reject("QSA indexer rejects foreign Context buffers",
                     foreign_view, kTailStride, kBlockPosStride, "not owned");
    passed &= reject_shared(
        "QSA shared indexer snapshot rejects foreign Context buffers",
        foreign_view, kTailStride, kBlockPosStride, "not owned");

    std::memset(tail_snapshot.view().contents, 0xa5, kTailBytes);
    std::memset(block_pos_snapshot.view().contents, 0xa5, kBlockPosBytes);
    QsaIndexerSnapshotBuffers late_short_tail = valid;
    late_short_tail.tails.length = tail_source_bytes - sizeof(float);
    auto late_failure = context.make_command_buffer(
        "QSA shared indexer snapshot late failure", error);
    if (!late_failure) {
        check(false, "QSA shared indexer snapshot late-failure command buffer",
              error);
        return false;
    }
    error.clear();
    bool first_dispatch_encoded = false;
    const bool late_failure_propagated = !Encoder::encode(
        context, *late_failure, "QSA shared indexer snapshot late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_indexer_snapshot(
                    context, encoder, kTailStride, kBlockPosStride, valid,
                    encode_error)) {
                return false;
            }
            first_dispatch_encoded = true;
            return encode_qsa_indexer_snapshot(
                context, encoder, kTailStride, kBlockPosStride,
                late_short_tail, encode_error);
        },
        error) &&
        first_dispatch_encoded &&
        error.find("undersized or misaligned") != std::string::npos;
    check(late_failure_propagated,
          "QSA shared indexer snapshot propagates a late segment failure",
          error);
    passed &= late_failure_propagated;
    const auto* tail_snapshot_bytes =
        static_cast<const unsigned char*>(tail_snapshot.view().contents);
    const auto* block_pos_snapshot_bytes =
        static_cast<const unsigned char*>(block_pos_snapshot.view().contents);
    const bool late_outputs_untouched =
        std::all_of(tail_snapshot_bytes, tail_snapshot_bytes + kTailBytes,
                    [](unsigned char value) { return value == 0xa5u; }) &&
        std::all_of(block_pos_snapshot_bytes,
                    block_pos_snapshot_bytes + kBlockPosBytes,
                    [](unsigned char value) { return value == 0xa5u; });
    check(late_outputs_untouched,
          "QSA shared indexer snapshot late failure leaves outputs untouched");
    passed &= late_outputs_untouched;
    error.clear();
    const bool late_failure_discarded =
        !late_failure->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_failure_discarded,
          "QSA shared indexer snapshot late failure discards earlier work",
          error);
    passed &= late_failure_discarded;

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA indexer foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA indexer foreign command buffer allocation", error);
        return false;
    }
    error.clear();
    const bool foreign_rejected =
        !encode_qsa_indexer_snapshot(
            context, *foreign_command, kTailStride, kBlockPosStride, valid,
            error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_rejected,
          "QSA indexer rejects a foreign command buffer", error);
    passed &= foreign_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA indexer submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA indexer submitted command buffer setup", error);
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_indexer_snapshot(
            context, *submitted, kTailStride, kBlockPosStride, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA indexer rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA indexer submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false,
                  "QSA indexer submitted command buffer completion");
            passed = false;
        }
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_qsa_test strata.metallib "
                     "[--seed N] [--input-set random|edge|fixture] "
                     "[--indexer-commit-batch]\n");
        return 2;
    }
    const bool indexer_commit_batch_only =
        argc == 3 &&
        std::string_view(argv[2]) == "--indexer-commit-batch";
    twin::Options options;
    std::string error;
    if (!indexer_commit_batch_only &&
        !twin::parse_options(argc, argv, 2, options, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    if (options.input_set == twin::InputSet::real) {
        std::fprintf(stderr,
                     "metal_qsa_test has no model-backed real input set\n");
        return 77;
    }

    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, error);
    check(context != nullptr, "open Metal context", error);
    if (!context) return 1;
    if (indexer_commit_batch_only) {
        const bool passed = test_indexer_commit_batch(*context);
        check(passed && g_failures == 0,
              "QSA indexer all-layer accepted-prefix commit batch");
        std::printf("metal_qsa_test: %d failures\n", g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    ContextOptions foreign_options = context_options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign validation context",
          error);
    if (!foreign_context) return 1;

    bool passed = test_gate_encoder_validation(*context, *foreign_context);
    passed &= test_encoder_validation(*context, *foreign_context);
    passed &= test_norm_rope_shared_encoder_validation(
        context_options, *foreign_context);
    passed &= run_prefill_norm_rope_case(*context, options);
    passed &= test_fwht_shared_encoder_validation(
        *context, *foreign_context);
    passed &= test_kv_encoder_validation(*context, *foreign_context);
    passed &= test_kv_i8_encoder_validation(*context, *foreign_context);
    passed &= test_kv_q4_encoder_validation(*context, *foreign_context);
    passed &= test_kv_k8v4_encoder_validation(*context, *foreign_context);
    passed &= test_prefill_kv_append_row_bounds(*context);
    passed &= test_indexer_snapshot_validation(*context, *foreign_context);
    passed &= test_indexer_append_validation(*context, *foreign_context);
    passed &= run_indexer_snapshot_case(
        *context, 3u * INDEXER_HEAD_DIM, 1u, "packed strides", false);
    passed &= run_indexer_snapshot_case(
        *context, 3u * INDEXER_HEAD_DIM + 17u, 3u, "padded strides", false);
    passed &= run_indexer_snapshot_case(
        *context, 3u * INDEXER_HEAD_DIM, 1u, "packed strides", true);
    passed &= run_indexer_snapshot_case(
        *context, 3u * INDEXER_HEAD_DIM + 17u, 3u, "padded strides", true);
    constexpr std::array indexer_sets{
        twin::InputSet::random, twin::InputSet::edge,
        twin::InputSet::fixture};
    for (const auto set : indexer_sets) {
        for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
            passed &= run_indexer_append_case(
                *context, rows, 0u, 16u, set, options, false);
            passed &= run_indexer_append_case(
                *context, rows, 3u, 16u, set, options, false);
        }
        passed &= run_indexer_append_case(
            *context, 8u, 1u, 16u, set, options, false);
        passed &= run_indexer_append_case(
            *context, 4u, 4u, 16u, set, options, false);
        passed &= run_indexer_append_case(
            *context, 2u, 6u, 16u, set, options, false);
    }
    passed &= run_indexer_append_case(
        *context, 8u, 1u, 16u, twin::InputSet::random, options, true, 1u,
        0u);
    passed &= run_indexer_append_case(
        *context, 8u, 1u, 16u, twin::InputSet::edge, options, true, 4u,
        5u);
    passed &= run_indexer_append_case(
        *context, 8u, 1u, 16u, twin::InputSet::fixture, options, true, 7u,
        11u);
    passed &= run_indexer_append_case(
        *context, 8u, 3u, 16u, twin::InputSet::random, options, true, 2u,
        5u);
    passed &= run_indexer_append_case(
        *context, 8u, 1u, 16u, twin::InputSet::edge, options, false, 0u, 11u,
        true);
    passed &= run_indexer_append_case(
        *context, 8u, 1u, 16u, twin::InputSet::fixture, options, true, 7u,
        11u, true);
    if (options.input_set_explicit) {
        passed &= run_set(*context, options.input_set, options);
        passed &= run_gate_set(*context, options.input_set, options);
        passed &= run_fwht_set(*context, options.input_set, options);
        passed &= run_kv_append_f16_set(
            *context, options.input_set, options);
        passed &= run_kv_append_i8_set(
            *context, options.input_set, options);
        passed &= run_kv_append_q4_set(
            *context, options.input_set, options);
        passed &= run_kv_append_k8v4_set(
            *context, options.input_set, options);
    } else {
        constexpr std::array sets{twin::InputSet::random,
                                  twin::InputSet::edge,
                                  twin::InputSet::fixture};
        for (const auto set : sets) {
            std::printf("QSA input set: %s\n", twin::input_set_name(set));
            passed &= run_set(*context, set, options);
            passed &= run_gate_set(*context, set, options);
            passed &= run_fwht_set(*context, set, options);
            passed &= run_kv_append_f16_set(*context, set, options);
            passed &= run_kv_append_i8_set(*context, set, options);
            passed &= run_kv_append_q4_set(*context, set, options);
            passed &= run_kv_append_k8v4_set(*context, set, options);
        }
    }
    passed &= run_gate_extreme_case(*context, options);
    passed &= run_kv_append_i8_case(
        *context, 1u, 0u, twin::InputSet::edge, options, true, false);
    passed &= run_kv_append_i8_case(
        *context, 1u, 0u, twin::InputSet::edge, options, false, true);
    passed &= run_kv_append_q4_case(
        *context, 1u, 0u, twin::InputSet::edge, options, true, false);
    passed &= run_kv_append_q4_case(
        *context, 1u, 0u, twin::InputSet::edge, options, false, true);
    passed &= run_kv_append_k8v4_case(
        *context, 1u, 0u, twin::InputSet::edge, options, true, false);
    passed &= run_kv_append_k8v4_case(
        *context, 1u, 0u, twin::InputSet::edge, options, false, true);
    passed &= test_fwht_impulse_golden(*context, options);
    passed &= test_fwht_dense_golden(*context, options);
    check(passed && g_failures == 0,
          "QSA norm/RoPE, attention gate, FWHT256, indexer snapshot and KV append encoder twins");
    std::printf("metal_qsa_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
