// Metal twin for MTP drafter stage 6 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "oracles/iq4_xs_mv.hpp"
#include "oracles/q5_k_mv.hpp"
#include "oracles/q6_k_mv.hpp"
#include "oracles/q8_1.hpp"
#include "twin.hpp"

#import <Metal/Metal.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kMaxRows = VERIFY_MAX_T;
constexpr std::size_t kH2Values =
    static_cast<std::size_t>(kMaxRows) * HC_DIM;
constexpr std::size_t kE2Values =
    static_cast<std::size_t>(kMaxRows) * N_EMBD;
constexpr std::size_t kH2Bytes = kH2Values * sizeof(float);
constexpr std::size_t kE2Bytes = kE2Values * sizeof(float);
constexpr std::size_t kOutputPayloadBytes = kH2Bytes + 8u;
constexpr std::size_t kH2PayloadBytes = kH2Bytes + 8u;
constexpr std::size_t kE2PayloadBytes = kE2Bytes + 8u;

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !condition;
}

std::span<float> floats(twin::GuardedBuffer& buffer) {
    return {reinterpret_cast<float*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(float)};
}

void set_params(twin::GuardedBuffer& buffer, std::uint32_t rows) {
    std::fill(buffer.payload().begin(), buffer.payload().end(), 0u);
    MtpParams params{};
    params.T = rows;
    std::memcpy(buffer.payload().data(), &params, sizeof(params));
}

void fill_inputs(twin::GuardedBuffer& h2,
                 twin::GuardedBuffer& e2) {
    auto hidden = floats(h2);
    auto embedding = floats(e2);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        const auto centered =
            static_cast<std::int32_t>(i % 257u) - 128;
        hidden[i] = static_cast<float>(centered) * 0.015625f;
    }
    for (std::size_t i = 0; i < embedding.size(); ++i) {
        const auto centered =
            static_cast<std::int32_t>(i % 71u) - 35;
        embedding[i] = static_cast<float>(centered) * 0.03125f;
    }

    // At this magnitude a binary16 sum rounds 1024.0 + 0.25 back to
    // 1024.0, while the required FP32 sum is exactly 1024.25.
    hidden[0] = 1024.0f;
    embedding[0] = 0.25f;
}

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     const std::string& label, std::string& error) {
    if (!command || !command->commit(error)) return false;
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        error = label + " command buffer became unavailable after commit";
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = label + " command buffer failed";
        if (native.error.localizedDescription.UTF8String != nullptr) {
            error += ": ";
            error += native.error.localizedDescription.UTF8String;
        }
        return false;
    }
    error.clear();
    return true;
}

bool execute_case(Context& context, twin::GuardedBuffer& params,
                  twin::GuardedBuffer& h2, twin::GuardedBuffer& e2,
                  twin::GuardedBuffer& output, std::uint32_t rows,
                  const std::string& label,
                  bool caller_owned_encoder = false,
                  ReadOnlyBufferRef alternate_embedding = {},
                  bool check_precision_witness = true) {
    set_params(params, rows);
    std::fill(output.payload().begin(), output.payload().end(), 0xa5u);
    params.capture_before_dispatch();
    h2.capture_before_dispatch();
    e2.capture_before_dispatch();
    output.capture_before_dispatch();

    MtpAddStreamsBuffers buffers{
        context.buffers().read_only(params.view()),
        context.buffers().read_only(h2.view()),
        alternate_embedding
            ? alternate_embedding
            : context.buffers().read_only(e2.view()),
        output.view(),
    };

    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;

    const bool encoded = caller_owned_encoder
        ? Encoder::encode(
              context, *command, label + " caller-owned segment",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_mtp_add_streams(context, encoder, buffers,
                                                encode_error);
              },
              error)
        : encode_mtp_add_streams(context, *command, buffers, error);
    check(encoded, label + " encode", error);
    if (!encoded) return false;

    const bool completed = submit_and_wait(command, label, error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto hidden = floats(h2);
    const auto embedding = alternate_embedding
        ? std::span<const float>(
              static_cast<const float*>(alternate_embedding.contents()),
              alternate_embedding.length() / sizeof(float))
        : std::span<const float>(floats(e2));
    const auto actual = floats(output);
    bool exact = true;
    bool preserves_fp32_witness = false;
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::size_t i = 0; i < HC_DIM; ++i) {
            const std::size_t output_index =
                static_cast<std::size_t>(row) * HC_DIM + i;
            const std::size_t embedding_index =
                static_cast<std::size_t>(row) * N_EMBD + i % N_EMBD;
            const float expected =
                hidden[output_index] + embedding[embedding_index];
            exact &= std::bit_cast<std::uint32_t>(actual[output_index]) ==
                     std::bit_cast<std::uint32_t>(expected);
            if (row == 0u && i == 0u) {
                preserves_fp32_witness =
                    std::bit_cast<std::uint32_t>(expected) ==
                        std::bit_cast<std::uint32_t>(1024.25f) &&
                    std::bit_cast<std::uint32_t>(actual[output_index]) ==
                        std::bit_cast<std::uint32_t>(1024.25f);
            }
        }
    }
    check(exact, label + " exact scalar oracle");
    if (check_precision_witness) {
        check(preserves_fp32_witness,
              label + " preserves FP32 precision witness");
    }

    const std::array<twin::ByteRange, 1> written{{
        {0u, static_cast<std::size_t>(rows) * HC_DIM * sizeof(float)},
    }};
    const auto output_guard = output.first_unwritten_change(written);
    check(!output_guard, label + " inactive rows and output guards",
          output_guard ? output_guard->reason : "");
    check(!params.first_unwritten_change({}),
          label + " parameter input and guards");
    check(!h2.first_unwritten_change({}), label + " h2 input and guards");
    check(!e2.first_unwritten_change({}), label + " e2 input and guards");
    return exact && !output_guard && !params.first_unwritten_change({}) &&
           !h2.first_unwritten_change({}) && !e2.first_unwritten_change({});
}

bool reject_case(Context& context, const MtpAddStreamsBuffers& buffers,
                 const std::string& label, const std::string& expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;

    const bool rejected =
        !encode_mtp_add_streams(context, *command, buffers, error);
    const bool expected =
        expected_error.empty() || error.find(expected_error) != std::string::npos;
    check(rejected && expected, label, error);
    return rejected && expected;
}

bool test_validation(Context& context, Context& foreign_context,
                     twin::GuardedBuffer& params,
                     twin::GuardedBuffer& h2,
                     twin::GuardedBuffer& e2,
                     twin::GuardedBuffer& output) {
    bool passed = true;
    std::string error;
    set_params(params, 8u);
    MtpAddStreamsBuffers valid{
        context.buffers().read_only(params.view()),
        context.buffers().read_only(h2.view()),
        context.buffers().read_only(e2.view()),
        output.view(),
    };

    set_params(params, 0u);
    auto invalid_rows = valid;
    invalid_rows.params = context.buffers().read_only(params.view());
    passed &= reject_case(context, invalid_rows, "reject T=0",
                          "T must be in 1..8");
    set_params(params, 9u);
    invalid_rows.params = context.buffers().read_only(params.view());
    passed &= reject_case(context, invalid_rows, "reject T=9",
                          "T must be in 1..8");
    set_params(params, 8u);

    auto short_params = valid;
    short_params.params =
        valid.params.slice(0u, sizeof(std::uint32_t));
    passed &= reject_case(context, short_params, "reject short MTP params",
                          "undersized or misaligned");
    auto misaligned_params = valid;
    misaligned_params.params =
        valid.params.slice(1u, sizeof(MtpParams));
    passed &= reject_case(context, misaligned_params,
                          "reject misaligned MTP params",
                          "undersized or misaligned");

    auto short_h2 = valid;
    short_h2.h2 = valid.h2.slice(0u, kH2Bytes - sizeof(float));
    passed &= reject_case(context, short_h2, "reject short h2");
    auto short_e2 = valid;
    short_e2.e2 = valid.e2.slice(0u, kE2Bytes - sizeof(float));
    passed &= reject_case(context, short_e2, "reject short e2");
    auto short_output = valid;
    short_output.output = valid.output.slice(0u, kH2Bytes - sizeof(float));
    passed &= reject_case(context, short_output, "reject short R");

    auto misaligned_h2 = valid;
    misaligned_h2.h2 = valid.h2.slice(1u, kH2Bytes);
    passed &= reject_case(context, misaligned_h2, "reject misaligned h2");
    auto misaligned_e2 = valid;
    misaligned_e2.e2 = valid.e2.slice(1u, kE2Bytes);
    passed &= reject_case(context, misaligned_e2, "reject misaligned e2");
    auto misaligned_output = valid;
    misaligned_output.output = valid.output.slice(1u, kH2Bytes);
    passed &= reject_case(context, misaligned_output,
                          "reject misaligned R");

    auto params_output_alias = valid;
    const std::uint32_t one = 1u;
    std::memcpy(output.payload().data(), &one, sizeof(one));
    params_output_alias.params =
        context.buffers().read_only(output.view());
    passed &= reject_case(context, params_output_alias,
                          "reject params/R overlap", "overlaps");
    set_params(params, 8u);

    auto h2_output_alias = valid;
    h2_output_alias.h2 = context.buffers().read_only(output.view());
    passed &= reject_case(context, h2_output_alias, "reject h2/R overlap",
                          "overlaps");
    auto partial_h2_output_alias = valid;
    const BufferRef shifted_output = output.view().slice(4u, kH2Bytes);
    partial_h2_output_alias.h2 =
        context.buffers().read_only(shifted_output);
    passed &= reject_case(context, partial_h2_output_alias,
                          "reject partial h2/R overlap", "overlaps");
    auto e2_output_alias = valid;
    e2_output_alias.e2 = context.buffers().read_only(output.view());
    passed &= reject_case(context, e2_output_alias, "reject e2/R overlap",
                          "overlaps");

    BufferRef foreign_h2 = foreign_context.buffers().allocate(
        "foreign MTP h2", kH2Bytes, error);
    check(static_cast<bool>(foreign_h2), "allocate foreign h2", error);
    if (foreign_h2) {
        auto foreign = valid;
        foreign.h2 = foreign_context.buffers().read_only(foreign_h2);
        passed &= reject_case(context, foreign,
                              "reject foreign-context h2",
                              "not owned by this Context");
    } else {
        passed = false;
    }
    BufferRef foreign_output = foreign_context.buffers().allocate(
        "foreign MTP output", kH2Bytes, error);
    check(static_cast<bool>(foreign_output), "allocate foreign output", error);
    if (foreign_output) {
        auto foreign = valid;
        foreign.output = foreign_output;
        passed &= reject_case(context, foreign,
                              "reject foreign-context output",
                              "not owned by this Context");
    } else {
        passed = false;
    }

    return passed;
}

bool test_caller_owned_rollback(Context& context,
                                twin::GuardedBuffer& params,
                                twin::GuardedBuffer& h2,
                                twin::GuardedBuffer& e2,
                                twin::GuardedBuffer& output) {
    const std::uint32_t rows = 3u;
    set_params(params, rows);
    std::fill(output.payload().begin(), output.payload().end(), 0x5au);
    params.capture_before_dispatch();
    h2.capture_before_dispatch();
    e2.capture_before_dispatch();
    output.capture_before_dispatch();

    const MtpAddStreamsBuffers valid{
        context.buffers().read_only(params.view()),
        context.buffers().read_only(h2.view()),
        context.buffers().read_only(e2.view()),
        output.view(),
    };
    MtpParams invalid_block{};
    invalid_block.T = 0u;
    twin::GuardedBuffer invalid_params;
    std::string error;
    check(invalid_params.allocate(context, "invalid rollback params",
                                  sizeof(MtpParams), error),
          "allocate invalid rollback params", error);
    if (!invalid_params.view()) return false;
    std::memcpy(invalid_params.payload().data(), &invalid_block,
                sizeof(invalid_block));
    const MtpAddStreamsBuffers invalid{
        context.buffers().read_only(invalid_params.view()),
        valid.h2,
        valid.e2,
        valid.output,
    };

    auto command = context.make_command_buffer("MTP caller rollback", error);
    check(command != nullptr, "create caller-owned rollback command", error);
    if (!command) return false;
    const bool encoded = Encoder::encode(
        context, *command, "MTP add streams late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_add_streams(context, encoder, valid,
                                        encode_error)) {
                return false;
            }
            return encode_mtp_add_streams(context, encoder, invalid,
                                          encode_error);
        },
        error);
    const bool discarded =
        !encoded && command->unsafe_native_handle() == nullptr;
    check(discarded, "discard caller-owned segment after late MTP failure",
          error);
    check(!params.first_unwritten_change({}),
          "caller rollback preserves valid MTP params");
    check(!h2.first_unwritten_change({}),
          "caller rollback preserves h2");
    check(!e2.first_unwritten_change({}),
          "caller rollback preserves e2");
    check(!output.first_unwritten_change({}),
          "caller rollback preserves R");
    return discarded && !params.first_unwritten_change({}) &&
           !h2.first_unwritten_change({}) &&
           !e2.first_unwritten_change({}) &&
           !output.first_unwritten_change({});
}

bool execute_draft_head_case(
    Context& context, const std::string& label, MatVecWeightFormat format,
    std::uint32_t head_rows, std::span<const std::uint8_t> weight_bytes,
    std::span<const float> sample_values,
    std::span<const std::int32_t> row_map,
    std::span<const Q81> expected_q8,
    std::span<const float> expected_logits, bool caller_owned_encoder) {
    constexpr std::size_t sample_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    constexpr std::size_t q8_bytes =
        static_cast<std::size_t>(N_EMBD / 32u) * sizeof(Q81);
    constexpr std::size_t sink_bytes = sizeof(float);
    if (sample_values.size() != N_EMBD || expected_q8.size() != N_EMBD / 32u ||
        weight_bytes.empty() ||
        expected_logits.empty() ||
        expected_logits.size() != (row_map.empty() ? head_rows : row_map.size())) {
        check(false, label + " fixture shape");
        return false;
    }

    std::string error;
    twin::GuardedBuffer params;
    twin::GuardedBuffer sample;
    twin::GuardedBuffer weights;
    twin::GuardedBuffer row_map_buffer;
    twin::GuardedBuffer q8;
    twin::GuardedBuffer logits;
    twin::GuardedBuffer sink;
    const std::size_t logits_capacity =
        static_cast<std::size_t>(head_rows) * sizeof(float);
    bool allocated =
        params.allocate(context, label + " params", 256u, error) &&
        sample.allocate(context, label + " sample", sample_bytes, error) &&
        weights.allocate(context, label + " weights", weight_bytes.size(),
                         error) &&
        q8.allocate(context, label + " Q8_1", q8_bytes, error) &&
        logits.allocate(context, label + " logits", logits_capacity, error) &&
        sink.allocate(context, label + " sink", sink_bytes, error);
    if (!row_map.empty()) {
        allocated &= row_map_buffer.allocate(
            context, label + " row map",
            row_map.size_bytes(), error);
    }
    check(allocated, label + " buffer allocation", error);
    if (!allocated) return false;

    set_params(params, 1u);
    std::memcpy(sample.payload().data(), sample_values.data(), sample_bytes);
    std::memcpy(weights.payload().data(), weight_bytes.data(),
                weight_bytes.size());
    if (!row_map.empty()) {
        std::memcpy(row_map_buffer.payload().data(), row_map.data(),
                    row_map.size_bytes());
    }
    std::fill(q8.payload().begin(), q8.payload().end(), 0xa5u);
    std::fill(logits.payload().begin(), logits.payload().end(), 0x5au);
    std::fill(sink.payload().begin(), sink.payload().end(), 0x3cu);

    params.capture_before_dispatch();
    sample.capture_before_dispatch();
    weights.capture_before_dispatch();
    if (!row_map.empty()) row_map_buffer.capture_before_dispatch();
    q8.capture_before_dispatch();
    logits.capture_before_dispatch();
    sink.capture_before_dispatch();

    const MtpDraftHeadBuffers buffers{
        context.buffers().read_only(params.view()),
        sample.view(),
        context.buffers().read_only(weights.view()),
        row_map.empty()
            ? ReadOnlyBufferRef{}
            : context.buffers().read_only(row_map_buffer.view()),
        q8.view(),
        logits.view(),
        sink.view(),
    };
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool encoded = caller_owned_encoder
        ? Encoder::encode(
              context, *command, label + " caller-owned segment",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_mtp_draft_head(
                      context, encoder, format, head_rows, buffers,
                      encode_error);
              },
              error)
        : encode_mtp_draft_head(context, *command, format, head_rows, buffers,
                                error);
    check(encoded, label + " encode", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(command, label, error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto q8_failure = twin::compare_bytes(
        q8.payload(),
        {reinterpret_cast<const std::uint8_t*>(expected_q8.data()),
         q8_bytes});
    check(!q8_failure, label + " exact Q8_1 scalar oracle",
          q8_failure ? q8_failure->reason : "");
    const std::size_t active_logits_bytes =
        expected_logits.size_bytes();
    const auto logits_failure = twin::compare_bytes(
        logits.payload().first(active_logits_bytes),
        {reinterpret_cast<const std::uint8_t*>(expected_logits.data()),
         active_logits_bytes});
    check(!logits_failure, label + " exact logits scalar oracle",
          logits_failure ? logits_failure->reason : "");
    const std::array<twin::ByteRange, 1> q8_written{{{0u, q8_bytes}}};
    const std::array<twin::ByteRange, 1> logits_written{{
        {0u, active_logits_bytes},
    }};
    const auto q8_guard = q8.first_unwritten_change(q8_written);
    const auto logits_guard = logits.first_unwritten_change(logits_written);
    check(!q8_guard, label + " Q8_1 guards",
          q8_guard ? q8_guard->reason : "");
    check(!logits_guard, label + " inactive logits and guards",
          logits_guard ? logits_guard->reason : "");
    check(!sink.first_unwritten_change({}), label + " unused sink unchanged");
    check(!params.first_unwritten_change({}),
          label + " parameter input unchanged");
    check(!sample.first_unwritten_change({}), label + " sample unchanged");
    check(!weights.first_unwritten_change({}), label + " weights unchanged");
    if (!row_map.empty()) {
        check(!row_map_buffer.first_unwritten_change({}),
              label + " row map unchanged");
    }
    return !q8_failure && !logits_failure && !q8_guard && !logits_guard &&
           !sink.first_unwritten_change({}) &&
           !params.first_unwritten_change({}) &&
           !sample.first_unwritten_change({}) &&
           !weights.first_unwritten_change({}) &&
           (row_map.empty() || !row_map_buffer.first_unwritten_change({}));
}

bool test_mtp_draft_head(Context& context) {
    constexpr std::uint32_t head_rows = 9u;
    constexpr std::uint32_t blocks_per_row = N_EMBD / 256u;
    constexpr std::array<std::int32_t, 4> draft_rows{7, 2, 5, 1};
    std::vector<float> sample_values(N_EMBD);
    for (std::size_t index = 0; index < sample_values.size(); ++index) {
        const auto centered =
            static_cast<std::int32_t>((index * 29u + 7u) % 127u) - 63;
        sample_values[index] = static_cast<float>(centered) * 0.03125f;
    }
    std::string error;
    std::vector<Q81> expected_q8;
    check(twin::quantize_q8_1_reference(sample_values, N_EMBD, 1u,
                                        expected_q8, error),
          "MTP stage 21 Q8_1 CPU oracle", error);
    if (expected_q8.size() != N_EMBD / 32u) return false;

    bool passed = true;
    {
        std::vector<block_q5_K> weights(
            static_cast<std::size_t>(head_rows) * blocks_per_row);
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_q5_K& weight = weights[block];
            const auto d = strata::kernels::f16_from_f32(0.25f);
            const auto dmin = strata::kernels::f16_from_f32(0.125f);
            weight.dm = static_cast<std::uint32_t>(d) |
                        (static_cast<std::uint32_t>(dmin) << 16u);
            std::fill_n(weight.scales, 12u,
                        static_cast<std::uint8_t>(0x11u + block % 3u));
            std::fill_n(weight.qh, 32u,
                        static_cast<std::uint8_t>(0x69u + block));
            std::fill_n(weight.qs, 128u,
                        static_cast<std::uint8_t>(0x3du + block * 7u));
        }
        std::vector<float> expected_logits;
        check(twin::q5_k_matvec_reference(
                  weights, expected_q8, N_EMBD, head_rows, 1u, 1u,
                  expected_logits, error),
              "MTP stage 21 full Q5_K CPU oracle", error);
        passed &= execute_draft_head_case(
            context, "MTP stage 21 full Q5_K head", MatVecWeightFormat::q5_k,
            head_rows,
            {reinterpret_cast<const std::uint8_t*>(weights.data()),
             weights.size() * sizeof(block_q5_K)},
            sample_values, {}, expected_q8, expected_logits, false);

        std::vector<block_q5_K> selected;
        selected.reserve(draft_rows.size() * blocks_per_row);
        for (const std::int32_t row : draft_rows) {
            selected.insert(
                selected.end(),
                weights.begin() + static_cast<std::size_t>(row) *
                                      blocks_per_row,
                weights.begin() + static_cast<std::size_t>(row + 1) *
                                      blocks_per_row);
        }
        expected_logits.clear();
        check(twin::q5_k_matvec_reference(
                  selected, expected_q8, N_EMBD,
                  static_cast<std::uint32_t>(draft_rows.size()), 1u, 1u,
                  expected_logits, error),
              "MTP stage 21 indexed Q5_K CPU oracle", error);
        passed &= execute_draft_head_case(
            context, "MTP stage 21 indexed Q5_K head",
            MatVecWeightFormat::q5_k, head_rows,
            {reinterpret_cast<const std::uint8_t*>(weights.data()),
             weights.size() * sizeof(block_q5_K)},
            sample_values, draft_rows, expected_q8, expected_logits, true);
    }

    {
        std::vector<block_q6_K> weights(
            static_cast<std::size_t>(head_rows) * blocks_per_row);
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_q6_K& weight = weights[block];
            std::fill_n(weight.ql, 128u,
                        static_cast<std::uint8_t>(block * 11u + 0x35u));
            std::fill_n(weight.qh, 64u,
                        static_cast<std::uint8_t>(block * 7u + 0x9bu));
            for (std::uint32_t scale = 0; scale < 16u; ++scale) {
                weight.scales[scale] =
                    static_cast<std::int8_t>((scale + block) % 17u) - 8;
            }
            weight.d = strata::kernels::f16_from_f32(0.1875f);
        }
        std::vector<block_q6_K> selected;
        selected.reserve(draft_rows.size() * blocks_per_row);
        for (const std::int32_t row : draft_rows) {
            selected.insert(
                selected.end(),
                weights.begin() + static_cast<std::size_t>(row) *
                                      blocks_per_row,
                weights.begin() + static_cast<std::size_t>(row + 1) *
                                      blocks_per_row);
        }
        std::vector<float> expected_logits;
        check(twin::q6_k_matvec_reference(
                  selected, expected_q8, N_EMBD,
                  static_cast<std::uint32_t>(draft_rows.size()), 1u, 1u,
                  expected_logits, error),
              "MTP stage 21 indexed Q6_K CPU oracle", error);
        passed &= execute_draft_head_case(
            context, "MTP stage 21 indexed Q6_K head",
            MatVecWeightFormat::q6_k, head_rows,
            {reinterpret_cast<const std::uint8_t*>(weights.data()),
             weights.size() * sizeof(block_q6_K)},
            sample_values, draft_rows, expected_q8, expected_logits, true);
    }

    {
        std::vector<block_iq4_xs> weights(
            static_cast<std::size_t>(head_rows) * blocks_per_row);
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq4_xs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(0.25f);
            weight.scales_h = static_cast<std::uint16_t>(block * 0x111u);
            for (std::size_t index = 0; index < 4u; ++index) {
                weight.scales_l[index] =
                    static_cast<std::uint8_t>(block * 13u + index * 37u);
            }
            for (std::size_t index = 0; index < 128u; ++index) {
                weight.qs[index] =
                    static_cast<std::uint8_t>(block * 17u + index * 29u);
            }
        }
        std::vector<block_iq4_xs> selected;
        selected.reserve(draft_rows.size() * blocks_per_row);
        for (const std::int32_t row : draft_rows) {
            selected.insert(
                selected.end(),
                weights.begin() + static_cast<std::size_t>(row) *
                                      blocks_per_row,
                weights.begin() + static_cast<std::size_t>(row + 1) *
                                      blocks_per_row);
        }
        std::vector<float> expected_logits;
        check(twin::iq4_xs_matvec_reference(
                  selected, expected_q8, N_EMBD,
                  static_cast<std::uint32_t>(draft_rows.size()), 1u, 1u,
                  expected_logits, error),
              "MTP stage 21 indexed IQ4_XS CPU oracle", error);
        passed &= execute_draft_head_case(
            context, "MTP stage 21 indexed IQ4_XS head",
            MatVecWeightFormat::iq4_xs, head_rows,
            {reinterpret_cast<const std::uint8_t*>(weights.data()),
             weights.size() * sizeof(block_iq4_xs)},
            sample_values, draft_rows, expected_q8, expected_logits, true);
    }
    return passed;
}

bool test_mtp_draft_head_validation(Context& context,
                                    Context& foreign_context) {
    constexpr std::uint32_t head_rows = 2u;
    constexpr std::size_t q8_bytes =
        static_cast<std::size_t>(N_EMBD / 32u) * sizeof(Q81);
    constexpr std::size_t sample_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(N_EMBD / 256u) * sizeof(block_q5_K) *
        head_rows;
    std::string error;
    twin::GuardedBuffer params;
    twin::GuardedBuffer sample;
    twin::GuardedBuffer weights;
    twin::GuardedBuffer map;
    twin::GuardedBuffer q8;
    twin::GuardedBuffer logits;
    twin::GuardedBuffer sink;
    bool allocated =
        params.allocate(context, "MTP draft validation params", 256u, error) &&
        sample.allocate(context, "MTP draft validation sample", sample_bytes,
                        error) &&
        weights.allocate(context, "MTP draft validation weights",
                         weight_bytes, error) &&
        map.allocate(context, "MTP draft validation map", 2u * sizeof(std::int32_t),
                     error) &&
        q8.allocate(context, "MTP draft validation Q8_1", q8_bytes, error) &&
        logits.allocate(context, "MTP draft validation logits",
                        head_rows * sizeof(float), error) &&
        sink.allocate(context, "MTP draft validation sink", sizeof(float),
                      error);
    check(allocated, "MTP stage 21 validation allocations", error);
    if (!allocated) return false;

    std::vector<float> sample_values(N_EMBD, 0.125f);
    std::vector<block_q5_K> weight_values(
        static_cast<std::size_t>(head_rows) * (N_EMBD / 256u));
    for (block_q5_K& weight : weight_values) {
        const auto d = strata::kernels::f16_from_f32(0.25f);
        const auto dmin = strata::kernels::f16_from_f32(0.125f);
        weight.dm = static_cast<std::uint32_t>(d) |
                    (static_cast<std::uint32_t>(dmin) << 16u);
        std::fill_n(weight.scales, 12u, std::uint8_t{0x11u});
        std::fill_n(weight.qh, 32u, std::uint8_t{0x55u});
        std::fill_n(weight.qs, 128u, std::uint8_t{0xaau});
    }
    std::memcpy(sample.payload().data(), sample_values.data(), sample_bytes);
    std::memcpy(weights.payload().data(), weight_values.data(), weight_bytes);
    const std::array<std::int32_t, 2> valid_map{1, 0};
    std::memcpy(map.payload().data(), valid_map.data(), sizeof(valid_map));
    set_params(params, 1u);
    std::fill(q8.payload().begin(), q8.payload().end(), 0x6bu);
    std::fill(logits.payload().begin(), logits.payload().end(), 0x7cu);
    std::fill(sink.payload().begin(), sink.payload().end(), 0x8du);

    MtpDraftHeadBuffers valid{
        context.buffers().read_only(params.view()),
        sample.view(),
        context.buffers().read_only(weights.view()),
        context.buffers().read_only(map.view()),
        q8.view(),
        logits.view(),
        sink.view(),
    };

    auto expect_rejected = [&](MtpDraftHeadBuffers candidate,
                               MatVecWeightFormat format,
                               std::uint32_t rows,
                               const std::string& label,
                               const std::string& expected_error) {
        params.capture_before_dispatch();
        sample.capture_before_dispatch();
        weights.capture_before_dispatch();
        map.capture_before_dispatch();
        q8.capture_before_dispatch();
        logits.capture_before_dispatch();
        sink.capture_before_dispatch();
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) return false;
        const bool rejected =
            !encode_mtp_draft_head(context, *command, format, rows, candidate,
                                   error);
        const bool expected =
            rejected && error.find(expected_error) != std::string::npos;
        check(expected, label, error);
        const bool unchanged =
            !params.first_unwritten_change({}) &&
            !sample.first_unwritten_change({}) &&
            !weights.first_unwritten_change({}) &&
            !map.first_unwritten_change({}) &&
            !q8.first_unwritten_change({}) &&
            !logits.first_unwritten_change({}) &&
            !sink.first_unwritten_change({});
        check(unchanged, label + " preserves all buffers");
        command->discard();
        return expected && unchanged;
    };

    bool passed = true;
    set_params(params, 2u);
    valid.params = context.buffers().read_only(params.view());
    passed &= expect_rejected(valid, MatVecWeightFormat::q5_k, head_rows,
                              "reject MTP draft T=2", "requires T=1");
    set_params(params, 1u);
    valid.params = context.buffers().read_only(params.view());

    const std::array<std::int32_t, 2> invalid_map{-1, 0};
    std::memcpy(map.payload().data(), invalid_map.data(), sizeof(invalid_map));
    valid.draft_vocab_ids = context.buffers().read_only(map.view());
    passed &= expect_rejected(
        valid, MatVecWeightFormat::q5_k, head_rows,
        "reject negative MTP draft row id", "outside the full head");
    const std::array<std::int32_t, 2> duplicate_map{1, 1};
    std::memcpy(map.payload().data(), duplicate_map.data(),
                sizeof(duplicate_map));
    valid.draft_vocab_ids = context.buffers().read_only(map.view());
    passed &= expect_rejected(
        valid, MatVecWeightFormat::q5_k, head_rows,
        "reject duplicate MTP draft row id", "duplicate head rows");
    std::memcpy(map.payload().data(), valid_map.data(), sizeof(valid_map));
    valid.draft_vocab_ids = context.buffers().read_only(map.view());

    passed &= expect_rejected(
        valid, MatVecWeightFormat::q8_0, head_rows,
        "reject unsupported MTP draft head format", "Q5_K, Q6_K or IQ4_XS");
    passed &= expect_rejected(
        valid, MatVecWeightFormat::q5_k, 0u,
        "reject empty MTP draft head", "1..248320");
    auto short_q8 = valid;
    short_q8.activations_q8_1 = q8.view().slice(0u, q8_bytes - 1u);
    passed &= expect_rejected(
        short_q8, MatVecWeightFormat::q5_k, head_rows,
        "reject short MTP draft Q8_1 scratch", "undersized or misaligned");
    auto output_alias = valid;
    output_alias.draft_vocab_ids = {};
    output_alias.logits = sample.view().slice(0u, sizeof(float));
    passed &= expect_rejected(
        output_alias, MatVecWeightFormat::q5_k, 1u,
        "reject MTP draft sample/logits alias", "overlap");

    BufferRef foreign_weights = foreign_context.buffers().allocate(
        "foreign MTP draft weights", weight_bytes, error);
    check(static_cast<bool>(foreign_weights),
          "allocate foreign MTP draft weights", error);
    if (foreign_weights) {
        auto foreign = valid;
        foreign.weights = foreign_context.buffers().read_only(foreign_weights);
        passed &= expect_rejected(
            foreign, MatVecWeightFormat::q5_k, head_rows,
            "reject foreign MTP draft weights",
            "not owned by this Context");
    } else {
        passed = false;
    }

    // A valid quantize→projection sequence followed by invalid MTP metadata
    // must discard the entire caller-owned segment before it can be submitted.
    set_params(params, 1u);
    valid.params = context.buffers().read_only(params.view());
    std::fill(q8.payload().begin(), q8.payload().end(), 0x31u);
    std::fill(logits.payload().begin(), logits.payload().end(), 0x42u);
    q8.capture_before_dispatch();
    logits.capture_before_dispatch();
    twin::GuardedBuffer bad_params;
    check(bad_params.allocate(context, "late-invalid MTP draft params",
                              256u, error),
          "allocate late-invalid MTP draft params", error);
    if (!bad_params.view()) return false;
    set_params(bad_params, 2u);
    auto invalid = valid;
    invalid.params = context.buffers().read_only(bad_params.view());
    auto command = context.make_command_buffer("MTP draft late rollback",
                                              error);
    check(command != nullptr, "MTP draft rollback command buffer", error);
    if (!command) return false;
    bool first_stage_encoded = false;
    bool expected_late_rejection = false;
    const bool transaction_succeeded = Encoder::encode(
        context, *command, "MTP draft rollback segment",
        [&](Encoder& encoder, std::string& encode_error) {
            first_stage_encoded = encode_mtp_draft_head(
                context, encoder, MatVecWeightFormat::q5_k, head_rows, valid,
                encode_error);
            if (!first_stage_encoded) return false;
            const bool invalid_stage_encoded = encode_mtp_draft_head(
                context, encoder, MatVecWeightFormat::q5_k, head_rows,
                invalid, encode_error);
            expected_late_rejection =
                !invalid_stage_encoded &&
                encode_error.find("requires T=1") != std::string::npos;
            return invalid_stage_encoded;
        },
        error);
    const bool discarded = first_stage_encoded && expected_late_rejection &&
                           !transaction_succeeded &&
                           command->unsafe_native_handle() == nullptr;
    const bool outputs_unchanged =
        !q8.first_unwritten_change({}) && !logits.first_unwritten_change({});
    check(discarded, "MTP draft caller-owned late failure discards segment",
          error);
    check(outputs_unchanged,
          "MTP draft rollback preserves Q8_1 and logits scratch");
    passed &= discarded && outputs_unchanged;
    command->discard();

    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        q8.capture_before_dispatch();
        logits.capture_before_dispatch();
        auto group_command =
            context.make_command_buffer("MTP draft group boundary", error);
        check(group_command != nullptr,
              "MTP draft group-boundary command buffer", error);
        if (!group_command) return false;
        const bool group_rejected = Encoder::encode(
            context, *group_command, "MTP draft group validation",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                std::string stage_error;
                const bool rejected = !encode_mtp_draft_head(
                    context, encoder, MatVecWeightFormat::q5_k, head_rows,
                    valid, stage_error);
                if (!encoder.group_end(encode_error)) return false;
                if (!rejected ||
                    stage_error.find("ordered concurrent-group boundary") ==
                        std::string::npos) {
                    encode_error = "MTP stage 21 accepted an open group";
                    return false;
                }
                return true;
            },
            error);
        const bool group_outputs_unchanged =
            !q8.first_unwritten_change({}) &&
            !logits.first_unwritten_change({});
        check(group_rejected,
              "MTP stage 21 rejects open concurrent group", error);
        check(group_outputs_unchanged,
              "MTP stage 21 open-group rejection preserves scratch");
        passed &= group_rejected && group_outputs_unchanged;
        group_command->discard();
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: metal_mtp_test <strata.metallib>\n");
        return 2;
    }

    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal context", error);
    if (!context) return 1;

    ContextOptions foreign_options = options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal context", error);
    if (!foreign_context) return 1;

    twin::GuardedBuffer params;
    twin::GuardedBuffer h2;
    twin::GuardedBuffer e2;
    twin::GuardedBuffer output;
    check(params.allocate(*context, "MTP params", 256u, error),
          "allocate MTP params", error);
    check(h2.allocate(*context, "MTP hidden", kH2PayloadBytes, error),
          "allocate h2", error);
    check(e2.allocate(*context, "MTP embedding", kE2PayloadBytes, error),
          "allocate e2", error);
    check(output.allocate(*context, "MTP result", kOutputPayloadBytes, error),
          "allocate R", error);
    if (!params.view() || !h2.view() || !e2.view() || !output.view()) {
        return 1;
    }
    fill_inputs(h2, e2);

    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= kMaxRows; ++rows) {
        passed &= execute_case(*context, params, h2, e2, output, rows,
                               "MTP add streams T=" + std::to_string(rows),
                               rows == 3u);
    }
    passed &= execute_case(
        *context, params, h2, e2, output, 8u,
        "MTP add streams with overlapping read-only inputs", false,
        context->buffers().read_only(h2.view()), false);
    passed &= test_validation(*context, *foreign_context, params, h2, e2,
                              output);
    passed &= test_caller_owned_rollback(*context, params, h2, e2, output);
    passed &= test_mtp_draft_head(*context);
    passed &= test_mtp_draft_head_validation(*context, *foreign_context);

    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("metal_mtp_test: MTP stages 6 and 21 OK");
    return 0;
}
