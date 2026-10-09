// Metal twin for MTP drafter stage 6 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "strata/metal/sampler_dispatch.hpp"
#include "oracles/iq4_xs_mv.hpp"
#include "oracles/q5_k_mv.hpp"
#include "oracles/q6_k_mv.hpp"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"
#include "twin.hpp"

#import <Metal/Metal.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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

#include "strata/kernels/bf16_bits.hpp"

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
constexpr std::uint32_t kImageTokenFlag = 0x80000000u;

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
    MtpDraftVocabularyMap validated_row_map;
    if (!row_map.empty()) {
        const bool map_valid = make_mtp_draft_vocabulary_map(
            context, context.buffers().read_only(row_map_buffer.view()),
            head_rows, validated_row_map, error);
        check(map_valid, label + " row-map validation", error);
        if (!map_valid) return false;
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
        validated_row_map,
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

bool test_mtp_input_projection(Context& context) {
    constexpr std::uint32_t vocabulary = 8u;
    constexpr std::size_t embedding_table_row_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(std::uint16_t);
    constexpr std::size_t embedding_capacity =
        static_cast<std::size_t>(kMaxRows) * N_EMBD * sizeof(float);
    constexpr std::size_t residual_capacity =
        static_cast<std::size_t>(kMaxRows) * HC_DIM * sizeof(float);
    constexpr std::size_t hidden_rows_capacity =
        static_cast<std::size_t>(kMaxRows) * 4u;
    constexpr std::size_t q8_row_bytes =
        static_cast<std::size_t>(N_EMBD / 32u) * sizeof(Q81);
    constexpr std::size_t q8_capacity =
        hidden_rows_capacity * q8_row_bytes;
    constexpr std::size_t hidden_output_capacity =
        hidden_rows_capacity * N_EMBD * sizeof(float);
    constexpr std::size_t sink_capacity =
        4u * static_cast<std::size_t>(N_EMBD) * sizeof(float);

    std::string error;
    twin::GuardedBuffer params;
    twin::GuardedBuffer selected_token;
    twin::GuardedBuffer table;
    twin::GuardedBuffer embedding_gamma;
    twin::GuardedBuffer fc_embedding;
    twin::GuardedBuffer residual;
    twin::GuardedBuffer hidden_gamma;
    twin::GuardedBuffer fc_hidden;
    twin::GuardedBuffer embedding;
    twin::GuardedBuffer embedding_norm;
    twin::GuardedBuffer xq;
    twin::GuardedBuffer e2;
    twin::GuardedBuffer hidden_norm;
    twin::GuardedBuffer h2;
    twin::GuardedBuffer sink;
    const bool allocated =
        params.allocate(context, "MTP input projection params", 256u,
                        error) &&
        selected_token.allocate(context, "MTP selected token",
                                sizeof(std::uint32_t), error) &&
        table.allocate(context, "MTP BF16 token embeddings",
                       vocabulary * embedding_table_row_bytes, error) &&
        embedding_gamma.allocate(context, "MTP embedding norm gamma",
                                 N_EMBD * sizeof(float), error) &&
        fc_embedding.allocate(
            context, "MTP fc_embedding Q8_0",
            static_cast<std::size_t>(N_EMBD) * (N_EMBD / 32u) *
                sizeof(block_q8_0),
            error) &&
        residual.allocate(context, "MTP input residual", residual_capacity,
                          error) &&
        hidden_gamma.allocate(context, "MTP hidden norm gamma",
                              HC_DIM * sizeof(float), error) &&
        fc_hidden.allocate(
            context, "MTP fc_hidden Q8_0",
            static_cast<std::size_t>(N_EMBD) * (N_EMBD / 32u) *
                sizeof(block_q8_0),
            error) &&
        embedding.allocate(context, "MTP embedding scratch",
                           embedding_capacity, error) &&
        embedding_norm.allocate(context, "MTP normalized embedding",
                                embedding_capacity, error) &&
        xq.allocate(context, "MTP input projection Q8_1", q8_capacity,
                    error) &&
        e2.allocate(context, "MTP projected embedding", embedding_capacity,
                    error) &&
        hidden_norm.allocate(context, "MTP normalized hidden",
                             residual_capacity, error) &&
        h2.allocate(context, "MTP projected hidden", hidden_output_capacity,
                    error) &&
        sink.allocate(context, "MTP input projection mat-vec sink",
                      sink_capacity, error);
    check(allocated, "MTP stages 1–5 allocations", error);
    if (!allocated) return false;

    auto table_values = std::span<std::uint16_t>(
        reinterpret_cast<std::uint16_t*>(table.payload().data()),
        table.payload().size() / sizeof(std::uint16_t));
    for (std::uint32_t token = 0u; token < vocabulary; ++token) {
        for (std::uint32_t column = 0u; column < N_EMBD; ++column) {
            const auto centered = static_cast<std::int32_t>(
                                      (token * 43u + column * 17u) % 251u) -
                                  125;
            const float value = static_cast<float>(centered) * 0.0078125f;
            table_values[static_cast<std::size_t>(token) * N_EMBD + column] =
                strata::kernels::bf16_from_f32(value);
        }
    }
    auto embedding_gamma_values =
        std::span<float>(reinterpret_cast<float*>(
                             embedding_gamma.payload().data()),
                         N_EMBD);
    for (std::size_t index = 0u; index < embedding_gamma_values.size();
         ++index) {
        embedding_gamma_values[index] =
            0.875f + static_cast<float>(index % 31u) * 0.0078125f;
    }
    auto hidden_gamma_values = std::span<float>(
        reinterpret_cast<float*>(hidden_gamma.payload().data()), HC_DIM);
    for (std::size_t index = 0u; index < hidden_gamma_values.size(); ++index) {
        hidden_gamma_values[index] =
            0.8125f + static_cast<float>(index % 47u) * 0.0078125f;
    }
    auto residual_values =
        std::span<float>(reinterpret_cast<float*>(residual.payload().data()),
                         residual_capacity / sizeof(float));
    for (std::size_t index = 0u; index < residual_values.size(); ++index) {
        const auto centered =
            static_cast<std::int32_t>((index * 29u + 13u) % 263u) - 131;
        residual_values[index] = static_cast<float>(centered) * 0.00390625f;
    }

    const std::size_t weight_blocks_per_row = N_EMBD / 32u;
    auto fill_q8_weights = [&](twin::GuardedBuffer& buffer,
                               std::uint32_t output_rows,
                               std::uint32_t salt) {
        auto weights = std::span<block_q8_0>(
            reinterpret_cast<block_q8_0*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(block_q8_0));
        for (std::uint32_t row = 0u; row < output_rows; ++row) {
            for (std::size_t block = 0u; block < weight_blocks_per_row;
                 ++block) {
                auto& weight =
                    weights[static_cast<std::size_t>(row) *
                                weight_blocks_per_row +
                            block];
                weight.d = strata::kernels::f16_from_f32(0.03125f);
                for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                    const auto centered = static_cast<std::int32_t>(
                                              (row * 11u + block * 7u +
                                               lane * 5u + salt) %
                                              15u) -
                                          7;
                    weight.qs[lane] = static_cast<std::int8_t>(centered);
                }
            }
        }
    };
    fill_q8_weights(fc_embedding, N_EMBD, 3u);
    fill_q8_weights(fc_hidden, N_EMBD, 9u);

    auto make_params = [&](std::uint32_t rows,
                           std::uint32_t first_token) {
        MtpParams value{};
        value.T = rows;
        for (std::uint32_t pass = 0u; pass < 4u; ++pass) {
            const std::uint32_t offset = pass * 8u;
            value.hid[pass] =
                offset < rows * 4u ? std::min(8u, rows * 4u - offset) : 0u;
        }
        for (std::uint32_t row = 0u; row < rows; ++row) {
            value.tokens[row] = (first_token + row * 3u) % vocabulary;
        }
        std::memset(params.payload().data(), 0, params.payload().size());
        std::memcpy(params.payload().data(), &value, sizeof(value));
        return value;
    };
    auto make_expected_norm = [](std::span<const float> input,
                                 std::span<const float> gamma,
                                 std::uint32_t rows,
                                 std::uint32_t width) {
        std::vector<twin::tf> result(
            static_cast<std::size_t>(rows) * width);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            const std::size_t row_offset =
                static_cast<std::size_t>(row) * width;
            double sum_squares = 0.0;
            for (std::uint32_t column = 0u; column < width; ++column) {
                const double value = input[row_offset + column];
                sum_squares += value * value;
            }
            const float scale = static_cast<float>(
                1.0 / std::sqrt(sum_squares / width + 1.0e-6));
            for (std::uint32_t column = 0u; column < width; ++column) {
                const std::size_t index = row_offset + column;
                result[index] =
                    twin::tf{(scale * input[index]) * gamma[column]};
            }
        }
        return result;
    };
    auto capture_inputs = [&](bool external_token) {
        params.capture_before_dispatch();
        table.capture_before_dispatch();
        embedding_gamma.capture_before_dispatch();
        fc_embedding.capture_before_dispatch();
        residual.capture_before_dispatch();
        hidden_gamma.capture_before_dispatch();
        fc_hidden.capture_before_dispatch();
        if (external_token) selected_token.capture_before_dispatch();
    };
    auto poison_outputs = [&] {
        std::fill(embedding.payload().begin(), embedding.payload().end(),
                  0xa5u);
        std::fill(embedding_norm.payload().begin(),
                  embedding_norm.payload().end(), 0xa5u);
        std::fill(xq.payload().begin(), xq.payload().end(), 0xa5u);
        std::fill(e2.payload().begin(), e2.payload().end(), 0xa5u);
        std::fill(hidden_norm.payload().begin(), hidden_norm.payload().end(),
                  0xa5u);
        std::fill(h2.payload().begin(), h2.payload().end(), 0xa5u);
        std::fill(sink.payload().begin(), sink.payload().end(), 0xa5u);
        embedding.capture_before_dispatch();
        embedding_norm.capture_before_dispatch();
        xq.capture_before_dispatch();
        e2.capture_before_dispatch();
        hidden_norm.capture_before_dispatch();
        h2.capture_before_dispatch();
        sink.capture_before_dispatch();
    };
    auto run_case = [&](std::uint32_t rows, std::uint32_t first_token,
                        MtpInputTokenSource token_source, bool in_place_norm,
                        bool caller_owned, const std::string& label) {
        MtpParams params_value = make_params(rows, first_token);
        const bool external =
            token_source == MtpInputTokenSource::selected_token;
        const std::uint32_t external_id = first_token;
        const std::uint32_t stale_token = vocabulary + 3u;
        const std::array<twin::ByteRange, 1> token_writer_range{{
            {0u, sizeof(std::uint32_t)},
        }};
        std::memcpy(selected_token.payload().data(), &stale_token,
                    sizeof(stale_token));
        const BufferRef embedding_output =
            in_place_norm ? embedding_norm.view() : embedding.view();
        MtpInputProjectionBuffers buffers{
            context.buffers().read_only(params.view()),
            token_source,
            external ? context.buffers().read_only(selected_token.view())
                     : ReadOnlyBufferRef{},
            EmbedRowsFormat::bf16,
            vocabulary,
            context.buffers().read_only(table.view()),
            context.buffers().read_only(embedding_gamma.view()),
            context.buffers().read_only(fc_embedding.view()),
            residual.view(),
            context.buffers().read_only(hidden_gamma.view()),
            context.buffers().read_only(fc_hidden.view()),
            embedding_output,
            in_place_norm ? embedding_output : embedding_norm.view(),
            xq.view(),
            e2.view(),
            hidden_norm.view(),
            h2.view(),
            sink.view(),
        };

        poison_outputs();
        capture_inputs(external);
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) return false;
        bool encoded = false;
        if (external) {
            const ReadOnlyBufferRef source_token =
                context.buffers().read_only(params.view().slice(
                    offsetof(MtpParams, tokens), sizeof(std::uint32_t)));
            const std::array token_output{
                EncoderBufferBinding{selected_token.view(), 1u,
                                     sizeof(std::uint32_t),
                                     alignof(std::uint32_t)},
            };
            const std::array token_input{
                EncoderReadOnlyBufferBinding{
                    source_token, 0u, sizeof(std::uint32_t),
                    alignof(std::uint32_t)},
            };
            constexpr std::array<EncoderBytesBinding, 0> no_bytes{};
            constexpr std::array<EncoderThreadgroupMemory, 0> no_memory{};
            encoded = Encoder::encode(
                context, *command, label + " GPU token producer and projection",
                [&](Encoder& encoder, std::string& encode_error) {
                    if (!encoder.dispatch(
                            "strata_selftest_copy_u32_one", {1u, 1u, 1u},
                            {1u, 1u, 1u}, token_output, token_input, no_bytes,
                            no_memory, encode_error)) {
                        return false;
                    }
                    return encode_mtp_input_projection(
                        context, encoder, buffers, encode_error);
                },
                error);
        } else {
            encoded =
                caller_owned
                    ? Encoder::encode(
                          context, *command,
                          label + " caller-owned segment",
                          [&](Encoder& encoder,
                              std::string& encode_error) {
                              return encode_mtp_input_projection(
                                  context, encoder, buffers, encode_error);
                          },
                          error)
                    : encode_mtp_input_projection(
                          context, *command, buffers, error);
        }
        check(encoded, label + " encode", error);
        if (!encoded) {
            command->discard();
            return false;
        }
        const bool completed = submit_and_wait(command, label, error);
        check(completed, label + " GPU completion", error);
        if (!completed) return false;

        std::vector<std::uint32_t> token_ids(rows);
        if (external) {
            token_ids[0] = external_id;
        } else {
            for (std::uint32_t row = 0u; row < rows; ++row) {
                token_ids[row] = params_value.tokens[row];
            }
        }
        std::vector<float> expected_embedding(
            static_cast<std::size_t>(rows) * N_EMBD);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t column = 0u; column < N_EMBD; ++column) {
                expected_embedding[static_cast<std::size_t>(row) * N_EMBD +
                                   column] =
                    strata::kernels::f32_from_bf16(
                        table_values[static_cast<std::size_t>(token_ids[row]) *
                                         N_EMBD +
                                     column]);
            }
        }
        const auto expected_embedding_norm = make_expected_norm(
            expected_embedding, embedding_gamma_values, rows, N_EMBD);
        const auto expected_hidden_norm = make_expected_norm(
            residual_values.first(static_cast<std::size_t>(rows) * HC_DIM),
            hidden_gamma_values, rows, HC_DIM);

        const std::size_t embedding_bytes =
            static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
        const std::size_t hidden_bytes =
            static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
        const std::size_t q8_bytes =
            static_cast<std::size_t>(rows) * 4u * q8_row_bytes;
        const std::uint32_t embedding_columns =
            rows <= 1u ? 1u : rows <= 2u ? 2u : rows <= 4u ? 4u : 8u;
        const std::size_t sink_written_bytes =
            static_cast<std::size_t>(embedding_columns - rows) * N_EMBD *
            sizeof(float);

        if (!in_place_norm) {
            const auto actual_embedding =
                std::span<const std::uint8_t>(embedding.payload().data(),
                                              embedding_bytes);
            const auto expected_embedding_bytes =
                std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        expected_embedding.data()),
                    embedding_bytes);
            const auto embedding_failure =
                twin::compare_bytes(actual_embedding,
                                    expected_embedding_bytes);
            check(!embedding_failure, label + " BF16 token gather oracle",
                  embedding_failure ? embedding_failure->reason : "");
        }
        const auto actual_embedding_norm = std::span<const float>(
            reinterpret_cast<const float*>(embedding_norm.payload().data()),
            static_cast<std::size_t>(rows) * N_EMBD);
        const auto embedding_norm_failure = twin::compare_floats(
            actual_embedding_norm, expected_embedding_norm,
            {twin::Comparison::units, 64.0, 0.0, 0u});
        check(!embedding_norm_failure, label + " embedding RMSNorm U(64)",
              embedding_norm_failure ? embedding_norm_failure->reason : "");
        const auto actual_hidden_norm = std::span<const float>(
            reinterpret_cast<const float*>(hidden_norm.payload().data()),
            static_cast<std::size_t>(rows) * HC_DIM);
        const auto hidden_norm_failure = twin::compare_floats(
            actual_hidden_norm, expected_hidden_norm,
            {twin::Comparison::units, 64.0, 0.0, 0u});
        check(!hidden_norm_failure, label + " hidden RMSNorm U(64)",
              hidden_norm_failure ? hidden_norm_failure->reason : "");

        std::vector<float> padded_embedding(
            static_cast<std::size_t>(embedding_columns) * N_EMBD);
        const auto actual_embedding_norm_values =
            std::span<const float>(actual_embedding_norm);
        for (std::uint32_t row = 0u; row < embedding_columns; ++row) {
            const std::uint32_t source_row = row < rows ? row : 0u;
            for (std::uint32_t column = 0u; column < N_EMBD; ++column) {
                padded_embedding[static_cast<std::size_t>(row) * N_EMBD +
                                 column] =
                    actual_embedding_norm_values[
                        static_cast<std::size_t>(source_row) * N_EMBD +
                        column];
            }
        }
        std::vector<Q81> embedding_activations;
        check(twin::quantize_q8_1_reference(
                  padded_embedding, N_EMBD, embedding_columns,
                  embedding_activations, error),
              label + " embedding Q8_1 scalar oracle", error);
        const auto embedding_weights = std::span<const block_q8_0>(
            reinterpret_cast<const block_q8_0*>(fc_embedding.payload().data()),
            fc_embedding.payload().size() / sizeof(block_q8_0));
        std::vector<float> expected_e2_padded;
        check(twin::q8_0_matvec_reference(
                  embedding_weights, embedding_activations, N_EMBD, N_EMBD,
                  embedding_columns, rows, expected_e2_padded, error),
              label + " fc_embedding Q8_0 scalar oracle", error);
        const std::size_t e2_bytes =
            static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
        const auto expected_e2_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(
                expected_e2_padded.data()),
            e2_bytes);
        const auto e2_failure = twin::compare_bytes(
            e2.payload().first(e2_bytes), expected_e2_bytes);
        check(!e2_failure, label + " fc_embedding exact oracle",
              e2_failure ? e2_failure->reason : "");

        // The downstream Q8_1 oracle consumes the GPU RMS output. The norm
        // itself is independently compared above; this avoids making a
        // floating reduction's final ulp change the input to a quantizer.
        std::vector<float> hidden_norm_values(actual_hidden_norm.begin(),
                                              actual_hidden_norm.end());
        std::vector<Q81> hidden_activations;
        check(twin::quantize_q8_1_reference(
                  hidden_norm_values, N_EMBD, rows * 4u,
                  hidden_activations, error),
              label + " fc_hidden Q8_1 scalar oracle", error);
        const auto hidden_weights = std::span<const block_q8_0>(
            reinterpret_cast<const block_q8_0*>(fc_hidden.payload().data()),
            fc_hidden.payload().size() / sizeof(block_q8_0));
        std::vector<float> expected_h2(
            static_cast<std::size_t>(rows) * 4u * N_EMBD);
        for (std::uint32_t pass = 0u; pass < (rows * 4u + 7u) / 8u;
             ++pass) {
            const std::uint32_t row_offset = pass * 8u;
            const std::uint32_t active_rows =
                std::min(8u, rows * 4u - row_offset);
            const auto pass_activations =
                std::span<const Q81>(hidden_activations)
                    .subspan(static_cast<std::size_t>(row_offset) *
                                 (N_EMBD / 32u),
                             static_cast<std::size_t>(active_rows) *
                                 (N_EMBD / 32u));
            std::vector<float> pass_output;
            check(twin::q8_0_matvec_reference(
                      hidden_weights, pass_activations, N_EMBD, N_EMBD,
                      active_rows, active_rows, pass_output, error),
                  label + " fc_hidden pass " + std::to_string(pass) +
                      " Q8_0 scalar oracle",
                  error);
            std::copy(pass_output.begin(), pass_output.end(),
                      expected_h2.begin() +
                          static_cast<std::size_t>(row_offset) * N_EMBD);
        }
        const auto q8_failure = twin::compare_bytes(
            xq.payload().first(q8_bytes),
            {reinterpret_cast<const std::uint8_t*>(
                 hidden_activations.data()),
             q8_bytes});
        check(!q8_failure, label + " fc_hidden Q8_1 exact oracle",
              q8_failure ? q8_failure->reason : "");
        const std::size_t h2_bytes =
            static_cast<std::size_t>(rows) * 4u * N_EMBD * sizeof(float);
        const auto expected_h2_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_h2.data()),
            h2_bytes);
        const auto h2_failure =
            twin::compare_bytes(h2.payload().first(h2_bytes),
                                expected_h2_bytes);
        check(!h2_failure, label + " fc_hidden exact oracle",
              h2_failure ? h2_failure->reason : "");

        const std::array<twin::ByteRange, 1> embedding_written{{
            {0u, in_place_norm ? 0u : embedding_bytes},
        }};
        const std::array<twin::ByteRange, 1> embedding_norm_written{{
            {0u, embedding_bytes},
        }};
        const std::array<twin::ByteRange, 1> q8_written{{{0u, q8_bytes}}};
        const std::array<twin::ByteRange, 1> e2_written{{{0u, e2_bytes}}};
        const std::array<twin::ByteRange, 1> hidden_norm_written{{
            {0u, hidden_bytes},
        }};
        const std::array<twin::ByteRange, 1> h2_written{{{0u, h2_bytes}}};
        const std::array<twin::ByteRange, 1> sink_written{{
            {0u, sink_written_bytes},
        }};
        const bool output_guards =
            !embedding.first_unwritten_change(
                in_place_norm ? std::span<const twin::ByteRange>{}
                              : std::span<const twin::ByteRange>(
                                    embedding_written)) &&
            !embedding_norm.first_unwritten_change(embedding_norm_written) &&
            !xq.first_unwritten_change(q8_written) &&
            !e2.first_unwritten_change(e2_written) &&
            !hidden_norm.first_unwritten_change(hidden_norm_written) &&
            !h2.first_unwritten_change(h2_written) &&
            !sink.first_unwritten_change(
                sink_written_bytes == 0u
                    ? std::span<const twin::ByteRange>{}
                    : std::span<const twin::ByteRange>(sink_written));
        check(output_guards, label + " output guards and inactive tails");
        const bool input_guards =
            !params.first_unwritten_change({}) &&
            !table.first_unwritten_change({}) &&
            !embedding_gamma.first_unwritten_change({}) &&
            !fc_embedding.first_unwritten_change({}) &&
            !residual.first_unwritten_change({}) &&
            !hidden_gamma.first_unwritten_change({}) &&
            !fc_hidden.first_unwritten_change({}) &&
            (!external ||
             (std::memcmp(selected_token.payload().data(), &external_id,
                          sizeof(external_id)) == 0 &&
              !selected_token.first_unwritten_change(token_writer_range)));
        check(input_guards, label + " immutable inputs");
        return !embedding_norm_failure && !hidden_norm_failure && !e2_failure &&
               !q8_failure && !h2_failure && output_guards && input_guards;
    };

    bool passed = true;
    passed &= run_case(1u, 1u, MtpInputTokenSource::params_tokens, false,
                       false, "MTP stages 1–5 K/V T=1");
    passed &= run_case(2u, 2u, MtpInputTokenSource::params_tokens, false,
                       false, "MTP stages 1–5 K/V T=2");
    passed &= run_case(3u, 3u, MtpInputTokenSource::params_tokens, false,
                       false, "MTP stages 1–5 K/V T=3");
    passed &= run_case(8u, 0u, MtpInputTokenSource::params_tokens, false,
                       true, "MTP stages 1–5 K/V T=8");
    passed &= run_case(1u, 2u, MtpInputTokenSource::selected_token, true,
                       false, "MTP stages 1–5 full-layer selected token");

    // Host-side shape and alias validation must fail before appending any
    // dispatches to the transactional command buffer.
    MtpParams invalid_hid = make_params(3u, 2u);
    invalid_hid.hid[1] = 8u;
    std::memcpy(params.payload().data(), &invalid_hid, sizeof(invalid_hid));
    MtpInputProjectionBuffers valid_candidate{
        context.buffers().read_only(params.view()),
        MtpInputTokenSource::params_tokens,
        {},
        EmbedRowsFormat::bf16,
        vocabulary,
        context.buffers().read_only(table.view()),
        context.buffers().read_only(embedding_gamma.view()),
        context.buffers().read_only(fc_embedding.view()),
        residual.view(),
        context.buffers().read_only(hidden_gamma.view()),
        context.buffers().read_only(fc_hidden.view()),
        embedding.view(),
        embedding_norm.view(),
        xq.view(),
        e2.view(),
        hidden_norm.view(),
        h2.view(),
        sink.view(),
    };
    poison_outputs();
    capture_inputs(false);
    auto reject_candidate = [&](const MtpInputProjectionBuffers& candidate,
                                const std::string& label,
                                const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) return false;
        const bool rejected = !encode_mtp_input_projection(
            context, *command, candidate, error);
        const bool expected =
            rejected && error.find(expected_error) != std::string::npos;
        const bool outputs_unchanged =
            !embedding.first_unwritten_change({}) &&
            !embedding_norm.first_unwritten_change({}) &&
            !xq.first_unwritten_change({}) && !e2.first_unwritten_change({}) &&
            !hidden_norm.first_unwritten_change({}) &&
            !h2.first_unwritten_change({}) && !sink.first_unwritten_change({});
        check(expected, label, error);
        check(outputs_unchanged, label + " preserves outputs");
        command->discard();
        return expected && outputs_unchanged;
    };
    passed &= reject_candidate(valid_candidate, "reject MTP input hid shape",
                               "hid");

    MtpParams invalid_token = make_params(3u, 2u);
    invalid_token.tokens[1] = vocabulary;
    std::memcpy(params.payload().data(), &invalid_token,
                sizeof(invalid_token));
    capture_inputs(false);
    passed &= reject_candidate(valid_candidate, "reject MTP input OOV token",
                               "outside the vocabulary");

    MtpParams image_marker_token = make_params(3u, 2u);
    image_marker_token.tokens[1] = kImageTokenFlag;
    std::memcpy(params.payload().data(), &image_marker_token,
                sizeof(image_marker_token));
    capture_inputs(false);
    passed &= reject_candidate(
        valid_candidate, "reject MTP input image-marker token",
        "reserved image-token flag");

    MtpParams valid_t3 = make_params(3u, 2u);
    std::memcpy(params.payload().data(), &valid_t3, sizeof(valid_t3));
    auto alias_candidate = valid_candidate;
    alias_candidate.e2 = alias_candidate.embedding;
    capture_inputs(false);
    passed &= reject_candidate(alias_candidate, "reject MTP input output alias",
                               "overlap");

    auto selected_multirow = valid_candidate;
    selected_multirow.token_source = MtpInputTokenSource::selected_token;
    selected_multirow.selected_token =
        context.buffers().read_only(selected_token.view());
    capture_inputs(true);
    passed &= reject_candidate(selected_multirow,
                               "reject multi-row selected MTP token",
                               "requires T=1");

    MtpParams valid_t1 = make_params(1u, 2u);
    std::memcpy(params.payload().data(), &valid_t1, sizeof(valid_t1));
    auto selected_params_alias = valid_candidate;
    selected_params_alias.params =
        context.buffers().read_only(params.view());
    selected_params_alias.token_source = MtpInputTokenSource::selected_token;
    selected_params_alias.selected_token =
        context.buffers().read_only(params.view().slice(
            offsetof(MtpParams, T), sizeof(std::uint32_t)));
    capture_inputs(false);
    passed &= reject_candidate(
        selected_params_alias, "reject selected MTP token/params alias",
        "overlaps the MtpParams block");

    // A downstream failure discards the already encoded stage-1–5 segment.
    MtpParams rollback_params = make_params(3u, 1u);
    std::memcpy(params.payload().data(), &rollback_params,
                sizeof(rollback_params));
    MtpInputProjectionBuffers rollback_buffers = valid_candidate;
    poison_outputs();
    capture_inputs(false);
    twin::GuardedBuffer invalid_params;
    twin::GuardedBuffer rollback_output;
    const bool rollback_allocated =
        invalid_params.allocate(context, "MTP stage rollback params", 256u,
                                error) &&
        rollback_output.allocate(context, "MTP stage rollback output",
                                 hidden_output_capacity, error);
    check(rollback_allocated, "MTP stage rollback buffers", error);
    if (!rollback_allocated) return false;
    MtpParams invalid_block{};
    std::memcpy(invalid_params.payload().data(), &invalid_block,
                sizeof(invalid_block));
    rollback_output.capture_before_dispatch();
    auto rollback_command =
        context.make_command_buffer("MTP input projection caller rollback",
                                    error);
    check(rollback_command != nullptr,
          "MTP input projection rollback command buffer", error);
    if (!rollback_command) return false;
    bool projection_encoded = false;
    bool later_rejected = false;
    const bool rollback_succeeded = Encoder::encode(
        context, *rollback_command, "MTP input projection rollback segment",
        [&](Encoder& encoder, std::string& encode_error) {
            projection_encoded = encode_mtp_input_projection(
                context, encoder, rollback_buffers, encode_error);
            if (!projection_encoded) return false;
            const MtpAddStreamsBuffers invalid_downstream{
                context.buffers().read_only(invalid_params.view()),
                context.buffers().read_only(h2.view()),
                context.buffers().read_only(e2.view()),
                rollback_output.view(),
            };
            later_rejected = !encode_mtp_add_streams(
                context, encoder, invalid_downstream, encode_error);
            return false;
        },
        error);
    const bool discarded = projection_encoded && later_rejected &&
                           !rollback_succeeded &&
                           rollback_command->unsafe_native_handle() == nullptr;
    check(discarded, "MTP stages 1–5 caller-owned rollback", error);
    const bool rollback_outputs_unchanged =
        !embedding.first_unwritten_change({}) &&
        !embedding_norm.first_unwritten_change({}) &&
        !xq.first_unwritten_change({}) && !e2.first_unwritten_change({}) &&
        !hidden_norm.first_unwritten_change({}) &&
        !h2.first_unwritten_change({}) &&
        !rollback_output.first_unwritten_change({});
    check(rollback_outputs_unchanged,
          "MTP stages 1–5 rollback preserves scratch");
    passed &= discarded && rollback_outputs_unchanged;
    rollback_command->discard();

    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        poison_outputs();
        capture_inputs(false);
        auto group_command =
            context.make_command_buffer("MTP input projection group boundary",
                                        error);
        check(group_command != nullptr,
              "MTP input projection group command buffer", error);
        if (!group_command) return false;
        const bool group_rejected = Encoder::encode(
            context, *group_command, "MTP input projection group validation",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                std::string stage_error;
                const bool rejected = !encode_mtp_input_projection(
                    context, encoder, valid_candidate, stage_error);
                if (!encoder.group_end(encode_error)) return false;
                if (!rejected ||
                    stage_error.find("ordered concurrent-group boundary") ==
                        std::string::npos) {
                    encode_error =
                        "MTP input projection accepted an open group";
                    return false;
                }
                return true;
            },
            error);
        const bool group_outputs_unchanged =
            !embedding.first_unwritten_change({}) &&
            !embedding_norm.first_unwritten_change({}) &&
            !xq.first_unwritten_change({}) && !e2.first_unwritten_change({}) &&
            !hidden_norm.first_unwritten_change({}) &&
            !h2.first_unwritten_change({}) && !sink.first_unwritten_change({});
        check(group_rejected, "MTP stages 1–5 reject open concurrent group",
              error);
        check(group_outputs_unchanged,
              "MTP stages 1–5 open-group refusal preserves scratch");
        passed &= group_rejected && group_outputs_unchanged;
        group_command->discard();
    }
    return passed;
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
        std::vector<float> expected_full_logits;
        check(twin::q6_k_matvec_reference(
                  weights, expected_q8, N_EMBD, head_rows, 1u, 1u,
                  expected_full_logits, error),
              "MTP stage 21 full Q6_K CPU oracle", error);
        passed &= execute_draft_head_case(
            context, "MTP stage 21 full Q6_K head",
            MatVecWeightFormat::q6_k, head_rows,
            {reinterpret_cast<const std::uint8_t*>(weights.data()),
             weights.size() * sizeof(block_q6_K)},
            sample_values, {}, expected_q8, expected_full_logits, false);

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
        std::vector<float> expected_full_logits;
        check(twin::iq4_xs_matvec_reference(
                  weights, expected_q8, N_EMBD, head_rows, 1u, 1u,
                  expected_full_logits, error),
              "MTP stage 21 full IQ4_XS CPU oracle", error);
        passed &= execute_draft_head_case(
            context, "MTP stage 21 full IQ4_XS head",
            MatVecWeightFormat::iq4_xs, head_rows,
            {reinterpret_cast<const std::uint8_t*>(weights.data()),
             weights.size() * sizeof(block_iq4_xs)},
            sample_values, {}, expected_q8, expected_full_logits, false);

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

    MtpDraftVocabularyMap validated_map;
    check(make_mtp_draft_vocabulary_map(
              context, context.buffers().read_only(map.view()), head_rows,
              validated_map, error),
          "validate MTP draft row map once", error);
    if (!validated_map) return false;
    MtpDraftHeadBuffers valid{
        context.buffers().read_only(params.view()),
        sample.view(),
        context.buffers().read_only(weights.view()),
        validated_map,
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

    auto expect_map_rejected = [&](const std::array<std::int32_t, 2>& ids,
                                   const std::string& label,
                                   const std::string& expected_error) {
        std::memcpy(map.payload().data(), ids.data(), sizeof(ids));
        map.capture_before_dispatch();
        MtpDraftVocabularyMap rejected_map;
        const bool accepted = make_mtp_draft_vocabulary_map(
            context, context.buffers().read_only(map.view()), head_rows,
            rejected_map, error);
        const bool expected = !accepted &&
                              error.find(expected_error) != std::string::npos;
        const bool unchanged = !map.first_unwritten_change({});
        check(expected, label, error);
        check(unchanged, label + " does not mutate the source map");
        return expected && unchanged;
    };
    passed &= expect_map_rejected(
        std::array<std::int32_t, 2>{-1, 0},
        "reject negative MTP draft row id", "outside the full head");
    passed &= expect_map_rejected(
        std::array<std::int32_t, 2>{1, 1},
        "reject duplicate MTP draft row id", "duplicate head rows");
    std::memcpy(map.payload().data(), valid_map.data(), sizeof(valid_map));
    map.capture_before_dispatch();
    check(make_mtp_draft_vocabulary_map(
              context, context.buffers().read_only(map.view()), head_rows,
              validated_map, error),
          "revalidate restored MTP draft map", error);
    valid.draft_vocabulary = validated_map;

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
    output_alias.draft_vocabulary = {};
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

bool test_mtp_draft_sampling(Context& context) {
    constexpr std::uint32_t output_rows = 7u;
    constexpr std::uint32_t full_head_size = output_rows;
    constexpr std::uint32_t draft_row_count = 4u;
    constexpr std::size_t logits_bytes =
        static_cast<std::size_t>(output_rows) * sizeof(float);
    constexpr std::array<float, output_rows> logits_values{
        -1.0f, 0.25f, 3.25f, 2.5f, -3.0f, 1.75f, 3.0f,
    };
    constexpr std::array<std::int32_t, draft_row_count> draft_ids{
        6, 2, 5, 3,
    };
    constexpr std::uint32_t expected_subset_id = 2u;
    constexpr std::int32_t expected_full_id = 5;

    std::string error;
    twin::GuardedBuffer params;
    twin::GuardedBuffer logits;
    twin::GuardedBuffer map;
    twin::GuardedBuffer output_ids;
    twin::GuardedBuffer probabilities;
    const bool allocated =
        params.allocate(context, "MTP draft sampling params", 256u, error) &&
        logits.allocate(context, "MTP draft sampling logits", logits_bytes,
                        error) &&
        map.allocate(context, "MTP draft sampling map",
                     sizeof(draft_ids), error) &&
        output_ids.allocate(context, "MTP draft sampling output ID",
                            sizeof(std::int32_t), error) &&
        probabilities.allocate(context, "MTP draft sampling probability",
                               sizeof(float), error);
    check(allocated, "MTP stage 22 allocations", error);
    if (!allocated) return false;

    MtpParams params_value{};
    params_value.T = 1u;
    params_value.samp.greedy = 1;
    std::memcpy(params.payload().data(), &params_value, sizeof(params_value));
    std::memcpy(logits.payload().data(), logits_values.data(), logits_bytes);
    std::memcpy(map.payload().data(), draft_ids.data(), sizeof(draft_ids));
    auto expected_argmax_probability = [&](std::uint32_t active_rows) {
        const float maximum = logits_values[expected_subset_id];
        double exponential_sum = 0.0;
        for (std::uint32_t row = 0u; row < active_rows; ++row) {
            exponential_sum += std::exp(static_cast<double>(
                logits_values[row] - maximum));
        }
        return static_cast<float>(1.0 / exponential_sum);
    };
    const float expected_full_probability =
        expected_argmax_probability(output_rows);
    const float expected_draft_probability =
        expected_argmax_probability(draft_row_count);

    MtpDraftVocabularyMap validated_map;
    const bool valid_map = make_mtp_draft_vocabulary_map(
        context, context.buffers().read_only(map.view()), full_head_size,
        validated_map, error);
    check(valid_map, "MTP stage 22 map validation", error);
    if (!valid_map) return false;

    auto run_valid_case = [&](const std::string& label,
                              MtpDraftVocabularyMap vocabulary,
                              std::int32_t expected_id,
                              float expected_probability,
                              float& actual_probability) {
        std::fill(output_ids.payload().begin(), output_ids.payload().end(),
                  0xa5u);
        std::fill(probabilities.payload().begin(),
                  probabilities.payload().end(), 0x5au);
        params.capture_before_dispatch();
        logits.capture_before_dispatch();
        map.capture_before_dispatch();
        output_ids.capture_before_dispatch();
        probabilities.capture_before_dispatch();
        const MtpDraftSamplingBuffers buffers{
            params.view(), logits.view(), vocabulary, output_ids.view(),
            probabilities.view(),
        };
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) return false;
        const bool encoded = encode_mtp_draft_sampling(
            context, *command, full_head_size, buffers, error);
        check(encoded, label + " encode", error);
        if (!encoded) return false;
        const bool completed = submit_and_wait(command, label, error);
        check(completed, label + " GPU completion", error);
        if (!completed) return false;

        std::int32_t actual_id = -1;
        std::memcpy(&actual_id, output_ids.payload().data(),
                    sizeof(actual_id));
        std::memcpy(&actual_probability, probabilities.payload().data(),
                    sizeof(actual_probability));
        check(actual_id == expected_id, label + " selected token ID");
        const bool probability_matches =
            std::isfinite(actual_probability) &&
            std::abs(actual_probability - expected_probability) < 1.0e-6f;
        check(probability_matches,
              label + " argmax softmax probability",
              "expected " + std::to_string(expected_probability) +
                  ", got " + std::to_string(actual_probability));
        const std::array<twin::ByteRange, 1> id_written{{
            {0u, sizeof(std::int32_t)},
        }};
        const std::array<twin::ByteRange, 1> probability_written{{
            {0u, sizeof(float)},
        }};
        const bool guards =
            !output_ids.first_unwritten_change(id_written) &&
            !probabilities.first_unwritten_change(probability_written) &&
            !params.first_unwritten_change({}) &&
            !logits.first_unwritten_change({}) &&
            !map.first_unwritten_change({});
        check(guards, label + " guards and immutable inputs");
        return actual_id == expected_id && probability_matches && guards;
    };

    float full_probability = 0.0f;
    const bool full_passed = run_valid_case(
        "MTP stage 22 full vocabulary", {}, expected_subset_id,
        expected_full_probability,
        full_probability);
    float mapped_probability = 0.0f;
    const bool mapped_passed = run_valid_case(
        "MTP stage 22 draft vocabulary", validated_map, expected_full_id,
        expected_draft_probability,
        mapped_probability);
    const bool subset_probability_differs =
        std::abs(full_probability - mapped_probability) > 1.0e-3f;
    check(subset_probability_differs,
          "MTP stage 22 probability uses only active draft rows");

    auto expect_rejected = [&](const MtpDraftSamplingBuffers& candidate,
                               std::uint32_t vocabulary,
                               const std::string& label,
                               const std::string& expected_error) {
        params.capture_before_dispatch();
        logits.capture_before_dispatch();
        map.capture_before_dispatch();
        output_ids.capture_before_dispatch();
        probabilities.capture_before_dispatch();
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) return false;
        const bool rejected = !encode_mtp_draft_sampling(
            context, *command, vocabulary, candidate, error);
        const bool expected =
            rejected && error.find(expected_error) != std::string::npos;
        const bool unchanged =
            !params.first_unwritten_change({}) &&
            !logits.first_unwritten_change({}) &&
            !map.first_unwritten_change({}) &&
            !output_ids.first_unwritten_change({}) &&
            !probabilities.first_unwritten_change({});
        check(expected, label, error);
        check(unchanged, label + " preserves buffers");
        command->discard();
        return expected && unchanged;
    };

    bool passed =
        full_passed && mapped_passed && subset_probability_differs;
    auto valid_buffers = MtpDraftSamplingBuffers{
        params.view(), logits.view(), validated_map, output_ids.view(),
        probabilities.view(),
    };
    params_value.T = 2u;
    std::memcpy(params.payload().data(), &params_value, sizeof(params_value));
    passed &= expect_rejected(valid_buffers, full_head_size,
                              "reject MTP stage 22 T=2", "requires T=1");
    params_value.T = 1u;
    params_value.samp.greedy = 0;
    std::memcpy(params.payload().data(), &params_value, sizeof(params_value));
    passed &= expect_rejected(
        valid_buffers, full_head_size, "reject non-greedy MTP stage 22",
        "requires greedy=1");
    params_value.samp.greedy = 1;
    params_value.samp.hist_off = 1u;
    params_value.samp.hist_len = 1u;
    std::memcpy(params.payload().data(), &params_value, sizeof(params_value));
    passed &= expect_rejected(
        valid_buffers, full_head_size, "reject MTP stage 22 history",
        "empty zero-offset history");
    params_value.samp.hist_off = 0u;
    params_value.samp.hist_len = 0u;
    std::memcpy(params.payload().data(), &params_value, sizeof(params_value));

    auto short_logits = valid_buffers;
    short_logits.logits = logits.view().slice(
        0u, static_cast<std::size_t>(draft_row_count) * sizeof(float) - 1u);
    passed &= expect_rejected(
        short_logits, full_head_size, "reject short MTP stage 22 logits",
        "undersized or misaligned");
    auto output_alias = valid_buffers;
    output_alias.probabilities =
        output_ids.view().slice(0u, sizeof(float));
    passed &= expect_rejected(
        output_alias, full_head_size, "reject MTP stage 22 output alias",
        "overlap");
    passed &= expect_rejected(
        valid_buffers, 0u, "reject empty MTP stage 22 head",
        "must be in 1..248320");

    // The valid greedy dispatch is discarded when the later MTP wrapper
    // rejects malformed T in the same caller-owned encoder segment.
    std::fill(output_ids.payload().begin(), output_ids.payload().end(), 0x61u);
    std::fill(probabilities.payload().begin(), probabilities.payload().end(),
              0x72u);
    output_ids.capture_before_dispatch();
    probabilities.capture_before_dispatch();
    auto rollback_command =
        context.make_command_buffer("MTP stage 22 late rollback", error);
    check(rollback_command != nullptr,
          "MTP stage 22 rollback command buffer", error);
    if (!rollback_command) return false;
    bool first_dispatch_encoded = false;
    bool expected_late_rejection = false;
    const bool transaction_succeeded = Encoder::encode(
        context, *rollback_command, "MTP stage 22 rollback segment",
        [&](Encoder& encoder, std::string& encode_error) {
            const SamplerBuffers greedy_buffers{
                logits.view(),
                params.view().slice(offsetof(MtpParams, T),
                                    sizeof(std::uint32_t)),
                params.view().slice(offsetof(MtpParams, samp),
                                    sizeof(SampRow)),
                {},
                {},
                output_ids.view(),
            };
            first_dispatch_encoded = encode_sampler_greedy(
                context, encoder, 1u, output_rows, greedy_buffers,
                encode_error);
            if (!first_dispatch_encoded) return false;
            MtpParams invalid_value = params_value;
            invalid_value.T = 2u;
            twin::GuardedBuffer invalid_params;
            if (!invalid_params.allocate(
                    context, "late-invalid MTP sampling params", 256u,
                    encode_error)) {
                return false;
            }
            std::memcpy(invalid_params.payload().data(), &invalid_value,
                        sizeof(invalid_value));
            auto invalid = valid_buffers;
            invalid.params = invalid_params.view();
            const bool second_stage_encoded = encode_mtp_draft_sampling(
                context, encoder, full_head_size, invalid, encode_error);
            expected_late_rejection =
                !second_stage_encoded &&
                encode_error.find("requires T=1") != std::string::npos;
            return second_stage_encoded;
        },
        error);
    const bool discarded = first_dispatch_encoded &&
                           expected_late_rejection &&
                           !transaction_succeeded &&
                           rollback_command->unsafe_native_handle() == nullptr;
    const bool rollback_unchanged =
        !output_ids.first_unwritten_change({}) &&
        !probabilities.first_unwritten_change({});
    check(discarded, "MTP stage 22 caller-owned late rollback", error);
    check(rollback_unchanged,
          "MTP stage 22 rollback preserves output IDs and probability");
    passed &= discarded && rollback_unchanged;
    rollback_command->discard();

    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        output_ids.capture_before_dispatch();
        probabilities.capture_before_dispatch();
        auto group_command =
            context.make_command_buffer("MTP stage 22 group boundary", error);
        check(group_command != nullptr,
              "MTP stage 22 group-boundary command buffer", error);
        if (!group_command) return false;
        const bool group_rejected = Encoder::encode(
            context, *group_command, "MTP stage 22 group validation",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                std::string stage_error;
                const bool rejected = !encode_mtp_draft_sampling(
                    context, encoder, full_head_size, valid_buffers,
                    stage_error);
                if (!encoder.group_end(encode_error)) return false;
                if (!rejected ||
                    stage_error.find("ordered concurrent-group boundary") ==
                        std::string::npos) {
                    encode_error = "MTP stage 22 accepted an open group";
                    return false;
                }
                return true;
            },
            error);
        const bool unchanged =
            !output_ids.first_unwritten_change({}) &&
            !probabilities.first_unwritten_change({});
        check(group_rejected,
              "MTP stage 22 rejects open concurrent group", error);
        check(unchanged,
              "MTP stage 22 open-group rejection preserves outputs");
        passed &= group_rejected && unchanged;
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
    passed &= test_mtp_input_projection(*context);
    passed &= test_mtp_draft_head(*context);
    passed &= test_mtp_draft_head_validation(*context, *foreign_context);
    passed &= test_mtp_draft_sampling(*context);

    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("metal_mtp_test: MTP stages 1–6, 21 and 22 OK");
    return 0;
}
