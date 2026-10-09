// Metal twin for MTP drafter stage 6 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "twin.hpp"

#import <Metal/Metal.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
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

    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("metal_mtp_test: MTP stage 6 OK");
    return 0;
}
