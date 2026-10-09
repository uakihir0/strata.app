// Metal twin for MTP drafter stage 10 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "twin.hpp"

#import <Metal/Metal.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

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
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kMaxRows = VERIFY_MAX_T;
constexpr std::uint32_t kOutputWidth = N_HEAD_Q * 2u * HEAD_DIM;
constexpr std::uint32_t kBlocksPerRow = N_EMBD / 32u;
constexpr std::size_t kWeightBlocks =
    static_cast<std::size_t>(kOutputWidth) * kBlocksPerRow;
constexpr std::size_t kActivationBytes =
    kBlocksPerRow * sizeof(Q81);
constexpr std::size_t kWeightBytes =
    kWeightBlocks * sizeof(block_q8_0);
constexpr std::size_t kOutputBytes =
    static_cast<std::size_t>(kOutputWidth) * sizeof(float);
constexpr std::uint32_t kPoisonBits = 0x7fc12345u;
constexpr float kPoison = std::bit_cast<float>(kPoisonBits);

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !condition;
}

template <class T>
std::span<T> typed(twin::GuardedBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
}

struct Fixture {
    twin::GuardedBuffer params;
    twin::GuardedBuffer xq;
    twin::GuardedBuffer weights;
    twin::GuardedBuffer q_full;
    twin::GuardedBuffer sink;

    bool allocate(Context& context, const std::string& label) {
        std::string error;
        const std::array<std::pair<twin::GuardedBuffer*, std::pair<const char*,
                                                                  std::size_t>>,
                         5>
            buffers{{
                {&params, {"params", 256u}},
                {&xq, {"stage-8 xq", kMaxRows * kActivationBytes}},
                {&weights, {"q_proj weights", kWeightBytes}},
                {&q_full, {"q_full", kOutputBytes}},
                {&sink, {"bo sink", N_EMBD * sizeof(float)}},
            }};
        for (const auto& [buffer, metadata] : buffers) {
            if (!buffer->allocate(context, label + " " + metadata.first,
                                  metadata.second, error)) {
                check(false, label + " " + metadata.first + " allocation",
                      error);
                return false;
            }
        }
        if (!initialize()) return false;
        return true;
    }

    bool initialize() {
        std::fill(xq.payload().begin(), xq.payload().end(), 0xa5u);
        std::vector<float> source(N_EMBD);
        for (std::size_t i = 0u; i < source.size(); ++i) {
            const auto centered =
                static_cast<std::int32_t>((i * 31u + i / 37u) % 257u) - 128;
            source[i] = static_cast<float>(centered) * 0.00390625f;
        }
        std::vector<Q81> quantized;
        std::string error;
        if (!twin::quantize_q8_1_reference(source, N_EMBD, 1u, quantized,
                                           error)) {
            check(false, "MTP stage-10 Q8_1 fixture oracle", error);
            return false;
        }
        std::memcpy(xq.payload().data(), quantized.data(), kActivationBytes);

        auto weight_blocks = typed<block_q8_0>(weights);
        for (std::size_t block = 0u; block < kWeightBlocks; ++block) {
            const float scale =
                0.03125f +
                static_cast<float>((block * 7u + block / 19u) % 13u) *
                    0.015625f;
            weight_blocks[block].d =
                strata::kernels::f16_from_f32(scale);
            for (std::size_t lane = 0u; lane < 32u; ++lane) {
                const int centered = static_cast<int>(
                    (block * 17u + lane * 11u + block / 7u) % 63u) - 31;
                weight_blocks[block].qs[lane] =
                    static_cast<std::int8_t>(centered);
            }
        }
        return reset();
    }

    bool reset() {
        std::fill(params.payload().begin(), params.payload().end(), 0u);
        MtpParams block{};
        block.T = 1u;
        std::memcpy(params.payload().data(), &block, sizeof(block));
        auto output = typed<float>(q_full);
        std::fill(output.begin(), output.end(), kPoison);
        std::fill(sink.payload().begin(), sink.payload().end(), 0x5au);
        return true;
    }

    MtpAttentionQueryProjectionBuffers mtp_buffers(Context& context) {
        return {context.buffers().read_only(params.view()),
                xq.view(),
                context.buffers().read_only(weights.view()),
                q_full.view(),
                sink.view()};
    }

    void capture() {
        params.capture_before_dispatch();
        xq.capture_before_dispatch();
        weights.capture_before_dispatch();
        q_full.capture_before_dispatch();
        sink.capture_before_dispatch();
    }

    bool inputs_unchanged() const {
        return !params.first_unwritten_change({}) &&
               !xq.first_unwritten_change({}) &&
               !weights.first_unwritten_change({});
    }
};

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     const std::string& label, std::string& error) {
    if (!command || !command->commit(error)) return false;
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        error = label + " command buffer became unavailable";
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

bool execute_projection(Context& context, Fixture& fixture,
                        bool caller_owned) {
    fixture.reset();
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP stage-10 query projection", error);
    check(command != nullptr, "MTP stage-10 command buffer", error);
    if (!command) return false;

    bool encoded = false;
    const auto buffers = fixture.mtp_buffers(context);
    if (caller_owned) {
        encoded = Encoder::encode(
            context, *command, "MTP stage-10 caller-owned",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_attention_query_projection(
                    context, encoder, buffers, encode_error);
            },
            error);
    } else {
        encoded = encode_mtp_attention_query_projection(
            context, *command, buffers, error);
    }
    check(encoded, "MTP stage-10 encode", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(
        command, "MTP stage-10 query projection", error);
    check(completed, "MTP stage-10 GPU completion", error);
    if (!completed) return false;

    std::vector<float> expected;
    const auto activations =
        std::span<const Q81>(typed<Q81>(fixture.xq)).first(kBlocksPerRow);
    const auto weights =
        std::span<const block_q8_0>(typed<block_q8_0>(fixture.weights))
            .first(kWeightBlocks);
    const bool oracle_ready = twin::q8_0_matvec_reference(
        weights, activations, N_EMBD, kOutputWidth, 1u, 1u, expected,
        error);
    check(oracle_ready, "MTP stage-10 Q8_0 scalar oracle", error);
    if (!oracle_ready) return false;
    const auto actual =
        std::span<const float>(typed<float>(fixture.q_full)).first(
            kOutputWidth);
    const bool exact =
        std::memcmp(actual.data(), expected.data(), kOutputBytes) == 0;
    check(exact, "MTP stage-10 q_full matches the scalar Q8_0 oracle");
    const bool finite = std::all_of(
        actual.begin(), actual.end(),
        [](float value) { return std::isfinite(value); });
    check(finite, "MTP stage-10 active q_full row is finite");

    const std::array<twin::ByteRange, 1> output_written{{
        {0u, kOutputBytes},
    }};
    const bool guards_ok =
        fixture.inputs_unchanged() &&
        !fixture.sink.first_unwritten_change({}) &&
        !fixture.q_full.first_unwritten_change(output_written);
    check(guards_ok,
          "MTP stage-10 preserves inputs, sink and q_full allocation guards");
    return exact && finite && guards_ok;
}

bool reject_case(Context& context,
                 const MtpAttentionQueryProjectionBuffers& buffers,
                 const std::string& label,
                 const std::string& expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_attention_query_projection(context, *command, buffers,
                                               error);
    const bool expected =
        expected_error.empty() ||
        error.find(expected_error) != std::string::npos;
    check(rejected && expected, label, error);
    command->discard();
    return rejected && expected;
}

bool test_validation(Context& context, Context& foreign_context,
                     Fixture& fixture) {
    bool passed = true;
    fixture.reset();
    auto buffers = fixture.mtp_buffers(context);
    MtpParams params{};
    twin::GuardedBuffer overlap_storage;
    std::string overlap_error;
    const bool overlap_allocated = overlap_storage.allocate(
        context, "MTP query projection alias fixture",
        kOutputBytes + kActivationBytes, overlap_error);
    check(overlap_allocated, "MTP stage-10 alias fixture allocation",
          overlap_error);
    if (!overlap_allocated) return false;
    const std::size_t partially_overlapping_xq_offset =
        (kOutputBytes - kActivationBytes + 255u) / 256u * 256u;

    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers(context);
    passed &= reject_case(context, buffers, "MTP stage 10 rejects T=0",
                          "T must be in 1..8");

    params.T = 2u;
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers(context);
    passed &= reject_case(context, buffers, "MTP stage 10 rejects T=2",
                          "requires T=1");

    params.T = 1u;
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers(context);
    buffers.q_weights = context.buffers().read_only(
        fixture.weights.view().slice(0u, kWeightBytes - 1u));
    passed &= reject_case(context, buffers,
                          "MTP stage 10 rejects short q_proj weights",
                          "q_proj weights buffer is undersized");

    buffers = fixture.mtp_buffers(context);
    buffers.sink = fixture.sink.view().slice(0u, sizeof(float) - 1u);
    passed &= reject_case(context, buffers,
                          "MTP stage 10 rejects a short sink",
                          "sink buffer is undersized");

    buffers = fixture.mtp_buffers(context);
    buffers.sink = fixture.q_full.view().slice(0u, sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 10 rejects q_full/sink overlap",
                          "q_full and sink outputs overlap");

    buffers = fixture.mtp_buffers(context);
    buffers.q_full = fixture.weights.view().slice(0u, kOutputBytes);
    passed &= reject_case(
        context, buffers, "MTP stage 10 rejects q_full/q_proj overlap",
        "q_full overlaps an input");

    buffers = fixture.mtp_buffers(context);
    buffers.q_full = overlap_storage.view().slice(0u, kOutputBytes);
    buffers.xq = overlap_storage.view().slice(
        partially_overlapping_xq_offset, kActivationBytes);
    passed &= reject_case(
        context, buffers, "MTP stage 10 rejects partial q_full/xq overlap",
        "q_full overlaps an input");

    buffers = fixture.mtp_buffers(context);
    buffers.xq = overlap_storage.view().slice(
        partially_overlapping_xq_offset, kActivationBytes);
    buffers.sink = overlap_storage.view().slice(
        partially_overlapping_xq_offset + 512u, sizeof(float));
    passed &= reject_case(
        context, buffers, "MTP stage 10 rejects sink/xq overlap",
        "sink overlaps an input");

    MtpParams aliased{};
    aliased.T = 1u;
    std::memcpy(fixture.q_full.payload().data(), &aliased, sizeof(aliased));
    buffers = fixture.mtp_buffers(context);
    buffers.params =
        context.buffers().read_only(fixture.q_full.view());
    passed &= reject_case(context, buffers,
                          "MTP stage 10 rejects params/q_full alias",
                          "q_full overlaps an input");

    twin::GuardedBuffer foreign_weights;
    std::string error;
    const bool foreign_allocated = foreign_weights.allocate(
        foreign_context, "foreign MTP query projection weights",
        kWeightBytes, error);
    check(foreign_allocated, "foreign MTP q_proj weight allocation", error);
    if (foreign_allocated) {
        buffers = fixture.mtp_buffers(context);
        buffers.q_weights =
            foreign_context.buffers().read_only(foreign_weights.view());
        passed &= reject_case(
            context, buffers, "MTP stage 10 rejects foreign q_proj weights",
            "q_proj weights buffer is not owned");
    } else {
        passed = false;
    }
    return passed;
}

bool test_concurrent_group_boundary(Context& context, Fixture& fixture) {
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode == nullptr ||
        std::string_view(dispatch_mode) != "concurrent") {
        return true;
    }
    fixture.reset();
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP stage-10 group boundary", error);
    if (!command) {
        check(false, "MTP stage-10 group-boundary command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers(context);
    const bool wrapper_checked = Encoder::encode(
        context, *command, "MTP stage-10 group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected =
                !encode_mtp_attention_query_projection(
                    context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP stage 10 accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.xq.first_unwritten_change({}) &&
        !fixture.weights.first_unwritten_change({}) &&
        !fixture.q_full.first_unwritten_change({}) &&
        !fixture.sink.first_unwritten_change({});
    check(wrapper_checked, "MTP stage 10 rejects an open concurrent group",
          error);
    check(unchanged,
          "MTP stage-10 open-group rejection preserves every buffer");
    command->discard();
    return wrapper_checked && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.reset();
    fixture.capture();
    const auto good = fixture.mtp_buffers(context);
    auto bad = good;
    bad.sink = bad.q_full.slice(0u, sizeof(float));

    std::string error;
    auto command = context.make_command_buffer("MTP stage-10 rollback", error);
    if (!command) {
        check(false, "MTP stage-10 rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stage-10 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_attention_query_projection(
                    context, encoder, good, encode_error)) {
                return false;
            }
            return encode_mtp_attention_query_projection(
                context, encoder, bad, encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.xq.first_unwritten_change({}) &&
        !fixture.weights.first_unwritten_change({}) &&
        !fixture.q_full.first_unwritten_change({}) &&
        !fixture.sink.first_unwritten_change({});
    check(rejected && discarded,
          "MTP stage-10 late failure discards the caller-owned segment",
          error);
    check(unchanged, "MTP stage-10 rollback preserves every buffer");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_query_projection_test <strata.metallib>\n");
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

    Fixture fixture;
    const bool allocated = fixture.allocate(*context, "MTP query projection");
    check(allocated, "allocate MTP stage-10 fixture");
    if (!allocated) return 1;

    bool passed = execute_projection(*context, fixture, false);
    passed &= execute_projection(*context, fixture, true);
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_concurrent_group_boundary(*context, fixture);
    passed &= test_late_rollback(*context, fixture);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_query_projection_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_query_projection_test: MTP stage 10 OK");
    return 0;
}
