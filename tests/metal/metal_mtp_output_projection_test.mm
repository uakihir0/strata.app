// Metal twin for MTP drafter stage 14 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"
#include "oracles/qsa.hpp"
#include "strata/kernels/f16_bits.hpp"
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

constexpr std::uint32_t kInputWidth = N_HEAD_Q * HEAD_DIM;
constexpr std::uint32_t kOutputWidth = N_EMBD;
constexpr std::uint32_t kQ8BlockValues = 32u;
constexpr std::size_t kAttentionBytes =
    static_cast<std::size_t>(kInputWidth) * sizeof(float);
constexpr std::size_t kQFullBytes =
    static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM * sizeof(float);
constexpr std::size_t kQ81Bytes =
    (static_cast<std::size_t>(kInputWidth) / kQ8BlockValues) * sizeof(Q81);
constexpr std::size_t kWeightBlocks =
    static_cast<std::size_t>(kOutputWidth) *
    (kInputWidth / kQ8BlockValues);
constexpr std::size_t kWeightBytes = kWeightBlocks * sizeof(block_q8_0);
constexpr std::size_t kOutputBytes =
    static_cast<std::size_t>(kOutputWidth) * sizeof(float);
constexpr std::size_t kPoisonXqBytes = 256u;

static_assert(kInputWidth == 6'144u);
static_assert(kQFullBytes == 12'288u * sizeof(float));
static_assert(sizeof(Q81) == 36u);

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
    twin::GuardedBuffer attn;
    twin::GuardedBuffer q_full;
    twin::GuardedBuffer attn32;
    twin::GuardedBuffer xq;
    twin::GuardedBuffer weights;
    twin::GuardedBuffer bo;
    twin::GuardedBuffer sink;

    bool allocate(Context& context, const std::string& label) {
        const std::array<std::pair<twin::GuardedBuffer*,
                                   std::pair<const char*, std::size_t>>,
                         8>
            buffers{{
                {&params, {"params", 256u}},
                {&attn, {"stage-13 attn", kAttentionBytes}},
                {&q_full, {"stage-10 q_full", kQFullBytes}},
                {&attn32, {"gated attn", kAttentionBytes}},
                {&xq, {"stage-14 xq", kQ81Bytes + kPoisonXqBytes}},
                {&weights, {"o_proj weights", kWeightBytes}},
                {&bo, {"bo", kOutputBytes}},
                {&sink, {"o_proj sink", kOutputBytes}},
            }};
        for (const auto& [buffer, metadata] : buffers) {
            std::string error;
            if (!buffer->allocate(context, label + " " + metadata.first,
                                  metadata.second, error)) {
                check(false, label + " " + metadata.first + " allocation",
                      error);
                return false;
            }
        }
        return initialize();
    }

    bool initialize() {
        auto attn_values = typed<float>(attn);
        for (std::size_t i = 0u; i < attn_values.size(); ++i) {
            const auto centered = static_cast<std::int32_t>(
                                     (i * 37u + i / 23u) % 251u) -
                                 125;
            attn_values[i] = static_cast<float>(centered) * 0.00390625f;
        }

        auto q_values = typed<float>(q_full);
        for (std::size_t i = 0u; i < q_values.size(); ++i) {
            q_values[i] =
                static_cast<float>(static_cast<std::int32_t>(i % 31u) - 15) *
                0.0625f;
        }
        for (std::size_t head = 0u; head < N_HEAD_Q; ++head) {
            std::fill_n(q_values.begin() + head * 2u * HEAD_DIM + HEAD_DIM,
                        HEAD_DIM, 0.0f);
        }

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
                    (block * 17u + lane * 11u + block / 7u) % 63u) -
                                     31;
                weight_blocks[block].qs[lane] =
                    static_cast<std::int8_t>(centered);
            }
        }
        return reset();
    }

    void set_rows(std::uint32_t rows) {
        std::fill(params.payload().begin(), params.payload().end(), 0u);
        MtpParams block{};
        block.T = rows;
        std::memcpy(params.payload().data(), &block, sizeof(block));
    }

    bool reset() {
        set_rows(1u);
        std::fill(attn32.payload().begin(), attn32.payload().end(), 0xa5u);
        std::fill(xq.payload().begin(), xq.payload().end(), 0x5au);
        std::fill(bo.payload().begin(), bo.payload().end(), 0x3cu);
        std::fill(sink.payload().begin(), sink.payload().end(), 0x69u);
        return true;
    }

    MtpAttentionOutputProjectionBuffers mtp_buffers(Context& context) {
        return {context.buffers().read_only(params.view()),
                attn.view(),
                q_full.view(),
                attn32.view(),
                xq.view(),
                context.buffers().read_only(weights.view()),
                bo.view(),
                sink.view()};
    }

    void capture() {
        params.capture_before_dispatch();
        attn.capture_before_dispatch();
        q_full.capture_before_dispatch();
        attn32.capture_before_dispatch();
        xq.capture_before_dispatch();
        weights.capture_before_dispatch();
        bo.capture_before_dispatch();
        sink.capture_before_dispatch();
    }

    bool inputs_unchanged() const {
        return !params.first_unwritten_change({}) &&
               !attn.first_unwritten_change({}) &&
               !q_full.first_unwritten_change({}) &&
               !weights.first_unwritten_change({});
    }

    bool all_buffers_unchanged() const {
        return inputs_unchanged() &&
               !attn32.first_unwritten_change({}) &&
               !xq.first_unwritten_change({}) &&
               !bo.first_unwritten_change({}) &&
               !sink.first_unwritten_change({});
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
        context.make_command_buffer("MTP stage-14 output projection", error);
    check(command != nullptr, "MTP stage-14 command buffer", error);
    if (!command) return false;

    const auto buffers = fixture.mtp_buffers(context);
    bool encoded = false;
    if (caller_owned) {
        encoded = Encoder::encode(
            context, *command, "MTP stage-14 caller-owned",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_attention_output_projection(
                    context, encoder, buffers, encode_error);
            },
            error);
    } else {
        encoded = encode_mtp_attention_output_projection(
            context, *command, buffers, error);
    }
    check(encoded, "MTP stage-14 encode", error);
    if (!encoded) return false;
    const bool completed =
        submit_and_wait(command, "MTP stage-14 output projection", error);
    check(completed, "MTP stage-14 GPU completion", error);
    if (!completed) return false;

    std::vector<twin::tf> expected_gate;
    oracle::qsa_gate(typed<float>(fixture.attn),
                     typed<float>(fixture.q_full), 1u, expected_gate);
    const auto actual_gate =
        std::span<const float>(typed<float>(fixture.attn32)).first(
            kInputWidth);
    const auto gate_mismatch = twin::compare_floats(
        actual_gate, expected_gate, {twin::Comparison::units, 8.0, 0.0, 0u});
    check(!gate_mismatch,
          "MTP stage-14 gate matches the accurate sigmoid oracle within U(8)");

    std::vector<float> expected_gate_values;
    expected_gate_values.reserve(expected_gate.size());
    for (const auto value : expected_gate) {
        expected_gate_values.push_back(value.value);
    }
    std::vector<Q81> expected_xq;
    const bool q8_oracle_ready = twin::quantize_q8_1_reference(
        expected_gate_values, kInputWidth, 1u, expected_xq, error);
    check(q8_oracle_ready, "MTP stage-14 Q8_1 scalar oracle", error);
    if (!q8_oracle_ready) return false;
    const auto actual_xq =
        std::span<const Q81>(typed<Q81>(fixture.xq)).first(expected_xq.size());
    const bool q8_exact =
        std::memcmp(actual_xq.data(), expected_xq.data(), kQ81Bytes) == 0;
    check(q8_exact, "MTP stage-14 Q8_1 row matches the scalar oracle");

    std::vector<float> expected_bo;
    const auto weight_blocks =
        std::span<const block_q8_0>(typed<block_q8_0>(fixture.weights))
            .first(kWeightBlocks);
    const bool mv_oracle_ready = twin::q8_0_matvec_reference(
        weight_blocks, expected_xq, kInputWidth, kOutputWidth, 1u, 1u,
        expected_bo, error);
    check(mv_oracle_ready, "MTP stage-14 O-projection scalar oracle", error);
    if (!mv_oracle_ready) return false;
    const auto actual_bo =
        std::span<const float>(typed<float>(fixture.bo)).first(kOutputWidth);
    const bool output_exact =
        std::memcmp(actual_bo.data(), expected_bo.data(), kOutputBytes) == 0;
    check(output_exact, "MTP stage-14 bo matches the scalar Q8_0 oracle");
    const bool finite =
        std::all_of(actual_gate.begin(), actual_gate.end(),
                    [](float value) { return std::isfinite(value); }) &&
        std::all_of(actual_bo.begin(), actual_bo.end(),
                    [](float value) { return std::isfinite(value); });
    check(finite, "MTP stage-14 active outputs are finite");

    const std::array<twin::ByteRange, 1> gate_written{{
        {0u, kAttentionBytes},
    }};
    const std::array<twin::ByteRange, 1> xq_written{{
        {0u, kQ81Bytes},
    }};
    const std::array<twin::ByteRange, 1> bo_written{{
        {0u, kOutputBytes},
    }};
    const bool guards_ok =
        fixture.inputs_unchanged() &&
        !fixture.attn32.first_unwritten_change(gate_written) &&
        !fixture.xq.first_unwritten_change(xq_written) &&
        !fixture.bo.first_unwritten_change(bo_written) &&
        !fixture.sink.first_unwritten_change({});
    check(guards_ok,
          "MTP stage-14 preserves inputs, output guards, Q8_1 padding and sink");
    return !gate_mismatch && q8_exact && output_exact && finite && guards_ok;
}

bool reject_case(Context& context,
                 const MtpAttentionOutputProjectionBuffers& buffers,
                 const std::string& label,
                 const std::string& expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_attention_output_projection(context, *command, buffers,
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

    fixture.set_rows(0u);
    buffers = fixture.mtp_buffers(context);
    passed &= reject_case(context, buffers, "MTP stage 14 rejects T=0",
                          "T must be in 1..8");
    fixture.set_rows(2u);
    buffers = fixture.mtp_buffers(context);
    passed &= reject_case(context, buffers, "MTP stage 14 rejects T=2",
                          "requires T=1");
    fixture.set_rows(1u);

    buffers = fixture.mtp_buffers(context);
    buffers.attn = buffers.attn.slice(0u, kAttentionBytes - sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects short attention input",
                          "attn buffer is undersized");
    buffers = fixture.mtp_buffers(context);
    buffers.q_full =
        buffers.q_full.slice(0u, kQFullBytes - sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects short q_full input",
                          "q_full buffer is undersized");
    buffers = fixture.mtp_buffers(context);
    buffers.attn32 =
        buffers.attn32.slice(0u, kAttentionBytes - sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects short gated-attention output",
                          "attn32 buffer is undersized");
    buffers = fixture.mtp_buffers(context);
    buffers.xq = buffers.xq.slice(0u, kQ81Bytes - 1u);
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects short Q8_1 output",
                          "stage-14 xq buffer is undersized");
    buffers = fixture.mtp_buffers(context);
    buffers.o_weights = context.buffers().read_only(
        fixture.weights.view().slice(0u, kWeightBytes - 1u));
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects short o_proj weights",
                          "o_proj weights buffer is undersized");
    buffers = fixture.mtp_buffers(context);
    buffers.bo = buffers.bo.slice(0u, kOutputBytes - sizeof(float));
    passed &= reject_case(context, buffers, "MTP stage 14 rejects short bo",
                          "bo buffer is undersized");
    buffers = fixture.mtp_buffers(context);
    buffers.sink = buffers.sink.slice(0u, sizeof(float) - 1u);
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects short sink",
                          "sink buffer is undersized");

    buffers = fixture.mtp_buffers(context);
    buffers.attn32 = buffers.attn;
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects attn32/attn alias",
                          "attn32 overlaps input attn");
    buffers = fixture.mtp_buffers(context);
    buffers.attn32 = buffers.q_full;
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects attn32/q_full alias",
                          "attn32 overlaps input q_full");
    buffers = fixture.mtp_buffers(context);
    buffers.xq = buffers.attn32.slice(0u, kQ81Bytes);
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects gated-output/Q8_1 alias",
                          "outputs attn32 and stage-14 xq overlap");
    buffers = fixture.mtp_buffers(context);
    buffers.bo = fixture.weights.view().slice(0u, kOutputBytes);
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects bo/o_proj alias",
                          "bo overlaps input o_proj weights");
    buffers = fixture.mtp_buffers(context);
    buffers.sink = buffers.bo.slice(0u, sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects bo/sink alias",
                          "outputs bo and sink overlap");

    twin::GuardedBuffer params_alias;
    std::string error;
    const bool params_alias_allocated = params_alias.allocate(
        context, "MTP stage-14 params alias", kQFullBytes, error);
    check(params_alias_allocated, "MTP stage-14 params alias allocation",
          error);
    if (!params_alias_allocated) return false;
    MtpParams params{};
    params.T = 1u;
    std::memcpy(params_alias.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers(context);
    buffers.params = context.buffers().read_only(
        params_alias.view().slice(0u, sizeof(MtpParams)));
    buffers.attn32 = params_alias.view();
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects params/output alias",
                          "attn32 overlaps input params");

    std::memcpy(fixture.weights.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers(context);
    buffers.params = context.buffers().read_only(
        fixture.weights.view().slice(0u, sizeof(MtpParams)));
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects params/o_proj alias",
                          "parameter block overlaps input o_proj weights");

    buffers = fixture.mtp_buffers(context);
    buffers.params = context.buffers().read_only(
        params_alias.view().slice(0u, sizeof(MtpParams)));
    buffers.attn = params_alias.view();
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects params/attention alias",
                          "parameter block overlaps input attn");

    buffers = fixture.mtp_buffers(context);
    buffers.params = context.buffers().read_only(
        params_alias.view().slice(0u, sizeof(MtpParams)));
    buffers.q_full = params_alias.view();
    passed &= reject_case(context, buffers,
                          "MTP stage 14 rejects params/q_full alias",
                          "parameter block overlaps input q_full");

    twin::GuardedBuffer foreign_weights;
    const bool foreign_allocated = foreign_weights.allocate(
        foreign_context, "foreign MTP stage-14 O weights", kWeightBytes,
        error);
    check(foreign_allocated, "foreign MTP stage-14 weights allocation",
          error);
    if (foreign_allocated) {
        buffers = fixture.mtp_buffers(context);
        buffers.o_weights =
            foreign_context.buffers().read_only(foreign_weights.view());
        passed &= reject_case(
            context, buffers, "MTP stage 14 rejects foreign o_proj weights",
            "o_proj weights buffer is not owned");
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
        context.make_command_buffer("MTP stage-14 group boundary", error);
    if (!command) {
        check(false, "MTP stage-14 group-boundary command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers(context);
    const bool wrapper_checked = Encoder::encode(
        context, *command, "MTP stage-14 group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected =
                !encode_mtp_attention_output_projection(
                    context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP stage 14 accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged = fixture.all_buffers_unchanged();
    check(wrapper_checked, "MTP stage 14 rejects an open concurrent group",
          error);
    check(unchanged,
          "MTP stage-14 open-group rejection preserves every buffer");
    command->discard();
    return wrapper_checked && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.reset();
    fixture.capture();
    const auto good = fixture.mtp_buffers(context);
    auto bad = good;
    bad.xq = bad.attn32.slice(0u, kQ81Bytes);

    std::string error;
    auto command = context.make_command_buffer("MTP stage-14 rollback", error);
    if (!command) {
        check(false, "MTP stage-14 rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stage-14 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_attention_output_projection(
                    context, encoder, good, encode_error)) {
                return false;
            }
            return encode_mtp_attention_output_projection(
                context, encoder, bad, encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged = fixture.all_buffers_unchanged();
    check(rejected && discarded,
          "MTP stage-14 late failure discards the caller-owned segment",
          error);
    check(unchanged, "MTP stage-14 rollback preserves every buffer");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_output_projection_test <strata.metallib>\n");
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
    const bool allocated =
        fixture.allocate(*context, "MTP attention output projection");
    check(allocated, "allocate MTP stage-14 fixture");
    if (!allocated) return 1;

    bool passed = execute_projection(*context, fixture, false);
    passed &= execute_projection(*context, fixture, true);
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_concurrent_group_boundary(*context, fixture);
    passed &= test_late_rollback(*context, fixture);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr,
                     "metal_mtp_output_projection_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_output_projection_test: MTP stage 14 OK");
    return 0;
}
