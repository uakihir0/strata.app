// Metal twin for MTP drafter stage 11 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "oracles/qsa.hpp"
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

constexpr std::uint32_t kMtpRows = 1u;
constexpr std::uint32_t kRopeCells = 16u;
constexpr std::uint32_t kRopePairs = 32u;
constexpr std::uint32_t kFullHeadWidth = HEAD_DIM * 2u;
constexpr std::size_t kQFullElements =
    static_cast<std::size_t>(N_HEAD_Q) * kFullHeadWidth;
constexpr std::size_t kQCurElements =
    static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM;
constexpr std::size_t kQFullBytes = kQFullElements * sizeof(float);
constexpr std::size_t kQCurBytes = kQCurElements * sizeof(float);
constexpr std::size_t kQNormBytes = HEAD_DIM * sizeof(float);
constexpr std::size_t kRopeBytes =
    static_cast<std::size_t>(kRopeCells) * kRopePairs * 2u * sizeof(float);
constexpr std::uint32_t kPosition = 7u;
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
    twin::GuardedBuffer q_full;
    twin::GuardedBuffer q_norm;
    twin::GuardedBuffer rope;
    twin::GuardedBuffer qcur;

    bool allocate(Context& context, const std::string& label) {
        std::string error;
        const std::array<std::pair<twin::GuardedBuffer*, std::pair<const char*,
                                                                  std::size_t>>,
                         5>
            buffers{{
                {&params, {"params", 256u}},
                {&q_full, {"q_full", kQFullBytes}},
                {&q_norm, {"q_norm", kQNormBytes}},
                {&rope, {"RoPE", kRopeBytes}},
                {&qcur, {"qcur", kQCurBytes}},
            }};
        for (const auto& [buffer, metadata] : buffers) {
            if (!buffer->allocate(context, label + " " + metadata.first,
                                  metadata.second, error)) {
                check(false, label + " " + metadata.first + " allocation",
                      error);
                return false;
            }
        }
        initialize();
        return true;
    }

    void initialize() {
        MtpParams block{};
        block.T = kMtpRows;
        block.step[0].pos = static_cast<std::int32_t>(kPosition);
        std::memcpy(params.payload().data(), &block, sizeof(block));

        auto q_values = typed<float>(q_full);
        for (std::size_t i = 0u; i < q_values.size(); ++i) {
            const auto centered =
                static_cast<std::int32_t>((i * 29u + i / 23u) % 257u) - 128;
            q_values[i] = static_cast<float>(centered) * 0.0078125f;
        }
        auto gamma_values = typed<float>(q_norm);
        for (std::size_t i = 0u; i < gamma_values.size(); ++i) {
            gamma_values[i] =
                0.75f + static_cast<float>((i * 7u) % 31u) * 0.015625f;
        }
        auto rope_values = typed<float>(rope);
        for (std::uint32_t cell = 0u; cell < kRopeCells; ++cell) {
            for (std::uint32_t pair = 0u; pair < kRopePairs; ++pair) {
                const std::size_t index =
                    (static_cast<std::size_t>(cell) * kRopePairs + pair) * 2u;
                const float angle =
                    static_cast<float>((cell + 1u) * (pair + 3u)) * 0.0007f;
                rope_values[index] = std::cos(angle);
                rope_values[index + 1u] = std::sin(angle);
            }
        }
        reset_output();
    }

    void reset_output() {
        std::fill(qcur.payload().begin(), qcur.payload().end(), 0u);
        auto values = typed<float>(qcur);
        std::fill(values.begin(), values.end(), kPoison);
    }

    MtpAttentionQueryNormBuffers mtp_buffers() {
        return {params.view(), q_full.view(), q_norm.view(), rope.view(),
                qcur.view(), false};
    }

    void capture() {
        params.capture_before_dispatch();
        q_full.capture_before_dispatch();
        q_norm.capture_before_dispatch();
        rope.capture_before_dispatch();
        qcur.capture_before_dispatch();
    }

    bool inputs_unchanged() const {
        return !params.first_unwritten_change({}) &&
               !q_full.first_unwritten_change({}) &&
               !q_norm.first_unwritten_change({}) &&
               !rope.first_unwritten_change({});
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

std::vector<twin::tf> expected_output(const Fixture& fixture, bool kv_rot) {
    const auto q_full =
        std::span<const float>(typed<float>(
            const_cast<twin::GuardedBuffer&>(fixture.q_full)));
    const auto q_norm =
        std::span<const float>(typed<float>(
            const_cast<twin::GuardedBuffer&>(fixture.q_norm)));
    const auto rope =
        std::span<const float>(typed<float>(
            const_cast<twin::GuardedBuffer&>(fixture.rope)));
    const std::array<std::int32_t, 1> positions{
        static_cast<std::int32_t>(kPosition)};
    std::vector<twin::tf> normalized;
    oracle::qsa_norm_rope(q_full, q_norm, positions, rope, kMtpRows,
                          QsaNormRopeMode::query, normalized);
    if (!kv_rot) return normalized;

    std::vector<float> normalized_values;
    normalized_values.reserve(normalized.size());
    for (const twin::tf value : normalized) {
        normalized_values.push_back(value.value);
    }
    std::vector<twin::tf> rotated;
    oracle::qsa_fwht256(normalized_values, N_HEAD_Q, rotated);
    return rotated;
}

bool execute_case(Context& context, Fixture& fixture, bool kv_rot,
                  bool caller_owned) {
    fixture.reset_output();
    fixture.capture();
    auto buffers = fixture.mtp_buffers();
    buffers.kv_rot = kv_rot;
    std::string error;
    auto command = context.make_command_buffer(
        "MTP stage-11 query norm/RoPE", error);
    check(command != nullptr, "MTP stage-11 command buffer", error);
    if (!command) return false;

    bool encoded = false;
    if (caller_owned) {
        encoded = Encoder::encode(
            context, *command, "MTP stage-11 caller-owned",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_attention_query_norm(
                    context, encoder, buffers, encode_error);
            },
            error);
    } else {
        encoded = encode_mtp_attention_query_norm(
            context, *command, buffers, error);
    }
    check(encoded, "MTP stage-11 encode", error);
    if (!encoded) return false;
    const bool completed =
        submit_and_wait(command, "MTP stage-11 query norm/RoPE", error);
    check(completed, "MTP stage-11 GPU completion", error);
    if (!completed) return false;

    const auto expected = expected_output(fixture, kv_rot);
    const auto actual =
        std::span<const float>(typed<float>(fixture.qcur)).first(
            kQCurElements);
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 64.0, 0.0, 0u});
    check(!mismatch,
          std::string("MTP stage-11 ") +
              (kv_rot ? "rotated" : "unrotated") +
              " qcur matches QSA scalar oracle");
    const bool finite = std::all_of(
        actual.begin(), actual.end(),
        [](float value) { return std::isfinite(value); });
    check(finite, "MTP stage-11 qcur is finite");
    const std::array<twin::ByteRange, 1> written{{
        {0u, kQCurBytes},
    }};
    const bool guards_ok =
        fixture.inputs_unchanged() &&
        !fixture.qcur.first_unwritten_change(written);
    check(guards_ok, "MTP stage-11 preserves inputs and allocation guards");
    return !mismatch && finite && guards_ok;
}

bool reject_case(Context& context,
                 const MtpAttentionQueryNormBuffers& buffers,
                 const std::string& label,
                 std::string_view expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_attention_query_norm(context, *command, buffers, error);
    const bool matched =
        expected_error.empty() ||
        error.find(expected_error) != std::string::npos;
    check(rejected && matched, label, error);
    command->discard();
    return rejected && matched;
}

bool test_validation(Context& context, Context& foreign_context,
                     Fixture& fixture) {
    bool passed = true;
    fixture.initialize();
    auto buffers = fixture.mtp_buffers();
    MtpParams params{};
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers();
    passed &= reject_case(context, buffers, "MTP stage 11 rejects T=0",
                          "T must be in 1..8");

    params.T = 2u;
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers();
    passed &= reject_case(context, buffers, "MTP stage 11 rejects T=2",
                          "requires T=1");

    params.T = 1u;
    params.step[0].pos = static_cast<std::int32_t>(kRopeCells);
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers();
    passed &= reject_case(
        context, buffers, "MTP stage 11 rejects an out-of-table position",
        "position is outside the RoPE table");

    fixture.initialize();
    buffers = fixture.mtp_buffers();
    buffers.q_full = fixture.q_full.view().slice(0u, kQFullBytes - 4u);
    passed &= reject_case(context, buffers, "MTP stage 11 rejects short q_full",
                          "q_full buffer is undersized");

    buffers = fixture.mtp_buffers();
    buffers.q_norm = fixture.q_norm.view().slice(0u, kQNormBytes - 4u);
    passed &= reject_case(context, buffers, "MTP stage 11 rejects short q_norm",
                          "q_norm buffer is undersized");

    buffers = fixture.mtp_buffers();
    buffers.qcur = fixture.qcur.view().slice(0u, kQCurBytes - 4u);
    passed &= reject_case(context, buffers, "MTP stage 11 rejects short qcur",
                          "qcur buffer is undersized");

    buffers = fixture.mtp_buffers();
    buffers.qcur = fixture.q_full.view().slice(0u, kQCurBytes);
    passed &= reject_case(
        context, buffers, "MTP stage 11 rejects q_full/qcur overlap",
        "query input and compact output must be disjoint");

    MtpParams aliased{};
    aliased.T = 1u;
    aliased.step[0].pos = static_cast<std::int32_t>(kPosition);
    std::memcpy(fixture.q_full.payload().data(), &aliased, sizeof(aliased));
    buffers = fixture.mtp_buffers();
    buffers.params = fixture.q_full.view();
    passed &= reject_case(context, buffers, "MTP stage 11 rejects params/q_full",
                          "parameter block overlaps q_full");

    twin::GuardedBuffer foreign_norm;
    std::string error;
    const bool foreign_allocated = foreign_norm.allocate(
        foreign_context, "foreign MTP q_norm", kQNormBytes, error);
    check(foreign_allocated, "foreign MTP q_norm allocation", error);
    if (foreign_allocated) {
        fixture.initialize();
        buffers = fixture.mtp_buffers();
        buffers.q_norm = foreign_norm.view();
        passed &= reject_case(
            context, buffers, "MTP stage 11 rejects foreign q_norm",
            "q_norm buffer is not owned by this Context");
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
    fixture.initialize();
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP stage-11 group boundary", error);
    if (!command) {
        check(false, "MTP stage-11 group-boundary command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers();
    const bool encoded = Encoder::encode(
        context, *command, "MTP stage-11 group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected = !encode_mtp_attention_query_norm(
                context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error = "MTP stage 11 accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.q_full.first_unwritten_change({}) &&
        !fixture.q_norm.first_unwritten_change({}) &&
        !fixture.rope.first_unwritten_change({}) &&
        !fixture.qcur.first_unwritten_change({});
    check(encoded, "MTP stage 11 rejects an open concurrent group", error);
    check(unchanged, "MTP stage-11 open-group rejection preserves buffers");
    command->discard();
    return encoded && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.initialize();
    fixture.capture();
    auto good = fixture.mtp_buffers();
    good.kv_rot = true;
    auto bad = good;
    twin::GuardedBuffer invalid_params;
    std::string error;
    const bool allocated = invalid_params.allocate(
        context, "invalid MTP query norm params", 256u, error);
    check(allocated, "MTP stage-11 invalid params allocation", error);
    if (!allocated) return false;
    MtpParams invalid{};
    invalid.T = 2u;
    std::memcpy(invalid_params.payload().data(), &invalid, sizeof(invalid));
    bad.params = invalid_params.view();

    auto command = context.make_command_buffer("MTP stage-11 rollback", error);
    if (!command) {
        check(false, "MTP stage-11 rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stage-11 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_attention_query_norm(
                    context, encoder, good, encode_error)) {
                return false;
            }
            return encode_mtp_attention_query_norm(
                context, encoder, bad, encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.q_full.first_unwritten_change({}) &&
        !fixture.q_norm.first_unwritten_change({}) &&
        !fixture.rope.first_unwritten_change({}) &&
        !fixture.qcur.first_unwritten_change({});
    check(rejected && discarded,
          "MTP stage-11 late failure discards the caller-owned segment",
          error);
    check(unchanged, "MTP stage-11 rollback preserves every buffer");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_query_norm_test <strata.metallib>\n");
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
    const bool allocated = fixture.allocate(*context, "MTP query norm");
    check(allocated, "allocate MTP stage-11 fixture");
    if (!allocated) return 1;

    bool passed = execute_case(*context, fixture, false, false);
    passed &= execute_case(*context, fixture, true, false);
    passed &= execute_case(*context, fixture, false, true);
    passed &= execute_case(*context, fixture, true, true);
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_concurrent_group_boundary(*context, fixture);
    passed &= test_late_rollback(*context, fixture);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_query_norm_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_query_norm_test: MTP stage 11 OK");
    return 0;
}
