// Metal twin for MTP drafter stage 12 (docs/mac/04 §5.11).
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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kIdCapacity = 64u;
constexpr std::uint32_t kExpandedIdCapacity = 4'097u;
constexpr std::uint32_t kDefaultWindow = 64u;
constexpr std::size_t kIdBytes =
    static_cast<std::size_t>(kIdCapacity) * sizeof(std::int32_t);
constexpr std::int32_t kSentinel = 0x01234567;

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
    twin::GuardedBuffer mtp_ids;

    bool allocate(Context& context, const std::string& label,
                  std::uint32_t capacity = kIdCapacity) {
        std::string error;
        if (!params.allocate(context, label + " params", 256u, error)) {
            check(false, label + " params allocation", error);
            return false;
        }
        if (!mtp_ids.allocate(
                context, label + " mtp_ids",
                static_cast<std::size_t>(capacity) * sizeof(std::int32_t),
                error)) {
            check(false, label + " mtp_ids allocation", error);
            return false;
        }
        initialize(3, kDefaultWindow);
        return true;
    }

    void initialize(std::int32_t n_kv, std::uint32_t window) {
        MtpParams block{};
        block.T = 1u;
        block.step[0].n_kv = n_kv;
        block.step[0].width =
            n_kv > 0
                ? static_cast<std::int32_t>(
                      window == 0u
                          ? static_cast<std::uint32_t>(n_kv)
                          : std::min<std::uint32_t>(
                                static_cast<std::uint32_t>(n_kv), window))
                : 0;
        std::memcpy(params.payload().data(), &block, sizeof(block));
        std::fill(typed<std::int32_t>(mtp_ids).begin(),
                  typed<std::int32_t>(mtp_ids).end(), kSentinel);
    }

    MtpWindowIdsBuffers buffers(Context& context, std::uint32_t window) {
        return {context.buffers().read_only(params.view()), mtp_ids.view(),
                window};
    }

    void capture() {
        params.capture_before_dispatch();
        mtp_ids.capture_before_dispatch();
    }

    bool inputs_unchanged() const {
        return !params.first_unwritten_change({});
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

bool execute_case(Context& context, Fixture& fixture, std::int32_t n_kv,
                  std::uint32_t window, bool caller_owned) {
    fixture.initialize(n_kv, window);
    fixture.capture();
    const auto buffers = fixture.buffers(context, window);
    const std::uint32_t width =
        window == 0u
            ? static_cast<std::uint32_t>(n_kv)
            : std::min(static_cast<std::uint32_t>(n_kv), window);
    const std::uint32_t start = static_cast<std::uint32_t>(n_kv) - width;

    std::string error;
    auto command =
        context.make_command_buffer("MTP stage-12 window IDs", error);
    check(command != nullptr, "MTP stage-12 command buffer", error);
    if (!command) return false;

    bool encoded = false;
    if (caller_owned) {
        encoded = Encoder::encode(
            context, *command, "MTP stage-12 caller-owned",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_window_ids(context, encoder, buffers,
                                             encode_error);
            },
            error);
    } else {
        encoded = encode_mtp_window_ids(context, *command, buffers, error);
    }
    check(encoded, "MTP stage-12 encode", error);
    if (!encoded) return false;
    const bool completed =
        submit_and_wait(command, "MTP stage-12 window IDs", error);
    check(completed, "MTP stage-12 GPU completion", error);
    if (!completed) return false;

    const auto actual = std::span<const std::int32_t>(
        typed<std::int32_t>(fixture.mtp_ids));
    bool ids_match = true;
    for (std::uint32_t index = 0u; index < width; ++index) {
        ids_match &= actual[index] ==
                     static_cast<std::int32_t>(start + index);
    }
    const bool tail_unchanged =
        std::all_of(actual.begin() + width, actual.end(),
                    [](std::int32_t value) { return value == kSentinel; });
    check(ids_match,
          "MTP stage-12 active IDs are the absolute KV window positions");
    check(tail_unchanged,
          "MTP stage-12 leaves the inactive ID suffix untouched");
    const std::array<twin::ByteRange, 1> written{{
        {0u, static_cast<std::size_t>(width) * sizeof(std::int32_t)},
    }};
    const bool guards_ok =
        fixture.inputs_unchanged() &&
        !fixture.mtp_ids.first_unwritten_change(written);
    check(guards_ok, "MTP stage-12 preserves params and allocation guards");
    return ids_match && tail_unchanged && guards_ok;
}

bool test_finite_to_unwindowed_reset(Context& context, Fixture& fixture) {
    if (!execute_case(context, fixture, 9, 4u, false)) {
        return false;
    }

    MtpParams block{};
    block.T = 1u;
    block.step[0].n_kv = 9;
    block.step[0].width = 9;
    std::memcpy(fixture.params.payload().data(), &block, sizeof(block));
    fixture.capture();
    const auto buffers = fixture.buffers(context, 0u);
    std::string error;
    auto command = context.make_command_buffer(
        "MTP stage-12 finite-to-unwindowed reset", error);
    const bool encoded = command &&
        encode_mtp_window_ids(context, *command, buffers, error);
    check(encoded, "MTP stage 12 accepts zero as the unwindowed pass", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(
        command, "MTP stage-12 finite-to-unwindowed reset", error);
    check(completed,
          "MTP stage-12 finite-to-unwindowed reset GPU completion", error);
    if (!completed) return false;

    const auto actual = std::span<const std::int32_t>(
        typed<std::int32_t>(fixture.mtp_ids));
    bool identity = true;
    for (std::uint32_t index = 0u; index < 9u; ++index) {
        identity &= actual[index] == static_cast<std::int32_t>(index);
    }
    const bool tail_unchanged =
        std::all_of(actual.begin() + 9u, actual.end(),
                    [](std::int32_t value) { return value == kSentinel; });
    const std::array<twin::ByteRange, 1> written{{
        {0u, 9u * sizeof(std::int32_t)},
    }};
    const bool guards_ok =
        fixture.inputs_unchanged() &&
        !fixture.mtp_ids.first_unwritten_change(written);
    check(identity,
          "MTP stage 12 refreshes stale finite-window IDs to identity");
    check(tail_unchanged,
          "MTP stage 12 leaves the unwindowed inactive suffix untouched");
    check(guards_ok,
          "MTP stage-12 reset preserves params and allocation guards");
    return identity && tail_unchanged && guards_ok;
}

bool reject_case(Context& context, const MtpWindowIdsBuffers& buffers,
                 const std::string& label,
                 std::string_view expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_window_ids(context, *command, buffers, error);
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
    fixture.initialize(3, kDefaultWindow);
    auto buffers = fixture.buffers(context, kDefaultWindow);
    MtpParams params{};
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    passed &= reject_case(context, fixture.buffers(context, kDefaultWindow),
                          "MTP stage 12 rejects T=0", "T must be in 1..8");

    params.T = 2u;
    params.step[0].n_kv = 3;
    params.step[0].width = 3;
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    passed &= reject_case(context, fixture.buffers(context, kDefaultWindow),
                          "MTP stage 12 rejects T=2",
                          "requires T=1");

    fixture.initialize(3, 0u);
    MtpParams zero_window_width_mismatch{};
    zero_window_width_mismatch.T = 1u;
    zero_window_width_mismatch.step[0].n_kv = 3;
    zero_window_width_mismatch.step[0].width = 0;
    std::memcpy(fixture.params.payload().data(),
                &zero_window_width_mismatch,
                sizeof(zero_window_width_mismatch));
    buffers = fixture.buffers(context, 0u);
    passed &= reject_case(context, buffers,
                          "MTP stage 12 rejects a zero-window width mismatch",
                          "StepRow width does not match");
    buffers = fixture.buffers(context, kDefaultWindow + 1u);
    passed &= reject_case(context, buffers, "MTP stage 12 rejects a window "
                                             "larger than ID capacity",
                          "window exceeds the mtp_ids capacity");
    constexpr std::uint32_t kTooLargeWindow =
        static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max()) + 1u;
    passed &= reject_case(
        context, fixture.buffers(context, kTooLargeWindow),
        "MTP stage 12 rejects a window outside StepRow range",
        "window exceeds the StepRow integer range");

    fixture.initialize(0, kDefaultWindow);
    passed &= reject_case(context, fixture.buffers(context, kDefaultWindow),
                          "MTP stage 12 rejects n_kv=0",
                          "n_kv must be positive");
    fixture.initialize(-1, kDefaultWindow);
    passed &= reject_case(context, fixture.buffers(context, kDefaultWindow),
                          "MTP stage 12 rejects negative n_kv",
                          "n_kv must be positive");

    fixture.initialize(3, kDefaultWindow);
    params = {};
    params.T = 1u;
    params.step[0].n_kv = 3;
    params.step[0].width = 2;
    std::memcpy(fixture.params.payload().data(), &params, sizeof(params));
    passed &= reject_case(
        context, fixture.buffers(context, kDefaultWindow),
        "MTP stage 12 rejects mismatched StepRow width",
        "StepRow width does not match");

    fixture.initialize(3, kDefaultWindow);
    buffers = fixture.buffers(context, kDefaultWindow);
    buffers.mtp_ids = fixture.mtp_ids.view().slice(0u, kIdBytes - 1u);
    passed &= reject_case(
        context, buffers, "MTP stage 12 rejects partial ID elements",
        "whole int32 elements");
    buffers = fixture.buffers(context, kDefaultWindow);
    buffers.mtp_ids = fixture.mtp_ids.view().slice(1u, kIdBytes - 1u);
    passed &= reject_case(context, buffers,
                          "MTP stage 12 rejects misaligned IDs",
                          "undersized or misaligned");

    twin::GuardedBuffer foreign_params;
    twin::GuardedBuffer foreign_ids;
    std::string error;
    const bool foreign_params_allocated = foreign_params.allocate(
        foreign_context, "foreign MTP window params", 256u, error);
    check(foreign_params_allocated,
          "foreign MTP window params allocation", error);
    if (foreign_params_allocated) {
        MtpParams foreign_block{};
        foreign_block.T = 1u;
        foreign_block.step[0].n_kv = 3;
        foreign_block.step[0].width = 3;
        std::memcpy(foreign_params.payload().data(), &foreign_block,
                    sizeof(foreign_block));
        auto foreign_buffers = fixture.buffers(context, kDefaultWindow);
        foreign_buffers.params =
            foreign_context.buffers().read_only(foreign_params.view());
        passed &= reject_case(
            context, foreign_buffers, "MTP stage 12 rejects foreign params",
            "not owned by this Context");
    } else {
        passed = false;
    }

    const bool foreign_ids_allocated = foreign_ids.allocate(
        foreign_context, "foreign MTP window IDs", kIdBytes, error);
    check(foreign_ids_allocated, "foreign MTP window IDs allocation", error);
    if (foreign_ids_allocated) {
        auto foreign_buffers = fixture.buffers(context, kDefaultWindow);
        foreign_buffers.mtp_ids = foreign_ids.view();
        passed &= reject_case(
            context, foreign_buffers, "MTP stage 12 rejects foreign IDs",
            "not owned by this Context");
    } else {
        passed = false;
    }

    fixture.initialize(3, kDefaultWindow);
    MtpParams aliased{};
    aliased.T = 1u;
    aliased.step[0].n_kv = 3;
    aliased.step[0].width = 3;
    std::memcpy(fixture.mtp_ids.payload().data(), &aliased, sizeof(aliased));
    buffers = fixture.buffers(context, kDefaultWindow);
    buffers.params = context.buffers().read_only(fixture.mtp_ids.view());
    passed &= reject_case(
        context, buffers, "MTP stage 12 rejects active IDs/params alias",
        "active ID output overlaps the parameter block");
    return passed;
}

bool test_concurrent_group_boundary(Context& context, Fixture& fixture) {
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode == nullptr ||
        std::string_view(dispatch_mode) != "concurrent") {
        return true;
    }
    fixture.initialize(3, kDefaultWindow);
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP stage-12 group boundary", error);
    if (!command) {
        check(false, "MTP stage-12 group-boundary command buffer", error);
        return false;
    }
    const auto buffers = fixture.buffers(context, kDefaultWindow);
    const bool encoded = Encoder::encode(
        context, *command, "MTP stage-12 group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected = !encode_mtp_window_ids(
                context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP stage 12 accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged =
        fixture.inputs_unchanged() &&
        !fixture.mtp_ids.first_unwritten_change({});
    check(encoded, "MTP stage 12 rejects an open concurrent group", error);
    check(unchanged, "MTP stage-12 open-group rejection preserves buffers");
    command->discard();
    return encoded && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.initialize(3, kDefaultWindow);
    fixture.capture();
    const auto good = fixture.buffers(context, kDefaultWindow);
    auto bad = good;
    bad.window = kIdCapacity + 1u;

    std::string error;
    auto command = context.make_command_buffer("MTP stage-12 rollback", error);
    if (!command) {
        check(false, "MTP stage-12 rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stage-12 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_window_ids(context, encoder, good, encode_error)) {
                return false;
            }
            return encode_mtp_window_ids(context, encoder, bad, encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        fixture.inputs_unchanged() &&
        !fixture.mtp_ids.first_unwritten_change({});
    check(rejected && discarded,
          "MTP stage-12 late failure discards the caller-owned segment",
          error);
    check(unchanged, "MTP stage-12 rollback preserves every buffer");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_window_ids_test <strata.metallib>\n");
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
        fixture.allocate(*context, "MTP window IDs fixture");
    check(allocated, "allocate MTP stage-12 fixture");
    if (!allocated) return 1;

    Fixture expanded_fixture;
    const bool expanded_allocated = expanded_fixture.allocate(
        *context, "MTP expanded window IDs fixture", kExpandedIdCapacity);
    check(expanded_allocated,
          "allocate expanded MTP stage-12 fixture");
    if (!expanded_allocated) return 1;

    bool passed = execute_case(*context, fixture, 3, 64u, false);
    passed &= execute_case(*context, fixture, 3, 0u, false);
    passed &= execute_case(*context, fixture, 64, 64u, false);
    passed &= execute_case(*context, fixture, 75, 64u, false);
    passed &= execute_case(*context, fixture, 262'144, 64u, false);
    passed &= execute_case(*context, expanded_fixture, 5'000,
                           kExpandedIdCapacity, false);
    passed &= execute_case(
        *context, fixture, std::numeric_limits<std::int32_t>::max(), 1u,
        false);
    passed &= execute_case(*context, fixture, 75, 64u, true);
    passed &= test_finite_to_unwindowed_reset(*context, fixture);
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_concurrent_group_boundary(*context, fixture);
    passed &= test_late_rollback(*context, fixture);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_window_ids_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_window_ids_test: MTP stage 12 OK");
    return 0;
}
