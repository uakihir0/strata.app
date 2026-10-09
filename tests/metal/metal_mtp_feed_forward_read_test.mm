// Metal twin for MTP drafter stage 15 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "twin.hpp"

#import <Metal/Metal.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#include <algorithm>
#include <array>
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

constexpr std::size_t kBufferCount = 12u;

enum BufferIndex : std::size_t {
    kResidual,
    kBo,
    kInj,
    kNorm,
    kNormalized,
    kRs,
    kHcDown,
    kHcInject,
    kLo,
    kInjectOut,
    kHcUp,
    kMixed,
};

constexpr std::array<const char*, kBufferCount> kNames{
    "residual", "bo", "inj", "norm", "normalized", "rs",
    "hc_down", "hc_inject", "lo", "inject_out", "hc_up", "mixed"};
constexpr std::array<BufferIndex, 6u> kOutputBuffers{
    kResidual, kNormalized, kRs, kLo, kInjectOut, kMixed};

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

std::size_t bytes_for(BufferIndex index) {
    switch (index) {
        case kResidual:
        case kNormalized:
            return static_cast<std::size_t>(HC_DIM) * sizeof(float);
        case kBo:
        case kMixed:
            return static_cast<std::size_t>(N_EMBD) * sizeof(float);
        case kInj:
        case kRs:
        case kInjectOut:
            return static_cast<std::size_t>(HC) * sizeof(float);
        case kNorm:
            return static_cast<std::size_t>(HC_DIM) * sizeof(float);
        case kHcDown:
        case kHcUp:
            return static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
                   sizeof(std::uint16_t);
        case kHcInject:
            return static_cast<std::size_t>(HC) * HC_DIM *
                   sizeof(std::uint16_t);
        case kLo:
            return static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float);
    }
    return 0u;
}

struct Fixture {
    twin::GuardedBuffer params;
    std::array<twin::GuardedBuffer, kBufferCount> storage;
    std::vector<std::uint8_t> initial_residual;

    BufferRef view(BufferIndex index) { return storage[index].view(); }

    bool allocate(Context& context, const std::string& label) {
        std::string error;
        if (!params.allocate(context, label + " params", 256u, error)) {
            check(false, label + " params allocation", error);
            return false;
        }
        for (std::size_t index = 0u; index < storage.size(); ++index) {
            const auto which = static_cast<BufferIndex>(index);
            if (!storage[index].allocate(context, label + " " + kNames[index],
                                         bytes_for(which), error)) {
                check(false, label + " " + kNames[index] + " allocation",
                      error);
                return false;
            }
        }
        initialize();
        return true;
    }

    void set_rows(std::uint32_t rows) {
        std::fill(params.payload().begin(), params.payload().end(), 0u);
        MtpParams block{};
        block.T = rows;
        std::memcpy(params.payload().data(), &block, sizeof(block));
    }

    void initialize() {
        set_rows(1u);
        for (std::size_t index = 0u; index < storage.size(); ++index) {
            const auto which = static_cast<BufferIndex>(index);
            auto payload = storage[index].payload();
            if (which == kHcDown || which == kHcInject || which == kHcUp) {
                auto* values = reinterpret_cast<std::uint16_t*>(payload.data());
                const std::size_t count =
                    payload.size() / sizeof(std::uint16_t);
                const float scale = which == kHcInject ? 0.0002f : 0.0001f;
                for (std::size_t i = 0u; i < count; ++i) {
                    const auto centered =
                        static_cast<std::int32_t>((i * 17u + index) % 9u) - 4;
                    values[i] = strata::kernels::bf16_from_f32(
                        scale * static_cast<float>(centered));
                }
                continue;
            }
            if (which == kNormalized || which == kRs || which == kLo ||
                which == kInjectOut || which == kMixed) {
                std::memset(payload.data(), 0xa5, payload.size());
                continue;
            }

            auto* values = reinterpret_cast<float*>(payload.data());
            const std::size_t count = payload.size() / sizeof(float);
            for (std::size_t i = 0u; i < count; ++i) {
                const auto centered =
                    static_cast<std::int32_t>((i * 13u + index * 7u) % 101u) -
                    50;
                values[i] = static_cast<float>(centered) * 0.001f;
            }
            if (which == kNorm) {
                for (std::size_t i = 0u; i < count; ++i) {
                    values[i] = 0.75f + static_cast<float>(i % 17u) * 0.01f;
                }
            } else if (which == kBo) {
                for (std::size_t i = 0u; i < count; ++i) {
                    values[i] =
                        static_cast<float>(static_cast<std::int32_t>(i % 37u) -
                                           18) *
                        0.02f;
                }
            }
        }
        const auto residual_payload = storage[kResidual].payload();
        initial_residual.assign(residual_payload.begin(),
                                residual_payload.end());
        reset_outputs();
    }

    void reset_outputs() {
        auto residual_payload = storage[kResidual].payload();
        std::copy(initial_residual.begin(), initial_residual.end(),
                  residual_payload.begin());
        for (const auto index :
             {kNormalized, kRs, kLo, kInjectOut, kMixed}) {
            std::memset(storage[index].payload().data(), 0xa5,
                        bytes_for(index));
        }
    }

    MtpFeedForwardReadBuffers mtp_buffers(Context& context) {
        return {context.buffers().read_only(params.view()),
                view(kResidual),
                view(kBo),
                view(kInj),
                view(kNorm),
                view(kNormalized),
                view(kRs),
                view(kHcDown),
                view(kHcInject),
                view(kLo),
                view(kInjectOut),
                view(kHcUp),
                view(kMixed)};
    }

    GRReadBuffers direct_gr_buffers() {
        return {view(kResidual), view(kBo), view(kInj), view(kNorm),
                view(kNormalized), view(kRs), view(kHcDown), view(kHcInject),
                view(kLo), view(kInjectOut), view(kHcUp), view(kMixed)};
    }

    void capture() {
        params.capture_before_dispatch();
        for (auto& buffer : storage) buffer.capture_before_dispatch();
    }

    bool read_only_inputs_unchanged() const {
        return !params.first_unwritten_change({}) &&
               !storage[kBo].first_unwritten_change({}) &&
               !storage[kInj].first_unwritten_change({}) &&
               !storage[kNorm].first_unwritten_change({}) &&
               !storage[kHcDown].first_unwritten_change({}) &&
               !storage[kHcInject].first_unwritten_change({}) &&
               !storage[kHcUp].first_unwritten_change({});
    }

    bool all_buffers_unchanged() const {
        if (params.first_unwritten_change({})) return false;
        for (const auto& buffer : storage) {
            if (buffer.first_unwritten_change({})) return false;
        }
        return true;
    }

    std::array<std::vector<std::uint8_t>, kOutputBuffers.size()>
    outputs() const {
        std::array<std::vector<std::uint8_t>, kOutputBuffers.size()> result;
        for (std::size_t i = 0u; i < kOutputBuffers.size(); ++i) {
            const auto payload = storage[kOutputBuffers[i]].payload();
            result[i].assign(payload.begin(), payload.end());
        }
        return result;
    }

    bool output_guards_unchanged() const {
        const std::array<std::size_t, kOutputBuffers.size()> output_bytes{
            bytes_for(kResidual), bytes_for(kNormalized), bytes_for(kRs),
            bytes_for(kLo), bytes_for(kInjectOut), bytes_for(kMixed)};
        for (std::size_t i = 0u; i < kOutputBuffers.size(); ++i) {
            const std::array<twin::ByteRange, 1> written{{
                {0u, output_bytes[i]},
            }};
            if (storage[kOutputBuffers[i]].first_unwritten_change(written)) {
                return false;
            }
        }
        return true;
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

bool run_pipeline(Context& context, Fixture& fixture, bool use_mtp,
                  bool caller_owned,
                  std::array<std::vector<std::uint8_t>,
                             kOutputBuffers.size()>& outputs) {
    fixture.set_rows(1u);
    fixture.reset_outputs();
    fixture.capture();
    std::string error;
    auto command = context.make_command_buffer(
        use_mtp ? "MTP stage-15 feed-forward read" : "direct GR apply read",
        error);
    check(command != nullptr, "MTP stage-15 command buffer", error);
    if (!command) return false;

    bool encoded = false;
    if (use_mtp) {
        const auto buffers = fixture.mtp_buffers(context);
        if (caller_owned) {
            encoded = Encoder::encode(
                context, *command, "MTP stage-15 caller-owned",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_mtp_feed_forward_read(
                        context, encoder, buffers, encode_error);
                },
                error);
        } else {
            encoded = encode_mtp_feed_forward_read(context, *command, buffers,
                                                   error);
        }
    } else {
        encoded = encode_gr_read(context, *command, 1u, true,
                                 fixture.direct_gr_buffers(), error);
    }
    check(encoded,
          use_mtp ? "MTP stage-15 ordered GR read encoding"
                  : "direct GR apply=true reference encoding",
          error);
    if (!encoded) return false;
    const bool completed =
        submit_and_wait(command, "MTP stage-15 GR read", error);
    check(completed, "MTP stage-15 GPU completion", error);
    if (!completed) return false;

    const bool finite = std::all_of(
        typed<float>(fixture.storage[kResidual]).begin(),
        typed<float>(fixture.storage[kResidual]).end(),
        [](float value) { return std::isfinite(value); });
    check(finite, "MTP stage-15 residual output is finite");
    const bool inputs_unchanged = fixture.read_only_inputs_unchanged();
    check(inputs_unchanged,
          "MTP stage-15 preserves params, bo, inj, norm and GR weights");
    const bool guards_ok = fixture.output_guards_unchanged();
    check(guards_ok,
          "MTP stage-15 writes only the active GR output extents");
    outputs = fixture.outputs();
    return finite && inputs_unchanged && guards_ok;
}

bool test_pipeline_parity(Context& context, Fixture& fixture) {
    std::array<std::vector<std::uint8_t>, kOutputBuffers.size()> standalone;
    std::array<std::vector<std::uint8_t>, kOutputBuffers.size()> caller;
    std::array<std::vector<std::uint8_t>, kOutputBuffers.size()> direct;
    bool passed = run_pipeline(context, fixture, true, false, standalone);
    passed &= run_pipeline(context, fixture, true, true, caller);
    passed &= run_pipeline(context, fixture, false, false, direct);
    for (std::size_t output = 0u; output < kOutputBuffers.size(); ++output) {
        const std::string label =
            std::string("MTP stage 15 ") + kNames[kOutputBuffers[output]];
        const bool caller_matches =
            standalone[output] == caller[output];
        const bool direct_matches =
            standalone[output] == direct[output];
        check(caller_matches, label + " standalone/caller-owned parity");
        check(direct_matches, label + " matches direct GR apply=true chain");
        passed &= caller_matches && direct_matches;
    }
    return passed;
}

bool reject_case(Context& context, const MtpFeedForwardReadBuffers& buffers,
                 const std::string& label,
                 const std::string& expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_feed_forward_read(context, *command, buffers, error);
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
    fixture.set_rows(0u);
    auto buffers = fixture.mtp_buffers(context);
    passed &= reject_case(context, buffers, "MTP stage 15 rejects T=0",
                          "T must be in 1..8");

    fixture.set_rows(2u);
    buffers = fixture.mtp_buffers(context);
    passed &= reject_case(context, buffers, "MTP stage 15 rejects T=2",
                          "requires T=1");

    fixture.set_rows(1u);
    buffers = fixture.mtp_buffers(context);
    buffers.bo = buffers.bo.slice(0u, bytes_for(kBo) - sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 15 rejects short bo");

    twin::GuardedBuffer params_alias;
    std::string error;
    const bool alias_allocated = params_alias.allocate(
        context, "MTP stage-15 params alias", bytes_for(kResidual), error);
    check(alias_allocated, "MTP stage-15 params alias allocation", error);
    if (!alias_allocated) return false;
    MtpParams params{};
    params.T = 1u;
    std::memcpy(params_alias.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers(context);
    buffers.params = context.buffers().read_only(
        params_alias.view().slice(0u, sizeof(MtpParams)));
    buffers.residual = params_alias.view();
    passed &= reject_case(context, buffers,
                          "MTP stage 15 rejects params/residual alias",
                          "parameter block overlaps active view residual");

    buffers = fixture.mtp_buffers(context);
    buffers.mixed = buffers.bo;
    passed &= reject_case(context, buffers,
                          "MTP stage 15 rejects output/input alias");

    twin::GuardedBuffer foreign_down;
    const bool foreign_allocated = foreign_down.allocate(
        foreign_context, "foreign MTP stage-15 hc_down",
        bytes_for(kHcDown), error);
    check(foreign_allocated, "foreign MTP stage-15 weight allocation", error);
    if (foreign_allocated) {
        buffers = fixture.mtp_buffers(context);
        buffers.hc_down = foreign_down.view();
        passed &= reject_case(context, buffers,
                              "MTP stage 15 rejects foreign GR weights");
    } else {
        passed = false;
    }
    fixture.set_rows(1u);
    return passed;
}

bool test_concurrent_group_boundary(Context& context, Fixture& fixture) {
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode == nullptr ||
        std::string_view(dispatch_mode) != "concurrent") {
        return true;
    }
    fixture.set_rows(1u);
    fixture.reset_outputs();
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP stage-15 group boundary", error);
    if (!command) {
        check(false, "MTP stage-15 group-boundary command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers(context);
    const bool wrapper_checked = Encoder::encode(
        context, *command, "MTP stage-15 group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected =
                !encode_mtp_feed_forward_read(context, encoder, buffers,
                                              stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP stage 15 accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged = fixture.all_buffers_unchanged();
    check(wrapper_checked, "MTP stage 15 rejects an open concurrent group",
          error);
    check(unchanged, "MTP stage-15 group rejection preserves every buffer");
    command->discard();
    return wrapper_checked && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.set_rows(1u);
    fixture.reset_outputs();
    fixture.capture();
    const auto good = fixture.mtp_buffers(context);
    auto bad = good;
    bad.mixed = bad.bo;
    std::string error;
    auto command = context.make_command_buffer("MTP stage-15 rollback", error);
    if (!command) {
        check(false, "MTP stage-15 rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stage-15 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_feed_forward_read(context, encoder, good,
                                              encode_error)) {
                return false;
            }
            return encode_mtp_feed_forward_read(context, encoder, bad,
                                                encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged = fixture.all_buffers_unchanged();
    check(rejected && discarded,
          "MTP stage-15 late failure discards the caller-owned segment",
          error);
    check(unchanged, "MTP stage-15 rollback preserves every buffer");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_feed_forward_read_test <strata.metallib>\n");
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
        fixture.allocate(*context, "MTP feed-forward read");
    check(allocated, "allocate MTP stage-15 fixture");
    if (!allocated) return 1;

    bool passed = test_pipeline_parity(*context, fixture);
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_concurrent_group_boundary(*context, fixture);
    passed &= test_late_rollback(*context, fixture);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr,
                     "metal_mtp_feed_forward_read_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_feed_forward_read_test: MTP stage 15 OK");
    return 0;
}
