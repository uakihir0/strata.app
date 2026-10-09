// Metal twin for MTP drafter stage 16 (docs/mac/04 §5.11).
#include "oracles/bf16_mv.hpp"
#include "oracles/moe_route.hpp"
#include "metallib_sha256.inc"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
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
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kRouterExperts = 512u;
constexpr std::uint32_t kTopK = 10u;
constexpr std::size_t kMixedBytes =
    static_cast<std::size_t>(N_EMBD) * sizeof(float);
constexpr std::size_t kRouterWeightBytes =
    static_cast<std::size_t>(kRouterExperts) * N_EMBD *
    sizeof(std::uint16_t);
constexpr std::size_t kLogitsBytes =
    static_cast<std::size_t>(kRouterExperts) * sizeof(float);
constexpr std::size_t kIdsBytes =
    static_cast<std::size_t>(kTopK) * sizeof(std::int32_t);
constexpr std::size_t kWeightsBytes =
    static_cast<std::size_t>(kTopK) * sizeof(float);

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
    twin::GuardedBuffer mixed;
    twin::GuardedBuffer router_weights;
    twin::GuardedBuffer logits;
    twin::GuardedBuffer ids;
    twin::GuardedBuffer weights;
    twin::GuardedBuffer sink;
    std::vector<float> expected_logits;
    std::vector<std::int32_t> expected_ids;
    std::vector<twin::tf> expected_weights;

    bool allocate(Context& context, const std::string& label) {
        std::string error;
        const bool allocated =
            params.allocate(context, label + " params", 256u, error) &&
            mixed.allocate(context, label + " mixed", kMixedBytes, error) &&
            router_weights.allocate(context, label + " router weights",
                                    kRouterWeightBytes, error) &&
            logits.allocate(context, label + " logits", kLogitsBytes, error) &&
            ids.allocate(context, label + " IDs", kIdsBytes, error) &&
            weights.allocate(context, label + " route weights", kWeightsBytes,
                             error) &&
            sink.allocate(context, label + " matvec sink", sizeof(float),
                          error);
        check(allocated, label + " allocation", error);
        if (!allocated) return false;

        MtpParams block{};
        block.T = 1u;
        std::memcpy(params.payload().data(), &block, sizeof(block));

        std::mt19937_64 random(0x16'512'10ull);
        std::uniform_real_distribution<float> input_distribution(-0.5f, 0.5f);
        std::uniform_real_distribution<float> weight_distribution(-0.02f,
                                                                   0.02f);
        auto mixed_values = typed<float>(mixed);
        for (float& value : mixed_values) value = input_distribution(random);
        auto weight_values = typed<std::uint16_t>(router_weights);
        for (auto& value : weight_values) {
            value = strata::kernels::bf16_from_f32(
                weight_distribution(random));
        }
        reset_outputs();
        return build_oracles();
    }

    void set_rows(std::uint32_t rows) {
        std::memset(params.payload().data(), 0, params.payload().size());
        MtpParams block{};
        block.T = rows;
        std::memcpy(params.payload().data(), &block, sizeof(block));
    }

    void reset_outputs() {
        std::memset(logits.payload().data(), 0xa5, logits.payload().size());
        std::memset(ids.payload().data(), 0xa5, ids.payload().size());
        std::memset(weights.payload().data(), 0xa5, weights.payload().size());
        std::memset(sink.payload().data(), 0xa5, sink.payload().size());
    }

    bool build_oracles() {
        std::string error;
        const bool matvec_ready = twin::bf16_matvec_reference(
            typed<std::uint16_t>(router_weights),
            typed<float>(mixed),
            N_EMBD,
            kRouterExperts,
            1u,
            1u,
            256u,
            expected_logits,
            error);
        check(matvec_ready, "MTP stage-16 BF16 matvec scalar oracle", error);
        if (!matvec_ready || expected_logits.size() != kRouterExperts) {
            return false;
        }
        const bool route_ready = twin::moe_route_reference(
            expected_logits, 1u, kRouterExperts, false, expected_ids,
            expected_weights, error);
        check(route_ready, "MTP stage-16 top-10 route scalar oracle", error);
        return route_ready && expected_ids.size() == kTopK &&
               expected_weights.size() == kTopK;
    }

    MtpRouterBuffers mtp_buffers(Context& context) {
        return {context.buffers().read_only(params.view()),
                context.buffers().read_only(mixed.view()),
                context.buffers().read_only(router_weights.view()),
                logits.view(),
                ids.view(),
                weights.view(),
                sink.view()};
    }

    ReadOnlyBF16MatVecBuffers matvec_buffers(Context& context) {
        const ReadOnlyBufferRef parameter_view =
            context.buffers().read_only(params.view());
        return {context.buffers().read_only(mixed.view()),
                context.buffers().read_only(router_weights.view()),
                parameter_view.slice(offsetof(MtpParams, T),
                                    sizeof(std::uint32_t)),
                logits.view(),
                sink.view()};
    }

    void capture() {
        params.capture_before_dispatch();
        mixed.capture_before_dispatch();
        router_weights.capture_before_dispatch();
        logits.capture_before_dispatch();
        ids.capture_before_dispatch();
        weights.capture_before_dispatch();
        sink.capture_before_dispatch();
    }

    bool inputs_unchanged() const {
        return !params.first_unwritten_change({}) &&
               !mixed.first_unwritten_change({}) &&
               !router_weights.first_unwritten_change({});
    }

    bool output_guards_unchanged() const {
        const std::array<twin::ByteRange, 1> logits_written{{
            {0u, kLogitsBytes},
        }};
        const std::array<twin::ByteRange, 1> ids_written{{
            {0u, kIdsBytes},
        }};
        const std::array<twin::ByteRange, 1> weights_written{{
            {0u, kWeightsBytes},
        }};
        return !logits.first_unwritten_change(logits_written) &&
               !ids.first_unwritten_change(ids_written) &&
               !weights.first_unwritten_change(weights_written) &&
               !sink.first_unwritten_change({});
    }

    bool all_buffers_unchanged() const {
        return !params.first_unwritten_change({}) &&
               !mixed.first_unwritten_change({}) &&
               !router_weights.first_unwritten_change({}) &&
               !logits.first_unwritten_change({}) &&
               !ids.first_unwritten_change({}) &&
               !weights.first_unwritten_change({}) &&
               !sink.first_unwritten_change({});
    }

    std::array<std::vector<std::uint8_t>, 3u> outputs() const {
        std::array<std::vector<std::uint8_t>, 3u> result;
        const std::array<const twin::GuardedBuffer*, 3u> buffers{
            &logits, &ids, &weights};
        for (std::size_t i = 0u; i < buffers.size(); ++i) {
            const auto bytes = buffers[i]->payload();
            result[i].assign(bytes.begin(), bytes.end());
        }
        return result;
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

bool encode_direct_components(Context& context, Encoder& encoder,
                              Fixture& fixture, std::string& error) {
    if (!encode_bf16_matvec(context, encoder, N_EMBD, kRouterExperts, 1u,
                            1u, fixture.matvec_buffers(context), error)) {
        return false;
    }
    const auto logits = context.buffers().read_only(fixture.logits.view());
    if (!logits) {
        error = "MTP stage-16 reference could not form the router-logits view";
        return false;
    }
    return encode_moe_route(
        context, encoder, 1u, kRouterExperts, false,
        {logits, fixture.ids.view(), fixture.weights.view()}, error);
}

bool run_pipeline(Context& context, Fixture& fixture, bool use_mtp,
                  bool caller_owned,
                  std::array<std::vector<std::uint8_t>, 3u>& outputs) {
    fixture.set_rows(1u);
    fixture.reset_outputs();
    fixture.capture();
    std::string error;
    auto command = context.make_command_buffer(
        use_mtp ? "MTP stage-16 router" : "direct router components", error);
    check(command != nullptr, "MTP stage-16 command buffer", error);
    if (!command) return false;

    bool encoded = false;
    if (use_mtp && caller_owned) {
        const auto buffers = fixture.mtp_buffers(context);
        encoded = Encoder::encode(
            context, *command, "MTP stage-16 caller-owned",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_router(context, encoder, buffers,
                                         encode_error);
            },
            error);
    } else if (use_mtp) {
        encoded = encode_mtp_router(context, *command,
                                    fixture.mtp_buffers(context), error);
    } else {
        encoded = Encoder::encode(
            context, *command, "direct MTP stage-16 reference",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_direct_components(context, encoder, fixture,
                                                encode_error);
            },
            error);
    }
    check(encoded,
          use_mtp ? "MTP stage-16 ordered router/route encoding"
                  : "direct BF16 projection/route reference encoding",
          error);
    if (!encoded) return false;
    const bool completed =
        submit_and_wait(command, "MTP stage-16 router", error);
    check(completed, "MTP stage-16 GPU completion", error);
    if (!completed) return false;

    const auto expected_logits_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(fixture.expected_logits.data()),
        kLogitsBytes);
    const auto logits_failure =
        twin::compare_bytes(fixture.logits.payload(), expected_logits_bytes);
    check(!logits_failure, "MTP stage-16 router logits match BF16 scalar oracle");

    const auto expected_ids_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(fixture.expected_ids.data()),
        kIdsBytes);
    const auto ids_failure =
        twin::compare_bytes(fixture.ids.payload(), expected_ids_bytes);
    check(!ids_failure, "MTP stage-16 route IDs match top-10 scalar oracle");

    const auto* actual_weights =
        reinterpret_cast<const float*>(fixture.weights.payload().data());
    const std::span<const float> actual_weight_span(actual_weights, kTopK);
    const twin::Tolerance route_tolerance{
        twin::Comparison::units, 16.0, 0.0, 0u};
    const auto weights_failure = twin::compare_floats(
        actual_weight_span, fixture.expected_weights, route_tolerance);
    check(!weights_failure,
          "MTP stage-16 route weights match top-10 oracle within U(16)");
    check(fixture.inputs_unchanged(),
          "MTP stage 16 preserves params, mixed row and router weights");
    check(fixture.output_guards_unchanged(),
          "MTP stage 16 writes only logits, top-10 IDs and weights");
    outputs = fixture.outputs();
    return !logits_failure && !ids_failure && !weights_failure &&
           fixture.inputs_unchanged() && fixture.output_guards_unchanged();
}

bool test_pipeline_parity(Context& context, Fixture& fixture) {
    std::array<std::vector<std::uint8_t>, 3u> standalone;
    std::array<std::vector<std::uint8_t>, 3u> caller;
    std::array<std::vector<std::uint8_t>, 3u> direct;
    bool passed = run_pipeline(context, fixture, true, false, standalone);
    passed &= run_pipeline(context, fixture, true, true, caller);
    passed &= run_pipeline(context, fixture, false, false, direct);
    constexpr std::array<const char*, 3u> names{
        "logits_r", "ids", "route weights"};
    for (std::size_t output = 0u; output < names.size(); ++output) {
        const bool caller_matches = standalone[output] == caller[output];
        const bool direct_matches = standalone[output] == direct[output];
        check(caller_matches, std::string("MTP stage-16 ") + names[output] +
                                  " standalone/caller parity");
        check(direct_matches, std::string("MTP stage-16 ") + names[output] +
                                  " matches direct component chain");
        passed &= caller_matches && direct_matches;
    }
    return passed;
}

bool reject_case(Context& context, const MtpRouterBuffers& buffers,
                 const std::string& label,
                 const std::string& expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected = !encode_mtp_router(context, *command, buffers, error);
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
    passed &= reject_case(context, buffers, "MTP stage 16 rejects T=0",
                          "T must be in 1..8");

    fixture.set_rows(2u);
    buffers = fixture.mtp_buffers(context);
    passed &= reject_case(context, buffers, "MTP stage 16 rejects T=2",
                          "requires T=1");

    fixture.set_rows(1u);
    buffers = fixture.mtp_buffers(context);
    buffers.router_weights =
        buffers.router_weights.slice(0u, kRouterWeightBytes - 2u);
    passed &= reject_case(context, buffers,
                          "MTP stage 16 rejects short router weights",
                          "undersized or misaligned");

    buffers = fixture.mtp_buffers(context);
    buffers.logits_r =
        buffers.logits_r.slice(0u, kLogitsBytes - sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 16 rejects short router logits",
                          "undersized or misaligned");

    twin::GuardedBuffer alias;
    std::string error;
    const bool alias_allocated =
        alias.allocate(context, "MTP stage-16 params/mixed alias",
                       kMixedBytes, error);
    check(alias_allocated, "MTP stage-16 alias allocation", error);
    if (!alias_allocated) return false;
    MtpParams params{};
    params.T = 1u;
    std::memcpy(alias.payload().data(), &params, sizeof(params));
    buffers = fixture.mtp_buffers(context);
    buffers.params = context.buffers().read_only(
        alias.view().slice(0u, sizeof(MtpParams)));
    buffers.mixed = context.buffers().read_only(alias.view());
    passed &= reject_case(context, buffers,
                          "MTP stage 16 rejects params/mixed alias",
                          "parameter block overlaps active view mixed");

    buffers = fixture.mtp_buffers(context);
    buffers.ids = buffers.weights;
    passed &= reject_case(context, buffers,
                          "MTP stage 16 rejects route output alias",
                          "outputs weights and ids overlap");

    twin::GuardedBuffer foreign_weights;
    const bool foreign_allocated = foreign_weights.allocate(
        foreign_context, "foreign MTP stage-16 router weights",
        kRouterWeightBytes, error);
    check(foreign_allocated, "foreign MTP stage-16 allocation", error);
    if (foreign_allocated) {
        buffers = fixture.mtp_buffers(context);
        buffers.router_weights =
            foreign_context.buffers().read_only(foreign_weights.view());
        passed &= reject_case(context, buffers,
                              "MTP stage 16 rejects foreign router weights",
                              "not owned by this Context");
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
        context.make_command_buffer("MTP stage-16 group boundary", error);
    if (!command) {
        check(false, "MTP stage-16 group-boundary command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers(context);
    const bool wrapper_checked = Encoder::encode(
        context, *command, "MTP stage-16 group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected =
                !encode_mtp_router(context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error = "MTP stage 16 accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged = fixture.all_buffers_unchanged();
    check(wrapper_checked, "MTP stage 16 rejects an open concurrent group",
          error);
    check(unchanged, "MTP stage-16 group rejection preserves every buffer");
    command->discard();
    return wrapper_checked && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.set_rows(1u);
    fixture.reset_outputs();
    fixture.capture();
    const auto good = fixture.mtp_buffers(context);
    auto bad = good;
    bad.router_weights = bad.router_weights.slice(0u, 2u);
    std::string error;
    auto command = context.make_command_buffer("MTP stage-16 rollback", error);
    if (!command) {
        check(false, "MTP stage-16 rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stage-16 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_router(context, encoder, good, encode_error)) {
                return false;
            }
            return encode_mtp_router(context, encoder, bad, encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged = fixture.all_buffers_unchanged();
    check(rejected && discarded,
          "MTP stage-16 late failure discards caller-owned segment", error);
    check(unchanged, "MTP stage-16 rollback preserves every buffer");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_router_test <strata.metallib>\n");
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
    if (!fixture.allocate(*context, "MTP router")) return 1;

    bool passed = test_pipeline_parity(*context, fixture);
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_concurrent_group_boundary(*context, fixture);
    passed &= test_late_rollback(*context, fixture);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_router_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_router_test: MTP stage 16 OK");
    return 0;
}
