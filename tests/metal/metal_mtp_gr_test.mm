// MTP stage 7 wrapper twin over the validated GR read pipeline.
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "strata/kernels/bf16_bits.hpp"
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
#include <string>
#include <string_view>
#include <utility>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kMaxRows = VERIFY_MAX_T;
constexpr std::size_t kBufferCount = 11u;
constexpr std::uint32_t kUnwrittenOutputBits = 0x7fc12345u;
constexpr float kUnwrittenOutput =
    std::bit_cast<float>(kUnwrittenOutputBits);
enum BufferIndex : std::size_t {
    kResidual,
    kH2Scratch,
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
    "residual", "dead h2", "norm", "normalized", "rs", "hc_down",
    "hc_inject", "lo", "inject_out", "hc_up", "mixed",
};

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !condition;
}

std::size_t bytes_for(BufferIndex index) {
    switch (index) {
        case kResidual:
        case kH2Scratch:
        case kNormalized:
            return static_cast<std::size_t>(kMaxRows) * HC_DIM * sizeof(float);
        case kNorm:
            return static_cast<std::size_t>(HC_DIM) * sizeof(float);
        case kRs:
        case kInjectOut:
            return static_cast<std::size_t>(kMaxRows) * HC * sizeof(float);
        case kHcDown:
        case kHcUp:
            return static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
                   sizeof(std::uint16_t);
        case kHcInject:
            return static_cast<std::size_t>(HC) * HC_DIM *
                   sizeof(std::uint16_t);
        case kLo:
            return static_cast<std::size_t>(kMaxRows) * HC_LOW_RANK *
                   sizeof(float);
        case kMixed:
            return static_cast<std::size_t>(kMaxRows) * N_EMBD * sizeof(float);
    }
    return 0u;
}

struct Fixture {
    twin::GuardedBuffer params;
    std::array<twin::GuardedBuffer, kBufferCount> storage;

    BufferRef view(BufferIndex index) { return storage[index].view(); }

    bool allocate(Context& context, const std::string& label) {
        std::string error;
        if (!params.allocate(context, label + " params", 256u, error)) {
            check(false, label + " params allocation", error);
            return false;
        }
        for (std::size_t i = 0u; i < storage.size(); ++i) {
            if (!storage[i].allocate(context,
                                     label + " " + kNames[i],
                                     bytes_for(static_cast<BufferIndex>(i)),
                                     error)) {
                check(false, label + " " + kNames[i] + " allocation", error);
                return false;
            }
        }
        initialize();
        return true;
    }

    void initialize() {
        for (std::size_t index = 0u; index < storage.size(); ++index) {
            const BufferIndex which = static_cast<BufferIndex>(index);
            if (which == kHcDown || which == kHcInject || which == kHcUp) {
                auto* values = reinterpret_cast<std::uint16_t*>(
                    storage[index].payload().data());
                const std::size_t count =
                    bytes_for(which) / sizeof(std::uint16_t);
                const float magnitude =
                    which == kHcInject ? 0.0002f : 0.0001f;
                for (std::size_t i = 0u; i < count; ++i) {
                    const auto centered =
                        static_cast<std::int32_t>((i * 17u) % 9u) - 4;
                    values[i] = strata::kernels::bf16_from_f32(
                        magnitude * static_cast<float>(centered));
                }
                continue;
            }
            if (which == kNormalized || which == kRs || which == kLo ||
                which == kInjectOut || which == kMixed) {
                std::memset(storage[index].payload().data(), 0xa5,
                            storage[index].payload().size());
                continue;
            }
            auto* values = reinterpret_cast<float*>(
                storage[index].payload().data());
            const std::size_t count =
                bytes_for(which) / sizeof(float);
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
            }
        }
        set_rows(1u);
    }

    void set_rows(std::uint32_t rows) {
        std::fill(params.payload().begin(), params.payload().end(), 0u);
        MtpParams block{};
        block.T = rows;
        std::memcpy(params.payload().data(), &block, sizeof(block));
    }

    ReadOnlyBufferRef params_view(Context& context) const {
        return context.buffers().read_only(params.view());
    }

    MtpAttentionReadBuffers mtp_buffers(Context& context) {
        return {
            params_view(context),
            view(kResidual),
            view(kH2Scratch),
            view(kNorm),
            view(kNormalized),
            view(kRs),
            view(kHcDown),
            view(kHcInject),
            view(kLo),
            view(kInjectOut),
            view(kHcUp),
            view(kMixed),
        };
    }

    GRReadBuffers direct_gr_buffers() {
        const std::size_t block_bytes =
            static_cast<std::size_t>(kMaxRows) * N_EMBD * sizeof(float);
        const std::size_t inject_bytes =
            static_cast<std::size_t>(kMaxRows) * HC * sizeof(float);
        const BufferRef scratch = view(kH2Scratch);
        return {
            view(kResidual),
            scratch.slice(0u, block_bytes),
            scratch.slice(block_bytes, inject_bytes),
            view(kNorm),
            view(kNormalized),
            view(kRs),
            view(kHcDown),
            view(kHcInject),
            view(kLo),
            view(kInjectOut),
            view(kHcUp),
            view(kMixed),
        };
    }

    void capture() {
        params.capture_before_dispatch();
        for (auto& buffer : storage) buffer.capture_before_dispatch();
    }

    bool unchanged() const {
        bool result = !params.first_unwritten_change({});
        for (const auto& buffer : storage) {
            result &= !buffer.first_unwritten_change({});
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

bool reset_outputs(Fixture& fixture) {
    for (const BufferIndex index :
         {kNormalized, kRs, kLo, kInjectOut, kMixed}) {
        auto payload = fixture.storage[index].payload();
        auto* values = reinterpret_cast<float*>(payload.data());
        std::fill_n(values, payload.size() / sizeof(float),
                    kUnwrittenOutput);
    }
    return true;
}

bool payloads_equal(const Fixture& left, const Fixture& right) {
    for (std::size_t i = 0u; i < kBufferCount; ++i) {
        const auto a = left.storage[i].payload();
        const auto b = right.storage[i].payload();
        if (a.size() != b.size() ||
            !std::equal(a.begin(), a.end(), b.begin())) {
            return false;
        }
    }
    return true;
}

bool check_active_guards(Fixture& fixture, std::uint32_t rows,
                         const std::string& label) {
    const std::array<std::pair<BufferIndex, std::size_t>, 5> written{{
        {kNormalized,
         static_cast<std::size_t>(rows) * HC_DIM * sizeof(float)},
        {kRs, static_cast<std::size_t>(rows) * HC * sizeof(float)},
        {kLo, static_cast<std::size_t>(rows) * HC_LOW_RANK * sizeof(float)},
        {kInjectOut, static_cast<std::size_t>(rows) * HC * sizeof(float)},
        {kMixed, static_cast<std::size_t>(rows) * N_EMBD * sizeof(float)},
    }};
    bool intact = true;
    for (const auto& [index, active_bytes] : written) {
        const std::array<twin::ByteRange, 1> range{{{0u, active_bytes}}};
        const auto changed = fixture.storage[index].first_unwritten_change(range);
        check(!changed, label + " " + kNames[index] + " active extent/guards",
              changed ? changed->reason : "");
        intact &= !changed;
    }
    check(fixture.storage[kResidual].first_unwritten_change({}) == std::nullopt,
          label + " apply=false preserves residual");
    intact &= !fixture.storage[kResidual].first_unwritten_change({});
    return intact;
}

bool check_active_outputs(Fixture& fixture, std::uint32_t rows,
                          const std::string& label) {
    const std::array<std::pair<BufferIndex, std::size_t>, 5> outputs{{
        {kNormalized, static_cast<std::size_t>(rows) * HC_DIM},
        {kRs, static_cast<std::size_t>(rows) * HC},
        {kLo, static_cast<std::size_t>(rows) * HC_LOW_RANK},
        {kInjectOut, static_cast<std::size_t>(rows) * HC},
        {kMixed, static_cast<std::size_t>(rows) * N_EMBD},
    }};
    bool valid = true;
    for (const auto& [index, element_count] : outputs) {
        const auto* values = reinterpret_cast<const float*>(
            fixture.storage[index].payload().data());
        bool all_written = true;
        bool finite = true;
        for (std::size_t i = 0u; i < element_count; ++i) {
            all_written &=
                std::bit_cast<std::uint32_t>(values[i]) !=
                kUnwrittenOutputBits;
            finite &= std::isfinite(values[i]);
        }
        check(all_written && finite,
              label + " " + kNames[index] +
                  " every active value is written and finite");
        valid &= all_written && finite;
    }
    return valid;
}

bool test_success(Context& context, Fixture& wrapped, Fixture& direct) {
    bool passed = true;
    std::string error;
    for (std::uint32_t rows = 1u; rows <= kMaxRows; ++rows) {
        const std::string label =
            "MTP attention read T=" + std::to_string(rows);
        wrapped.set_rows(rows);
        direct.set_rows(rows);
        reset_outputs(wrapped);
        reset_outputs(direct);
        wrapped.capture();
        direct.capture();

        auto wrapped_command = context.make_command_buffer(
            label + " MTP wrapper", error);
        check(wrapped_command != nullptr, label + " wrapper command", error);
        if (!wrapped_command) return false;
        const bool wrapped_encoded = encode_mtp_attention_read(
            context, *wrapped_command, wrapped.mtp_buffers(context), error);
        check(wrapped_encoded, label + " wrapper encode", error);
        if (!wrapped_encoded) return false;
        const bool wrapped_completed =
            submit_and_wait(wrapped_command, label + " wrapper", error);
        check(wrapped_completed, label + " wrapper GPU completion", error);
        if (!wrapped_completed) return false;

        auto direct_command = context.make_command_buffer(
            label + " direct GR", error);
        check(direct_command != nullptr, label + " direct command", error);
        if (!direct_command) return false;
        const bool direct_encoded = encode_gr_read(
            context, *direct_command, rows, false,
            direct.direct_gr_buffers(), error);
        check(direct_encoded, label + " direct GR encode", error);
        if (!direct_encoded) return false;
        const bool direct_completed =
            submit_and_wait(direct_command, label + " direct GR", error);
        check(direct_completed, label + " direct GR GPU completion", error);
        if (!direct_completed) return false;

        const bool exact = payloads_equal(wrapped, direct);
        check(exact, label + " bitwise matches the underlying GR chain");
        check(check_active_guards(wrapped, rows, label),
              label + " active output extents and guards");
        const bool outputs_valid =
            check_active_outputs(wrapped, rows, label);
        check(!wrapped.params.first_unwritten_change({}),
              label + " MtpParams unchanged");
        passed &= exact && outputs_valid &&
                  !wrapped.params.first_unwritten_change({});
    }
    return passed;
}

bool test_concurrent_group_boundary(Context& context, Fixture& fixture) {
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode == nullptr ||
        std::string_view(dispatch_mode) != "concurrent") {
        return true;
    }

    fixture.set_rows(1u);
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP attention read group boundary", error);
    if (!command) {
        check(false, "MTP concurrent group command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers(context);
    const bool wrapper_rejected = Encoder::encode(
        context, *command, "MTP attention read group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected = !encode_mtp_attention_read(
                context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP attention read accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged = fixture.unchanged();
    check(wrapper_rejected,
          "MTP attention read rejects an open concurrent group", error);
    check(unchanged,
          "MTP open-group rejection preserves all buffers");
    command->discard();
    return wrapper_rejected && unchanged;
}

bool test_parameter_output_alias(Context& context, Fixture& fixture) {
    fixture.set_rows(1u);
    BufferRef shared = fixture.view(kNormalized);
    MtpParams block{};
    block.T = 1u;
    std::memcpy(shared.contents, &block, sizeof(block));

    MtpAttentionReadBuffers aliased = fixture.mtp_buffers(context);
    aliased.params = context.buffers().read_only(shared);
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP params/output alias", error);
    if (!command) {
        check(false, "MTP params/output alias command buffer", error);
        return false;
    }
    const bool rejected = !encode_mtp_attention_read(
        context, *command, aliased, error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged = fixture.unchanged();
    check(rejected && discarded &&
              error.find("overlaps the parameter block") != std::string::npos,
          "MTP attention read rejects params/output alias before dispatch",
          error);
    check(unchanged,
          "MTP params/output alias rejection preserves all buffers");
    bool passed = rejected && discarded && unchanged;

    BufferRef shared_norm = fixture.view(kNorm);
    std::memcpy(shared_norm.contents, &block, sizeof(block));
    MtpAttentionReadBuffers aliased_input = fixture.mtp_buffers(context);
    aliased_input.params = context.buffers().read_only(shared_norm);
    fixture.capture();
    auto input_command =
        context.make_command_buffer("MTP params/input alias", error);
    if (!input_command) {
        check(false, "MTP params/input alias command buffer", error);
        return false;
    }
    const bool input_rejected = !encode_mtp_attention_read(
        context, *input_command, aliased_input, error);
    const bool input_discarded =
        input_command->unsafe_native_handle() == nullptr;
    const bool input_unchanged = fixture.unchanged();
    check(input_rejected && input_discarded &&
              error.find("overlaps the parameter block") != std::string::npos,
          "MTP attention read rejects params/input alias before dispatch",
          error);
    check(input_unchanged,
          "MTP params/input alias rejection preserves all buffers");
    passed &= input_rejected && input_discarded && input_unchanged;

    fixture.capture();
    auto caller_owned =
        context.make_command_buffer("MTP params/output alias rollback", error);
    if (!caller_owned) {
        check(false, "MTP params/output alias caller command buffer", error);
        return false;
    }
    const MtpAddStreamsBuffers preceding_stage{
        fixture.params_view(context),
        context.buffers().read_only(fixture.view(kH2Scratch)),
        context.buffers().read_only(fixture.view(kNorm)),
        fixture.view(kResidual),
    };
    const bool caller_rejected = !Encoder::encode(
        context, *caller_owned, "MTP params/output alias caller segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_add_streams(context, encoder, preceding_stage,
                                        encode_error)) {
                return false;
            }
            return encode_mtp_attention_read(context, encoder, aliased,
                                             encode_error);
        },
        error);
    const bool caller_discarded =
        caller_owned->unsafe_native_handle() == nullptr;
    const bool caller_unchanged = fixture.unchanged();
    check(caller_rejected && caller_discarded &&
              error.find("overlaps the parameter block") != std::string::npos,
          "MTP caller segment rejects params/output alias after stage 6",
          error);
    check(caller_unchanged,
          "MTP parameter/output alias failure rolls back the prior stage");
    passed &= caller_rejected && caller_discarded && caller_unchanged;
    return passed;
}

bool test_rejections(Context& context, Context& foreign_context,
                     Fixture& fixture) {
    bool passed = true;
    std::string error;
    fixture.set_rows(1u);
    const MtpAttentionReadBuffers valid = fixture.mtp_buffers(context);

    for (const std::uint32_t rows : {0u, 9u}) {
        fixture.set_rows(rows);
        fixture.capture();
        auto command = context.make_command_buffer(
            "MTP attention read invalid T", error);
        if (!command) {
            check(false, "MTP invalid-T command buffer", error);
            return false;
        }
        const bool rejected = !encode_mtp_attention_read(
            context, *command, fixture.mtp_buffers(context), error);
        const bool discarded =
            command->unsafe_native_handle() == nullptr;
        check(rejected && discarded,
              "MTP attention read rejects T=" + std::to_string(rows), error);
        passed &= rejected && discarded && fixture.unchanged();
    }
    fixture.set_rows(1u);

    fixture.capture();
    MtpAttentionReadBuffers short_scratch = valid;
    short_scratch.dead_h2_scratch =
        fixture.view(kH2Scratch).slice(0u, sizeof(float));
    auto short_command =
        context.make_command_buffer("MTP short dead h2", error);
    if (!short_command) {
        check(false, "MTP short scratch command buffer", error);
        return false;
    }
    const bool short_rejected = !encode_mtp_attention_read(
        context, *short_command, short_scratch, error);
    check(short_rejected &&
              short_command->unsafe_native_handle() == nullptr &&
              error.find("undersized or misaligned") != std::string::npos,
          "MTP attention read rejects short dead h2 scratch", error);
    passed &= short_rejected && fixture.unchanged();

    fixture.capture();
    MtpAttentionReadBuffers misaligned = valid;
    const BufferRef h2 = fixture.view(kH2Scratch);
    misaligned.dead_h2_scratch =
        h2.slice(1u, h2.length - 1u);
    auto misaligned_command =
        context.make_command_buffer("MTP misaligned dead h2", error);
    if (!misaligned_command) {
        check(false, "MTP misaligned scratch command buffer", error);
        return false;
    }
    const bool misaligned_rejected = !encode_mtp_attention_read(
        context, *misaligned_command, misaligned, error);
    check(misaligned_rejected &&
              misaligned_command->unsafe_native_handle() == nullptr &&
              error.find("undersized or misaligned") != std::string::npos,
          "MTP attention read rejects misaligned dead h2 scratch", error);
    passed &= misaligned_rejected && fixture.unchanged();

    std::string foreign_error;
    BufferRef foreign_params = foreign_context.buffers().allocate(
        "foreign MTP attention params", 256u, foreign_error);
    check(static_cast<bool>(foreign_params), "allocate foreign MTP params",
          foreign_error);
    if (!foreign_params) return false;
    MtpParams foreign_block{};
    foreign_block.T = 1u;
    std::memcpy(foreign_params.contents, &foreign_block, sizeof(foreign_block));
    MtpAttentionReadBuffers foreign = valid;
    foreign.params = foreign_context.buffers().read_only(foreign_params);
    fixture.capture();
    auto foreign_command =
        context.make_command_buffer("MTP foreign params", error);
    if (!foreign_command) {
        check(false, "MTP foreign params command buffer", error);
        return false;
    }
    const bool foreign_rejected = !encode_mtp_attention_read(
        context, *foreign_command, foreign, error);
    check(foreign_rejected &&
              foreign_command->unsafe_native_handle() == nullptr &&
              error.find("not owned") != std::string::npos,
          "MTP attention read rejects foreign params", error);
    passed &= foreign_rejected && fixture.unchanged();
    fixture.set_rows(1u);
    return passed;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    bool passed = true;
    std::string error;
    fixture.set_rows(1u);
    auto good = fixture.mtp_buffers(context);
    auto bad = good;
    bad.hc_down = good.hc_down.slice(0u, 16u);
    fixture.capture();

    auto standalone =
        context.make_command_buffer("MTP stage-7 late standalone failure",
                                   error);
    if (!standalone) {
        check(false, "MTP late standalone command buffer", error);
        return false;
    }
    const bool standalone_rejected =
        !encode_mtp_attention_read(context, *standalone, bad, error);
    const bool standalone_discarded =
        standalone->unsafe_native_handle() == nullptr;
    check(standalone_rejected && standalone_discarded,
          "MTP stage-7 standalone late failure discards command", error);
    const bool standalone_unchanged = fixture.unchanged();
    check(standalone_unchanged,
          "MTP stage-7 standalone rollback preserves all buffers");
    passed &= standalone_rejected && standalone_discarded &&
              standalone_unchanged;

    auto caller_owned =
        context.make_command_buffer("MTP stage-7 caller rollback", error);
    if (!caller_owned) {
        check(false, "MTP caller-owned command buffer", error);
        return false;
    }
    const bool caller_rejected = !Encoder::encode(
        context, *caller_owned, "MTP stage-7 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_attention_read(context, encoder, good,
                                           encode_error)) {
                return false;
            }
            return encode_mtp_attention_read(context, encoder, bad,
                                             encode_error);
        },
        error);
    const bool caller_discarded =
        caller_owned->unsafe_native_handle() == nullptr;
    check(caller_rejected && caller_discarded,
          "MTP stage-7 caller-owned late failure discards segment", error);
    const bool caller_unchanged = fixture.unchanged();
    check(caller_unchanged,
          "MTP stage-7 caller rollback preserves all buffers");
    passed &= caller_rejected && caller_discarded && caller_unchanged;
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: metal_mtp_gr_test <strata.metallib>\n");
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

    Fixture wrapped;
    Fixture direct;
    check(wrapped.allocate(*context, "MTP GR wrapper"),
          "allocate MTP wrapper fixture");
    check(direct.allocate(*context, "MTP GR direct"),
          "allocate direct GR fixture");
    if (!wrapped.view(kResidual) || !direct.view(kResidual)) return 1;

    bool passed = test_success(*context, wrapped, direct);
    passed &= test_concurrent_group_boundary(*context, wrapped);
    passed &= test_parameter_output_alias(*context, wrapped);
    passed &= test_rejections(*context, *foreign_context, wrapped);
    passed &= test_late_rollback(*context, wrapped);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_gr_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("metal_mtp_gr_test: MTP stage 7 OK");
    return 0;
}
