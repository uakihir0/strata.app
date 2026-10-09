// Metal twin for MTP drafter stage 8 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"
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
constexpr std::uint32_t kOutputWidth = 512u;
constexpr std::size_t kProjectionCount = 7u;
constexpr std::uint32_t kPoisonBits = 0x7fc12345u;
constexpr float kPoison = std::bit_cast<float>(kPoisonBits);

enum ProjectionBuffer : std::size_t {
    kMixed,
    kXq,
    kKeyWeights,
    kValueWeights,
    kKcur,
    kVcur,
    kSink,
};

constexpr std::array<const char*, kProjectionCount> kNames{
    "mixed", "xq", "key weights", "value weights",
    "kcur", "vcur", "sink"};

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !condition;
}

std::size_t bytes_for(ProjectionBuffer index) {
    switch (index) {
        case kMixed:
            return static_cast<std::size_t>(kMaxRows) * N_EMBD * sizeof(float);
        case kXq:
            return static_cast<std::size_t>(kMaxRows) * (N_EMBD / 32u) *
                   sizeof(Q81);
        case kKeyWeights:
        case kValueWeights:
            return static_cast<std::size_t>(kOutputWidth) *
                   (N_EMBD / 32u) * sizeof(block_q8_0);
        case kKcur:
        case kVcur:
            return static_cast<std::size_t>(kMaxRows) * kOutputWidth *
                   sizeof(float);
        case kSink:
            return static_cast<std::size_t>(N_EMBD) * sizeof(float);
    }
    return 0u;
}

std::span<float> floats(twin::GuardedBuffer& buffer) {
    return {reinterpret_cast<float*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(float)};
}

template <class T>
std::span<T> objects(twin::GuardedBuffer& buffer) {
    return {reinterpret_cast<T*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(T)};
}

bool bytes_equal(std::span<const std::uint8_t> actual,
                 const void* expected, std::size_t bytes) {
    return actual.size() >= bytes &&
           std::memcmp(actual.data(), expected, bytes) == 0;
}

std::size_t q8_1_bytes(std::uint32_t rows) {
    return static_cast<std::size_t>(rows) * (N_EMBD / 32u) * sizeof(Q81);
}

struct Fixture {
    twin::GuardedBuffer params;
    std::array<twin::GuardedBuffer, kProjectionCount> storage;

    BufferRef view(ProjectionBuffer index) {
        return storage[index].view();
    }

    bool allocate(Context& context, const std::string& label) {
        std::string error;
        if (!params.allocate(context, label + " params", 256u, error)) {
            check(false, label + " params allocation", error);
            return false;
        }
        for (std::size_t i = 0u; i < storage.size(); ++i) {
            const auto index = static_cast<ProjectionBuffer>(i);
            if (!storage[i].allocate(context, label + " " + kNames[i],
                                     bytes_for(index), error)) {
                check(false, label + " " + kNames[i] + " allocation", error);
                return false;
            }
        }
        initialize();
        return true;
    }

    void initialize() {
        auto input = floats(storage[kMixed]);
        for (std::size_t i = 0u; i < input.size(); ++i) {
            const auto centered =
                static_cast<std::int32_t>((i * 31u + i / 257u) % 257u) - 128;
            input[i] = static_cast<float>(centered) * 0.00390625f;
        }
        for (const auto index : {kKeyWeights, kValueWeights}) {
            auto weights = objects<block_q8_0>(storage[index]);
            const int projection_bias = index == kKeyWeights ? 0 : 5;
            for (std::size_t block = 0u; block < weights.size(); ++block) {
                const float scale =
                    0.0625f +
                    static_cast<float>(
                        (block + static_cast<std::size_t>(index)) % 7u) *
                        0.03125f;
                weights[block].d =
                    strata::kernels::f16_from_f32(scale);
                for (std::size_t lane = 0u; lane < 32u; ++lane) {
                    const int centered = static_cast<int>(
                        (block * 13u + lane * 7u +
                         static_cast<std::size_t>(projection_bias)) %
                        31u) - 15;
                    weights[block].qs[lane] =
                        static_cast<std::int8_t>(centered);
                }
            }
        }
        set_rows(1u);
        reset_outputs();
    }

    void set_rows(std::uint32_t rows) {
        std::fill(params.payload().begin(), params.payload().end(), 0u);
        MtpParams block{};
        block.T = rows;
        std::memcpy(params.payload().data(), &block, sizeof(block));
    }

    void reset_outputs() {
        std::fill(storage[kXq].payload().begin(),
                  storage[kXq].payload().end(), 0xa5u);
        for (const auto index : {kKcur, kVcur}) {
            auto output = floats(storage[index]);
            std::fill(output.begin(), output.end(), kPoison);
        }
        std::fill(storage[kSink].payload().begin(),
                  storage[kSink].payload().end(), 0x5au);
    }

    MtpAttentionKvProjectionBuffers mtp_buffers(Context& context) {
        return {
            context.buffers().read_only(params.view()),
            view(kMixed),
            view(kXq),
            context.buffers().read_only(view(kKeyWeights)),
            context.buffers().read_only(view(kValueWeights)),
            view(kKcur),
            view(kVcur),
            view(kSink),
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

bool execute_rows(Context& context, Fixture& fixture, std::uint32_t rows) {
    const std::uint32_t compiled_columns =
        rows <= 1u ? 1u : rows <= 2u ? 2u : rows <= 4u ? 4u : 8u;
    fixture.set_rows(rows);
    fixture.reset_outputs();
    fixture.capture();

    const std::string label =
        "MTP stage-8 projection T=" + std::to_string(rows);
    auto buffers = fixture.mtp_buffers(context);
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;

    const bool encoded = encode_mtp_attention_kv_projection(
        context, *command, buffers, error);
    check(encoded, label + " encode", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(command, label, error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    std::vector<Q81> quantized;
    const auto input = std::span<const float>(floats(fixture.storage[kMixed]))
                           .first(static_cast<std::size_t>(rows) * N_EMBD);
    const bool quantized_ready = twin::quantize_q8_1_reference(
        input, N_EMBD, rows, quantized, error);
    check(quantized_ready, label + " Q8_1 scalar oracle", error);
    if (!quantized_ready) return false;
    const std::size_t activation_bytes = q8_1_bytes(rows);
    const bool quantized_exact =
        bytes_equal(fixture.storage[kXq].payload(), quantized.data(),
                    activation_bytes);
    check(quantized_exact, label + " Q8_1 rows match the scalar oracle");

    std::vector<Q81> matvec_activations(
        static_cast<std::size_t>(compiled_columns) * (N_EMBD / 32u));
    std::copy(quantized.begin(), quantized.end(),
              matvec_activations.begin());
    for (std::uint32_t row = rows; row < compiled_columns; ++row) {
        std::copy_n(quantized.begin(), N_EMBD / 32u,
                    matvec_activations.begin() +
                        static_cast<std::ptrdiff_t>(row) *
                            (N_EMBD / 32u));
    }
    const auto activations =
        std::span<const Q81>(matvec_activations);
    const std::size_t weight_count =
        static_cast<std::size_t>(kOutputWidth) * (N_EMBD / 32u);
    const auto key_weights =
        std::span<const block_q8_0>(objects<block_q8_0>(
            fixture.storage[kKeyWeights]).first(weight_count));
    const auto value_weights =
        std::span<const block_q8_0>(objects<block_q8_0>(
            fixture.storage[kValueWeights]).first(weight_count));
    std::vector<float> key_expected;
    std::vector<float> value_expected;
    bool oracle_ready = twin::q8_0_matvec_reference(
        key_weights, activations, N_EMBD, kOutputWidth, compiled_columns,
        rows, key_expected, error);
    check(oracle_ready, label + " key projection scalar oracle", error);
    if (!oracle_ready) return false;
    oracle_ready = twin::q8_0_matvec_reference(
        value_weights, activations, N_EMBD, kOutputWidth, compiled_columns,
        rows, value_expected, error);
    check(oracle_ready, label + " value projection scalar oracle", error);
    if (!oracle_ready) return false;

    const std::size_t active_output_bytes =
        static_cast<std::size_t>(rows) * kOutputWidth * sizeof(float);
    const bool key_exact =
        bytes_equal(fixture.storage[kKcur].payload(), key_expected.data(),
                    active_output_bytes);
    const bool value_exact =
        bytes_equal(fixture.storage[kVcur].payload(), value_expected.data(),
                    active_output_bytes);
    check(key_exact, label + " kcur matches the scalar oracle");
    check(value_exact, label + " vcur matches the scalar oracle");

    bool outputs_finite = true;
    for (const auto index : {kKcur, kVcur}) {
        const auto output = std::span<const float>(floats(fixture.storage[index]))
                                .first(static_cast<std::size_t>(rows) *
                                       kOutputWidth);
        outputs_finite &= std::all_of(
            output.begin(), output.end(),
            [](float value) { return std::isfinite(value); });
    }
    check(outputs_finite, label + " every active K/V result is finite");

    const std::size_t padded_columns = compiled_columns - rows;
    const std::size_t sink_bytes =
        padded_columns == 0u
            ? 0u
            : padded_columns * kOutputWidth * sizeof(float);
    bool sink_exact = true;
    if (sink_bytes != 0u) {
        const std::size_t active_count =
            static_cast<std::size_t>(rows) * kOutputWidth;
        sink_exact = bytes_equal(
            fixture.storage[kSink].payload(),
            value_expected.data() + active_count, sink_bytes);
    }
    check(sink_exact, label + " padded columns use the shared sink");

    const std::array<twin::ByteRange, 1> q8_written{{
        {0u, activation_bytes},
    }};
    const std::array<twin::ByteRange, 1> kcur_written{{
        {0u, active_output_bytes},
    }};
    const std::array<twin::ByteRange, 1> vcur_written{{
        {0u, active_output_bytes},
    }};
    const std::array<twin::ByteRange, 1> sink_written{{
        {0u, sink_bytes},
    }};
    const bool guards_ok =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.storage[kMixed].first_unwritten_change({}) &&
        !fixture.storage[kKeyWeights].first_unwritten_change({}) &&
        !fixture.storage[kValueWeights].first_unwritten_change({}) &&
        !fixture.storage[kXq].first_unwritten_change(q8_written) &&
        !fixture.storage[kKcur].first_unwritten_change(kcur_written) &&
        !fixture.storage[kVcur].first_unwritten_change(vcur_written) &&
        !fixture.storage[kSink].first_unwritten_change(sink_written);
    check(guards_ok, label + " input immutability, inactive tails, and guards");
    return quantized_exact && key_exact && value_exact && outputs_finite &&
           sink_exact && guards_ok;
}

bool reject_case(Context& context,
                 const MtpAttentionKvProjectionBuffers& buffers,
                 const std::string& label,
                 const std::string& expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_attention_kv_projection(context, *command, buffers, error);
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
    fixture.set_rows(1u);
    auto buffers = fixture.mtp_buffers(context);

    fixture.set_rows(0u);
    passed &= reject_case(context, buffers, "MTP stage 8 rejects T=0",
                          "T must be in 1..8");

    fixture.set_rows(8u);
    buffers = fixture.mtp_buffers(context);
    buffers.key_weights = context.buffers().read_only(
        fixture.view(kKeyWeights).slice(0u, bytes_for(kKeyWeights) - 1u));
    passed &= reject_case(context, buffers,
                          "MTP stage 8 rejects short key weights",
                          "key weights buffer is undersized");

    buffers = fixture.mtp_buffers(context);
    buffers.kcur = buffers.vcur;
    passed &= reject_case(context, buffers,
                          "MTP stage 8 rejects overlapping K/V outputs",
                          "kcur and vcur overlap");

    fixture.set_rows(5u);
    buffers = fixture.mtp_buffers(context);
    buffers.sink = fixture.view(kSink).slice(0u, sizeof(float));
    passed &= reject_case(context, buffers,
                          "MTP stage 8 rejects a short padded sink",
                          "sink buffer is undersized");

    fixture.set_rows(1u);
    MtpParams aliased_params{};
    aliased_params.T = 1u;
    std::memcpy(fixture.view(kXq).contents, &aliased_params,
                sizeof(aliased_params));
    buffers = fixture.mtp_buffers(context);
    buffers.params = context.buffers().read_only(fixture.view(kXq));
    passed &= reject_case(context, buffers,
                          "MTP stage 8 rejects params/output alias",
                          "parameter block overlaps xq");

    twin::GuardedBuffer foreign_weights;
    std::string error;
    const bool foreign_allocated = foreign_weights.allocate(
        foreign_context, "foreign MTP K projection", bytes_for(kKeyWeights),
        error);
    check(foreign_allocated, "foreign MTP K/V weights allocation", error);
    if (foreign_allocated) {
        buffers = fixture.mtp_buffers(context);
        buffers.key_weights =
            foreign_context.buffers().read_only(foreign_weights.view());
        passed &= reject_case(context, buffers,
                              "MTP stage 8 rejects foreign weights",
                              "key weights buffer is not owned");
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

    fixture.set_rows(1u);
    fixture.reset_outputs();
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP K/V projection group boundary", error);
    if (!command) {
        check(false, "MTP projection concurrent-group command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers(context);
    const bool wrapper_checked = Encoder::encode(
        context, *command, "MTP projection group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected = !encode_mtp_attention_kv_projection(
                context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP K/V projection accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged = fixture.unchanged();
    check(wrapper_checked,
          "MTP K/V projection rejects an open concurrent group", error);
    check(unchanged,
          "MTP projection open-group rejection preserves every buffer");
    command->discard();
    return wrapper_checked && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.set_rows(5u);
    fixture.reset_outputs();
    fixture.capture();
    const auto good = fixture.mtp_buffers(context);
    auto bad = good;
    bad.kcur = bad.vcur;

    std::string error;
    auto command =
        context.make_command_buffer("MTP K/V projection caller rollback", error);
    if (!command) {
        check(false, "MTP projection rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP K/V projection caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_attention_kv_projection(context, encoder, good,
                                                   encode_error)) {
                return false;
            }
            return encode_mtp_attention_kv_projection(context, encoder, bad,
                                                      encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged = fixture.unchanged();
    check(rejected && discarded,
          "MTP projection caller-owned late failure discards segment", error);
    check(unchanged,
          "MTP projection caller rollback leaves every buffer unchanged");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_kv_projection_test <strata.metallib>\n");
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
    check(fixture.allocate(*context, "MTP K/V projection"),
          "allocate MTP stage-8 fixture");
    if (!fixture.view(kMixed)) return 1;

    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= kMaxRows; ++rows) {
        passed &= execute_rows(*context, fixture, rows);
    }
    passed &= test_concurrent_group_boundary(*context, fixture);
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_late_rollback(*context, fixture);
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_kv_projection_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_kv_projection_test: MTP stage 8 OK");
    return 0;
}
