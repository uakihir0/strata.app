// Full-grid numeric and ABI checks for the Metal common helpers (docs/mac/04 §3 and §10.5).
#include "strata/metal/context.hpp"
#include "../../src/metal/engine/metal_fence.hpp"
#include "strata/kernels/f16_bits.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

struct SelftestMathParams {
    std::uint32_t count;
    std::uint32_t operation;
    float lower;
    float upper;
};
static_assert(sizeof(SelftestMathParams) == 16);

struct Dot4Input {
    std::int32_t a;
    std::int32_t b;
};
static_assert(sizeof(Dot4Input) == 8);

constexpr std::uint32_t kGridCount = 1u << 24;
constexpr std::uint32_t kGridChunk = 1u << 20;
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kPartialLanes = 8;
int g_failures = 0;

void check(bool ok, const char* label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", label,
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !ok;
}

BufferRef allocate(Context& context, const std::string& label, std::size_t bytes, std::string& err) {
    auto buffer = context.buffers().allocate(label, bytes, err, true);
    check(static_cast<bool>(buffer), label.c_str(), err);
    return buffer;
}

bool dispatch(Context& context, const char* name, BufferRef input, BufferRef output,
              const SelftestMathParams& params, std::uint32_t threads, std::uint32_t threads_per_group,
              std::string& err) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(name);
    if (pipeline == nil) {
        err = std::string("missing pipeline ") + name;
        return false;
    }
    auto command = context.make_command_buffer(name, err);
    if (!command) return false;
    id<MTLCommandBuffer> native_command =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native_command);
    if (encoder == nil) {
        command->discard();
        err = std::string("cannot create encoder for ") + name;
        return false;
    }
    encoder.label = [NSString stringWithUTF8String:name];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input.handle offset:input.offset atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output.handle offset:output.offset atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    end_compute_encoder(context, encoder);
    if (!command->commit(err)) return false;
    [native_command waitUntilCompleted];
    if (native_command.status != MTLCommandBufferStatusCompleted) {
        const char* message = native_command.error.localizedDescription.UTF8String;
        err = std::string(name) + " command buffer failed";
        if (message != nullptr && message[0] != '\0') err += ": " + std::string(message);
        return false;
    }
    err.clear();
    return true;
}

bool dispatch_partials(Context& context, BufferRef input, BufferRef output,
                       const SelftestMathParams& params, std::uint32_t groups, std::string& err) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline("strata_selftest_partials");
    if (pipeline == nil) {
        err = "missing pipeline strata_selftest_partials";
        return false;
    }
    auto command = context.make_command_buffer("strata_selftest_partials", err);
    if (!command) return false;
    id<MTLCommandBuffer> native_command =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native_command);
    if (encoder == nil) {
        command->discard();
        err = "cannot create partial-sum encoder";
        return false;
    }
    encoder.label = @"strata selftest partial sum";
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input.handle offset:input.offset atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output.handle offset:output.offset atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder setThreadgroupMemoryLength:kPartialLanes * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kPartialLanes, 1, 1)];
    end_compute_encoder(context, encoder);
    if (!command->commit(err)) return false;
    [native_command waitUntilCompleted];
    if (native_command.status != MTLCommandBufferStatusCompleted) {
        const char* message = native_command.error.localizedDescription.UTF8String;
        err = "partial-sum command buffer failed";
        if (message != nullptr && message[0] != '\0') err += ": " + std::string(message);
        return false;
    }
    err.clear();
    return true;
}

float accurate_exp(float x) {
    return static_cast<float>(std::exp(static_cast<double>(x)));
}

float accurate_log(float x) {
    return static_cast<float>(std::log(static_cast<double>(x)));
}

float oracle_log1p(float x) {
    const float u = 1.0f + x;
    return u == 1.0f ? x : accurate_log(u) * (x / (u - 1.0f));
}

float math_oracle(std::uint32_t operation, float x) {
    switch (operation) {
        case 0: return accurate_exp(x);
        case 1: return 1.0f / (1.0f + accurate_exp(-x));
        case 2: return x / (1.0f + accurate_exp(-x));
        case 3: return x > 20.0f ? x : oracle_log1p(accurate_exp(x));
        case 4: return oracle_log1p(x);
        default: return std::numeric_limits<float>::quiet_NaN();
    }
}

bool test_math_grids(Context& context) {
    struct Spec {
        const char* name;
        std::uint32_t operation;
        float lower;
        float upper;
        double n;
    };
    constexpr std::array<Spec, 5> specs = {{
        {"exp", 0, -30.0f, 30.0f, 2.0},
        {"sigmoid_k", 1, -30.0f, 30.0f, 8.0},
        {"silu_k", 2, -30.0f, 30.0f, 8.0},
        {"softplus_k", 3, -30.0f, 30.0f, 10.0},
        {"log1p_k", 4, -0.999f, 100.0f, 6.0},
    }};
    std::string err;
    BufferRef input = allocate(context, "math grid input", kGridChunk * sizeof(float), err);
    BufferRef output = allocate(context, "math grid output", kGridChunk * sizeof(float), err);
    if (!input || !output) return false;

    bool all_ok = true;
    for (const Spec& spec : specs) {
        double worst_fraction = 0.0;
        std::uint32_t worst_index = 0;
        bool ok = true;
        for (std::uint32_t base = 0; base < kGridCount; base += kGridChunk) {
            const std::uint32_t count = std::min(kGridChunk, kGridCount - base);
            auto* x = static_cast<float*>(input.contents);
            auto* y = static_cast<float*>(output.contents);
            for (std::uint32_t j = 0; j < count; ++j) {
                const std::uint32_t index = base + j;
                const float fraction = static_cast<float>(
                    static_cast<double>(index) / static_cast<double>(kGridCount - 1));
                x[j] = spec.lower + (spec.upper - spec.lower) * fraction;
            }
            std::memset(y, 0xff, count * sizeof(float));
            const SelftestMathParams params{count, spec.operation, spec.lower, spec.upper};
            if (!dispatch(context, "strata_selftest_math", input, output, params, count, kThreads, err)) {
                check(false, spec.name, err);
                return false;
            }
            for (std::uint32_t j = 0; j < count; ++j) {
                const std::uint32_t index = base + j;
                const float expected = math_oracle(spec.operation, x[j]);
                const double actual = static_cast<double>(y[j]);
                const double reference = static_cast<double>(expected);
                if (!std::isfinite(actual) || !std::isfinite(reference)) {
                    ok = false;
                    worst_index = index;
                    break;
                }
                const double tolerance =
                    spec.n * std::ldexp(std::fabs(reference), -24) + std::ldexp(1.0, -126);
                const double error = std::fabs(actual - reference);
                const double fraction = error / tolerance;
                if (fraction > worst_fraction) {
                    worst_fraction = fraction;
                    worst_index = index;
                }
                if (error > tolerance) ok = false;
            }
        }
        char detail[192];
        std::snprintf(detail, sizeof(detail), "2^24 inputs; worst tolerance fraction %.6g at index %u",
                      worst_fraction, worst_index);
        check(ok, spec.name, detail);
        all_ok &= ok;
    }
    return all_ok;
}

bool test_bf16_to_f32(Context& context) {
    constexpr std::uint32_t count = 1u << 16;
    std::string err;
    BufferRef input = allocate(context, "bf16 input", count * sizeof(std::uint16_t), err);
    BufferRef output = allocate(context, "bf16 widened output", count * sizeof(std::uint32_t), err);
    if (!input || !output) return false;
    auto* values = static_cast<std::uint16_t*>(input.contents);
    for (std::uint32_t i = 0; i < count; ++i) values[i] = static_cast<std::uint16_t>(i);
    const SelftestMathParams params{count, 0, 0.0f, 0.0f};
    if (!dispatch(context, "strata_selftest_bf16_to_f32", input, output, params, count, kThreads, err)) {
        check(false, "bf16_to_f32 dispatch", err);
        return false;
    }
    const auto* actual = static_cast<const std::uint32_t*>(output.contents);
    bool ok = true;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (actual[i] != (i << 16)) {
            ok = false;
            std::printf("bf16_to_f32 differs at 0x%04x: got 0x%08x, want 0x%08x\n",
                        i, actual[i], i << 16);
            break;
        }
    }
    check(ok, "bf16_to_f32 exact over all 65,536 bit patterns");
    return ok;
}

bool test_f32_to_bf16(Context& context) {
    constexpr std::array<std::uint16_t, 6> low_bits = {0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff};
    constexpr std::uint32_t count = (1u << 16) * static_cast<std::uint32_t>(low_bits.size());
    std::vector<float> inputs;
    inputs.reserve(count);
    for (std::uint32_t high = 0; high < (1u << 16); ++high) {
        for (std::uint16_t low : low_bits) {
            inputs.push_back(std::bit_cast<float>((high << 16) | low));
        }
    }
    std::string err;
    BufferRef input = allocate(context, "f32-to-bf16 input", inputs.size() * sizeof(float), err);
    BufferRef output = allocate(context, "f32-to-bf16 output", inputs.size() * sizeof(std::uint16_t), err);
    if (!input || !output) return false;
    std::memcpy(input.contents, inputs.data(), inputs.size() * sizeof(float));
    const SelftestMathParams params{count, 0, 0.0f, 0.0f};
    if (!dispatch(context, "strata_selftest_f32_to_bf16", input, output, params, count, kThreads, err)) {
        check(false, "f32_to_bf16_rne dispatch", err);
        return false;
    }
    const auto* actual = static_cast<const std::uint16_t*>(output.contents);
    bool ok = true;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(inputs[i]);
        const std::uint32_t upper = bits >> 16;
        const bool nan = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
        const std::uint16_t expected = static_cast<std::uint16_t>(
            nan ? upper | 64u : (bits + ((bits >> 16) & 1u) + 0x7fffu) >> 16);
        if (actual[i] != expected) {
            ok = false;
            std::printf("f32_to_bf16_rne differs at input 0x%08x: got 0x%04x, want 0x%04x\n",
                        bits, actual[i], expected);
            break;
        }
    }
    check(ok, "f32_to_bf16_rne ties, carry, subnormal, infinity and NaN cases");
    return ok;
}

std::uint16_t oracle_half_sat(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) return 0x7e00;
    if ((bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0) return 0;
    const float clamped = std::clamp(value, -65504.0f, 65504.0f);
    return strata::kernels::f16_from_f32(clamped);
}

bool test_half_sat(Context& context) {
    std::vector<float> inputs;
    inputs.reserve(500000);
    for (std::uint32_t half_bits = 0; half_bits < 0x7c00u; ++half_bits) {
        const float lower = strata::kernels::f32_from_f16(static_cast<std::uint16_t>(half_bits));
        inputs.push_back(lower);
        if (half_bits + 1 < 0x7c00u) {
            const float upper = strata::kernels::f32_from_f16(static_cast<std::uint16_t>(half_bits + 1));
            const float midpoint = static_cast<float>((static_cast<double>(lower) + upper) * 0.5);
            const float around[] = {
                std::nextafter(midpoint, -std::numeric_limits<float>::infinity()),
                midpoint,
                std::nextafter(midpoint, std::numeric_limits<float>::infinity()),
            };
            for (float value : around) {
                inputs.push_back(value);
                inputs.push_back(-value);
            }
        }
    }
    const std::uint32_t special_bits[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu,
        0x00800000u, 0x80800000u, 0x477fe000u, 0xc77fe000u, 0x477fe001u, 0xc77fe001u,
        0x477ff000u, 0xc77ff000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u,
        0x7f800001u, 0xff800001u,
    };
    for (std::uint32_t bits : special_bits) inputs.push_back(std::bit_cast<float>(bits));

    std::string err;
    BufferRef input = allocate(context, "half_sat input", inputs.size() * sizeof(float), err);
    BufferRef output = allocate(context, "half_sat output", inputs.size() * sizeof(std::uint16_t), err);
    if (!input || !output) return false;
    std::memcpy(input.contents, inputs.data(), inputs.size() * sizeof(float));
    const SelftestMathParams params{static_cast<std::uint32_t>(inputs.size()), 0, 0.0f, 0.0f};
    if (!dispatch(context, "strata_selftest_half_sat", input, output, params, params.count, kThreads, err)) {
        check(false, "half_sat dispatch", err);
        return false;
    }
    const auto* actual = static_cast<const std::uint16_t*>(output.contents);
    bool ok = true;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::uint16_t expected = oracle_half_sat(inputs[i]);
        if (actual[i] != expected) {
            ok = false;
            std::printf("half_sat differs at %zu (f32 0x%08x): got 0x%04x, want 0x%04x\n",
                        i, std::bit_cast<std::uint32_t>(inputs[i]), actual[i], expected);
            break;
        }
    }
    check(ok, "half_sat RNE, saturation, subnormal, signed zero and NaN cases");
    return ok;
}

bool test_round_half_away(Context& context) {
    std::vector<float> inputs;
    inputs.reserve(70000);
    for (int integer = -10000; integer <= 10000; ++integer) {
        const float tie = static_cast<float>(integer) + 0.5f;
        inputs.push_back(std::nextafter(tie, -std::numeric_limits<float>::infinity()));
        inputs.push_back(tie);
        inputs.push_back(std::nextafter(tie, std::numeric_limits<float>::infinity()));
    }
    for (float value : {-0.0f, 0.0f, -0.5f, 0.5f, -1.5f, 1.5f, -1000000.0f, 1000000.0f}) {
        inputs.push_back(value);
    }
    std::string err;
    BufferRef input = allocate(context, "round input", inputs.size() * sizeof(float), err);
    BufferRef output = allocate(context, "round output", inputs.size() * sizeof(std::int32_t), err);
    if (!input || !output) return false;
    std::memcpy(input.contents, inputs.data(), inputs.size() * sizeof(float));
    const SelftestMathParams params{static_cast<std::uint32_t>(inputs.size()), 0, 0.0f, 0.0f};
    if (!dispatch(context, "strata_selftest_round_half_away", input, output, params, params.count, kThreads, err)) {
        check(false, "round_half_away dispatch", err);
        return false;
    }
    const auto* actual = static_cast<const std::int32_t*>(output.contents);
    bool ok = true;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto expected = static_cast<std::int32_t>(std::round(inputs[i]));
        if (actual[i] != expected) {
            ok = false;
            std::printf("round_half_away differs at %zu: got %d, want %d\n", i, actual[i], expected);
            break;
        }
    }
    check(ok, "round_half_away around 20,001 positive and negative ties");
    return ok;
}

template <typename T, typename Op>
std::array<T, 32> butterfly_reference(std::array<T, 32> values, Op op) {
    for (std::uint32_t offset = 16; offset > 0; offset >>= 1) {
        const auto previous = values;
        for (std::uint32_t lane = 0; lane < 32; ++lane) {
            values[lane] = op(previous[lane], previous[lane ^ offset]);
        }
    }
    return values;
}

bool test_butterflies(Context& context) {
    constexpr std::uint32_t groups = 64;
    constexpr std::uint32_t count = groups * 32;
    std::mt19937 rng(0x51a7u);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> inputs(count);
    std::vector<std::int32_t> int_inputs(count);
    for (std::uint32_t group = 0; group < groups; ++group) {
        for (std::uint32_t lane = 0; lane < 32; ++lane) {
            float value = dist(rng);
            if (group % 4 == 0) {
                constexpr float pattern[] = {1.0e20f, 1.0f, -1.0e20f, -0.0f, 0.0f, 3.0f, -3.0f, 0x1p-20f};
                value = pattern[lane % (sizeof(pattern) / sizeof(pattern[0]))];
            }
            inputs[group * 32 + lane] = value;
            int_inputs[group * 32 + lane] =
                (group % 2 == 0) ? static_cast<std::int32_t>((lane * 313u) ^ (group * 65537u))
                                 : std::numeric_limits<std::int32_t>::min() + static_cast<int>(lane);
        }
    }
    std::string err;
    BufferRef input = allocate(context, "butterfly float input", count * sizeof(float), err);
    BufferRef output = allocate(context, "butterfly float output", count * sizeof(float), err);
    if (!input || !output) return false;
    std::memcpy(input.contents, inputs.data(), count * sizeof(float));
    bool all_ok = true;
    for (std::uint32_t operation = 0; operation < 2; ++operation) {
        const SelftestMathParams params{count, operation, 0.0f, 0.0f};
        if (!dispatch(context, "strata_selftest_bfly", input, output, params, count, kThreads, err)) {
            check(false, operation == 0 ? "bfly_sum dispatch" : "bfly_max dispatch", err);
            return false;
        }
        const auto* actual = static_cast<const float*>(output.contents);
        bool ok = true;
        for (std::uint32_t group = 0; group < groups && ok; ++group) {
            std::array<float, 32> values;
            std::copy_n(inputs.data() + group * 32, 32, values.data());
            const auto expected = butterfly_reference(values, [operation](float a, float b) {
                return operation == 0 ? a + b : std::fmax(a, b);
            });
            for (std::uint32_t lane = 0; lane < 32; ++lane) {
                const std::uint32_t index = group * 32 + lane;
                if (std::bit_cast<std::uint32_t>(actual[index]) !=
                    std::bit_cast<std::uint32_t>(expected[lane])) {
                    ok = false;
                    std::printf("butterfly %s differs at %u: got 0x%08x, want 0x%08x\n",
                                operation == 0 ? "sum" : "max", index,
                                std::bit_cast<std::uint32_t>(actual[index]),
                                std::bit_cast<std::uint32_t>(expected[lane]));
                    break;
                }
            }
        }
        check(ok, operation == 0 ? "bfly_sum explicit xor order" : "bfly_max explicit xor order");
        all_ok &= ok;
    }

    BufferRef int_input = allocate(context, "butterfly int input", count * sizeof(std::int32_t), err);
    BufferRef int_output = allocate(context, "butterfly int output", count * sizeof(std::int32_t), err);
    if (!int_input || !int_output) return false;
    std::memcpy(int_input.contents, int_inputs.data(), count * sizeof(std::int32_t));
    const SelftestMathParams int_params{count, 0, 0.0f, 0.0f};
    if (!dispatch(context, "strata_selftest_bfly_int_max", int_input, int_output,
                  int_params, count, kThreads, err)) {
        check(false, "bfly_max(int) dispatch", err);
        return false;
    }
    const auto* actual_int = static_cast<const std::int32_t*>(int_output.contents);
    bool int_ok = true;
    for (std::uint32_t group = 0; group < groups && int_ok; ++group) {
        std::array<std::int32_t, 32> values;
        std::copy_n(int_inputs.data() + group * 32, 32, values.data());
        const auto expected = butterfly_reference(values, [](std::int32_t a, std::int32_t b) {
            return std::max(a, b);
        });
        for (std::uint32_t lane = 0; lane < 32; ++lane) {
            const std::uint32_t index = group * 32 + lane;
            if (actual_int[index] != expected[lane]) {
                int_ok = false;
                std::printf("bfly_max(int) differs at %u: got %d, want %d\n",
                            index, actual_int[index], expected[lane]);
                break;
            }
        }
    }
    check(int_ok, "bfly_max(int) exact");
    return all_ok && int_ok;
}

bool test_partial_sums(Context& context) {
    constexpr std::uint32_t groups = 32;
    constexpr std::uint32_t input_count = groups * kPartialLanes;
    std::vector<float> inputs(input_count);
    for (std::uint32_t i = 0; i < input_count; ++i) {
        inputs[i] = (i % 11 == 0) ? -0.0f : static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    std::string err;
    BufferRef input = allocate(context, "partial sums input", input_count * sizeof(float), err);
    BufferRef output = allocate(context, "partial sums output", groups * sizeof(float), err);
    if (!input || !output) return false;
    std::memcpy(input.contents, inputs.data(), input_count * sizeof(float));
    bool all_ok = true;
    for (std::uint32_t active = 0; active <= kPartialLanes; ++active) {
        const SelftestMathParams params{input_count, active, 0.0f, 0.0f};
        if (!dispatch_partials(context, input, output, params, groups, err)) {
            check(false, "sum_partials_in_order dispatch", err);
            return false;
        }
        const auto* actual = static_cast<const float*>(output.contents);
        bool ok = true;
        for (std::uint32_t group = 0; group < groups; ++group) {
            float expected = active == 0 ? 0.0f : inputs[group * kPartialLanes];
            for (std::uint32_t i = 1; i < active; ++i) expected += inputs[group * kPartialLanes + i];
            if (std::bit_cast<std::uint32_t>(actual[group]) != std::bit_cast<std::uint32_t>(expected)) {
                ok = false;
                std::printf("sum_partials count %u group %u: got 0x%08x, want 0x%08x\n",
                            active, group, std::bit_cast<std::uint32_t>(actual[group]),
                            std::bit_cast<std::uint32_t>(expected));
                break;
            }
        }
        all_ok &= ok;
        if (!ok) break;
    }
    check(all_ok, "sum_partials_in_order counts 0..8 and signed-zero preservation");
    return all_ok;
}

std::int32_t dot4_reference(std::int32_t a, std::int32_t b) {
    std::int32_t result = 0;
    const std::uint32_t ua = std::bit_cast<std::uint32_t>(a);
    const std::uint32_t ub = std::bit_cast<std::uint32_t>(b);
    for (unsigned i = 0; i < 4; ++i) {
        const auto ax = std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(ua >> (8 * i)));
        const auto bx = std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(ub >> (8 * i)));
        result += static_cast<std::int32_t>(ax) * static_cast<std::int32_t>(bx);
    }
    return result;
}

bool test_dot4_i8(Context& context) {
    constexpr std::uint32_t count = 8192;
    std::mt19937 rng(0xd047a8u);
    std::uniform_int_distribution<std::uint32_t> word;
    std::vector<Dot4Input> inputs(count);
    for (auto& item : inputs) item = {std::bit_cast<std::int32_t>(word(rng)),
                                      std::bit_cast<std::int32_t>(word(rng))};
    inputs[0] = {std::bit_cast<std::int32_t>(0x807f01ffu), std::bit_cast<std::int32_t>(0x7f80fe02u)};
    inputs[1] = {std::bit_cast<std::int32_t>(0x80808080u), std::bit_cast<std::int32_t>(0x7f7f7f7fu)};
    std::string err;
    BufferRef input = allocate(context, "dot4 input", count * sizeof(Dot4Input), err);
    BufferRef output = allocate(context, "dot4 output", count * sizeof(std::int32_t), err);
    if (!input || !output) return false;
    std::memcpy(input.contents, inputs.data(), count * sizeof(Dot4Input));
    const SelftestMathParams params{count, 0, 0.0f, 0.0f};
    if (!dispatch(context, "strata_selftest_dot4_i8", input, output, params, count, kThreads, err)) {
        check(false, "dot4_i8 dispatch", err);
        return false;
    }
    const auto* actual = static_cast<const std::int32_t*>(output.contents);
    bool ok = true;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::int32_t expected = dot4_reference(inputs[i].a, inputs[i].b);
        if (actual[i] != expected) {
            ok = false;
            std::printf("dot4_i8 differs at %u: got %d, want %d\n", i, actual[i], expected);
            break;
        }
    }
    check(ok, "dot4_i8 signed-byte products and sums");
    return ok;
}

}  // namespace

int main() {
    ContextOptions options;
    std::string err;
    auto context = Context::open(options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;
    check(context->run_selftest(err), "selftest arithmetic and C++/MSL parameter ABI", err);
    if (!context->buffers().commit_residency(err)) {
        check(false, "commit math-test buffers", err);
        return 1;
    }
    test_math_grids(*context);
    test_bf16_to_f32(*context);
    test_f32_to_bf16(*context);
    test_half_sat(*context);
    test_round_half_away(*context);
    test_butterflies(*context);
    test_partial_sums(*context);
    test_dot4_i8(*context);
    std::printf("metal_math_test: %s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
