// Metal twins for weighted RMS norm and GR kernels (docs/mac/04 §5.4).
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/window_dispatch.hpp"
#include "strata/metal/window_pending_dispatch.hpp"
#include "strata/metal/window_head_mixer.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "twin.hpp"

#include "../../src/metal/kernels/common/params.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kThreads = 1024u;
constexpr std::uint32_t kGrNormThreads = 256u;
constexpr std::uint32_t kGrDownTileValues = 512u;
constexpr float kEpsilon = 1.0e-6f;

struct GRUpParams {
    std::uint32_t token_count;
    std::uint32_t apply;
};
static_assert(sizeof(GRUpParams) == 8u);

struct GRCvecParams {
    std::uint32_t layer;
    std::uint32_t token_count;
    std::uint32_t mode;
    std::uint32_t write;
};
static_assert(sizeof(GRCvecParams) == 16u);

struct SelftestMathParams {
    std::uint32_t count;
    std::uint32_t operation;
    float lower;
    float upper;
};
static_assert(sizeof(SelftestMathParams) == 16u);

struct FusedSensitiveValues {
    float candidate;
    float rounded_product;
    float fused_residual;
    float multiplier;
};

struct RMSNormCase {
    std::uint32_t n_cols;
    std::uint32_t gp;
    std::uint32_t rows;
};

constexpr std::array<RMSNormCase, 6> kCases{{
    {2560u, 4u, 4u},
    {2560u, 4u, 32u},
    {2560u, 1u, 1u},
    {2560u, 1u, 8u},
    {10240u, 1u, 1u},
    {10240u, 1u, 8u},
}};

int g_failures = 0;

void check(bool condition, const std::string& label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
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

bool probe_sigmoid(Context& context, float input, float& output) {
    const std::string label = "gr_pre_gated sigmoid probe";
    std::string err;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer output_buffer;
    check(input_buffer.allocate(context, label + " input", sizeof(float), err),
          label + " input buffer", err);
    check(output_buffer.allocate(context, label + " output", sizeof(float), err),
          label + " output buffer", err);
    if (!input_buffer.view() || !output_buffer.view()) return false;
    std::memcpy(input_buffer.payload().data(), &input, sizeof(input));
    std::memset(output_buffer.payload().data(), 0xff, sizeof(float));
    input_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threads;
    work.grid = {1u, 1u, 1u};
    work.threads_per_group = {0u, 0u, 0u};
    work.buffers = {{0u, input_buffer.view()}, {1u, output_buffer.view()}};
    const SelftestMathParams params{1u, 1u, 0.0f, 0.0f};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 2u;
    const bool dispatched =
        twin::dispatch_manifest(context, "strata_selftest_math", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;
    std::memcpy(&output, output_buffer.payload().data(), sizeof(output));
    check(std::isfinite(output) && output > 0.0f && output < 1.0f,
          label + " finite sigmoid", std::to_string(output));
    check(!input_buffer.first_unwritten_change({}),
          label + " input remains read-only");
    const std::array<twin::ByteRange, 1> written{{{0u, sizeof(float)}}};
    const auto changed = output_buffer.first_unwritten_change(written);
    check(!changed, label + " output guards remain unchanged",
          changed ? changed->reason : "");
    return std::isfinite(output) && output > 0.0f && output < 1.0f;
}

bool choose_fused_sensitive_values(float multiplier,
                                   FusedSensitiveValues& values) {
    if (!std::isfinite(multiplier) || multiplier <= 0.0f) {
        return false;
    }
    constexpr float target = 16777216.0f;
    const float center = target / multiplier;
    const std::int64_t center_bits =
        static_cast<std::int64_t>(std::bit_cast<std::uint32_t>(center));
    for (std::int64_t offset = -256; offset <= 256; ++offset) {
        const auto bits = static_cast<std::uint32_t>(center_bits + offset);
        const float candidate = std::bit_cast<float>(bits);
        if (!std::isfinite(candidate) || candidate <= 0.0f) continue;
        const float product = candidate * multiplier;
        if (!std::isfinite(product) || product < target * 0.5f) continue;
        const float residual = std::fma(candidate, multiplier, -product);
        if (std::fabs(residual) < 0.5f) continue;
        values = {candidate, product, residual, multiplier};
        return true;
    }
    return false;
}

float input_value(twin::InputSet set, std::uint32_t row, std::uint32_t col,
                  std::mt19937_64& rng) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 12> edge{
            0.0f, -0.0f, 0.25f, -0.25f, 0.5f, -0.5f,
            1.0f, -1.0f, 3.0f, -3.0f, 64.0f, -64.0f};
        return edge[(col + 3u * row) % edge.size()];
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t numerator =
            static_cast<std::int32_t>((col * 17u + row * 29u) % 1021u) - 510;
        return static_cast<float>(numerator) * (1.0f / 256.0f);
    }
    const std::int32_t numerator =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(numerator) * (1.0f / 8192.0f);
}

float gamma_value(twin::InputSet set, std::uint32_t row, std::uint32_t col,
                  std::mt19937_64& rng) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 8> edge{
            0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f, 1.5f, -2.0f};
        return edge[(col + 5u * row) % edge.size()];
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t numerator =
            static_cast<std::int32_t>((col * 31u + row * 47u) % 513u) - 256;
        return static_cast<float>(numerator) * (1.0f / 256.0f);
    }
    const std::int32_t numerator =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(numerator) * (1.0f / 65536.0f);
}

std::array<twin::tf, 32> butterfly(std::array<twin::tf, 32> values) {
    for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values;
}

std::array<float, 32> butterfly(std::array<float, 32> values) {
    for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values;
}

void rms_norm_reference(std::span<const float> input, std::span<const float> gamma,
                        std::uint32_t rows, std::uint32_t n_cols, std::uint32_t gp,
                        float eps, std::vector<twin::tf>& output) {
    output.resize(static_cast<std::size_t>(rows) * n_cols);
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::array<std::array<twin::tf, 32>, 32> warp_lanes{};
        const std::size_t row_offset = static_cast<std::size_t>(row) * n_cols;
        for (std::uint32_t tid = 0; tid < kThreads; ++tid) {
            twin::tf partial{};
            for (std::uint32_t col = tid; col < n_cols; col += kThreads) {
                const twin::tf value{input[row_offset + col]};
                partial = twin::fma(value, value, partial);
            }
            warp_lanes[tid / 32u][tid % 32u] = partial;
        }
        std::array<twin::tf, 32> sums{};
        for (std::uint32_t warp = 0; warp < sums.size(); ++warp) {
            sums[warp] = butterfly(warp_lanes[warp])[0];
        }
        const twin::tf sum = butterfly(sums)[0];
        const twin::tf mean = sum / twin::tf{static_cast<float>(n_cols)};
        const twin::tf variance = mean + twin::tf{eps};
        // The scalar value follows safe Metal's correctly rounded rsqrt contract.
        const twin::tf scale{static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(variance.value)))};
        const std::size_t gamma_offset = static_cast<std::size_t>(row % gp) * n_cols;
        for (std::uint32_t col = 0; col < n_cols; ++col) {
            output[row_offset + col] =
                (scale * twin::tf{input[row_offset + col]}) *
                twin::tf{gamma[gamma_offset + col]};
        }
    }
}

void gr_post_reference(std::span<const float> residual,
                       std::span<const float> block_out,
                       std::span<const float> inject, std::uint32_t rows,
                       std::vector<twin::tf>& output) {
    output.resize(static_cast<std::size_t>(rows) * HC_DIM);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t index = 0; index < HC_DIM; ++index) {
            const std::uint32_t stream = index / N_EMBD;
            const std::uint32_t col = index % N_EMBD;
            const twin::tf scaled_inject = twin::fma(
                twin::tf{0.25f}, twin::tf{inject[row * HC + stream]},
                twin::tf{});
            const twin::tf sigmoid =
                twin::tf{1.0f} / (twin::tf{1.0f} + twin::accurate_exp(-scaled_inject));
            const twin::tf weight =
                twin::fma(twin::tf{2.0f}, sigmoid, twin::tf{});
            const std::size_t residual_index =
                static_cast<std::size_t>(row) * HC_DIM + index;
            const std::size_t block_index =
                static_cast<std::size_t>(row) * N_EMBD + col;
            output[residual_index] = twin::fma(
                twin::tf{block_out[block_index]}, weight,
                twin::tf{residual[residual_index]});
        }
    }
}

void gr_pre_gated_reference(std::span<const float> xn,
                            std::span<const float> gate,
                            std::uint32_t rows,
                            std::vector<twin::tf>& output) {
    output.resize(static_cast<std::size_t>(rows) * N_EMBD);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < N_EMBD; ++col) {
            twin::tf sum{};
            for (std::uint32_t stream = 0; stream < HC; ++stream) {
                const std::size_t index =
                    static_cast<std::size_t>(row) * HC_DIM +
                    static_cast<std::size_t>(stream) * N_EMBD + col;
                const twin::tf sigmoid =
                    twin::tf{1.0f} /
                    (twin::tf{1.0f} + twin::accurate_exp(-twin::tf{gate[index]}));
                const twin::tf product = twin::tf{xn[index]} * sigmoid;
                sum = stream == 0u ? product : sum + product;
            }
            output[static_cast<std::size_t>(row) * N_EMBD + col] =
                twin::fma(twin::tf{0.25f}, sum, twin::tf{});
        }
    }
}

void gr_norm_reference(std::span<const float> residual,
                       std::span<const float> block_out,
                       std::span<const float> norm,
                       std::span<const float> inject_prev,
                       std::uint32_t rows, bool apply,
                       std::vector<twin::tf>& xn,
                       std::vector<twin::tf>& rs_out) {
    constexpr std::uint32_t vectors_per_stream = N_EMBD / 4u;
    xn.resize(static_cast<std::size_t>(rows) * HC_DIM);
    rs_out.resize(static_cast<std::size_t>(rows) * HC);

    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            twin::tf gate_weight{};
            if (apply) {
                const twin::tf injection{
                    inject_prev[static_cast<std::size_t>(row) * HC + stream]};
                const twin::tf scaled = injection / twin::tf{4.0f};
                const twin::tf sigmoid =
                    twin::tf{1.0f} /
                    (twin::tf{1.0f} + twin::accurate_exp(-scaled));
                gate_weight = twin::tf{2.0f} * sigmoid;
            }

            std::array<twin::tf, kGrNormThreads> thread_sums{};
            const std::size_t row_offset =
                static_cast<std::size_t>(row) * HC_DIM;
            for (std::uint32_t local_vector = 0;
                 local_vector < vectors_per_stream; ++local_vector) {
                const std::uint32_t thread =
                    (stream * vectors_per_stream + local_vector) %
                    kGrNormThreads;
                const std::size_t vector_offset =
                    row_offset + static_cast<std::size_t>(stream) * N_EMBD +
                    static_cast<std::size_t>(local_vector) * 4u;
                std::array<twin::tf, 4> value{};
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    value[component] =
                        twin::tf{residual[vector_offset + component]};
                }
                if (apply) {
                    const std::size_t block_offset =
                        static_cast<std::size_t>(row) * N_EMBD +
                        static_cast<std::size_t>(local_vector) * 4u;
                    for (std::uint32_t component = 0; component < 4u; ++component) {
                        value[component] = twin::fma(
                            twin::tf{block_out[block_offset + component]},
                            gate_weight, value[component]);
                    }
                }

                const twin::tf square_sum = twin::fma(
                    value[3], value[3],
                    twin::fma(value[2], value[2],
                              twin::fma(value[1], value[1],
                                        value[0] * value[0])));
                thread_sums[thread] = thread_sums[thread] + square_sum;

                const std::size_t norm_offset =
                    static_cast<std::size_t>(stream) * N_EMBD +
                    static_cast<std::size_t>(local_vector) * 4u;
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    xn[vector_offset + component] =
                        value[component] *
                        twin::tf{norm[norm_offset + component]};
                }
            }

            std::array<std::array<twin::tf, 32>, 8> lanes{};
            for (std::uint32_t thread = 0; thread < kGrNormThreads; ++thread) {
                lanes[thread / 32u][thread % 32u] = thread_sums[thread];
            }
            std::array<twin::tf, 8> warp_sums{};
            for (std::uint32_t warp = 0; warp < warp_sums.size(); ++warp) {
                warp_sums[warp] = butterfly(lanes[warp])[0];
            }
            twin::tf total{};
            for (const twin::tf& warp_sum : warp_sums) {
                total = total + warp_sum;
            }
            const twin::tf mean = total / twin::tf{static_cast<float>(N_EMBD)};
            const twin::tf variance = mean + twin::tf{1.0e-6f};
            const twin::tf scale{static_cast<float>(
                1.0 / std::sqrt(static_cast<double>(variance.value)))};
            rs_out[static_cast<std::size_t>(row) * HC + stream] = scale;

            for (std::uint32_t local_vector = 0;
                 local_vector < vectors_per_stream; ++local_vector) {
                const std::size_t vector_offset =
                    row_offset + static_cast<std::size_t>(stream) * N_EMBD +
                    static_cast<std::size_t>(local_vector) * 4u;
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    xn[vector_offset + component] =
                        xn[vector_offset + component] * scale;
                }
            }
        }
    }
}

void gr_down_reference(std::span<const float> xn,
                       std::span<const std::uint16_t> hc_down,
                       std::span<const std::uint16_t> hc_inject,
                       std::uint32_t rows,
                       std::vector<twin::tf>& lo,
                       std::vector<twin::tf>& inject) {
    lo.resize(static_cast<std::size_t>(rows) * HC_LOW_RANK);
    inject.resize(static_cast<std::size_t>(rows) * HC);
    constexpr std::uint32_t chunks_per_tile = kGrDownTileValues / 256u;

    for (std::uint32_t token = 0; token < rows; ++token) {
        for (std::uint32_t projection_row = 0;
             projection_row < HC_LOW_RANK + HC; ++projection_row) {
            const bool is_inject = projection_row >= HC_LOW_RANK;
            const std::uint32_t weight_row = is_inject
                ? projection_row - HC_LOW_RANK
                : projection_row;
            const std::span<const std::uint16_t> matrix =
                is_inject ? hc_inject : hc_down;
            const std::size_t weight_row_offset =
                static_cast<std::size_t>(weight_row) * HC_DIM;
            const std::size_t input_row_offset =
                static_cast<std::size_t>(token) * HC_DIM;
            std::array<float, 32> lane_accum{};
            double magnitude = 0.0;

            for (std::uint32_t base = 0; base < HC_DIM;
                 base += kGrDownTileValues) {
                for (std::uint32_t chunk = 0;
                     chunk < chunks_per_tile; ++chunk) {
                    for (std::uint32_t lane = 0; lane < 32u; ++lane) {
                        const std::uint32_t first_value =
                            base + (lane + 32u * chunk) * 8u;
                        float dot = 0.0f;
                        for (std::uint32_t pair = 0; pair < 4u; ++pair) {
                            const std::uint32_t value0 = first_value + 2u * pair;
                            const std::uint32_t value1 = value0 + 1u;
                            const float weight0 = strata::kernels::f32_from_bf16(
                                matrix[weight_row_offset + value0]);
                            const float weight1 = strata::kernels::f32_from_bf16(
                                matrix[weight_row_offset + value1]);
                            const float input0 = xn[input_row_offset + value0];
                            const float input1 = xn[input_row_offset + value1];
                            magnitude += std::fabs(
                                static_cast<double>(weight0) * input0);
                            magnitude += std::fabs(
                                static_cast<double>(weight1) * input1);
                            dot = std::fma(
                                weight0, input0, dot);
                            dot = std::fma(
                                weight1, input1, dot);
                        }
                        lane_accum[lane] = lane_accum[lane] + dot;
                    }
                }
            }

            const float sum = butterfly(lane_accum)[0];
            const twin::tf sum_reference{sum, magnitude};
            if (is_inject) {
                inject[static_cast<std::size_t>(token) * HC + weight_row] =
                    sum_reference;
            } else {
                const twin::tf scaled = sum_reference / twin::tf{4.0f};
                const twin::tf silu =
                    scaled / (twin::tf{1.0f} + twin::accurate_exp(-scaled));
                lo[static_cast<std::size_t>(token) * HC_LOW_RANK +
                   weight_row] = silu;
            }
        }
    }
}

void gr_up_reference(std::span<const float> residual,
                     std::span<const float> block_out,
                     std::span<const float> inject_prev,
                     std::span<const float> norm,
                     std::span<const float> rs_out,
                     std::span<const float> lo,
                     std::span<const std::uint16_t> hc_up,
                     std::uint32_t rows, bool apply,
                     std::vector<twin::tf>& expected_residual,
                     std::vector<twin::tf>& expected_mixed) {
    expected_residual.resize(static_cast<std::size_t>(rows) * HC_DIM);
    expected_mixed.resize(static_cast<std::size_t>(rows) * N_EMBD);
    for (std::size_t index = 0; index < expected_residual.size(); ++index) {
        expected_residual[index] = twin::tf{residual[index]};
    }

    std::array<std::array<twin::tf, 16>, HC> gated{};
    for (std::uint32_t token = 0; token < rows; ++token) {
        for (std::uint32_t column = 0; column < N_EMBD; ++column) {
            for (std::uint32_t stream = 0; stream < HC; ++stream) {
                const std::uint32_t weight_row =
                    stream * N_EMBD + column;
                const std::size_t weight_offset =
                    static_cast<std::size_t>(weight_row) * HC_LOW_RANK;
                std::array<twin::tf, 32> lane_sums{};

                for (std::uint32_t lane = 0; lane < 32u; ++lane) {
                    twin::tf first{};
                    twin::tf second{};
                    for (std::uint32_t pair = 0; pair < 4u; ++pair) {
                        const std::uint32_t first_index =
                            lane * 8u + pair * 2u;
                        const float first_weight =
                            strata::kernels::f32_from_bf16(
                                hc_up[weight_offset + first_index]);
                        const float second_weight =
                            strata::kernels::f32_from_bf16(
                                hc_up[weight_offset + first_index + 1u]);
                        const std::size_t lo_offset =
                            static_cast<std::size_t>(token) * HC_LOW_RANK +
                            first_index;
                        first = twin::fma(
                            twin::tf{first_weight},
                            twin::tf{lo[lo_offset]}, first);
                        first = twin::fma(
                            twin::tf{second_weight},
                            twin::tf{lo[lo_offset + 1u]}, first);
                    }

                    if (lane < 8u) {
                        for (std::uint32_t pair = 0; pair < 4u; ++pair) {
                            const std::uint32_t second_index =
                                (32u + lane) * 8u + pair * 2u;
                            const float first_weight =
                                strata::kernels::f32_from_bf16(
                                    hc_up[weight_offset + second_index]);
                            const float second_weight =
                                strata::kernels::f32_from_bf16(
                                    hc_up[weight_offset + second_index + 1u]);
                            const std::size_t lo_offset =
                                static_cast<std::size_t>(token) * HC_LOW_RANK +
                                second_index;
                            second = twin::fma(
                                twin::tf{first_weight},
                                twin::tf{lo[lo_offset]}, second);
                            second = twin::fma(
                                twin::tf{second_weight},
                                twin::tf{lo[lo_offset + 1u]}, second);
                        }
                        first = first + second;
                    }
                    lane_sums[lane] = first;
                }

                const twin::tf projection = butterfly(lane_sums)[token];
                const twin::tf sigmoid =
                    twin::tf{1.0f} /
                    (twin::tf{1.0f} +
                     twin::accurate_exp(-projection));
                const std::size_t residual_index =
                    static_cast<std::size_t>(token) * HC_DIM +
                    static_cast<std::size_t>(stream) * N_EMBD + column;
                twin::tf value{residual[residual_index]};
                if (apply) {
                    const twin::tf injection{
                        inject_prev[static_cast<std::size_t>(token) * HC +
                                    stream]};
                    const twin::tf gate_weight =
                        twin::tf{2.0f} *
                        (twin::tf{1.0f} /
                         (twin::tf{1.0f} +
                          twin::accurate_exp(-(injection / twin::tf{4.0f}))));
                    value = twin::fma(
                        twin::tf{block_out[
                            static_cast<std::size_t>(token) * N_EMBD +
                            column]},
                        gate_weight, value);
                    expected_residual[residual_index] = value;
                }

                const twin::tf normalized =
                    (value * twin::tf{norm[stream * N_EMBD + column]}) *
                    twin::tf{rs_out[token * HC + stream]};
                gated[stream][column % 16u] =
                    normalized * sigmoid;
            }

            const std::uint32_t tile_column = column % 16u;
            twin::tf sum{};
            for (std::uint32_t stream = 0; stream < HC; ++stream) {
                sum = sum + gated[stream][tile_column];
            }
            expected_mixed[static_cast<std::size_t>(token) * N_EMBD +
                           column] = sum / twin::tf{4.0f};
        }
    }
}

void gr_cvec_reference(std::span<const float> residual,
                       std::span<const float> direction,
                       std::span<const float> layer_scale,
                       std::span<const float> block_out,
                       std::span<const float> inject_prev,
                       std::uint32_t layer, std::uint32_t rows,
                       std::uint32_t mode, bool on, bool write,
                       std::vector<twin::tf>& expected) {
    expected.resize(residual.size());
    for (std::size_t index = 0; index < residual.size(); ++index) {
        expected[index] = twin::tf{residual[index]};
    }

    const float scale_value = layer_scale[layer];
    const bool steer = on && scale_value != 0.0f;
    if (!steer && !write) return;
    const std::size_t direction_offset =
        static_cast<std::size_t>(layer) * N_EMBD;

    for (std::uint32_t token = 0; token < rows; ++token) {
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            twin::tf write_weight{};
            if (write) {
                const twin::tf scaled =
                    twin::tf{inject_prev[token * HC + stream]} /
                    twin::tf{static_cast<float>(HC)};
                const twin::tf sigmoid =
                    twin::tf{1.0f} /
                    (twin::tf{1.0f} + twin::accurate_exp(-scaled));
                write_weight = twin::tf{2.0f} * sigmoid;
            }

            std::array<std::array<twin::tf, 10>, 256> values{};
            std::array<twin::tf, 256> thread_dots{};
            for (std::uint32_t thread = 0; thread < 256u; ++thread) {
                twin::tf dot{};
                for (std::uint32_t k = 0; k < 10u; ++k) {
                    const std::uint32_t column = thread + 256u * k;
                    const std::size_t residual_index =
                        static_cast<std::size_t>(token) * HC_DIM +
                        static_cast<std::size_t>(stream) * N_EMBD + column;
                    twin::tf value{residual[residual_index]};
                    if (write) {
                        value = twin::fma(
                            twin::tf{block_out[
                                static_cast<std::size_t>(token) * N_EMBD +
                                column]},
                            write_weight, value);
                    }
                    values[thread][k] = value;
                    if (steer && mode == 0u) {
                        dot = twin::fma(
                            value,
                            twin::tf{direction[direction_offset + column]},
                            dot);
                    }
                }
                thread_dots[thread] = dot;
            }

            twin::tf dot{};
            if (steer && mode == 0u) {
                std::array<twin::tf, 32> warp_partials{};
                for (std::uint32_t warp = 0; warp < 8u; ++warp) {
                    std::array<twin::tf, 32> lanes{};
                    for (std::uint32_t lane = 0; lane < 32u; ++lane) {
                        lanes[lane] = thread_dots[warp * 32u + lane];
                    }
                    warp_partials[warp] = butterfly(lanes)[0];
                }
                std::array<twin::tf, 32> partial_lanes{};
                for (std::uint32_t warp = 0; warp < 8u; ++warp) {
                    partial_lanes[warp] = warp_partials[warp];
                }
                dot = butterfly(partial_lanes)[0] *
                    twin::tf{scale_value};
            }

            for (std::uint32_t thread = 0; thread < 256u; ++thread) {
                for (std::uint32_t k = 0; k < 10u; ++k) {
                    const std::uint32_t column = thread + 256u * k;
                    const std::size_t residual_index =
                        static_cast<std::size_t>(token) * HC_DIM +
                        static_cast<std::size_t>(stream) * N_EMBD + column;
                    twin::tf value = values[thread][k];
                    if (steer) {
                        if (mode == 0u) {
                            const twin::tf negative_dot{
                                -dot.value, dot.magnitude};
                            value = twin::fma(
                                negative_dot,
                                twin::tf{direction[direction_offset + column]},
                                value);
                        } else {
                            value = value +
                                twin::tf{direction[direction_offset + column]};
                        }
                    }
                    expected[residual_index] = value;
                }
            }
        }
    }
}

bool run_case(Context& context, RMSNormCase shape, twin::InputSet set,
              const twin::Options& options, bool in_place) {
    const std::string label = "rms_norm_w n_cols=" + std::to_string(shape.n_cols) +
        " gp=" + std::to_string(shape.gp) + " rows=" + std::to_string(shape.rows) +
        " set=" + twin::input_set_name(set) + (in_place ? " in-place" : "");
    const std::size_t input_count =
        static_cast<std::size_t>(shape.rows) * shape.n_cols;
    const std::size_t gamma_count =
        static_cast<std::size_t>(shape.gp) * shape.n_cols;
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t gamma_bytes = gamma_count * sizeof(float);

    std::vector<float> input(input_count);
    std::vector<float> gamma(gamma_count);
    std::mt19937_64 rng(options.seed ^
        (static_cast<std::uint64_t>(shape.n_cols) << 32u) ^
        (static_cast<std::uint64_t>(shape.gp) << 16u) ^ shape.rows);
    for (std::uint32_t row = 0; row < shape.rows; ++row) {
        for (std::uint32_t col = 0; col < shape.n_cols; ++col) {
            input[static_cast<std::size_t>(row) * shape.n_cols + col] =
                input_value(set, row, col, rng);
        }
    }
    for (std::uint32_t row = 0; row < shape.gp; ++row) {
        for (std::uint32_t col = 0; col < shape.n_cols; ++col) {
            gamma[static_cast<std::size_t>(row) * shape.n_cols + col] =
                gamma_value(set, row, col, rng);
        }
    }

    std::vector<twin::tf> expected;
    rms_norm_reference(input, gamma, shape.rows, shape.n_cols, shape.gp,
                       kEpsilon, expected);

    std::string err;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer gamma_buffer;
    twin::GuardedBuffer separate_output;
    const bool shared_encoder_case =
        !in_place && set == twin::InputSet::fixture &&
        shape.n_cols == 2560u && shape.gp == 4u && shape.rows == 4u;
    twin::GuardedBuffer shared_output;
    check(input_buffer.allocate(context, label + " input", input_bytes, err),
          label + " input buffer", err);
    check(gamma_buffer.allocate(context, label + " gamma", gamma_bytes, err),
          label + " gamma buffer", err);
    if (!in_place) {
        check(separate_output.allocate(context, label + " output", input_bytes, err),
              label + " output buffer", err);
    }
    if (shared_encoder_case) {
        check(shared_output.allocate(context, label + " shared output",
                                     input_bytes, err),
              label + " shared output buffer", err);
    }
    if (!input_buffer.view() || !gamma_buffer.view() ||
        (!in_place && !separate_output.view()) ||
        (shared_encoder_case && !shared_output.view())) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), input.data(), input_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(), gamma_bytes);
    if (!in_place) {
        check(separate_output.poison_floats_against(
                  expected, twin::Comparison::units, err),
              label + " output poison", err);
    }
    if (shared_encoder_case) {
        check(shared_output.poison_floats_against(
                  expected, twin::Comparison::units, err),
              label + " shared output poison", err);
    }
    input_buffer.capture_before_dispatch();
    gamma_buffer.capture_before_dispatch();
    if (!in_place) separate_output.capture_before_dispatch();
    if (shared_encoder_case) shared_output.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const WeightedRMSNormBuffers buffers{
        input_buffer.view(), gamma_buffer.view(),
        in_place ? input_buffer.view() : separate_output.view()};
    const bool encoded = shared_encoder_case
        ? Encoder::encode(
              context, *command, "Weighted RMS norm shared segment",
              [&](Encoder& segment, std::string& encode_error) {
                  if (!encode_weighted_rms_norm(
                          context, segment, shape.rows, shape.n_cols,
                          shape.gp, kEpsilon, buffers, encode_error)) {
                      return false;
                  }
                  return encode_weighted_rms_norm(
                      context, segment, shape.rows, shape.n_cols,
                      shape.gp, kEpsilon,
                      {input_buffer.view(), gamma_buffer.view(),
                       shared_output.view()},
                      encode_error);
              },
              err)
        : encode_weighted_rms_norm(
              context, *command, shape.rows, shape.n_cols, shape.gp, kEpsilon,
              buffers, err);
    check(encoded,
          label + (shared_encoder_case
                       ? " production weighted RMS norm shared-segment encode"
                       : " production weighted RMS norm encode"),
          err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto output = in_place ? input_buffer.payload() : separate_output.payload();
    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(output.data()), input_count);
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 64.0, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, shape.rows, *mismatch);
    std::size_t exact_count = 0u;
    double max_units = 0.0;
    for (std::size_t i = 0; i < input_count; ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) ==
            std::bit_cast<std::uint32_t>(expected[i].value)) {
            ++exact_count;
        }
        const double bound =
            std::ldexp(expected[i].magnitude, -24) + std::ldexp(1.0, -126);
        max_units = std::max(
            max_units, std::fabs(static_cast<double>(actual[i]) - expected[i].value) / bound);
    }
    check(!mismatch, label + " U(64) ordered RMSNorm result",
          "max observed U=" + std::to_string(max_units) + ", exact bits=" +
              std::to_string(exact_count) + "/" + std::to_string(input_count));
    bool shared_mismatch = false;
    bool outputs_bitwise_equal = true;
    if (shared_encoder_case) {
        const auto shared_actual = std::span<const float>(
            reinterpret_cast<const float*>(shared_output.payload().data()),
            input_count);
        const auto shared_difference = twin::compare_floats(
            shared_actual, expected,
            {twin::Comparison::units, 64.0, 0.0, 0u});
        if (shared_difference) {
            twin::report_failure(label.c_str(), options, shape.rows,
                                 *shared_difference);
        }
        shared_mismatch = shared_difference.has_value();
        check(!shared_mismatch,
              label + " shared-segment U(64) ordered RMSNorm result");
        for (std::size_t i = 0; i < input_count; ++i) {
            outputs_bitwise_equal &=
                std::bit_cast<std::uint32_t>(actual[i]) ==
                std::bit_cast<std::uint32_t>(shared_actual[i]);
        }
        check(outputs_bitwise_equal,
              label + " shared-segment outputs are bitwise equal");
    }

    check(!gamma_buffer.first_unwritten_change({}),
          label + " gamma remains read-only");
    const std::array<twin::ByteRange, 1> written{{{0u, input_bytes}}};
    if (in_place) {
        const auto changed = input_buffer.first_unwritten_change(written);
        check(!changed, label + " in-place output guards remain unchanged",
              changed ? changed->reason : "");
    } else {
        check(!input_buffer.first_unwritten_change({}),
              label + " input remains read-only");
        const auto changed = separate_output.first_unwritten_change(written);
        check(!changed, label + " output guards remain unchanged",
              changed ? changed->reason : "");
        if (shared_encoder_case) {
            const auto shared_changed =
                shared_output.first_unwritten_change(written);
            check(!shared_changed,
                  label + " shared output guards remain unchanged",
                  shared_changed ? shared_changed->reason : "");
        }
    }
    return !mismatch && !shared_mismatch && outputs_bitwise_equal;
}

bool test_weighted_rms_norm_encoder_validation(
    Context& context, Context& foreign_context) {
    std::string error;
    auto rejected_command = context.make_command_buffer(
        "Weighted RMS norm rejected-input validation", error);
    if (!rejected_command) {
        check(false, "Weighted RMS norm rejection-validation command buffer",
              error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            std::uint32_t n_cols, std::uint32_t gamma_period,
                            float epsilon,
                            const WeightedRMSNormBuffers& buffers,
                            const std::string& expected_error) {
        const bool rejected =
            !encode_weighted_rms_norm(
                context, *rejected_command, rows, n_cols, gamma_period,
                epsilon, buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    const WeightedRMSNormBuffers empty{};
    bool passed = true;
    passed &= reject("Weighted RMS norm encoder rejects zero rows",
                     0u, 2560u, 1u, kEpsilon, empty, "row count");
    passed &= reject("Weighted RMS norm encoder rejects zero columns",
                     1u, 0u, 1u, kEpsilon, empty, "column count");
    passed &= reject("Weighted RMS norm encoder rejects zero gamma period",
                     1u, 2560u, 0u, kEpsilon, empty, "gamma period");
    passed &= reject("Weighted RMS norm encoder rejects unsupported columns",
                     1u, 512u, 1u, kEpsilon, empty, "supports");
    passed &= reject("Weighted RMS norm encoder rejects unsupported gamma pair",
                     1u, 10240u, 4u, kEpsilon, empty, "supports");
    passed &= reject("Weighted RMS norm encoder rejects zero epsilon",
                     1u, 2560u, 1u, 0.0f, empty, "finite and positive");
    passed &= reject("Weighted RMS norm encoder rejects negative epsilon",
                     1u, 2560u, 1u, -kEpsilon, empty, "finite and positive");
    passed &= reject(
        "Weighted RMS norm encoder rejects non-finite epsilon",
        1u, 2560u, 1u, std::numeric_limits<float>::infinity(),
        empty, "finite and positive");
    passed &= reject(
        "Weighted RMS norm encoder rejects NaN epsilon",
        1u, 2560u, 1u, std::numeric_limits<float>::quiet_NaN(),
        empty, "finite and positive");

    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t n_cols = 2560u;
    constexpr std::uint32_t gamma_period = 4u;
    const std::size_t input_row_bytes =
        static_cast<std::size_t>(n_cols) * sizeof(float);
    const std::size_t input_bytes = rows * input_row_bytes;
    const std::size_t gamma_bytes =
        static_cast<std::size_t>(gamma_period) * n_cols * sizeof(float);
    BufferRef input = context.buffers().allocate(
        "Weighted RMS norm validation input", input_bytes, error);
    BufferRef gamma = context.buffers().allocate(
        "Weighted RMS norm validation gamma", gamma_bytes, error);
    BufferRef output = context.buffers().allocate(
        "Weighted RMS norm validation output", input_bytes, error);
    BufferRef foreign_input = foreign_context.buffers().allocate(
        "Weighted RMS norm foreign-context input", input_bytes, error);
    BufferRef partial_backing = context.buffers().allocate(
        "Weighted RMS norm partial-alias backing",
        std::max(input_bytes, gamma_bytes) + 16u, error);
    if (!input || !gamma || !output || !foreign_input || !partial_backing) {
        check(false, "Weighted RMS norm encoder validation buffers", error);
        rejected_command->discard();
        return false;
    }
    const WeightedRMSNormBuffers valid{input, gamma, output};
    WeightedRMSNormBuffers foreign_view = valid;
    foreign_view.input = foreign_input;
    passed &= reject("Weighted RMS norm encoder rejects foreign buffer",
                     rows, n_cols, gamma_period, kEpsilon,
                     foreign_view, "not owned by this Context");

    WeightedRMSNormBuffers short_input = valid;
    short_input.input.length = input_row_bytes;
    passed &= reject("Weighted RMS norm encoder rejects one-row input for T=2",
                     rows, n_cols, gamma_period, kEpsilon,
                     short_input, "undersized");
    WeightedRMSNormBuffers short_gamma = valid;
    short_gamma.gamma.length = gamma_bytes - sizeof(float);
    passed &= reject("Weighted RMS norm encoder rejects short gamma",
                     rows, n_cols, gamma_period, kEpsilon,
                     short_gamma, "undersized");
    WeightedRMSNormBuffers short_output = valid;
    short_output.output.length = input_row_bytes;
    passed &= reject("Weighted RMS norm encoder rejects one-row output for T=2",
                     rows, n_cols, gamma_period, kEpsilon,
                     short_output, "undersized");

    WeightedRMSNormBuffers misaligned = valid;
    misaligned.gamma = gamma.slice(1u, gamma.length - 1u);
    passed &= reject("Weighted RMS norm encoder rejects misaligned gamma",
                     rows, n_cols, gamma_period, kEpsilon,
                     misaligned, "misaligned");

    BufferRef partial_input = partial_backing.slice(0u, input_bytes);
    BufferRef partial_output =
        partial_backing.slice(sizeof(float), input_bytes);
    WeightedRMSNormBuffers partial_alias{partial_input, gamma, partial_output};
    passed &= reject("Weighted RMS norm encoder rejects partial input/output alias",
                     rows, n_cols, gamma_period, kEpsilon,
                     partial_alias, "exact input/output aliasing");

    WeightedRMSNormBuffers gamma_output_alias = valid;
    gamma_output_alias.output = gamma;
    passed &= reject("Weighted RMS norm encoder rejects output/gamma overlap",
                     rows, n_cols, gamma_period, kEpsilon,
                     gamma_output_alias, "must not overlap gamma");

    WeightedRMSNormBuffers out_of_range{input, gamma, output};
    out_of_range.output = output.slice(output.length, sizeof(float));
    passed &= reject("Weighted RMS norm encoder rejects a view past allocation",
                     rows, n_cols, gamma_period, kEpsilon,
                     out_of_range, "not owned by this Context");

    WeightedRMSNormBuffers read_only_alias{gamma, gamma, output};
    const bool accepts_read_only_alias = encode_weighted_rms_norm(
        context, *rejected_command, rows, n_cols, gamma_period, kEpsilon,
        read_only_alias, error);
    check(accepts_read_only_alias,
          "Weighted RMS norm permits overlapping read-only inputs", error);
    passed &= accepts_read_only_alias;

    const ReadOnlyWeightedRMSNormBuffers read_only_gamma_alias{
        partial_input,
        context.buffers().read_only(
            partial_backing.slice(0u, gamma_bytes)),
        output};
    const bool accepts_read_only_gamma_alias = encode_weighted_rms_norm(
        context, *rejected_command, rows, n_cols, gamma_period, kEpsilon,
        read_only_gamma_alias, error);
    check(accepts_read_only_gamma_alias,
          "Weighted RMS norm permits read-only input/gamma overlap", error);
    passed &= accepts_read_only_gamma_alias;
    rejected_command->discard();

    auto foreign_command = foreign_context.make_command_buffer(
        "Weighted RMS norm foreign command-buffer validation", error);
    if (!foreign_command) {
        check(false, "Weighted RMS norm foreign command buffer", error);
        return false;
    }
    const bool rejected_foreign_command =
        !encode_weighted_rms_norm(
            context, *foreign_command, rows, n_cols, gamma_period, kEpsilon,
            valid, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "Weighted RMS norm rejects a foreign command buffer", error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    auto submitted = context.make_command_buffer(
        "Weighted RMS norm submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "Weighted RMS norm command-buffer submission", error);
        return false;
    }
    const bool rejected_submitted =
        !encode_weighted_rms_norm(
            context, *submitted, rows, n_cols, gamma_period, kEpsilon,
            valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "Weighted RMS norm rejects a submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "Weighted RMS norm validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "Weighted RMS norm enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "Weighted RMS norm command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_weighted_rms_norm(
            context, *enqueued, rows, n_cols, gamma_period, kEpsilon,
            valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "Weighted RMS norm rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(
        enqueued, "Weighted RMS norm enqueued validation", error);
    check(drained,
          "Weighted RMS norm enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool run_gr_down_case(Context& context, std::uint32_t rows,
                      twin::InputSet set, const twin::Options& options,
                      bool shared_encoder = false) {
    const std::string label = "gr_down T=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared-encoder" : "");
    const std::size_t input_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t down_weight_count =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM;
    const std::size_t inject_weight_count =
        static_cast<std::size_t>(HC) * HC_DIM;
    const std::size_t lo_count =
        static_cast<std::size_t>(rows) * HC_LOW_RANK;
    const std::size_t inject_count =
        static_cast<std::size_t>(rows) * HC;
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t down_weight_bytes =
        down_weight_count * sizeof(std::uint16_t);
    const std::size_t inject_weight_bytes =
        inject_weight_count * sizeof(std::uint16_t);
    const std::size_t lo_bytes = lo_count * sizeof(float);
    const std::size_t inject_bytes = inject_count * sizeof(float);

    std::vector<float> xn(input_count);
    std::vector<std::uint16_t> down_weights(down_weight_count);
    std::vector<std::uint16_t> inject_weights(inject_weight_count);
    std::mt19937_64 rng(options.seed ^ 0x4752444f574eull ^
                        (static_cast<std::uint64_t>(rows) << 32u));
    for (std::uint32_t token = 0; token < rows; ++token) {
        for (std::uint32_t index = 0; index < HC_DIM; ++index) {
            xn[static_cast<std::size_t>(token) * HC_DIM + index] =
                input_value(set, token, index, rng);
        }
    }
    for (std::uint32_t row = 0; row < HC_LOW_RANK; ++row) {
        for (std::uint32_t col = 0; col < HC_DIM; ++col) {
            down_weights[static_cast<std::size_t>(row) * HC_DIM + col] =
                strata::kernels::bf16_from_f32(
                    gamma_value(set, row, col, rng));
        }
    }
    for (std::uint32_t row = 0; row < HC; ++row) {
        for (std::uint32_t col = 0; col < HC_DIM; ++col) {
            inject_weights[static_cast<std::size_t>(row) * HC_DIM + col] =
                strata::kernels::bf16_from_f32(
                    gamma_value(set, HC_LOW_RANK + row, col, rng));
        }
    }

    std::vector<twin::tf> expected_lo;
    std::vector<twin::tf> expected_inject;
    gr_down_reference(xn, down_weights, inject_weights, rows,
                      expected_lo, expected_inject);

    twin::GuardedBuffer xn_buffer;
    twin::GuardedBuffer down_buffer;
    twin::GuardedBuffer inject_weight_buffer;
    twin::GuardedBuffer lo_buffer;
    twin::GuardedBuffer inject_output_buffer;
    std::string err;
    check(xn_buffer.allocate(context, label + " xn", input_bytes, err),
          label + " xn buffer", err);
    check(down_buffer.allocate(context, label + " hc_down",
                               down_weight_bytes, err),
          label + " hc_down buffer", err);
    check(inject_weight_buffer.allocate(context, label + " hc_inject",
                                        inject_weight_bytes, err),
          label + " hc_inject buffer", err);
    check(lo_buffer.allocate(context, label + " lo", lo_bytes, err),
          label + " lo buffer", err);
    check(inject_output_buffer.allocate(context, label + " inj_out",
                                        inject_bytes, err),
          label + " inj_out buffer", err);
    if (!xn_buffer.view() || !down_buffer.view() ||
        !inject_weight_buffer.view() || !lo_buffer.view() ||
        !inject_output_buffer.view()) {
        return false;
    }
    std::memcpy(xn_buffer.payload().data(), xn.data(), input_bytes);
    std::memcpy(down_buffer.payload().data(),
                down_weights.data(), down_weight_bytes);
    std::memcpy(inject_weight_buffer.payload().data(),
                inject_weights.data(), inject_weight_bytes);
    check(lo_buffer.poison_floats_against(
              expected_lo, twin::Comparison::units, err),
          label + " lo output poison", err);
    check(inject_output_buffer.poison_floats_against(
              expected_inject, twin::Comparison::units, err),
          label + " inject output poison", err);
    xn_buffer.capture_before_dispatch();
    down_buffer.capture_before_dispatch();
    inject_weight_buffer.capture_before_dispatch();
    lo_buffer.capture_before_dispatch();
    inject_output_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const GRDownBuffers buffers{
        xn_buffer.view(), down_buffer.view(), inject_weight_buffer.view(),
        lo_buffer.view(), inject_output_buffer.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared GR down and injection",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_gr_down(context, encoder, rows, buffers,
                                        encode_error);
              },
              err)
        : encode_gr_down(context, *command, rows, buffers, err);
    check(encoded, label + " production GR down encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual_lo = std::span<const float>(
        reinterpret_cast<const float*>(lo_buffer.payload().data()), lo_count);
    const auto actual_inject = std::span<const float>(
        reinterpret_cast<const float*>(inject_output_buffer.payload().data()),
        inject_count);
    const auto lo_mismatch = twin::compare_floats(
        actual_lo, expected_lo, {twin::Comparison::units, 16.0, 0.0, 0u});
    const auto inject_mismatch = twin::compare_floats(
        actual_inject, expected_inject,
        {twin::Comparison::bits, 0.0, 0.0, 0u});
    if (lo_mismatch) twin::report_failure(label.c_str(), options, rows, *lo_mismatch);
    if (inject_mismatch) {
        twin::report_failure((label + " inj_out").c_str(), options, rows,
                             *inject_mismatch);
    }
    check(!lo_mismatch, label + " U(16) BF16 down projection and SiLU");
    check(!inject_mismatch, label + " bit-exact BF16 injection projection");

    const bool xn_unchanged = !xn_buffer.first_unwritten_change({});
    const bool down_unchanged = !down_buffer.first_unwritten_change({});
    const bool inject_weights_unchanged =
        !inject_weight_buffer.first_unwritten_change({});
    check(xn_unchanged, label + " xn remains read-only");
    check(down_unchanged, label + " hc_down remains read-only");
    check(inject_weights_unchanged, label + " hc_inject remains read-only");
    const std::array<twin::ByteRange, 1> lo_written{{{0u, lo_bytes}}};
    const std::array<twin::ByteRange, 1> inject_written{{{0u, inject_bytes}}};
    const bool lo_guards_unchanged =
        !lo_buffer.first_unwritten_change(lo_written);
    const bool inject_guards_unchanged =
        !inject_output_buffer.first_unwritten_change(inject_written);
    check(lo_guards_unchanged, label + " lo guards remain unchanged");
    check(inject_guards_unchanged, label + " inj_out guards remain unchanged");
    return !lo_mismatch && !inject_mismatch && xn_unchanged &&
           down_unchanged && inject_weights_unchanged && lo_guards_unchanged &&
           inject_guards_unchanged;
}

bool test_gr_down_encoder_validation(Context& context,
                                    Context& foreign_context) {
    std::string error;
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRDownBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !encode_gr_down(context, *command, rows, buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    const GRDownBuffers empty{};
    bool passed = true;
    passed &= reject("GR down encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR down encoder rejects T>8", 9u, empty, "1..8");

    const std::size_t input_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t down_weight_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
        sizeof(std::uint16_t);
    const std::size_t inject_weight_bytes =
        static_cast<std::size_t>(HC) * HC_DIM * sizeof(std::uint16_t);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    BufferRef xn = context.buffers().allocate(
        "GR down validation normalized input", input_bytes, error);
    BufferRef hc_down = context.buffers().allocate(
        "GR down validation down weights", down_weight_bytes, error);
    BufferRef hc_inject = context.buffers().allocate(
        "GR down validation injection weights", inject_weight_bytes, error);
    BufferRef lo = context.buffers().allocate(
        "GR down validation low-rank output", low_rank_bytes, error);
    BufferRef inj_out = context.buffers().allocate(
        "GR down validation injection output", inject_bytes, error);
    if (!xn || !hc_down || !hc_inject || !lo || !inj_out) {
        check(false, "GR down encoder validation buffers", error);
        return false;
    }
    const GRDownBuffers valid{xn, hc_down, hc_inject, lo, inj_out};
    BufferRef foreign_xn = foreign_context.buffers().allocate(
        "GR down foreign-context input", input_bytes, error);
    if (!foreign_xn) {
        check(false, "GR down foreign-context validation buffer", error);
        return false;
    }
    check(context.buffers().owns(xn),
          "GR down encoder recognizes a local buffer view");
    check(!context.buffers().owns(foreign_xn),
          "GR down encoder distinguishes a foreign buffer view");
    GRDownBuffers foreign_view = valid;
    foreign_view.xn = foreign_xn;
    passed &= reject("GR down encoder rejects a foreign-context buffer",
                     1u, foreign_view, "not owned by this Context");

    auto foreign_command = foreign_context.make_command_buffer(
        "GR down foreign-context command-buffer validation", error);
    if (!foreign_command) {
        check(false, "GR down foreign-context command buffer", error);
        return false;
    }
    const bool rejected_foreign_command =
        !encode_gr_down(context, *foreign_command, 1u, valid, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "GR down encoder rejects a foreign-context command buffer", error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    const auto reject_short = [&](const std::string& label,
                                  GRDownBuffers buffers,
                                  BufferRef GRDownBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR down encoder rejects short input",
                           valid, &GRDownBuffers::xn);
    passed &= reject_short("GR down encoder rejects short down weights",
                           valid, &GRDownBuffers::hc_down);
    passed &= reject_short("GR down encoder rejects short injection weights",
                           valid, &GRDownBuffers::hc_inject);
    passed &= reject_short("GR down encoder rejects short low-rank output",
                           valid, &GRDownBuffers::lo);
    passed &= reject_short("GR down encoder rejects short injection output",
                           valid, &GRDownBuffers::inj_out);

    GRDownBuffers misaligned = valid;
    misaligned.hc_down =
        hc_down.slice(1u, hc_down.length - 1u);
    passed &= reject("GR down encoder rejects misaligned BF16 weights",
                     1u, misaligned, "misaligned");

    GRDownBuffers overlapping = valid;
    overlapping.lo = overlapping.xn;
    passed &= reject("GR down encoder rejects overlapping views",
                     1u, overlapping, "overlap");

    GRDownBuffers out_of_range = valid;
    out_of_range.hc_inject =
        hc_inject.slice(hc_inject.length, sizeof(std::uint16_t));
    passed &= reject("GR down encoder rejects a weight view past its allocation",
                     1u, out_of_range, "not owned by this Context");

    auto submitted = context.make_command_buffer(
        "GR down submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "GR down validation command-buffer submission", error);
        return false;
    }
    const bool rejected_submitted =
        !encode_gr_down(context, *submitted, 1u, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "GR down encoder rejects a submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "GR down validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "GR down enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GR down validation command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_gr_down(context, *enqueued, 1u, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "GR down encoder rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(
        enqueued, "GR down enqueued validation", error);
    check(drained, "GR down enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool test_gr_down_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GR down shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    const std::size_t input_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * HC_DIM * sizeof(float);
    const std::size_t down_weight_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
        sizeof(std::uint16_t);
    const std::size_t inject_weight_bytes =
        static_cast<std::size_t>(HC) * HC_DIM * sizeof(std::uint16_t);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * HC_LOW_RANK * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * HC * sizeof(float);
    BufferRef xn = context.buffers().allocate(
        "GR down shared validation normalized input", input_bytes, error);
    BufferRef hc_down_storage = context.buffers().allocate(
        "GR down shared validation aligned down weights",
        down_weight_bytes + 32u, error);
    BufferRef hc_inject_storage = context.buffers().allocate(
        "GR down shared validation aligned injection weights",
        inject_weight_bytes + 32u, error);
    BufferRef lo = context.buffers().allocate(
        "GR down shared validation low-rank output", low_rank_bytes, error);
    BufferRef inj_out = context.buffers().allocate(
        "GR down shared validation injection output", inject_bytes, error);
    BufferRef foreign_xn = foreign_context.buffers().allocate(
        "GR down shared foreign normalized input", input_bytes, error);
    if (!xn || !hc_down_storage || !hc_inject_storage || !lo || !inj_out ||
        !foreign_xn) {
        check(false, "GR down shared Encoder validation buffers", error);
        return false;
    }
    const BufferRef hc_down =
        hc_down_storage.slice(16u, down_weight_bytes);
    const BufferRef hc_inject =
        hc_inject_storage.slice(16u, inject_weight_bytes);
    const GRDownBuffers valid{xn, hc_down, hc_inject, lo, inj_out};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRDownBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gr_down(context, encoder, rows, buffers,
                                          encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const GRDownBuffers empty{};
    passed &= reject("GR down shared Encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR down shared Encoder rejects T>8", 9u, empty, "1..8");
    GRDownBuffers foreign_view = valid;
    foreign_view.xn = foreign_xn;
    passed &= reject("GR down shared Encoder rejects foreign Context view",
                     1u, foreign_view, "not owned by this Context");

    const auto reject_short = [&](const std::string& label,
                                  GRDownBuffers buffers,
                                  BufferRef GRDownBuffers::*member) {
        (buffers.*member).length = sizeof(float);
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR down shared Encoder rejects short input",
                           valid, &GRDownBuffers::xn);
    passed &= reject_short("GR down shared Encoder rejects short down weights",
                           valid, &GRDownBuffers::hc_down);
    passed &= reject_short(
        "GR down shared Encoder rejects short injection weights",
        valid, &GRDownBuffers::hc_inject);
    passed &= reject_short(
        "GR down shared Encoder rejects short low-rank output",
        valid, &GRDownBuffers::lo);
    passed &= reject_short(
        "GR down shared Encoder rejects short injection output",
        valid, &GRDownBuffers::inj_out);

    GRDownBuffers one_row_for_two = valid;
    one_row_for_two.xn.length =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    passed &= reject(
        "GR down shared Encoder rejects one-row input for T=2",
        2u, one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.lo.length =
        static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float);
    passed &= reject(
        "GR down shared Encoder rejects one-row low-rank output for T=2",
        2u, one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.inj_out.length =
        static_cast<std::size_t>(HC) * sizeof(float);
    passed &= reject(
        "GR down shared Encoder rejects one-row injection output for T=2",
        2u, one_row_for_two, "undersized");

    GRDownBuffers misaligned = valid;
    misaligned.hc_down =
        hc_down_storage.slice(4u, down_weight_bytes);
    passed &= reject(
        "GR down shared Encoder rejects misaligned down weights",
        1u, misaligned, "misaligned");
    misaligned = valid;
    misaligned.hc_inject =
        hc_inject_storage.slice(4u, inject_weight_bytes);
    passed &= reject(
        "GR down shared Encoder rejects misaligned injection weights",
        1u, misaligned, "misaligned");
    misaligned = valid;
    misaligned.xn = xn.slice(1u, xn.length - 1u);
    passed &= reject("GR down shared Encoder rejects misaligned input",
                     1u, misaligned, "misaligned");

    GRDownBuffers out_of_range = valid;
    out_of_range.hc_inject =
        hc_inject.slice(hc_inject.length, sizeof(std::uint16_t));
    passed &= reject(
        "GR down shared Encoder rejects out-of-range weight view",
        1u, out_of_range, "not owned by this Context");

    GRDownBuffers input_overlap = valid;
    input_overlap.xn = hc_down;
    passed &= reject("GR down shared Encoder rejects input/input overlap",
                     1u, input_overlap, "overlap");
    GRDownBuffers output_overlap = valid;
    output_overlap.lo = output_overlap.xn;
    passed &= reject("GR down shared Encoder rejects input/output overlap",
                     1u, output_overlap, "overlap");
    output_overlap = valid;
    output_overlap.inj_out = output_overlap.lo;
    passed &= reject("GR down shared Encoder rejects output/output overlap",
                     1u, output_overlap, "overlap");

    BufferRef partial_storage = context.buffers().allocate(
        "GR down shared validation partial-alias storage",
        static_cast<std::size_t>(HC_DIM) * sizeof(float) +
            static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float),
        error);
    if (!partial_storage) {
        check(false, "GR down shared partial-alias buffer", error);
        return false;
    }
    GRDownBuffers partial_overlap = valid;
    partial_overlap.xn = partial_storage.slice(
        0u, static_cast<std::size_t>(HC_DIM) * sizeof(float));
    partial_overlap.lo = partial_storage.slice(
        sizeof(float), static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float));
    passed &= reject("GR down shared Encoder rejects partial input/output overlap",
                     1u, partial_overlap, "overlap");
    return passed;
}

bool run_gr_up_case(Context& context, std::uint32_t rows, bool apply,
                    twin::InputSet set, const twin::Options& options,
                    bool shared_encoder = false) {
    const std::string label = "gr_up T=" + std::to_string(rows) +
        (apply ? " apply" : " no-apply") +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared-encoder" : "");
    const std::size_t residual_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t block_count =
        static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t inject_count =
        static_cast<std::size_t>(rows) * HC;
    const std::size_t lo_count =
        static_cast<std::size_t>(rows) * HC_LOW_RANK;
    const std::size_t weight_count =
        static_cast<std::size_t>(HC_DIM) * HC_LOW_RANK;
    const std::size_t residual_bytes = residual_count * sizeof(float);
    const std::size_t block_bytes = block_count * sizeof(float);
    const std::size_t inject_bytes = inject_count * sizeof(float);
    const std::size_t norm_bytes = HC_DIM * sizeof(float);
    const std::size_t lo_bytes = lo_count * sizeof(float);
    const std::size_t weight_bytes = weight_count * sizeof(std::uint16_t);
    const std::size_t mixed_bytes = block_count * sizeof(float);

    std::vector<float> residual(residual_count);
    std::vector<float> block_out(block_count);
    std::vector<float> inject_prev(inject_count);
    std::vector<float> norm(HC_DIM);
    std::vector<float> lo(lo_count);
    std::vector<std::uint16_t> up_weights(weight_count);
    std::mt19937_64 rng(options.seed ^ 0x47525550ull ^
                        (static_cast<std::uint64_t>(rows) << 32u) ^
                        (apply ? 0x4150504c59ull : 0x4e4f415050ull));
    for (std::uint32_t token = 0; token < rows; ++token) {
        for (std::uint32_t index = 0; index < HC_DIM; ++index) {
            residual[static_cast<std::size_t>(token) * HC_DIM + index] =
                input_value(set, token, index, rng);
        }
        for (std::uint32_t column = 0; column < N_EMBD; ++column) {
            block_out[static_cast<std::size_t>(token) * N_EMBD + column] =
                gamma_value(set, token, column, rng);
        }
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            inject_prev[static_cast<std::size_t>(token) * HC + stream] =
                input_value(set, token, stream, rng);
        }
        for (std::uint32_t rank = 0; rank < HC_LOW_RANK; ++rank) {
            lo[static_cast<std::size_t>(token) * HC_LOW_RANK + rank] =
                input_value(set, token, rank, rng);
        }
    }
    for (std::uint32_t index = 0; index < HC_DIM; ++index) {
        norm[index] = gamma_value(set, index / N_EMBD, index % N_EMBD, rng);
    }
    for (std::uint32_t row = 0; row < HC_DIM; ++row) {
        for (std::uint32_t rank = 0; rank < HC_LOW_RANK; ++rank) {
            up_weights[static_cast<std::size_t>(row) * HC_LOW_RANK + rank] =
                strata::kernels::bf16_from_f32(
                    gamma_value(set, row, rank, rng));
        }
    }

    // Use scales from the documented GR norm oracle so this up pass receives
    // the reciprocal scales paired with the same residual and pending write.
    std::vector<twin::tf> ignored_xn;
    std::vector<twin::tf> expected_rs;
    gr_norm_reference(residual, block_out, norm, inject_prev, rows, apply,
                      ignored_xn, expected_rs);
    std::vector<float> rs_out(inject_count);
    for (std::size_t index = 0; index < rs_out.size(); ++index) {
        rs_out[index] = expected_rs[index].value;
    }

    if (set == twin::InputSet::fixture && !apply) {
        constexpr float order_residual = 5.934396743774414f;
        constexpr float order_norm = 1.6455072164535522f;
        constexpr float order_rs = 0.9459876418113708f;
        constexpr std::array<float, HC> stream_values{
            2.0e20f, 6.0f, -2.0e20f, 6.0f};
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            const std::size_t row =
                static_cast<std::size_t>(stream) * N_EMBD;
            residual[row] = stream_values[stream];
            residual[row + 1u] = stream == 0u ? order_residual : 0.0f;
            norm[row] = 1.0f;
            norm[row + 1u] = stream == 0u ? order_norm : 1.0f;
            rs_out[stream] =
                (stream == 0u || stream == 2u) ? order_rs : 1.0f;
            for (std::uint32_t rank = 0; rank < HC_LOW_RANK; ++rank) {
                up_weights[row * HC_LOW_RANK + rank] = 0u;
                up_weights[(row + 1u) * HC_LOW_RANK + rank] = 0u;
            }
        }
    }

    std::vector<twin::tf> expected_residual;
    std::vector<twin::tf> expected_mixed;
    gr_up_reference(residual, block_out, inject_prev, norm, rs_out, lo,
                    up_weights, rows, apply, expected_residual,
                    expected_mixed);

    twin::GuardedBuffer residual_buffer;
    twin::GuardedBuffer block_buffer;
    twin::GuardedBuffer inject_buffer;
    twin::GuardedBuffer norm_buffer;
    twin::GuardedBuffer rs_buffer;
    twin::GuardedBuffer lo_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer mixed_buffer;
    std::string err;
    check(residual_buffer.allocate(context, label + " residual",
                                   residual_bytes, err),
          label + " residual buffer", err);
    check(block_buffer.allocate(context, label + " block_out", block_bytes, err),
          label + " block_out buffer", err);
    check(inject_buffer.allocate(context, label + " inject_prev",
                                 inject_bytes, err),
          label + " inject_prev buffer", err);
    check(norm_buffer.allocate(context, label + " norm", norm_bytes, err),
          label + " norm buffer", err);
    check(rs_buffer.allocate(context, label + " rs_out", inject_bytes, err),
          label + " rs_out buffer", err);
    check(lo_buffer.allocate(context, label + " lo", lo_bytes, err),
          label + " lo buffer", err);
    check(up_weight_buffer.allocate(context, label + " hc_up", weight_bytes, err),
          label + " hc_up buffer", err);
    check(mixed_buffer.allocate(context, label + " mixed", mixed_bytes, err),
          label + " mixed buffer", err);
    if (!residual_buffer.view() || !block_buffer.view() ||
        !inject_buffer.view() || !norm_buffer.view() || !rs_buffer.view() ||
        !lo_buffer.view() || !up_weight_buffer.view() ||
        !mixed_buffer.view()) {
        return false;
    }
    std::memcpy(residual_buffer.payload().data(), residual.data(),
                residual_bytes);
    std::memcpy(block_buffer.payload().data(), block_out.data(), block_bytes);
    std::memcpy(inject_buffer.payload().data(), inject_prev.data(),
                inject_bytes);
    std::memcpy(norm_buffer.payload().data(), norm.data(), norm_bytes);
    std::memcpy(rs_buffer.payload().data(), rs_out.data(), inject_bytes);
    std::memcpy(lo_buffer.payload().data(), lo.data(), lo_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    check(mixed_buffer.poison_floats_against(
              expected_mixed, twin::Comparison::units, err),
          label + " mixed output poison", err);
    residual_buffer.capture_before_dispatch();
    block_buffer.capture_before_dispatch();
    inject_buffer.capture_before_dispatch();
    norm_buffer.capture_before_dispatch();
    rs_buffer.capture_before_dispatch();
    lo_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    mixed_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const GRUpBuffers buffers{
        residual_buffer.view(), block_buffer.view(), inject_buffer.view(),
        norm_buffer.view(), rs_buffer.view(), lo_buffer.view(),
        up_weight_buffer.view(), mixed_buffer.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared GR up",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_gr_up(context, encoder, rows, apply, buffers,
                                      encode_error);
              },
              err)
        : encode_gr_up(context, *command, rows, apply, buffers, err);
    check(encoded, label + " production GR up encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual_residual = std::span<const float>(
        reinterpret_cast<const float*>(residual_buffer.payload().data()),
        residual_count);
    const auto actual_mixed = std::span<const float>(
        reinterpret_cast<const float*>(mixed_buffer.payload().data()),
        block_count);
    const auto residual_mismatch = twin::compare_floats(
        actual_residual, expected_residual,
        {apply ? twin::Comparison::units : twin::Comparison::bits,
         apply ? 16.0 : 0.0, 0.0, 0u});
    const auto mixed_mismatch = twin::compare_floats(
        actual_mixed, expected_mixed,
        {twin::Comparison::units, 64.0, 0.0, N_EMBD});
    if (residual_mismatch) {
        twin::report_failure((label + " residual").c_str(), options, rows,
                             *residual_mismatch);
    }
    if (mixed_mismatch) {
        twin::report_failure((label + " mixed").c_str(), options, rows,
                             *mixed_mismatch);
    }
    check(!residual_mismatch,
          label + (apply ? " U(16) folded residual write"
                         : " residual remains bit-exact"));
    check(!mixed_mismatch, label + " U(64) up projection and stream mean");
    if (set == twin::InputSet::fixture && !apply) {
        constexpr float order_residual = 5.934396743774414f;
        constexpr float order_norm = 1.6455072164535522f;
        constexpr float order_rs = 0.9459876418113708f;
        const float correct_normalized =
            ((order_residual * order_norm) * order_rs) * 0.5f / 4.0f;
        const float wrong_normalized =
            ((order_residual * order_rs) * order_norm) * 0.5f / 4.0f;
        const float stream0 = (2.0e20f * order_rs) * 0.5f;
        const float stream1 = (6.0f * 1.0f) * 1.0f * 0.5f;
        const float stream2 = (-2.0e20f * order_rs) * 0.5f;
        const float stream3 = (6.0f * 1.0f) * 1.0f * 0.5f;
        const float wrong_stream_order =
            ((stream0 + stream2) + (stream1 + stream3)) / 4.0f;
        const bool exact_normalization_order =
            std::bit_cast<std::uint32_t>(actual_mixed[1]) ==
            std::bit_cast<std::uint32_t>(correct_normalized);
        const bool exact_stream_order =
            std::bit_cast<std::uint32_t>(actual_mixed[0]) ==
            std::bit_cast<std::uint32_t>(0.75f);
        const bool normalization_negative_control =
            std::bit_cast<std::uint32_t>(wrong_normalized) !=
            std::bit_cast<std::uint32_t>(correct_normalized);
        const bool stream_negative_control =
            std::bit_cast<std::uint32_t>(wrong_stream_order) !=
            std::bit_cast<std::uint32_t>(0.75f);
        check(exact_normalization_order,
              label + " exact (R*g)*rs order fixture");
        check(exact_stream_order,
              label + " exact ascending stream mixer fixture");
        check(normalization_negative_control,
              label + " reversed normalization order negative control");
        check(stream_negative_control,
              label + " reordered stream mixer negative control");
    }

    check(!block_buffer.first_unwritten_change({}),
          label + " block_out remains read-only");
    check(!inject_buffer.first_unwritten_change({}),
          label + " inject_prev remains read-only");
    check(!norm_buffer.first_unwritten_change({}),
          label + " norm remains read-only");
    check(!rs_buffer.first_unwritten_change({}),
          label + " rs_out remains read-only");
    check(!lo_buffer.first_unwritten_change({}),
          label + " lo remains read-only");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " hc_up remains read-only");
    if (apply) {
        const std::array<twin::ByteRange, 1> residual_written{{
            {0u, residual_bytes}}};
        const auto changed =
            residual_buffer.first_unwritten_change(residual_written);
        check(!changed, label + " residual guards remain unchanged",
              changed ? changed->reason : "");
    } else {
        check(!residual_buffer.first_unwritten_change({}),
              label + " residual remains completely unchanged");
    }
    const std::array<twin::ByteRange, 1> mixed_written{{
        {0u, mixed_bytes}}};
    const auto mixed_changed =
        mixed_buffer.first_unwritten_change(mixed_written);
    check(!mixed_changed, label + " mixed guards remain unchanged",
          mixed_changed ? mixed_changed->reason : "");
    return !residual_mismatch && !mixed_mismatch;
}

bool run_gr_read_pipeline_case(Context& context, std::uint32_t rows,
                               bool apply, const twin::Options& options) {
    const std::string label = "gr_read_pipeline T=" + std::to_string(rows) +
        (apply ? " apply" : " no-apply");
    constexpr std::size_t kResidual = 0u;
    constexpr std::size_t kBlockOut = 1u;
    constexpr std::size_t kInjectPrev = 2u;
    constexpr std::size_t kNorm = 3u;
    constexpr std::size_t kNormalized = 4u;
    constexpr std::size_t kRs = 5u;
    constexpr std::size_t kHcDown = 6u;
    constexpr std::size_t kHcInject = 7u;
    constexpr std::size_t kLo = 8u;
    constexpr std::size_t kInjectOut = 9u;
    constexpr std::size_t kHcUp = 10u;
    constexpr std::size_t kMixed = 11u;

    const std::size_t residual_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t block_count =
        static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t inject_count =
        static_cast<std::size_t>(rows) * HC;
    const std::size_t low_rank_count =
        static_cast<std::size_t>(rows) * HC_LOW_RANK;
    const std::size_t down_weight_count =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM;
    const std::size_t inject_weight_count =
        static_cast<std::size_t>(HC) * HC_DIM;
    const std::size_t up_weight_count =
        static_cast<std::size_t>(HC_DIM) * HC_LOW_RANK;
    const std::array<std::size_t, 12u> byte_sizes{
        residual_count * sizeof(float),
        block_count * sizeof(float),
        inject_count * sizeof(float),
        static_cast<std::size_t>(HC_DIM) * sizeof(float),
        residual_count * sizeof(float),
        inject_count * sizeof(float),
        down_weight_count * sizeof(std::uint16_t),
        inject_weight_count * sizeof(std::uint16_t),
        low_rank_count * sizeof(float),
        inject_count * sizeof(float),
        up_weight_count * sizeof(std::uint16_t),
        block_count * sizeof(float),
    };

    std::vector<float> residual(residual_count);
    std::vector<float> block_out(block_count);
    std::vector<float> inject_prev(inject_count);
    std::vector<float> norm(HC_DIM);
    std::vector<std::uint16_t> hc_down(down_weight_count);
    std::vector<std::uint16_t> hc_inject(inject_weight_count);
    std::vector<std::uint16_t> hc_up(up_weight_count);
    std::mt19937_64 rng(options.seed ^ 0x475252454144ull ^
                        (static_cast<std::uint64_t>(rows) << 32u) ^
                        (apply ? 0x4150504c59ull : 0x4e4f415050ull));
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t index = 0u; index < HC_DIM; ++index) {
            residual[static_cast<std::size_t>(row) * HC_DIM + index] =
                input_value(twin::InputSet::fixture, row, index, rng);
        }
        for (std::uint32_t index = 0u; index < N_EMBD; ++index) {
            block_out[static_cast<std::size_t>(row) * N_EMBD + index] =
                gamma_value(twin::InputSet::fixture, row, index, rng);
        }
        for (std::uint32_t stream = 0u; stream < HC; ++stream) {
            inject_prev[static_cast<std::size_t>(row) * HC + stream] =
                input_value(twin::InputSet::fixture, row, stream, rng);
        }
    }
    for (std::uint32_t index = 0u; index < HC_DIM; ++index) {
        norm[index] = gamma_value(twin::InputSet::fixture,
                                  index / N_EMBD, index % N_EMBD, rng);
    }
    for (std::uint32_t row = 0u; row < HC_LOW_RANK; ++row) {
        for (std::uint32_t column = 0u; column < HC_DIM; ++column) {
            hc_down[static_cast<std::size_t>(row) * HC_DIM + column] =
                strata::kernels::bf16_from_f32(gamma_value(
                    twin::InputSet::fixture, row, column, rng));
        }
    }
    for (std::uint32_t row = 0u; row < HC; ++row) {
        for (std::uint32_t column = 0u; column < HC_DIM; ++column) {
            hc_inject[static_cast<std::size_t>(row) * HC_DIM + column] =
                strata::kernels::bf16_from_f32(gamma_value(
                    twin::InputSet::fixture, HC_LOW_RANK + row, column, rng));
        }
    }
    for (std::uint32_t row = 0u; row < HC_DIM; ++row) {
        for (std::uint32_t rank = 0u; rank < HC_LOW_RANK; ++rank) {
            hc_up[static_cast<std::size_t>(row) * HC_LOW_RANK + rank] =
                strata::kernels::bf16_from_f32(gamma_value(
                    twin::InputSet::fixture, row, rank, rng));
        }
    }

    std::array<twin::GuardedBuffer, 12u> storage;
    std::array<BufferRef, 12u> views{};
    const std::array<const char*, 12u> names{
        "residual", "block_out", "inject_prev", "norm", "normalized",
        "rs", "hc_down", "hc_inject", "lo", "inject_out", "hc_up",
        "mixed",
    };
    std::string error;
    for (std::size_t index = 0u; index < storage.size(); ++index) {
        check(storage[index].allocate(context,
                                      label + " " + names[index],
                                      byte_sizes[index], error),
              label + " " + names[index] + " buffer", error);
        views[index] = storage[index].view();
        if (!views[index]) return false;
    }
    const auto copy_input = [&](std::size_t index, const void* source) {
        std::memcpy(storage[index].payload().data(), source, byte_sizes[index]);
    };
    copy_input(kResidual, residual.data());
    copy_input(kBlockOut, block_out.data());
    copy_input(kInjectPrev, inject_prev.data());
    copy_input(kNorm, norm.data());
    copy_input(kHcDown, hc_down.data());
    copy_input(kHcInject, hc_inject.data());
    copy_input(kHcUp, hc_up.data());
    for (const std::size_t index :
         {kNormalized, kRs, kLo, kInjectOut, kMixed}) {
        std::memset(storage[index].payload().data(), 0xa5, byte_sizes[index]);
    }
    for (auto& buffer : storage) buffer.capture_before_dispatch();

    const GRReadBuffers buffers{
        views[kResidual], views[kBlockOut], views[kInjectPrev], views[kNorm],
        views[kNormalized], views[kRs], views[kHcDown], views[kHcInject],
        views[kLo], views[kInjectOut], views[kHcUp], views[kMixed],
    };
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode_gr_read(context, *command, rows, apply,
                                        buffers, error);
    check(encoded, label + " ordered GR read encoding", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool completed = submit_and_wait(command, label, error);
    check(completed, label + " ordered GR read execution", error);
    if (!completed) return false;

    const auto floats = [&](std::size_t index, std::size_t count) {
        return std::span<const float>(
            reinterpret_cast<const float*>(storage[index].payload().data()),
            count);
    };
    std::vector<twin::tf> expected_xn;
    std::vector<twin::tf> expected_rs;
    gr_norm_reference(residual, block_out, norm, inject_prev, rows, apply,
                      expected_xn, expected_rs);
    const auto actual_xn = floats(kNormalized, residual_count);
    const auto actual_rs = floats(kRs, inject_count);
    const twin::Comparison norm_comparison = apply
        ? twin::Comparison::units : twin::Comparison::bits;
    const auto xn_mismatch = twin::compare_floats(
        actual_xn, expected_xn,
        {norm_comparison, apply ? 64.0 : 0.0, 0.0, 0u});
    const auto rs_mismatch = twin::compare_floats(
        actual_rs, expected_rs,
        {norm_comparison, apply ? 64.0 : 0.0, 0.0, 0u});
    check(!xn_mismatch, label + " normalized rows match the GR norm oracle");
    check(!rs_mismatch, label + " reciprocal scales match the GR norm oracle");

    std::vector<twin::tf> expected_lo;
    std::vector<twin::tf> expected_inject_out;
    gr_down_reference(actual_xn, hc_down, hc_inject, rows,
                      expected_lo, expected_inject_out);
    const auto actual_lo = floats(kLo, low_rank_count);
    const auto actual_inject_out = floats(kInjectOut, inject_count);
    const auto lo_mismatch = twin::compare_floats(
        actual_lo, expected_lo,
        {twin::Comparison::units, 16.0, 0.0, 0u});
    const auto inject_mismatch = twin::compare_floats(
        actual_inject_out, expected_inject_out,
        {twin::Comparison::bits, 0.0, 0.0, 0u});
    check(!lo_mismatch, label + " low-rank rows match the chained GR oracle");
    check(!inject_mismatch,
          label + " injection output matches the chained GR oracle");

    std::vector<twin::tf> expected_residual;
    std::vector<twin::tf> expected_mixed;
    gr_up_reference(residual, block_out, inject_prev, norm, actual_rs,
                    actual_lo, hc_up, rows, apply, expected_residual,
                    expected_mixed);
    const auto actual_residual = floats(kResidual, residual_count);
    const auto actual_mixed = floats(kMixed, block_count);
    const auto residual_mismatch = twin::compare_floats(
        actual_residual, expected_residual,
        {apply ? twin::Comparison::units : twin::Comparison::bits,
         apply ? 16.0 : 0.0, 0.0, 0u});
    const auto mixed_mismatch = twin::compare_floats(
        actual_mixed, expected_mixed,
        {twin::Comparison::units, 64.0, 0.0, N_EMBD});
    check(!residual_mismatch,
          label + (apply ? " residual write matches the GR oracle"
                         : " residual remains bit-exact"));
    check(!mixed_mismatch, label + " mixed rows match the GR oracle");

    const std::array<std::size_t, 6u> writable_views{
        kResidual, kNormalized, kRs, kLo, kInjectOut, kMixed};
    bool shared_encoder_matches = true;
    if (rows == 1u) {
        std::array<std::vector<unsigned char>, writable_views.size()>
            command_buffer_outputs;
        for (std::size_t index = 0u; index < writable_views.size(); ++index) {
            const auto payload = storage[writable_views[index]].payload();
            command_buffer_outputs[index].assign(payload.begin(), payload.end());
        }
        auto shared_command = context.make_command_buffer(
            label + " caller-owned Encoder", error);
        if (!shared_command) {
            check(false, label + " caller-owned Encoder command buffer", error);
            shared_encoder_matches = false;
        } else {
            const bool shared_encoded = Encoder::encode(
                context, *shared_command, "GR read caller-owned segment",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gr_read(context, encoder, rows, apply,
                                          buffers, encode_error);
                }, error);
            check(shared_encoded,
                  label + " caller-owned Encoder GR read encoding", error);
            bool shared_completed = false;
            if (shared_encoded) {
                shared_completed = submit_and_wait(
                    shared_command, label + " caller-owned Encoder", error);
                check(shared_completed,
                      label + " caller-owned Encoder GR read execution", error);
            }
            bool outputs_match = shared_completed;
            if (shared_completed) {
                for (std::size_t index = 0u;
                     index < writable_views.size(); ++index) {
                    const auto payload = storage[writable_views[index]].payload();
                    outputs_match &= std::equal(
                        command_buffer_outputs[index].begin(),
                        command_buffer_outputs[index].end(),
                        payload.begin(), payload.end());
                }
            } else {
                shared_command->discard();
            }
            check(outputs_match,
                  label + " caller-owned Encoder matches standalone outputs");
            shared_encoder_matches = shared_encoded && shared_completed &&
                                     outputs_match;
        }
    }

    bool inputs_unchanged = true;
    for (const std::size_t index :
         {kBlockOut, kInjectPrev, kNorm, kHcDown, kHcInject, kHcUp}) {
        const bool unchanged = !storage[index].first_unwritten_change({});
        check(unchanged, label + " " + names[index] + " remains read-only");
        inputs_unchanged &= unchanged;
    }
    const std::array<twin::ByteRange, 1u> residual_written{{
        {0u, apply ? byte_sizes[kResidual] : 0u}}};
    const std::array<twin::ByteRange, 1u> normalized_written{{
        {0u, byte_sizes[kNormalized]}}};
    const std::array<twin::ByteRange, 1u> rs_written{{
        {0u, byte_sizes[kRs]}}};
    const std::array<twin::ByteRange, 1u> lo_written{{
        {0u, byte_sizes[kLo]}}};
    const std::array<twin::ByteRange, 1u> inject_written{{
        {0u, byte_sizes[kInjectOut]}}};
    const std::array<twin::ByteRange, 1u> mixed_written{{
        {0u, byte_sizes[kMixed]}}};
    const bool output_guards =
        !storage[kResidual].first_unwritten_change(apply
            ? std::span<const twin::ByteRange>(residual_written)
            : std::span<const twin::ByteRange>{}) &&
        !storage[kNormalized].first_unwritten_change(normalized_written) &&
        !storage[kRs].first_unwritten_change(rs_written) &&
        !storage[kLo].first_unwritten_change(lo_written) &&
        !storage[kInjectOut].first_unwritten_change(inject_written) &&
        !storage[kMixed].first_unwritten_change(mixed_written);
    check(output_guards, label + " outputs stay inside their declared views");

    bool late_failure_discards = true;
    std::array<std::vector<unsigned char>, writable_views.size()> before_failure;
    for (std::size_t index = 0u; index < writable_views.size(); ++index) {
        const std::size_t view_index = writable_views[index];
        auto payload = storage[view_index].payload();
        std::fill(payload.begin(), payload.end(), static_cast<unsigned char>(0x6d));
        before_failure[index].assign(payload.begin(), payload.end());
    }
    GRReadBuffers malformed = buffers;
    malformed.hc_up = views[kHcUp].slice(0u, sizeof(std::uint16_t));
    auto rejected_command = context.make_command_buffer(
        label + " late-stage rejection", error);
    if (!rejected_command) {
        check(false, label + " late-stage rejection command buffer", error);
        late_failure_discards = false;
    } else {
        const bool rejected = !encode_gr_read(
            context, *rejected_command, rows, apply, malformed, error) &&
            error.find("undersized") != std::string::npos;
        const bool discarded =
            rejected_command->unsafe_native_handle() == nullptr;
        bool outputs_unchanged = true;
        for (std::size_t index = 0u; index < writable_views.size(); ++index) {
            const auto payload = storage[writable_views[index]].payload();
            outputs_unchanged &= std::equal(
                before_failure[index].begin(), before_failure[index].end(),
                payload.begin(), payload.end());
        }
        check(rejected, label + " late GR up validation rejects short weights",
              error);
        check(discarded,
              label + " late GR up validation discards earlier dispatches");
        check(outputs_unchanged,
              label + " failed pipeline leaves every output untouched");
        late_failure_discards = rejected && discarded && outputs_unchanged;
    }

    bool caller_owned_late_failure_discards = true;
    auto rejected_shared_command = context.make_command_buffer(
        label + " caller-owned late-stage rejection", error);
    if (!rejected_shared_command) {
        check(false, label + " caller-owned late-rejection command buffer",
              error);
        caller_owned_late_failure_discards = false;
    } else {
        const bool rejected = !Encoder::encode(
            context, *rejected_shared_command,
            "GR read caller-owned late-stage rejection",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_gr_read(context, encoder, rows, apply,
                                      malformed, encode_error);
            }, error) && error.find("undersized") != std::string::npos;
        const bool discarded =
            rejected_shared_command->unsafe_native_handle() == nullptr;
        bool outputs_unchanged = true;
        for (std::size_t index = 0u; index < writable_views.size(); ++index) {
            const auto payload = storage[writable_views[index]].payload();
            outputs_unchanged &= std::equal(
                before_failure[index].begin(), before_failure[index].end(),
                payload.begin(), payload.end());
        }
        check(rejected,
              label + " caller-owned GR up validation rejects short weights",
              error);
        check(discarded,
              label + " caller-owned failure discards earlier dispatches");
        check(outputs_unchanged,
              label + " caller-owned failure leaves outputs untouched");
        caller_owned_late_failure_discards = rejected && discarded &&
                                             outputs_unchanged;
        if (!discarded) rejected_shared_command->discard();
    }

    bool alias_rejected = true;
    GRReadBuffers aliased = buffers;
    aliased.lo = aliased.normalized;
    auto alias_command = context.make_command_buffer(
        label + " cross-stage alias rejection", error);
    if (!alias_command) {
        check(false, label + " cross-stage alias command buffer", error);
        alias_rejected = false;
    } else {
        alias_rejected = !encode_gr_read(context, *alias_command, rows, apply,
                                         aliased, error) &&
            error.find("overlap across dispatches") != std::string::npos &&
            alias_command->unsafe_native_handle() == nullptr;
        check(alias_rejected,
              label + " cross-stage writable alias is rejected", error);
    }

    bool concurrent_group_rejected = true;
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        auto group_command = context.make_command_buffer(
            label + " concurrent-group boundary", error);
        if (!group_command) {
            check(false, label + " concurrent-group command buffer", error);
            concurrent_group_rejected = false;
        } else {
            concurrent_group_rejected = Encoder::encode(
                context, *group_command, "GR read group-boundary validation",
                [&](Encoder& encoder, std::string& encode_error) {
                    if (!encoder.group_begin(encode_error)) return false;
                    std::string stage_error;
                    const bool rejected = !encode_gr_read(
                        context, encoder, rows, apply, buffers, stage_error) &&
                        stage_error.find("ordered concurrent-group boundary") !=
                            std::string::npos;
                    if (!encoder.group_end(encode_error)) return false;
                    if (!rejected) {
                        encode_error = "GR read pipeline accepted an open concurrent group";
                    }
                    return rejected;
                }, error);
            check(concurrent_group_rejected,
                  label + " pipeline refuses an open concurrent group", error);
            group_command->discard();
        }
    }

    bool pending_moe_entry_matches = true;
    if (!apply) {
        MoeRoutedSharedFFNBuffers previous_ffn{};
        previous_ffn.shared.output = views[kBlockOut];
        GRReadBuffers previous_gr_read{};
        previous_gr_read.residual = views[kResidual];
        previous_gr_read.inject_out = views[kInjectPrev];
        GRReadBuffers attention_entry = buffers;
        attention_entry.block_out = views[kLo];
        attention_entry.inject_prev = views[kRs];
        const WindowMoeAttentionEntryBuffers pending_entry{
            previous_ffn, previous_gr_read, attention_entry};
        const std::array<std::size_t, 2u> pending_inputs{
            kBlockOut, kInjectPrev};
        const std::array<std::size_t, 6u> pending_outputs{
            kResidual, kNormalized, kRs, kLo, kInjectOut, kMixed};
        const auto restore_pending_inputs = [&] {
            std::memcpy(storage[kResidual].payload().data(), residual.data(),
                        residual_count * sizeof(float));
            for (const std::size_t index :
                 {kNormalized, kRs, kLo, kInjectOut, kMixed}) {
                std::fill(storage[index].payload().begin(),
                          storage[index].payload().end(), 0xa5u);
            }
        };
        const auto capture_pending_outputs = [&] {
            std::array<std::vector<unsigned char>, pending_outputs.size()> copy;
            for (std::size_t index = 0u; index < pending_outputs.size(); ++index) {
                const auto payload = storage[pending_outputs[index]].payload();
                copy[index].assign(payload.begin(), payload.end());
            }
            return copy;
        };
        const auto capture_pending_inputs = [&] {
            std::array<std::vector<unsigned char>, pending_inputs.size()> copy;
            for (std::size_t index = 0u; index < pending_inputs.size(); ++index) {
                const auto payload = storage[pending_inputs[index]].payload();
                copy[index].assign(payload.begin(), payload.end());
            }
            return copy;
        };
        const auto pending_outputs_match_reference = [&] {
            const auto actual_norm = floats(kNormalized, residual_count);
            const auto actual_rs = floats(kRs, inject_count);
            const auto actual_lo = floats(kLo, low_rank_count);
            const auto actual_inject = floats(kInjectOut, inject_count);
            const auto actual_residual = floats(kResidual, residual_count);
            const auto actual_mixed = floats(kMixed, block_count);
            std::vector<twin::tf> expected_norm;
            std::vector<twin::tf> expected_rs;
            gr_norm_reference(residual, block_out, norm, inject_prev, rows,
                              true, expected_norm, expected_rs);
            std::vector<twin::tf> expected_lo;
            std::vector<twin::tf> expected_inject;
            gr_down_reference(actual_norm, hc_down, hc_inject, rows,
                              expected_lo, expected_inject);
            std::vector<twin::tf> expected_residual;
            std::vector<twin::tf> expected_mixed;
            gr_up_reference(residual, block_out, inject_prev, norm, actual_rs,
                            actual_lo, hc_up, rows, true, expected_residual,
                            expected_mixed);
            const auto norm_diff = twin::compare_floats(
                actual_norm, expected_norm,
                {twin::Comparison::units, 64.0, 0.0, 0u});
            const auto rs_diff = twin::compare_floats(
                actual_rs, expected_rs,
                {twin::Comparison::units, 64.0, 0.0, 0u});
            const auto lo_diff = twin::compare_floats(
                actual_lo, expected_lo,
                {twin::Comparison::units, 16.0, 0.0, 0u});
            const auto inject_diff = twin::compare_floats(
                actual_inject, expected_inject,
                {twin::Comparison::bits, 0.0, 0.0, 0u});
            const auto residual_diff = twin::compare_floats(
                actual_residual, expected_residual,
                {twin::Comparison::units, 16.0, 0.0, 0u});
            const auto mixed_diff = twin::compare_floats(
                actual_mixed, expected_mixed,
                {twin::Comparison::units, 64.0, 0.0, N_EMBD});
            return !norm_diff && !rs_diff && !lo_diff && !inject_diff &&
                   !residual_diff && !mixed_diff;
        };

        const auto inputs_before_consumers = capture_pending_inputs();
        restore_pending_inputs();
        auto producer = context.make_command_buffer(
            label + " completed synthetic pending producer", error);
        const bool producer_completed =
            producer && producer->commit(error) &&
            producer->wait_until_completed(error);
        check(producer_completed,
              label + " pending producer succeeds before standalone consumer",
              error);
        auto consumer = producer_completed
            ? context.make_command_buffer(
                  label + " typed no-PLE pending consumer", error)
            : nullptr;
        const bool dependency_registered =
            consumer && consumer->depends_on(*producer, error);
        check(dependency_registered,
              label + " standalone consumer registers its producer dependency",
              error);
        const bool pending_encoded =
            dependency_registered &&
            encode_window_moe_attention_entry(
                context, *consumer, rows, pending_entry, error);
        check(pending_encoded,
              label + " typed no-PLE standalone consumer encodes", error);
        const bool pending_completed =
            pending_encoded &&
            submit_and_wait(consumer, label + " typed no-PLE consumer", error);
        check(pending_completed,
              label + " typed no-PLE standalone consumer executes", error);
        pending_moe_entry_matches &= producer_completed &&
                                     dependency_registered &&
                                     pending_encoded && pending_completed;
        const bool pending_reference_matches =
            pending_completed && pending_outputs_match_reference();
        check(pending_reference_matches,
              label + " typed no-PLE consumer matches the apply=true GR oracle");
        pending_moe_entry_matches &= pending_reference_matches;
        const auto standalone_outputs = capture_pending_outputs();
        const bool standalone_inputs_unchanged =
            inputs_before_consumers == capture_pending_inputs();
        check(standalone_inputs_unchanged,
              label + " standalone no-PLE consumer preserves producer views");
        pending_moe_entry_matches &= standalone_inputs_unchanged;

        restore_pending_inputs();
        auto shared_pending = context.make_command_buffer(
            label + " typed no-PLE caller-owned consumer", error);
        const bool shared_pending_encoded =
            shared_pending && Encoder::encode(
                context, *shared_pending, "typed no-PLE caller-owned segment",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_window_moe_pending_consumer(
                        context, encoder, rows, false,
                        WindowMoePendingConsumer{pending_entry}, encode_error);
                },
                error);
        check(shared_pending_encoded,
              label + " typed no-PLE caller-owned consumer encodes", error);
        const bool shared_pending_completed =
            shared_pending_encoded &&
            submit_and_wait(shared_pending,
                            label + " typed no-PLE caller-owned consumer",
                            error);
        check(shared_pending_completed,
              label + " typed no-PLE caller-owned consumer executes", error);
        const auto caller_owned_outputs = capture_pending_outputs();
        const bool caller_owned_inputs_unchanged =
            inputs_before_consumers == capture_pending_inputs();
        const bool caller_owned_matches =
            shared_pending_completed &&
            standalone_outputs == caller_owned_outputs &&
            pending_outputs_match_reference() &&
            caller_owned_inputs_unchanged;
        check(caller_owned_matches,
              label + " caller-owned no-PLE entry matches standalone outputs");
        check(caller_owned_inputs_unchanged,
              label + " caller-owned no-PLE consumer preserves producer views");
        pending_moe_entry_matches &= caller_owned_matches;

        const std::array<twin::ByteRange, 1u> pending_residual_written{{
            {0u, byte_sizes[kResidual]}}};
        const std::array<twin::ByteRange, 1u> pending_norm_written{{
            {0u, byte_sizes[kNormalized]}}};
        const std::array<twin::ByteRange, 1u> pending_rs_written{{
            {0u, byte_sizes[kRs]}}};
        const std::array<twin::ByteRange, 1u> pending_lo_written{{
            {0u, byte_sizes[kLo]}}};
        const std::array<twin::ByteRange, 1u> pending_inject_written{{
            {0u, byte_sizes[kInjectOut]}}};
        const std::array<twin::ByteRange, 1u> pending_mixed_written{{
            {0u, byte_sizes[kMixed]}}};
        const bool pending_entry_guards =
            !storage[kResidual].first_unwritten_change(
                pending_residual_written) &&
            !storage[kNormalized].first_unwritten_change(pending_norm_written) &&
            !storage[kRs].first_unwritten_change(pending_rs_written) &&
            !storage[kLo].first_unwritten_change(pending_lo_written) &&
            !storage[kInjectOut].first_unwritten_change(
                pending_inject_written) &&
            !storage[kMixed].first_unwritten_change(pending_mixed_written);
        check(pending_entry_guards,
              label + " typed no-PLE consumer stays inside output guards");
        pending_moe_entry_matches &= pending_entry_guards;

        const auto before_rejection = capture_pending_outputs();
        WindowMoeAttentionEntryBuffers wrong_residual_entry = pending_entry;
        wrong_residual_entry.previous_gr_read.residual = views[kNorm];
        auto rejected_pending = context.make_command_buffer(
            label + " mismatched no-PLE residual", error);
        const bool wrong_residual_rejected =
            rejected_pending &&
            !encode_window_moe_attention_entry(
                context, *rejected_pending, rows, wrong_residual_entry, error) &&
            rejected_pending->unsafe_native_handle() == nullptr &&
            error.find("previous GR residual") != std::string::npos;
        const bool rejection_preserved_outputs =
            before_rejection == capture_pending_outputs();
        const bool rejection_preserved_inputs =
            inputs_before_consumers == capture_pending_inputs();
        check(wrong_residual_rejected,
              label + " no-PLE entry rejects a different GR residual", error);
        check(rejection_preserved_outputs,
              label + " mismatched no-PLE residual preserves output buffers");
        check(rejection_preserved_inputs,
              label + " mismatched no-PLE residual preserves producer views");
        pending_moe_entry_matches &= wrong_residual_rejected &&
                                     rejection_preserved_outputs &&
                                     rejection_preserved_inputs;
    }

    return !xn_mismatch && !rs_mismatch && !lo_mismatch &&
           !inject_mismatch && !residual_mismatch && !mixed_mismatch &&
           inputs_unchanged && output_guards && shared_encoder_matches &&
           late_failure_discards && caller_owned_late_failure_discards &&
           alias_rejected && concurrent_group_rejected &&
           pending_moe_entry_matches;
}

bool test_gr_read_encoder_validation(Context& context) {
    std::string error;
    const GRReadBuffers empty{};
    bool passed = true;
    for (const std::uint32_t rows : {0u, 9u}) {
        const std::string label = "GR read encoder rejects T=" +
                                  std::to_string(rows);
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            passed = false;
            continue;
        }
        const bool rejected = !encode_gr_read(
            context, *command, rows, false, empty, error) &&
            error.find("1..8") != std::string::npos;
        const bool discarded = command->unsafe_native_handle() == nullptr;
        check(rejected, label, error);
        check(discarded, label + " discards its empty encoder segment");
        if (!discarded) command->discard();
        passed &= rejected && discarded;
    }
    for (const std::uint32_t rows : {0u, 9u}) {
        const std::string label =
            "Window pending-MoE consumer rejects T=" +
            std::to_string(rows);
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            passed = false;
            continue;
        }
        const WindowMoePendingConsumer consumer{
            WindowMoeAttentionEntryBuffers{}};
        const bool rejected = !Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_window_moe_pending_consumer(
                    context, encoder, rows, false, consumer, encode_error);
            },
            error) &&
            error.find("rows must be in 1..8") != std::string::npos;
        const bool discarded = command->unsafe_native_handle() == nullptr;
        check(rejected, label, error);
        check(discarded, label + " discards its empty encoder segment");
        if (!discarded) command->discard();
        passed &= rejected && discarded;
    }
    return passed;
}

bool test_gr_up_encoder_validation(Context& context) {
    std::string error;
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRUpBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !encode_gr_up(context, *command, rows, false, buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    const GRUpBuffers empty{};
    bool passed = true;
    passed &= reject("GR up encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR up encoder rejects T>8", 9u, empty, "1..8");

    const std::size_t residual_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(HC_DIM) * HC_LOW_RANK *
        sizeof(std::uint16_t);
    BufferRef residual = context.buffers().allocate(
        "GR up validation residual", residual_bytes, error);
    BufferRef block_out = context.buffers().allocate(
        "GR up validation block output", block_bytes, error);
    BufferRef inject_prev = context.buffers().allocate(
        "GR up validation injection", injection_bytes, error);
    BufferRef norm = context.buffers().allocate(
        "GR up validation norm", norm_bytes, error);
    BufferRef rs = context.buffers().allocate(
        "GR up validation reciprocal scale", injection_bytes, error);
    BufferRef lo = context.buffers().allocate(
        "GR up validation low-rank input", low_rank_bytes, error);
    BufferRef hc_up = context.buffers().allocate(
        "GR up validation up weights", weight_bytes, error);
    BufferRef mixed = context.buffers().allocate(
        "GR up validation mixed output", block_bytes, error);
    if (!residual || !block_out || !inject_prev || !norm || !rs || !lo ||
        !hc_up || !mixed) {
        check(false, "GR up encoder validation buffers", error);
        return false;
    }
    const GRUpBuffers valid{
        residual, block_out, inject_prev, norm, rs, lo, hc_up, mixed};
    const auto reject_short = [&](const std::string& label,
                                  GRUpBuffers buffers,
                                  BufferRef GRUpBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR up encoder rejects short residual",
                           valid, &GRUpBuffers::residual);
    passed &= reject_short("GR up encoder rejects short block output",
                           valid, &GRUpBuffers::block_out);
    passed &= reject_short("GR up encoder rejects short injection",
                           valid, &GRUpBuffers::inject_prev);
    passed &= reject_short("GR up encoder rejects short norm",
                           valid, &GRUpBuffers::norm);
    passed &= reject_short("GR up encoder rejects short reciprocal scale",
                           valid, &GRUpBuffers::rs);
    passed &= reject_short("GR up encoder rejects short low-rank input",
                           valid, &GRUpBuffers::lo);
    passed &= reject_short("GR up encoder rejects short up weights",
                           valid, &GRUpBuffers::hc_up);
    passed &= reject_short("GR up encoder rejects short mixed output",
                           valid, &GRUpBuffers::mixed);

    GRUpBuffers misaligned = valid;
    misaligned.hc_up =
        hc_up.slice(1u, hc_up.length - 1u);
    passed &= reject("GR up encoder rejects misaligned BF16 weights",
                     1u, misaligned, "misaligned");

    GRUpBuffers overlapping = valid;
    overlapping.mixed = overlapping.residual;
    passed &= reject("GR up encoder rejects overlapping views",
                     1u, overlapping, "overlap");

    GRUpBuffers out_of_range = valid;
    out_of_range.hc_up =
        hc_up.slice(hc_up.length, sizeof(std::uint16_t));
    passed &= reject("GR up encoder rejects a weight view past its allocation",
                     1u, out_of_range, "not owned by this Context");

    auto submitted = context.make_command_buffer(
        "GR up submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "GR up validation command-buffer submission", error);
        return false;
    }
    const bool rejected_submitted =
        !encode_gr_up(context, *submitted, 1u, false, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "GR up encoder rejects a submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed, "GR up validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "GR up enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GR up validation command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_gr_up(context, *enqueued, 1u, false, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "GR up encoder rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(
        enqueued, "GR up enqueued validation", error);
    check(drained, "GR up enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool test_gr_up_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GR up shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    const std::size_t residual_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * HC_DIM * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * N_EMBD * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * HC * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * HC_LOW_RANK * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(HC_DIM) * HC_LOW_RANK *
        sizeof(std::uint16_t);
    BufferRef residual = context.buffers().allocate(
        "GR up shared validation residual", residual_bytes, error);
    BufferRef block_out = context.buffers().allocate(
        "GR up shared validation block output", block_bytes, error);
    BufferRef inject_prev = context.buffers().allocate(
        "GR up shared validation injection", injection_bytes, error);
    BufferRef norm = context.buffers().allocate(
        "GR up shared validation norm", norm_bytes, error);
    BufferRef rs = context.buffers().allocate(
        "GR up shared validation reciprocal scale", injection_bytes, error);
    BufferRef lo = context.buffers().allocate(
        "GR up shared validation low-rank input", low_rank_bytes, error);
    BufferRef hc_up_storage = context.buffers().allocate(
        "GR up shared validation aligned weight storage",
        weight_bytes + 32u, error);
    BufferRef mixed = context.buffers().allocate(
        "GR up shared validation mixed output", block_bytes, error);
    BufferRef foreign_residual = foreign_context.buffers().allocate(
        "GR up shared foreign residual", residual_bytes, error);
    if (!residual || !block_out || !inject_prev || !norm || !rs || !lo ||
        !hc_up_storage || !mixed || !foreign_residual) {
        check(false, "GR up shared Encoder validation buffers", error);
        return false;
    }
    const BufferRef hc_up = hc_up_storage.slice(16u, weight_bytes);
    const GRUpBuffers valid{
        residual, block_out, inject_prev, norm, rs, lo, hc_up, mixed};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRUpBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gr_up(context, encoder, rows, false,
                                        buffers, encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const GRUpBuffers empty{};
    passed &= reject("GR up shared Encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR up shared Encoder rejects T>8", 9u, empty, "1..8");
    const auto reject_short = [&](const std::string& label,
                                  GRUpBuffers buffers,
                                  BufferRef GRUpBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR up shared Encoder rejects short residual",
                           valid, &GRUpBuffers::residual);
    passed &= reject_short("GR up shared Encoder rejects short block output",
                           valid, &GRUpBuffers::block_out);
    passed &= reject_short("GR up shared Encoder rejects short injection",
                           valid, &GRUpBuffers::inject_prev);
    passed &= reject_short("GR up shared Encoder rejects short norm",
                           valid, &GRUpBuffers::norm);
    passed &= reject_short("GR up shared Encoder rejects short reciprocal scale",
                           valid, &GRUpBuffers::rs);
    passed &= reject_short("GR up shared Encoder rejects short low-rank input",
                           valid, &GRUpBuffers::lo);
    passed &= reject_short("GR up shared Encoder rejects short up weights",
                           valid, &GRUpBuffers::hc_up);
    passed &= reject_short("GR up shared Encoder rejects short mixed output",
                           valid, &GRUpBuffers::mixed);

    GRUpBuffers misaligned = valid;
    misaligned.hc_up = hc_up_storage.slice(4u, weight_bytes);
    passed &= reject("GR up shared Encoder rejects misaligned BF16 weights",
                     1u, misaligned, "misaligned");
    misaligned = valid;
    misaligned.norm = norm.slice(1u, norm.length - 1u);
    passed &= reject("GR up shared Encoder rejects misaligned norm",
                     1u, misaligned, "misaligned");

    GRUpBuffers out_of_range = valid;
    out_of_range.hc_up = hc_up.slice(hc_up.length, sizeof(std::uint16_t));
    passed &= reject(
        "GR up shared Encoder rejects out-of-range weight view",
        1u, out_of_range, "not owned by this Context");

    GRUpBuffers output_overlap = valid;
    output_overlap.mixed = residual;
    passed &= reject("GR up shared Encoder rejects output/input overlap",
                     1u, output_overlap, "overlap");
    GRUpBuffers input_overlap = valid;
    input_overlap.block_out = residual;
    passed &= reject("GR up shared Encoder rejects input/input overlap",
                     1u, input_overlap, "overlap");

    BufferRef partial_storage = context.buffers().allocate(
        "GR up shared validation partial-alias storage",
        residual_bytes + block_bytes, error);
    if (!partial_storage) {
        check(false, "GR up shared partial-alias buffer", error);
        return false;
    }
    GRUpBuffers partial_overlap = valid;
    partial_overlap.residual = partial_storage.slice(0u, residual_bytes);
    partial_overlap.mixed =
        partial_storage.slice(sizeof(float), block_bytes);
    passed &= reject("GR up shared Encoder rejects partial output alias",
                     1u, partial_overlap, "overlap");

    GRUpBuffers foreign_view = valid;
    foreign_view.residual = foreign_residual;
    passed &= reject("GR up shared Encoder rejects foreign Context view",
                     1u, foreign_view, "not owned by this Context");
    return passed;
}

bool run_gr_cvec_case(Context& context, std::uint32_t rows, std::uint32_t mode,
                      std::uint32_t layer, bool on, bool write,
                      bool zero_scale, const FusedSensitiveValues* fused_probe,
                      twin::InputSet set, const twin::Options& options,
                      bool shared_encoder = false) {
    constexpr std::uint32_t kLayer = 7u;
    const std::string label = "gr_cvec T=" + std::to_string(rows) +
        " layer=" + std::to_string(layer) +
        " mode=" + std::to_string(mode) +
        (on ? " on" : " off") + (write ? " write" : " no-write") +
        (zero_scale ? (layer == kLayer ? " zero-scale"
                                       : " uncovered-layer") : "") +
        (fused_probe != nullptr ? " fused-probe" : "") +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared-encoder" : "");
    const std::size_t residual_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t block_count =
        static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t inject_count =
        static_cast<std::size_t>(rows) * HC;
    const std::size_t direction_count =
        static_cast<std::size_t>(N_LAYER) * N_EMBD;
    const std::size_t residual_bytes = residual_count * sizeof(float);
    const std::size_t block_bytes = block_count * sizeof(float);
    const std::size_t inject_bytes = inject_count * sizeof(float);
    const std::size_t direction_bytes = direction_count * sizeof(float);
    const std::size_t scale_bytes =
        static_cast<std::size_t>(N_LAYER) * sizeof(float);

    std::vector<float> residual(residual_count);
    std::vector<float> block_out(block_count);
    std::vector<float> inject_prev(inject_count);
    std::vector<float> direction(direction_count, 0.0f);
    std::vector<float> layer_scale(N_LAYER, 0.0f);
    std::mt19937_64 rng(options.seed ^ 0x475243564543ull ^
                        (static_cast<std::uint64_t>(rows) << 32u) ^
                        (static_cast<std::uint64_t>(mode) << 24u) ^
                        (on ? 0x4f4eull : 0x4f4646ull) ^
                        (write ? 0x5752495445ull : 0x4e4f5752495445ull));
    for (std::uint32_t token = 0; token < rows; ++token) {
        for (std::uint32_t index = 0; index < HC_DIM; ++index) {
            residual[static_cast<std::size_t>(token) * HC_DIM + index] =
                input_value(set, token, index, rng);
        }
        for (std::uint32_t column = 0; column < N_EMBD; ++column) {
            block_out[static_cast<std::size_t>(token) * N_EMBD + column] =
                gamma_value(set, token, column, rng);
        }
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            inject_prev[static_cast<std::size_t>(token) * HC + stream] =
                (set == twin::InputSet::fixture && write &&
                 (!on || zero_scale))
                    ? 0.0f
                    : input_value(set, token, stream, rng);
        }
    }

    double direction_norm_squared = 0.0;
    for (std::uint32_t column = 0; column < N_EMBD; ++column) {
        const float value = gamma_value(set, layer, column, rng);
        direction_norm_squared +=
            static_cast<double>(value) * static_cast<double>(value);
        direction[static_cast<std::size_t>(layer) * N_EMBD + column] =
            value;
    }
    if (mode == 0u) {
        const double direction_norm = std::sqrt(direction_norm_squared);
        for (std::uint32_t column = 0; column < N_EMBD; ++column) {
            direction[static_cast<std::size_t>(layer) * N_EMBD + column] =
                static_cast<float>(
                    static_cast<double>(
                        direction[static_cast<std::size_t>(layer) * N_EMBD +
                                  column]) /
                    direction_norm);
        }
        if (!zero_scale) {
            layer_scale[layer] = layer == kLayer ? 1.75f : 0.625f;
        }
    } else {
        for (std::uint32_t column = 0; column < N_EMBD; ++column) {
            direction[static_cast<std::size_t>(layer) * N_EMBD + column] *=
                0.1f;
        }
        if (!zero_scale) {
            layer_scale[layer] = 1.0f;
        }
    }
    if (fused_probe != nullptr) {
        inject_prev[0] = 4.0f;
        block_out[0] = fused_probe->candidate;
        residual[0] = -fused_probe->rounded_product;
    }

    std::vector<twin::tf> expected;
    gr_cvec_reference(residual, direction, layer_scale, block_out,
                      inject_prev, layer, rows, mode, on, write, expected);
    if (fused_probe != nullptr) {
        expected[0] = twin::tf{fused_probe->fused_residual};
    }
    twin::GuardedBuffer residual_buffer;
    twin::GuardedBuffer direction_buffer;
    twin::GuardedBuffer scale_buffer;
    twin::GuardedBuffer on_buffer;
    twin::GuardedBuffer block_buffer;
    twin::GuardedBuffer inject_buffer;
    std::string err;
    check(residual_buffer.allocate(context, label + " residual",
                                   residual_bytes, err),
          label + " residual buffer", err);
    check(direction_buffer.allocate(context, label + " direction",
                                    direction_bytes, err),
          label + " direction buffer", err);
    check(scale_buffer.allocate(context, label + " layer_scale",
                                scale_bytes, err),
          label + " layer_scale buffer", err);
    check(on_buffer.allocate(context, label + " on", sizeof(std::int32_t), err),
          label + " on buffer", err);
    check(block_buffer.allocate(context, label + " block_out", block_bytes, err),
          label + " block_out buffer", err);
    check(inject_buffer.allocate(context, label + " inject_prev",
                                 inject_bytes, err),
          label + " inject_prev buffer", err);
    if (!residual_buffer.view() || !direction_buffer.view() ||
        !scale_buffer.view() || !on_buffer.view() || !block_buffer.view() ||
        !inject_buffer.view()) {
        return false;
    }
    std::memcpy(residual_buffer.payload().data(), residual.data(),
                residual_bytes);
    std::memcpy(direction_buffer.payload().data(), direction.data(),
                direction_bytes);
    std::memcpy(scale_buffer.payload().data(), layer_scale.data(),
                scale_bytes);
    const std::int32_t on_word = on ? 1 : 0;
    std::memcpy(on_buffer.payload().data(), &on_word, sizeof(on_word));
    std::memcpy(block_buffer.payload().data(), block_out.data(), block_bytes);
    std::memcpy(inject_buffer.payload().data(), inject_prev.data(),
                inject_bytes);
    residual_buffer.capture_before_dispatch();
    direction_buffer.capture_before_dispatch();
    scale_buffer.capture_before_dispatch();
    on_buffer.capture_before_dispatch();
    block_buffer.capture_before_dispatch();
    inject_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const GRControlVectorBuffers buffers{
        residual_buffer.view(), direction_buffer.view(), scale_buffer.view(),
        on_buffer.view(), block_buffer.view(), inject_buffer.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared GR cvec",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_gr_cvec(
                      context, encoder, rows, layer,
                      static_cast<GRControlMode>(mode), write, buffers,
                      encode_error);
              },
              err)
        : encode_gr_cvec(
              context, *command, rows, layer,
              static_cast<GRControlMode>(mode), write, buffers, err);
    check(encoded, label + " production GR cvec encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(residual_buffer.payload().data()),
        residual_count);
    const bool steer = on && layer_scale[layer] != 0.0f;
    const bool exact_result =
        (!steer && !write) || (steer && mode == 1u && !write) ||
        (set == twin::InputSet::fixture && write && !steer);
    const auto mismatch = twin::compare_floats(
        actual, expected,
        {exact_result ? twin::Comparison::bits : twin::Comparison::units,
         exact_result ? 0.0 : 64.0, 0.0, 0u});
    if (mismatch) {
        twin::report_failure(label.c_str(), options, rows, *mismatch);
    }
    check(!mismatch,
          label + (exact_result ? " bit-exact residual result"
                                : " U(64) ordered residual result"));
    if (fused_probe != nullptr) {
        const twin::tf separate =
            twin::tf{block_out[0]} * twin::tf{fused_probe->multiplier} +
            twin::tf{residual[0]};
        check(std::bit_cast<std::uint32_t>(actual[0]) ==
                  std::bit_cast<std::uint32_t>(
                      fused_probe->fused_residual) &&
                  std::bit_cast<std::uint32_t>(separate.value) !=
                      std::bit_cast<std::uint32_t>(
                          fused_probe->fused_residual),
              label + " fused-write FMA negative control",
              "separate=" + std::to_string(separate.value) +
                  ", fused=" +
                  std::to_string(fused_probe->fused_residual));
    }

    check(!direction_buffer.first_unwritten_change({}),
          label + " direction remains read-only");
    check(!scale_buffer.first_unwritten_change({}),
          label + " layer_scale remains read-only");
    check(!on_buffer.first_unwritten_change({}),
          label + " on flag remains read-only");
    check(!block_buffer.first_unwritten_change({}),
          label + " block_out remains read-only");
    check(!inject_buffer.first_unwritten_change({}),
          label + " inject_prev remains read-only");
    if (!steer && !write) {
        check(!residual_buffer.first_unwritten_change({}),
              label + " no-op residual is untouched");
    } else {
        const std::array<twin::ByteRange, 1> residual_written{{
            {0u, residual_bytes}}};
        const auto changed =
            residual_buffer.first_unwritten_change(residual_written);
        check(!changed, label + " residual guards remain unchanged",
              changed ? changed->reason : "");
    }
    return !mismatch;
}

bool run_gr_norm_case(Context& context, std::uint32_t rows, bool apply,
                      twin::InputSet set, const twin::Options& options,
                      bool shared_encoder = false) {
    const std::string label = "gr_norm T=" + std::to_string(rows) +
        (apply ? " apply" : " no-apply") +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared-encoder" : "");
    const std::size_t residual_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t block_count =
        static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t norm_count = HC_DIM;
    const std::size_t inject_count =
        static_cast<std::size_t>(rows) * HC;
    const std::size_t residual_bytes = residual_count * sizeof(float);
    const std::size_t block_bytes = block_count * sizeof(float);
    const std::size_t norm_bytes = norm_count * sizeof(float);
    const std::size_t inject_bytes = inject_count * sizeof(float);

    std::vector<float> residual(residual_count);
    std::vector<float> block_out(block_count);
    std::vector<float> norm(norm_count);
    std::vector<float> inject_prev(inject_count);
    std::mt19937_64 rng(options.seed ^ 0x47524e4f524dull ^
                        (static_cast<std::uint64_t>(rows) << 32u) ^
                        (apply ? 0x4150504c59ull : 0x4e4f415050ull));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t index = 0; index < HC_DIM; ++index) {
            residual[static_cast<std::size_t>(row) * HC_DIM + index] =
                input_value(set, row, index, rng);
        }
        for (std::uint32_t col = 0; col < N_EMBD; ++col) {
            block_out[static_cast<std::size_t>(row) * N_EMBD + col] =
                gamma_value(set, row, col, rng);
        }
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            inject_prev[static_cast<std::size_t>(row) * HC + stream] =
                input_value(set, row, stream, rng);
        }
    }
    for (std::uint32_t index = 0; index < HC_DIM; ++index) {
        norm[index] = gamma_value(set, 0u, index, rng);
    }

    std::vector<twin::tf> expected_xn;
    std::vector<twin::tf> expected_rs;
    gr_norm_reference(residual, block_out, norm, inject_prev, rows, apply,
                      expected_xn, expected_rs);

    twin::GuardedBuffer residual_buffer;
    twin::GuardedBuffer block_buffer;
    twin::GuardedBuffer norm_buffer;
    twin::GuardedBuffer inject_buffer;
    twin::GuardedBuffer xn_buffer;
    twin::GuardedBuffer rs_buffer;
    std::string err;
    check(residual_buffer.allocate(context, label + " residual",
                                   residual_bytes, err),
          label + " residual buffer", err);
    check(block_buffer.allocate(context, label + " block_out", block_bytes, err),
          label + " block_out buffer", err);
    check(norm_buffer.allocate(context, label + " norm", norm_bytes, err),
          label + " norm buffer", err);
    check(inject_buffer.allocate(context, label + " inject_prev",
                                 inject_bytes, err),
          label + " inject_prev buffer", err);
    check(xn_buffer.allocate(context, label + " xn", residual_bytes, err),
          label + " xn buffer", err);
    check(rs_buffer.allocate(context, label + " rs_out", inject_bytes, err),
          label + " rs_out buffer", err);
    if (!residual_buffer.view() || !block_buffer.view() || !norm_buffer.view() ||
        !inject_buffer.view() || !xn_buffer.view() || !rs_buffer.view()) {
        return false;
    }
    std::memcpy(residual_buffer.payload().data(), residual.data(), residual_bytes);
    std::memcpy(block_buffer.payload().data(), block_out.data(), block_bytes);
    std::memcpy(norm_buffer.payload().data(), norm.data(), norm_bytes);
    std::memcpy(inject_buffer.payload().data(), inject_prev.data(), inject_bytes);
    check(xn_buffer.poison_floats_against(
              expected_xn, twin::Comparison::units, err),
          label + " xn output poison", err);
    check(rs_buffer.poison_floats_against(
              expected_rs, twin::Comparison::units, err),
          label + " rs output poison", err);
    residual_buffer.capture_before_dispatch();
    block_buffer.capture_before_dispatch();
    norm_buffer.capture_before_dispatch();
    inject_buffer.capture_before_dispatch();
    xn_buffer.capture_before_dispatch();
    rs_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const GRNormBuffers buffers{
        residual_buffer.view(), block_buffer.view(), norm_buffer.view(),
        inject_buffer.view(), xn_buffer.view(), rs_buffer.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared GR norm",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_gr_norm(context, encoder, rows, apply, buffers,
                                        encode_error);
              },
              err)
        : encode_gr_norm(context, *command, rows, apply, buffers, err);
    check(encoded, label + " production GR norm encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual_xn = std::span<const float>(
        reinterpret_cast<const float*>(xn_buffer.payload().data()),
        residual_count);
    const auto actual_rs = std::span<const float>(
        reinterpret_cast<const float*>(rs_buffer.payload().data()),
        inject_count);
    const twin::Comparison comparison = apply
        ? twin::Comparison::units
        : twin::Comparison::bits;
    const auto xn_mismatch = twin::compare_floats(
        actual_xn, expected_xn,
        {comparison, apply ? 64.0 : 0.0, 0.0, 0u});
    const auto rs_mismatch = twin::compare_floats(
        actual_rs, expected_rs,
        {comparison, apply ? 64.0 : 0.0, 0.0, 0u});
    if (xn_mismatch) {
        twin::report_failure(label.c_str(), options, rows, *xn_mismatch);
    }
    if (rs_mismatch) {
        twin::report_failure((label + " rs_out").c_str(), options, rows,
                             *rs_mismatch);
    }

    std::size_t xn_exact = 0u;
    double xn_max_units = 0.0;
    for (std::size_t index = 0; index < residual_count; ++index) {
        if (std::bit_cast<std::uint32_t>(actual_xn[index]) ==
            std::bit_cast<std::uint32_t>(expected_xn[index].value)) {
            ++xn_exact;
        }
        const double bound =
            std::ldexp(expected_xn[index].magnitude, -24) +
            std::ldexp(1.0, -126);
        xn_max_units = std::max(
            xn_max_units,
            std::fabs(static_cast<double>(actual_xn[index]) -
                      expected_xn[index].value) / bound);
    }
    std::size_t rs_exact = 0u;
    double rs_max_units = 0.0;
    for (std::size_t index = 0; index < inject_count; ++index) {
        if (std::bit_cast<std::uint32_t>(actual_rs[index]) ==
            std::bit_cast<std::uint32_t>(expected_rs[index].value)) {
            ++rs_exact;
        }
        const double bound =
            std::ldexp(expected_rs[index].magnitude, -24) +
            std::ldexp(1.0, -126);
        rs_max_units = std::max(
            rs_max_units,
            std::fabs(static_cast<double>(actual_rs[index]) -
                      expected_rs[index].value) / bound);
    }
    check(!xn_mismatch, label + " documented GR norm output",
          "max observed U=" + std::to_string(xn_max_units) +
              ", exact bits=" + std::to_string(xn_exact) + "/" +
              std::to_string(residual_count));
    check(!rs_mismatch, label + " per-stream reciprocal scales",
          "max observed U=" + std::to_string(rs_max_units) +
              ", exact bits=" + std::to_string(rs_exact) + "/" +
              std::to_string(inject_count));

    const bool residual_unchanged =
        !residual_buffer.first_unwritten_change({});
    const bool block_unchanged = !block_buffer.first_unwritten_change({});
    const bool norm_unchanged = !norm_buffer.first_unwritten_change({});
    const bool inject_unchanged = !inject_buffer.first_unwritten_change({});
    check(residual_unchanged, label + " residual remains read-only");
    check(block_unchanged, label + " block_out remains read-only");
    check(norm_unchanged, label + " norm remains read-only");
    check(inject_unchanged, label + " inject_prev remains read-only");
    const std::array<twin::ByteRange, 1> xn_written{{{0u, residual_bytes}}};
    const std::array<twin::ByteRange, 1> rs_written{{{0u, inject_bytes}}};
    const bool xn_guards_unchanged =
        !xn_buffer.first_unwritten_change(xn_written);
    const bool rs_guards_unchanged =
        !rs_buffer.first_unwritten_change(rs_written);
    check(xn_guards_unchanged, label + " xn guards remain unchanged");
    check(rs_guards_unchanged, label + " rs_out guards remain unchanged");
    return !xn_mismatch && !rs_mismatch && residual_unchanged &&
           block_unchanged && norm_unchanged && inject_unchanged &&
           xn_guards_unchanged && rs_guards_unchanged;
}

bool test_gr_norm_encoder_validation(Context& context) {
    std::string error;
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRNormBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !encode_gr_norm(context, *command, rows, false, buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    const GRNormBuffers empty{};
    bool passed = true;
    passed &= reject("GR norm encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR norm encoder rejects T>8", 9u, empty, "1..8");

    const std::size_t residual_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    BufferRef residual = context.buffers().allocate(
        "GR norm validation residual", residual_bytes, error);
    BufferRef block_out = context.buffers().allocate(
        "GR norm validation block output", block_bytes, error);
    BufferRef norm = context.buffers().allocate(
        "GR norm validation norm", norm_bytes, error);
    BufferRef inject_prev = context.buffers().allocate(
        "GR norm validation injection", injection_bytes, error);
    BufferRef xn = context.buffers().allocate(
        "GR norm validation normalized output", residual_bytes, error);
    BufferRef rs_out = context.buffers().allocate(
        "GR norm validation scale output", injection_bytes, error);
    if (!residual || !block_out || !norm || !inject_prev || !xn || !rs_out) {
        check(false, "GR norm encoder validation buffers", error);
        return false;
    }
    const GRNormBuffers valid{
        residual, block_out, norm, inject_prev, xn, rs_out};
    const auto reject_short = [&](const std::string& label,
                                  GRNormBuffers buffers,
                                  BufferRef GRNormBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR norm encoder rejects short residual",
                           valid, &GRNormBuffers::residual);
    passed &= reject_short("GR norm encoder rejects short block output",
                           valid, &GRNormBuffers::block_out);
    passed &= reject_short("GR norm encoder rejects short norm",
                           valid, &GRNormBuffers::norm);
    passed &= reject_short("GR norm encoder rejects short injection",
                           valid, &GRNormBuffers::inject_prev);
    passed &= reject_short("GR norm encoder rejects short normalized output",
                           valid, &GRNormBuffers::xn);
    passed &= reject_short("GR norm encoder rejects short scale output",
                           valid, &GRNormBuffers::rs_out);

    GRNormBuffers misaligned = valid;
    misaligned.norm =
        norm.slice(1u, norm.length - 1u);
    passed &= reject("GR norm encoder rejects misaligned norm",
                     1u, misaligned, "misaligned");

    GRNormBuffers overlapping = valid;
    overlapping.xn = overlapping.residual;
    passed &= reject("GR norm encoder rejects overlapping views",
                     1u, overlapping, "overlap");

    GRNormBuffers out_of_range = valid;
    out_of_range.norm = norm.slice(norm.length, sizeof(float));
    passed &= reject("GR norm encoder rejects a view past its allocation",
                     1u, out_of_range, "not owned by this Context");

    auto submitted = context.make_command_buffer(
        "GR norm submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "GR norm validation command-buffer submission", error);
        return false;
    }
    const bool rejected_submitted =
        !encode_gr_norm(context, *submitted, 1u, false, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "GR norm encoder rejects a submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "GR norm validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "GR norm enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GR norm validation command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_gr_norm(context, *enqueued, 1u, false, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "GR norm encoder rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(
        enqueued, "GR norm enqueued validation", error);
    check(drained, "GR norm enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool test_gr_norm_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GR norm shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    const std::size_t residual_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    BufferRef residual = context.buffers().allocate(
        "GR norm shared validation residual", residual_bytes, error);
    BufferRef block_out = context.buffers().allocate(
        "GR norm shared validation block output", block_bytes, error);
    BufferRef norm = context.buffers().allocate(
        "GR norm shared validation norm", norm_bytes, error);
    BufferRef inject_prev = context.buffers().allocate(
        "GR norm shared validation injection", injection_bytes, error);
    BufferRef xn = context.buffers().allocate(
        "GR norm shared validation normalized output", residual_bytes, error);
    BufferRef rs_out = context.buffers().allocate(
        "GR norm shared validation scale output", injection_bytes, error);
    if (!residual || !block_out || !norm || !inject_prev || !xn || !rs_out) {
        check(false, "GR norm shared Encoder validation buffers", error);
        return false;
    }
    const GRNormBuffers valid{
        residual, block_out, norm, inject_prev, xn, rs_out};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRNormBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gr_norm(context, encoder, rows, false,
                                          buffers, encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = reject("GR norm shared Encoder rejects T=0",
                         0u, GRNormBuffers{}, "1..8");
    passed &= reject("GR norm shared Encoder rejects T>8",
                     9u, GRNormBuffers{}, "1..8");
    const auto reject_short = [&](const std::string& label,
                                  BufferRef GRNormBuffers::*member) {
        GRNormBuffers short_view = valid;
        (short_view.*member).length = 1u;
        return reject(label, 1u, short_view, "undersized");
    };
    passed &= reject_short("GR norm shared Encoder rejects short residual",
                           &GRNormBuffers::residual);
    passed &= reject_short("GR norm shared Encoder rejects short block output",
                           &GRNormBuffers::block_out);
    passed &= reject_short("GR norm shared Encoder rejects short norm",
                           &GRNormBuffers::norm);
    passed &= reject_short("GR norm shared Encoder rejects short injection",
                           &GRNormBuffers::inject_prev);
    passed &= reject_short(
        "GR norm shared Encoder rejects short normalized output",
        &GRNormBuffers::xn);
    passed &= reject_short("GR norm shared Encoder rejects short scale output",
                           &GRNormBuffers::rs_out);

    GRNormBuffers misaligned = valid;
    misaligned.norm = norm.slice(1u, norm.length - 1u);
    passed &= reject("GR norm shared Encoder rejects misaligned norm",
                     1u, misaligned, "misaligned");
    GRNormBuffers out_of_range = valid;
    out_of_range.norm = norm.slice(norm.length, sizeof(float));
    passed &= reject("GR norm shared Encoder rejects out-of-range view",
                     1u, out_of_range, "not owned by this Context");

    GRNormBuffers exact_alias = valid;
    exact_alias.xn = exact_alias.residual;
    passed &= reject("GR norm shared Encoder rejects exact output alias",
                     1u, exact_alias, "overlap");
    BufferRef partial_storage = context.buffers().allocate(
        "GR norm shared validation partial-alias storage",
        residual_bytes + sizeof(float), error);
    if (!partial_storage) {
        check(false, "GR norm shared partial-alias buffer", error);
        return false;
    }
    GRNormBuffers partial_alias = valid;
    partial_alias.residual = partial_storage;
    partial_alias.xn = partial_storage.slice(sizeof(float), residual_bytes);
    passed &= reject("GR norm shared Encoder rejects partial output alias",
                     1u, partial_alias, "overlap");

    BufferRef foreign_residual = foreign_context.buffers().allocate(
        "GR norm shared foreign residual", residual_bytes, error);
    if (!foreign_residual) {
        check(false, "GR norm shared foreign buffer", error);
        return false;
    }
    GRNormBuffers foreign_view = valid;
    foreign_view.residual = foreign_residual;
    passed &= reject("GR norm shared Encoder rejects foreign Context view",
                     1u, foreign_view, "not owned by this Context");
    return passed;
}

bool run_gr_post_case(Context& context, std::uint32_t rows, twin::InputSet set,
                      const twin::Options& options, bool in_place,
                      bool shared_encoder = false) {
    const std::string label = "gr_post rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) + (in_place ? " in-place" : "") +
        (shared_encoder ? " shared-encoder" : "");
    const std::size_t residual_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t block_count =
        static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t inject_count =
        static_cast<std::size_t>(rows) * HC;
    const std::size_t residual_bytes = residual_count * sizeof(float);
    const std::size_t block_bytes = block_count * sizeof(float);
    const std::size_t inject_bytes = inject_count * sizeof(float);

    std::vector<float> residual(residual_count);
    std::vector<float> block_out(block_count);
    std::vector<float> inject(inject_count);
    std::mt19937_64 rng(options.seed ^ 0x4752504f5354ull ^
                        (static_cast<std::uint64_t>(rows) << 32u));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t index = 0; index < HC_DIM; ++index) {
            residual[static_cast<std::size_t>(row) * HC_DIM + index] =
                input_value(set, row, index, rng);
        }
        for (std::uint32_t col = 0; col < N_EMBD; ++col) {
            block_out[static_cast<std::size_t>(row) * N_EMBD + col] =
                gamma_value(set, row, col, rng);
        }
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            inject[static_cast<std::size_t>(row) * HC + stream] =
                input_value(set, row, stream, rng);
        }
    }
    std::vector<twin::tf> expected;
    gr_post_reference(residual, block_out, inject, rows, expected);

    std::string err;
    twin::GuardedBuffer residual_buffer;
    twin::GuardedBuffer block_buffer;
    twin::GuardedBuffer inject_buffer;
    twin::GuardedBuffer separate_output;
    check(residual_buffer.allocate(context, label + " residual",
                                   residual_bytes, err),
          label + " residual buffer", err);
    check(block_buffer.allocate(context, label + " block_out", block_bytes, err),
          label + " block_out buffer", err);
    check(inject_buffer.allocate(context, label + " inject", inject_bytes, err),
          label + " inject buffer", err);
    if (!in_place) {
        check(separate_output.allocate(context, label + " output",
                                       residual_bytes, err),
              label + " output buffer", err);
    }
    if (!residual_buffer.view() || !block_buffer.view() ||
        !inject_buffer.view() || (!in_place && !separate_output.view())) {
        return false;
    }
    std::memcpy(residual_buffer.payload().data(), residual.data(), residual_bytes);
    std::memcpy(block_buffer.payload().data(), block_out.data(), block_bytes);
    std::memcpy(inject_buffer.payload().data(), inject.data(), inject_bytes);
    if (!in_place) {
        check(separate_output.poison_floats_against(
                  expected, twin::Comparison::units, err),
              label + " output poison", err);
    }
    residual_buffer.capture_before_dispatch();
    block_buffer.capture_before_dispatch();
    inject_buffer.capture_before_dispatch();
    if (!in_place) separate_output.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const GRPostBuffers buffers{
        residual_buffer.view(), block_buffer.view(), inject_buffer.view(),
        in_place ? residual_buffer.view() : separate_output.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared GR post",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_gr_post(context, encoder, rows, buffers,
                                        encode_error);
              },
              err)
        : encode_gr_post(context, *command, rows, buffers, err);
    check(encoded, label + " production GR post encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto output = in_place ? residual_buffer.payload()
                                 : separate_output.payload();
    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(output.data()), residual_count);
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 16.0, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    std::size_t exact_count = 0u;
    double max_units = 0.0;
    for (std::size_t i = 0; i < residual_count; ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) ==
            std::bit_cast<std::uint32_t>(expected[i].value)) {
            ++exact_count;
        }
        const double bound =
            std::ldexp(expected[i].magnitude, -24) + std::ldexp(1.0, -126);
        max_units = std::max(
            max_units, std::fabs(static_cast<double>(actual[i]) - expected[i].value) / bound);
    }
    check(!mismatch, label + " U(16) native-post result",
          "max observed U=" + std::to_string(max_units) + ", exact bits=" +
              std::to_string(exact_count) + "/" + std::to_string(residual_count));

    check(!block_buffer.first_unwritten_change({}),
          label + " block_out remains read-only");
    check(!inject_buffer.first_unwritten_change({}),
          label + " inject remains read-only");
    if (in_place) {
        const std::array<twin::ByteRange, 1> written{{{0u, residual_bytes}}};
        const auto changed = residual_buffer.first_unwritten_change(written);
        check(!changed, label + " in-place output guards remain unchanged",
              changed ? changed->reason : "");
    } else {
        check(!residual_buffer.first_unwritten_change({}),
              label + " residual remains read-only");
        const std::array<twin::ByteRange, 1> written{{{0u, residual_bytes}}};
        const auto changed = separate_output.first_unwritten_change(written);
        check(!changed, label + " output guards remain unchanged",
              changed ? changed->reason : "");
    }
    return !mismatch;
}

bool test_gr_post_encoder_validation(Context& context,
                                     Context& foreign_context) {
    std::string error;
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRPostBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !encode_gr_post(context, *command, rows, buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    const GRPostBuffers empty{};
    bool passed = true;
    passed &= reject("GR post encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR post encoder rejects T>8", 9u, empty, "1..8");

    const std::size_t residual_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    BufferRef residual = context.buffers().allocate(
        "GR post validation residual", residual_bytes, error);
    BufferRef block_out = context.buffers().allocate(
        "GR post validation block output", block_bytes, error);
    BufferRef inject = context.buffers().allocate(
        "GR post validation injection", injection_bytes, error);
    BufferRef output = context.buffers().allocate(
        "GR post validation output", residual_bytes, error);
    BufferRef partial_backing = context.buffers().allocate(
        "GR post validation partial-alias backing", residual_bytes + 16u,
        error);
    if (!residual || !block_out || !inject || !output || !partial_backing) {
        check(false, "GR post encoder validation buffers", error);
        return false;
    }
    const GRPostBuffers valid{residual, block_out, inject, output};
    BufferRef foreign_residual = foreign_context.buffers().allocate(
        "GR post foreign-context residual", residual_bytes, error);
    if (!foreign_residual) {
        check(false, "GR post foreign-context validation buffer", error);
        return false;
    }
    GRPostBuffers foreign_view = valid;
    foreign_view.residual = foreign_residual;
    passed &= reject("GR post encoder rejects a foreign-context buffer",
                     1u, foreign_view, "not owned by this Context");

    auto foreign_command = foreign_context.make_command_buffer(
        "GR post foreign-context command-buffer validation", error);
    if (!foreign_command) {
        check(false, "GR post foreign-context command buffer", error);
        return false;
    }
    const bool rejected_foreign_command =
        !encode_gr_post(context, *foreign_command, 1u, valid, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "GR post encoder rejects a foreign-context command buffer", error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    const auto reject_short = [&](const std::string& label,
                                  GRPostBuffers buffers,
                                  BufferRef GRPostBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR post encoder rejects short residual",
                           valid, &GRPostBuffers::residual);
    passed &= reject_short("GR post encoder rejects short block output",
                           valid, &GRPostBuffers::block_out);
    passed &= reject_short("GR post encoder rejects short injection",
                           valid, &GRPostBuffers::inject);
    passed &= reject_short("GR post encoder rejects short output",
                           valid, &GRPostBuffers::output);

    GRPostBuffers misaligned = valid;
    misaligned.inject =
        inject.slice(1u, inject.length - 1u);
    passed &= reject("GR post encoder rejects misaligned injection",
                     1u, misaligned, "misaligned");

    GRPostBuffers cross_role_overlap = valid;
    cross_role_overlap.block_out = cross_role_overlap.residual;
    passed &= reject("GR post encoder rejects residual/block overlap",
                     1u, cross_role_overlap,
                     "exact residual/output alias");

    BufferRef partial_residual =
        partial_backing.slice(0u, residual_bytes);
    BufferRef partial_output =
        partial_backing.slice(sizeof(float), residual_bytes);
    GRPostBuffers partial_alias = valid;
    partial_alias.residual = partial_residual;
    partial_alias.output = partial_output;
    passed &= reject("GR post encoder rejects partial residual/output alias",
                     1u, partial_alias,
                     "exact residual/output alias");

    GRPostBuffers out_of_range = valid;
    out_of_range.output = output.slice(output.length, sizeof(float));
    passed &= reject("GR post encoder rejects a view past its allocation",
                     1u, out_of_range, "not owned by this Context");

    auto submitted = context.make_command_buffer(
        "GR post submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "GR post validation command-buffer submission", error);
        return false;
    }
    const bool rejected_submitted =
        !encode_gr_post(context, *submitted, 1u, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "GR post encoder rejects a submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "GR post validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "GR post enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GR post validation command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_gr_post(context, *enqueued, 1u, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "GR post encoder rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(
        enqueued, "GR post enqueued validation", error);
    check(drained, "GR post enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool run_gr_down_silu_case(Context& context, std::uint32_t rows,
                           twin::InputSet set, const twin::Options& options,
                           bool shared_encoder = false) {
    const std::string label = "gr_down_silu rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared-encoder" : "");
    const std::uint32_t n_values = rows * HC_LOW_RANK;
    const std::size_t bytes = static_cast<std::size_t>(n_values) * sizeof(float);
    std::vector<float> input(n_values);
    std::vector<twin::tf> expected(n_values);
    std::mt19937_64 rng(options.seed ^ 0x47524453494c55ull ^
                        (static_cast<std::uint64_t>(rows) << 32u));
    for (std::uint32_t index = 0; index < n_values; ++index) {
        const std::uint32_t row = index / HC_LOW_RANK;
        const std::uint32_t col = index % HC_LOW_RANK;
        input[index] = input_value(set, row, col, rng);
        const twin::tf scaled =
            twin::fma(twin::tf{0.25f}, twin::tf{input[index]}, twin::tf{});
        expected[index] =
            scaled / (twin::tf{1.0f} + twin::accurate_exp(-scaled));
    }

    std::string err;
    twin::GuardedBuffer buffer;
    check(buffer.allocate(context, label + " in-place", bytes, err),
          label + " buffer", err);
    if (!buffer.view()) return false;
    std::memcpy(buffer.payload().data(), input.data(), bytes);
    buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared GR down SiLU",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_gr_down_silu(context, encoder, rows,
                                             buffer.view(), encode_error);
              },
              err)
        : encode_gr_down_silu(context, *command, rows, buffer.view(), err);
    check(encoded, label + " production GR down SiLU encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(buffer.payload().data()), n_values);
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 16.0, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    std::size_t exact_count = 0u;
    std::size_t negative_zero_count = 0u;
    std::size_t negative_zero_matches = 0u;
    double max_units = 0.0;
    for (std::size_t i = 0; i < n_values; ++i) {
        const std::uint32_t actual_bits = std::bit_cast<std::uint32_t>(actual[i]);
        const std::uint32_t expected_bits =
            std::bit_cast<std::uint32_t>(expected[i].value);
        if (actual_bits == expected_bits) {
            ++exact_count;
        }
        if (set == twin::InputSet::edge &&
            std::bit_cast<std::uint32_t>(input[i]) == 0x80000000u) {
            ++negative_zero_count;
            if (actual_bits == 0u && expected_bits == 0u) {
                ++negative_zero_matches;
            }
        }
        const double bound =
            std::ldexp(expected[i].magnitude, -24) + std::ldexp(1.0, -126);
        max_units = std::max(
            max_units, std::fabs(static_cast<double>(actual[i]) - expected[i].value) / bound);
    }
    check(!mismatch, label + " U(16) in-place SiLU result",
          "max observed U=" + std::to_string(max_units) + ", exact bits=" +
              std::to_string(exact_count) + "/" + std::to_string(n_values));
    check(set != twin::InputSet::edge ||
              (negative_zero_count > 0u &&
               negative_zero_matches == negative_zero_count),
          label + " negative-zero FMA bits",
          std::to_string(negative_zero_matches) + "/" +
              std::to_string(negative_zero_count) + " mapped to positive zero");
    const std::array<twin::ByteRange, 1> written{{{0u, bytes}}};
    const auto changed = buffer.first_unwritten_change(written);
    check(!changed, label + " guards remain unchanged",
          changed ? changed->reason : "");
    return !mismatch;
}

bool test_gr_down_silu_encoder_validation(Context& context,
                                          Context& foreign_context) {
    std::string error;
    auto rejected_command = context.make_command_buffer(
        "GR down SiLU rejected-input validation", error);
    if (!rejected_command) {
        check(false, "GR down SiLU rejection-validation command buffer", error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            BufferRef low_rank,
                            const std::string& expected_error) {
        const bool rejected =
            !encode_gr_down_silu(context, *rejected_command, rows, low_rank,
                                  error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    const BufferRef empty{};
    bool passed = true;
    passed &= reject("GR down SiLU encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR down SiLU encoder rejects T>8", 9u, empty, "1..8");

    const std::size_t required_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float);
    BufferRef low_rank = context.buffers().allocate(
        "GR down SiLU validation low-rank buffer", required_bytes, error);
    BufferRef foreign_low_rank = foreign_context.buffers().allocate(
        "GR down SiLU foreign-context buffer", required_bytes, error);
    if (!low_rank || !foreign_low_rank) {
        check(false, "GR down SiLU encoder validation buffers", error);
        rejected_command->discard();
        return false;
    }
    passed &= reject("GR down SiLU encoder rejects a foreign-context buffer",
                     1u, foreign_low_rank, "not owned by this Context");

    BufferRef short_view = low_rank;
    short_view.length = sizeof(float);
    passed &= reject("GR down SiLU encoder rejects a short buffer",
                     1u, short_view, "undersized");
    BufferRef one_row_view = low_rank;
    passed &= reject("GR down SiLU encoder rejects one-row storage for T=2",
                     2u, one_row_view, "undersized");
    BufferRef misaligned = low_rank;
    misaligned = low_rank.slice(1u, low_rank.length - 1u);
    passed &= reject("GR down SiLU encoder rejects a misaligned buffer",
                     1u, misaligned, "misaligned");
    BufferRef out_of_range = low_rank;
    out_of_range = low_rank.slice(low_rank.length, sizeof(float));
    passed &= reject("GR down SiLU encoder rejects a view past its allocation",
                     1u, out_of_range, "not owned by this Context");
    rejected_command->discard();

    auto foreign_command = foreign_context.make_command_buffer(
        "GR down SiLU foreign command-buffer validation", error);
    if (!foreign_command) {
        check(false, "GR down SiLU foreign command buffer", error);
        return false;
    }
    const bool rejected_foreign_command =
        !encode_gr_down_silu(context, *foreign_command, 1u, low_rank, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "GR down SiLU encoder rejects a foreign-context command buffer",
          error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    auto submitted = context.make_command_buffer(
        "GR down SiLU submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "GR down SiLU validation command-buffer submission",
              error);
        return false;
    }
    const bool rejected_submitted =
        !encode_gr_down_silu(context, *submitted, 1u, low_rank, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "GR down SiLU encoder rejects a submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "GR down SiLU validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "GR down SiLU enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GR down SiLU validation command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_gr_down_silu(context, *enqueued, 1u, low_rank, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "GR down SiLU encoder rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(
        enqueued, "GR down SiLU enqueued validation", error);
    check(drained, "GR down SiLU enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool test_gr_down_silu_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GR down SiLU shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    const std::size_t one_row_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * sizeof(float);
    const std::size_t all_rows_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * one_row_bytes;
    BufferRef low_rank = context.buffers().allocate(
        "GR down SiLU shared validation low-rank buffer",
        all_rows_bytes, error);
    BufferRef foreign_low_rank = foreign_context.buffers().allocate(
        "GR down SiLU shared foreign low-rank buffer",
        one_row_bytes, error);
    if (!low_rank || !foreign_low_rank) {
        check(false, "GR down SiLU shared Encoder validation buffers", error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            BufferRef view, const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gr_down_silu(context, encoder, rows, view,
                                               encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const BufferRef empty{};
    passed &= reject("GR down SiLU shared Encoder rejects T=0", 0u, empty,
                     "1..8");
    passed &= reject("GR down SiLU shared Encoder rejects T>8", 9u, empty,
                     "1..8");
    passed &= reject(
        "GR down SiLU shared Encoder rejects foreign Context view", 1u,
        foreign_low_rank, "not owned by this Context");

    BufferRef short_view = low_rank;
    short_view.length = sizeof(float);
    passed &= reject("GR down SiLU shared Encoder rejects short view",
                     1u, short_view, "undersized");
    BufferRef one_row_view = low_rank;
    one_row_view.length = one_row_bytes;
    passed &= reject(
        "GR down SiLU shared Encoder rejects one-row storage for T=2",
        2u, one_row_view, "undersized");
    BufferRef misaligned = low_rank.slice(1u, low_rank.length - 1u);
    passed &= reject("GR down SiLU shared Encoder rejects misaligned view",
                     1u, misaligned, "misaligned");
    BufferRef out_of_range = low_rank.slice(low_rank.length, sizeof(float));
    passed &= reject(
        "GR down SiLU shared Encoder rejects out-of-range view",
        1u, out_of_range, "not owned by this Context");
    return passed;
}

bool run_gr_pre_gated_case(Context& context, std::uint32_t rows,
                           twin::InputSet set, const twin::Options& options,
                           FusedSensitiveValues fused_sensitive,
                           bool shared_encoder = false) {
    const std::string label = "gr_pre_gated rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) +
        (shared_encoder ? " shared-encoder" : "");
    const std::size_t input_count = static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t output_count = static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t output_bytes = output_count * sizeof(float);
    std::vector<float> xn(input_count);
    std::vector<float> gate(input_count);
    std::mt19937_64 rng(options.seed ^ 0x4752505245474154ull ^
                        (static_cast<std::uint64_t>(rows) << 32u));
    for (std::size_t index = 0; index < input_count; ++index) {
        const std::uint32_t row =
            static_cast<std::uint32_t>(index / HC_DIM);
        const std::uint32_t col =
            static_cast<std::uint32_t>(index % HC_DIM);
        xn[index] = input_value(set, row, col, rng);
        gate[index] = input_value(set, row, col + HC_DIM, rng);
        if (set == twin::InputSet::edge) {
            const std::uint32_t stream = col / N_EMBD;
            const std::uint32_t output_col = col % N_EMBD;
            if (output_col == 0u) {
                xn[index] = -0.0f;
                gate[index] = 0.0f;
            } else if (output_col == 1u) {
                constexpr std::array<float, HC> order_sensitive_xn{
                    33554432.0f, 2.0f, -33554432.0f, 2.0f};
                xn[index] = order_sensitive_xn[stream];
                gate[index] = 0.0f;
            } else if (output_col == 2u) {
                if (stream == 0u) {
                    xn[index] = -fused_sensitive.rounded_product;
                    gate[index] = 64.0f;
                } else if (stream == 1u) {
                    xn[index] = fused_sensitive.candidate;
                    gate[index] = 1.0f;
                } else {
                    xn[index] = 0.0f;
                    gate[index] = 0.0f;
                }
            }
        }
    }

    std::vector<twin::tf> expected;
    gr_pre_gated_reference(xn, gate, rows, expected);
    if (set == twin::InputSet::edge) {
        const std::size_t order_index = 1u;
        const twin::tf p0{xn[1u] * 0.5f};
        const twin::tf p1{xn[N_EMBD + 1u] * 0.5f};
        const twin::tf p2{xn[2u * N_EMBD + 1u] * 0.5f};
        const twin::tf p3{xn[3u * N_EMBD + 1u] * 0.5f};
        const twin::tf reordered_sum = (p0 + p2) + (p1 + p3);
        const twin::tf reordered_output =
            twin::fma(twin::tf{0.25f}, reordered_sum, twin::tf{});
        check(std::bit_cast<std::uint32_t>(reordered_output.value) !=
                  std::bit_cast<std::uint32_t>(expected[order_index].value) &&
                  std::bit_cast<std::uint32_t>(expected[order_index].value) ==
                      std::bit_cast<std::uint32_t>(0.25f),
              label + " reordered-stream negative control",
              "ordered=" + std::to_string(expected[order_index].value) +
                  ", reordered=" + std::to_string(reordered_output.value));

        const float fused_output =
            std::fma(0.25f, fused_sensitive.fused_residual, 0.0f);
        const float separate_output = std::fma(
            0.25f, -fused_sensitive.rounded_product +
                       fused_sensitive.rounded_product,
            0.0f);
        check(std::bit_cast<std::uint32_t>(separate_output) == 0u &&
                  fused_output != separate_output,
              label + " fused-product negative control",
              "separate product/sum=" + std::to_string(separate_output) +
                  ", fused=" + std::to_string(fused_output));
    }

    std::string err;
    twin::GuardedBuffer xn_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer output_buffer;
    check(xn_buffer.allocate(context, label + " xn", input_bytes, err),
          label + " xn buffer", err);
    check(gate_buffer.allocate(context, label + " gate", input_bytes, err),
          label + " gate buffer", err);
    check(output_buffer.allocate(context, label + " output", output_bytes, err),
          label + " output buffer", err);
    if (!xn_buffer.view() || !gate_buffer.view() || !output_buffer.view()) {
        return false;
    }
    std::memcpy(xn_buffer.payload().data(), xn.data(), input_bytes);
    std::memcpy(gate_buffer.payload().data(), gate.data(), input_bytes);
    check(output_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " output poison", err);
    xn_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const GRPreGatedBuffers buffers{
        xn_buffer.view(), gate_buffer.view(), output_buffer.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared GR pre-gated",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_gr_pre_gated(context, encoder, rows, buffers,
                                             encode_error);
              },
              err)
        : encode_gr_pre_gated(context, *command, rows, buffers, err);
    check(encoded, label + " production GR pre-gated encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(output_buffer.payload().data()), output_count);
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 16.0, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    std::size_t exact_count = 0u;
    std::size_t negative_zero_count = 0u;
    std::size_t negative_zero_matches = 0u;
    std::size_t order_sensitive_exact_count = 0u;
    std::size_t fused_sensitive_exact_count = 0u;
    double max_units = 0.0;
    for (std::size_t index = 0; index < output_count; ++index) {
        const std::uint32_t actual_bits = std::bit_cast<std::uint32_t>(actual[index]);
        const std::uint32_t expected_bits =
            std::bit_cast<std::uint32_t>(expected[index].value);
        if (actual_bits == expected_bits) ++exact_count;
        if (set == twin::InputSet::edge && index % N_EMBD == 0u) {
            ++negative_zero_count;
            if (actual_bits == 0u && expected_bits == 0u) {
                ++negative_zero_matches;
            }
        }
        if (set == twin::InputSet::edge && index % N_EMBD == 1u &&
            actual_bits == std::bit_cast<std::uint32_t>(0.25f)) {
            ++order_sensitive_exact_count;
        }
        if (set == twin::InputSet::edge && index % N_EMBD == 2u &&
            actual_bits == 0u) {
            ++fused_sensitive_exact_count;
        }
        const double bound =
            std::ldexp(expected[index].magnitude, -24) + std::ldexp(1.0, -126);
        max_units = std::max(
            max_units,
            std::fabs(static_cast<double>(actual[index]) - expected[index].value) / bound);
    }
    check(!mismatch, label + " U(16) pre-gated result",
          "max observed U=" + std::to_string(max_units) + ", exact bits=" +
              std::to_string(exact_count) + "/" + std::to_string(output_count));
    check(set != twin::InputSet::edge ||
              (negative_zero_count > 0u &&
               negative_zero_matches == negative_zero_count),
          label + " negative-zero sum/FMA bits",
          std::to_string(negative_zero_matches) + "/" +
              std::to_string(negative_zero_count) + " mapped to positive zero");
    if (set == twin::InputSet::edge) {
        check(order_sensitive_exact_count == rows,
              label + " stream-order sensitive exact result",
              std::to_string(order_sensitive_exact_count) + "/" +
                  std::to_string(rows) + " matched 0.25 exactly");
        check(fused_sensitive_exact_count == rows,
              label + " separate-product exact result",
              std::to_string(fused_sensitive_exact_count) + "/" +
                  std::to_string(rows) + " matched positive zero exactly");
    }

    check(!xn_buffer.first_unwritten_change({}),
          label + " xn remains read-only");
    check(!gate_buffer.first_unwritten_change({}),
          label + " gate remains read-only");
    const std::array<twin::ByteRange, 1> written{{{0u, output_bytes}}};
    const auto changed = output_buffer.first_unwritten_change(written);
    check(!changed, label + " output guards remain unchanged",
          changed ? changed->reason : "");
    return !mismatch;
}

bool test_gr_pre_gated_encoder_validation(Context& context,
                                          Context& foreign_context) {
    std::string error;
    auto rejected_command = context.make_command_buffer(
        "GR pre-gated rejected-input validation", error);
    if (!rejected_command) {
        check(false, "GR pre-gated rejection-validation command buffer",
              error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRPreGatedBuffers& buffers,
                            const std::string& expected_error) {
        const bool rejected =
            !encode_gr_pre_gated(context, *rejected_command, rows, buffers,
                                 error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    const GRPreGatedBuffers empty{};
    bool passed = true;
    passed &= reject("GR pre-gated encoder rejects T=0", 0u, empty, "1..8");
    passed &= reject("GR pre-gated encoder rejects T>8", 9u, empty, "1..8");

    const std::size_t input_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    BufferRef xn = context.buffers().allocate(
        "GR pre-gated validation normalized input", 2u * input_bytes, error);
    BufferRef gate = context.buffers().allocate(
        "GR pre-gated validation gate", 2u * input_bytes, error);
    BufferRef mixed = context.buffers().allocate(
        "GR pre-gated validation output", 2u * output_bytes, error);
    BufferRef foreign_xn = foreign_context.buffers().allocate(
        "GR pre-gated foreign-context normalized input", input_bytes, error);
    if (!xn || !gate || !mixed || !foreign_xn) {
        check(false, "GR pre-gated encoder validation buffers", error);
        rejected_command->discard();
        return false;
    }
    const GRPreGatedBuffers valid{xn, gate, mixed};
    GRPreGatedBuffers foreign_view = valid;
    foreign_view.xn = foreign_xn;
    passed &= reject("GR pre-gated encoder rejects a foreign-context buffer",
                     1u, foreign_view, "not owned by this Context");

    const auto reject_short = [&](const std::string& label,
                                  GRPreGatedBuffers buffers,
                                  BufferRef GRPreGatedBuffers::*member) {
        (buffers.*member).length = sizeof(float);
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR pre-gated encoder rejects short xn",
                           valid, &GRPreGatedBuffers::xn);
    passed &= reject_short("GR pre-gated encoder rejects short gate",
                           valid, &GRPreGatedBuffers::gate);
    passed &= reject_short("GR pre-gated encoder rejects short output",
                           valid, &GRPreGatedBuffers::mixed);
    const auto reject_one_row_for_two = [&](
        const std::string& label, GRPreGatedBuffers buffers,
        BufferRef GRPreGatedBuffers::*member, std::size_t one_row_bytes) {
        (buffers.*member).length = one_row_bytes;
        return reject(label, 2u, buffers, "undersized");
    };
    passed &= reject_one_row_for_two(
        "GR pre-gated encoder rejects one-row xn storage for T=2",
        valid, &GRPreGatedBuffers::xn, input_bytes);
    passed &= reject_one_row_for_two(
        "GR pre-gated encoder rejects one-row gate storage for T=2",
        valid, &GRPreGatedBuffers::gate, input_bytes);
    passed &= reject_one_row_for_two(
        "GR pre-gated encoder rejects one-row output storage for T=2",
        valid, &GRPreGatedBuffers::mixed, output_bytes);

    GRPreGatedBuffers misaligned = valid;
    misaligned.gate =
        gate.slice(1u, gate.length - 1u);
    passed &= reject("GR pre-gated encoder rejects a misaligned gate",
                     1u, misaligned, "misaligned");

    GRPreGatedBuffers xn_output_overlap = valid;
    xn_output_overlap.mixed = xn_output_overlap.xn;
    passed &= reject("GR pre-gated encoder rejects output/xn overlap",
                     1u, xn_output_overlap, "must not overlap");
    GRPreGatedBuffers gate_output_overlap = valid;
    gate_output_overlap.mixed = gate_output_overlap.gate;
    passed &= reject("GR pre-gated encoder rejects output/gate overlap",
                     1u, gate_output_overlap, "must not overlap");

    GRPreGatedBuffers out_of_range = valid;
    out_of_range.mixed = mixed.slice(mixed.length, sizeof(float));
    passed &= reject("GR pre-gated encoder rejects output past its allocation",
                     1u, out_of_range, "not owned by this Context");

    const GRPreGatedBuffers read_only_alias{xn, xn, mixed};
    const bool accepts_read_only_alias =
        encode_gr_pre_gated(context, *rejected_command, 1u, read_only_alias,
                            error);
    check(accepts_read_only_alias,
          "GR pre-gated encoder permits overlapping read-only inputs", error);
    passed &= accepts_read_only_alias;
    rejected_command->discard();

    auto foreign_command = foreign_context.make_command_buffer(
        "GR pre-gated foreign command-buffer validation", error);
    if (!foreign_command) {
        check(false, "GR pre-gated foreign command buffer", error);
        return false;
    }
    const bool rejected_foreign_command =
        !encode_gr_pre_gated(context, *foreign_command, 1u, valid, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "GR pre-gated encoder rejects a foreign-context command buffer",
          error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    auto submitted = context.make_command_buffer(
        "GR pre-gated submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "GR pre-gated validation command-buffer submission",
              error);
        return false;
    }
    const bool rejected_submitted =
        !encode_gr_pre_gated(context, *submitted, 1u, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "GR pre-gated encoder rejects a submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "GR pre-gated validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "GR pre-gated enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GR pre-gated validation command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_gr_pre_gated(context, *enqueued, 1u, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "GR pre-gated encoder rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(
        enqueued, "GR pre-gated enqueued validation", error);
    check(drained, "GR pre-gated enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool test_gr_pre_gated_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GR pre-gated shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    const std::size_t input_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    BufferRef xn = context.buffers().allocate(
        "GR pre-gated shared validation normalized input",
        2u * input_bytes, error);
    BufferRef gate = context.buffers().allocate(
        "GR pre-gated shared validation gate", 2u * input_bytes, error);
    BufferRef mixed = context.buffers().allocate(
        "GR pre-gated shared validation output", 2u * output_bytes, error);
    BufferRef foreign_xn = foreign_context.buffers().allocate(
        "GR pre-gated shared foreign normalized input", input_bytes, error);
    if (!xn || !gate || !mixed || !foreign_xn) {
        check(false, "GR pre-gated shared Encoder validation buffers", error);
        return false;
    }
    const GRPreGatedBuffers valid{xn, gate, mixed};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GRPreGatedBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gr_pre_gated(context, encoder, rows, buffers,
                                               encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const GRPreGatedBuffers empty{};
    passed &= reject("GR pre-gated shared Encoder rejects T=0", 0u, empty,
                     "1..8");
    passed &= reject("GR pre-gated shared Encoder rejects T>8", 9u, empty,
                     "1..8");

    const auto reject_short = [&](const std::string& label,
                                  GRPreGatedBuffers buffers,
                                  BufferRef GRPreGatedBuffers::*member) {
        (buffers.*member).length = sizeof(float);
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GR pre-gated shared Encoder rejects short xn",
                           valid, &GRPreGatedBuffers::xn);
    passed &= reject_short("GR pre-gated shared Encoder rejects short gate",
                           valid, &GRPreGatedBuffers::gate);
    passed &= reject_short("GR pre-gated shared Encoder rejects short output",
                           valid, &GRPreGatedBuffers::mixed);

    const auto reject_one_row_for_two = [&](
        const std::string& label, GRPreGatedBuffers buffers,
        BufferRef GRPreGatedBuffers::*member, std::size_t one_row_bytes) {
        (buffers.*member).length = one_row_bytes;
        return reject(label, 2u, buffers, "undersized");
    };
    passed &= reject_one_row_for_two(
        "GR pre-gated shared Encoder rejects one-row xn storage for T=2",
        valid, &GRPreGatedBuffers::xn, input_bytes);
    passed &= reject_one_row_for_two(
        "GR pre-gated shared Encoder rejects one-row gate storage for T=2",
        valid, &GRPreGatedBuffers::gate, input_bytes);
    passed &= reject_one_row_for_two(
        "GR pre-gated shared Encoder rejects one-row output storage for T=2",
        valid, &GRPreGatedBuffers::mixed, output_bytes);

    GRPreGatedBuffers misaligned = valid;
    misaligned.gate = gate.slice(1u, gate.length - 1u);
    passed &= reject("GR pre-gated shared Encoder rejects misaligned gate",
                     1u, misaligned, "misaligned");

    GRPreGatedBuffers out_of_range = valid;
    out_of_range.mixed = mixed.slice(mixed.length, sizeof(float));
    passed &= reject("GR pre-gated shared Encoder rejects out-of-range view",
                     1u, out_of_range, "not owned by this Context");

    GRPreGatedBuffers xn_output_overlap = valid;
    xn_output_overlap.mixed = xn_output_overlap.xn;
    passed &= reject("GR pre-gated shared Encoder rejects output/xn overlap",
                     1u, xn_output_overlap, "must not overlap");
    GRPreGatedBuffers gate_output_overlap = valid;
    gate_output_overlap.mixed = gate_output_overlap.gate;
    passed &= reject("GR pre-gated shared Encoder rejects output/gate overlap",
                     1u, gate_output_overlap, "must not overlap");

    BufferRef partial_storage = context.buffers().allocate(
        "GR pre-gated shared validation partial-alias storage",
        input_bytes + output_bytes, error);
    if (!partial_storage) {
        check(false, "GR pre-gated shared partial-alias buffer", error);
        return false;
    }
    GRPreGatedBuffers partial_alias = valid;
    partial_alias.xn = partial_storage;
    partial_alias.mixed = partial_storage.slice(sizeof(float), output_bytes);
    passed &= reject("GR pre-gated shared Encoder rejects partial output/xn overlap",
                     1u, partial_alias, "must not overlap");
    partial_alias = valid;
    partial_alias.gate = partial_storage;
    partial_alias.mixed = partial_storage.slice(sizeof(float), output_bytes);
    passed &= reject("GR pre-gated shared Encoder rejects partial output/gate overlap",
                     1u, partial_alias, "must not overlap");

    GRPreGatedBuffers foreign_view = valid;
    foreign_view.xn = foreign_xn;
    passed &= reject("GR pre-gated shared Encoder rejects foreign Context view",
                     1u, foreign_view, "not owned by this Context");

    const GRPreGatedBuffers read_only_alias{xn, xn, mixed};
    auto alias_command = context.make_command_buffer(
        "GR pre-gated shared read-only alias validation", error);
    if (!alias_command) {
        check(false, "GR pre-gated shared read-only alias command buffer",
              error);
        return false;
    }
    const bool accepts_read_only_alias =
        Encoder::encode(
            context, *alias_command, "GR pre-gated shared read-only alias",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_gr_pre_gated(context, encoder, 1u,
                                           read_only_alias, encode_error);
            },
            error);
    check(accepts_read_only_alias,
          "GR pre-gated shared Encoder permits overlapping read-only inputs",
          error);
    alias_command->discard();
    passed &= accepts_read_only_alias;
    return passed;
}

bool test_window_head_mixer(Context& context,
                            const twin::Options& options) {
    constexpr std::uint32_t max_rows = 8u;
    constexpr std::size_t hc_dim = HC_DIM;
    constexpr std::size_t low_rank = HC_LOW_RANK;
    constexpr std::size_t down_weight_count = low_rank * hc_dim;
    constexpr std::size_t up_weight_count = hc_dim * low_rank;
    constexpr std::size_t max_input_count = max_rows * hc_dim;
    constexpr std::size_t max_low_rank_count = max_rows * low_rank;
    constexpr std::size_t max_output_count = max_rows * N_EMBD;
    constexpr std::size_t max_sink_count = max_rows * hc_dim;
    const std::size_t weight_bytes =
        down_weight_count * sizeof(std::uint16_t);

    std::string error;
    BufferRef residual = context.buffers().allocate(
        "Window head mixer test residual",
        max_input_count * sizeof(float), error);
    BufferRef gamma = context.buffers().allocate(
        "Window head mixer test norm gamma",
        hc_dim * sizeof(float), error);
    BufferRef down_weights = context.buffers().allocate(
        "Window head mixer test down weights", weight_bytes, error);
    BufferRef up_weights = context.buffers().allocate(
        "Window head mixer test up weights", weight_bytes, error);
    BufferRef active_rows = context.buffers().allocate(
        "Window head mixer test active rows", sizeof(std::uint32_t), error);
    BufferRef normalized = context.buffers().allocate(
        "Window head mixer test normalized",
        max_input_count * sizeof(float), error);
    BufferRef low_rank_output = context.buffers().allocate(
        "Window head mixer test low rank",
        max_low_rank_count * sizeof(float), error);
    BufferRef gate = context.buffers().allocate(
        "Window head mixer test gate",
        max_input_count * sizeof(float), error);
    BufferRef mixed = context.buffers().allocate(
        "Window head mixer test mixed",
        max_output_count * sizeof(float), error);
    BufferRef sink = context.buffers().allocate(
        "Window head mixer test matvec sink",
        max_sink_count * sizeof(float), error);
    if (!residual || !gamma || !down_weights || !up_weights ||
        !active_rows || !normalized || !low_rank_output || !gate ||
        !mixed || !sink) {
        check(false, "Window head mixer test allocations", error);
        return false;
    }

    std::vector<std::uint16_t> down_matrix(down_weight_count, 0u);
    std::vector<std::uint16_t> up_matrix(up_weight_count, 0u);
    for (std::size_t row = 0u; row < low_rank; ++row) {
        down_matrix[row * hc_dim + row] =
            strata::kernels::bf16_from_f32(1.0f);
    }
    for (std::size_t row = 0u; row < hc_dim; ++row) {
        up_matrix[row * low_rank + row % low_rank] =
            strata::kernels::bf16_from_f32(1.0f);
    }
    std::memcpy(down_weights.contents, down_matrix.data(), weight_bytes);
    std::memcpy(up_weights.contents, up_matrix.data(), weight_bytes);

    std::vector<float> gamma_values(hc_dim);
    std::mt19937_64 rng(options.seed ^ 0x484541444d495845ull);
    for (std::uint32_t stream = 0u; stream < HC; ++stream) {
        for (std::uint32_t col = 0u; col < N_EMBD; ++col) {
            gamma_values[static_cast<std::size_t>(stream) * N_EMBD + col] =
                gamma_value(twin::InputSet::fixture, stream, col, rng);
        }
    }
    std::memcpy(gamma.contents, gamma_values.data(),
                gamma_values.size() * sizeof(float));

    const WindowHeadMixerBuffers buffers{
        residual,
        context.buffers().read_only(gamma),
        context.buffers().read_only(down_weights),
        context.buffers().read_only(up_weights),
        context.buffers().read_only(active_rows),
        normalized,
        low_rank_output,
        gate,
        mixed,
        sink,
    };

    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= max_rows; ++rows) {
        const std::size_t input_count =
            static_cast<std::size_t>(rows) * hc_dim;
        const std::size_t output_count =
            static_cast<std::size_t>(rows) * N_EMBD;
        std::vector<float> input(input_count);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t col = 0u; col < hc_dim; ++col) {
                input[static_cast<std::size_t>(row) * hc_dim + col] =
                    input_value(twin::InputSet::fixture, row, col, rng);
            }
        }
        std::memcpy(residual.contents, input.data(),
                    input.size() * sizeof(float));
        std::memcpy(active_rows.contents, &rows, sizeof(rows));

        std::vector<twin::tf> normalized_reference;
        rms_norm_reference(input, gamma_values, rows * HC, N_EMBD, HC,
                           kEpsilon, normalized_reference);
        std::vector<float> normalized_values(input_count);
        for (std::size_t index = 0u; index < input_count; ++index) {
            normalized_values[index] = normalized_reference[index].value;
        }

        std::vector<twin::tf> low_rank_reference(
            static_cast<std::size_t>(rows) * low_rank);
        std::vector<float> gate_reference(input_count);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t col = 0u; col < low_rank; ++col) {
                const twin::tf down{
                    normalized_values[static_cast<std::size_t>(row) *
                                         hc_dim +
                                     col]};
                const twin::tf scaled =
                    twin::fma(twin::tf{0.25f}, down, twin::tf{});
                low_rank_reference[static_cast<std::size_t>(row) *
                                       low_rank +
                                   col] =
                    scaled /
                    (twin::tf{1.0f} + twin::accurate_exp(-scaled));
            }
            for (std::uint32_t col = 0u; col < hc_dim; ++col) {
                gate_reference[static_cast<std::size_t>(row) * hc_dim + col] =
                    low_rank_reference[static_cast<std::size_t>(row) *
                                           low_rank +
                                       col % low_rank]
                        .value;
            }
        }
        std::vector<twin::tf> expected;
        gr_pre_gated_reference(normalized_values, gate_reference, rows,
                               expected);

        const std::uint32_t poison = 0x7fc00000u;
        std::memcpy(normalized.contents, &poison, sizeof(poison));
        std::memcpy(low_rank_output.contents, &poison, sizeof(poison));
        std::memcpy(gate.contents, &poison, sizeof(poison));
        std::memcpy(mixed.contents, &poison, sizeof(poison));
        std::memcpy(sink.contents, &poison, sizeof(poison));

        const std::string label =
            "Window head mixer T=" + std::to_string(rows);
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool encoded =
            rows == max_rows
                ? Encoder::encode(
                      context, *command, "Window head mixer shared Encoder",
                      [&](Encoder& encoder, std::string& encode_error) {
                          return encode_window_head_mixer(
                              context, encoder, rows, buffers, encode_error);
                      },
                      error)
                : encode_window_head_mixer(
                      context, *command, rows, buffers, error);
        check(encoded, label + " ordered production encode", error);
        if (!encoded) {
            command->discard();
            return false;
        }
        const bool dispatched = submit_and_wait(command, label, error);
        check(dispatched, label + " Metal dispatch", error);
        if (!dispatched) return false;

        const auto actual = std::span<const float>(
            static_cast<const float*>(mixed.contents), output_count);
        const auto mismatch = twin::compare_floats(
            actual, expected,
            {twin::Comparison::units, 256.0, 0.0, 0u});
        if (mismatch) twin::report_failure(label.c_str(), options, rows,
                                          *mismatch);
        check(!mismatch, label + " scalar head-mixer oracle U(256)");
        passed &= !mismatch;
        check(std::memcmp(residual.contents, input.data(),
                          input.size() * sizeof(float)) == 0,
              label + " keeps residual input unchanged");
        check(std::memcmp(gamma.contents, gamma_values.data(),
                          gamma_values.size() * sizeof(float)) == 0 &&
                  std::memcmp(down_weights.contents, down_matrix.data(),
                              weight_bytes) == 0 &&
                  std::memcmp(up_weights.contents, up_matrix.data(),
                              weight_bytes) == 0,
              label + " keeps model inputs unchanged");
    }

    std::uint32_t invalid_active_rows = 7u;
    std::memcpy(active_rows.contents, &invalid_active_rows,
                sizeof(invalid_active_rows));
    auto invalid_command = context.make_command_buffer(
        "Window head mixer rejects mismatched active rows", error);
    if (!invalid_command) {
        check(false, "Window head mixer invalid-input command buffer", error);
        return false;
    }
    const bool rejects_mismatched_rows =
        !encode_window_head_mixer(context, *invalid_command, 8u, buffers,
                                  error) &&
        error.find("does not match") != std::string::npos;
    check(rejects_mismatched_rows,
          "Window head mixer rejects active-row count mismatch", error);
    invalid_command->discard();
    passed &= rejects_mismatched_rows;

    WindowHeadMixerBuffers overlapping = buffers;
    overlapping.gate = overlapping.normalized;
    auto overlap_command = context.make_command_buffer(
        "Window head mixer rejects scratch overlap", error);
    if (!overlap_command) {
        check(false, "Window head mixer overlap command buffer", error);
        return false;
    }
    const bool rejects_overlapping_scratch =
        !encode_window_head_mixer(context, *overlap_command, 8u, overlapping,
                                  error) &&
        error.find("writable scratch views must be disjoint") !=
            std::string::npos;
    check(rejects_overlapping_scratch,
          "Window head mixer rejects writable scratch alias", error);
    overlap_command->discard();
    passed &= rejects_overlapping_scratch;

    std::memcpy(active_rows.contents, &max_rows, sizeof(max_rows));
    WindowHeadMixerBuffers late_failure = buffers;
    late_failure.up_weights = context.buffers().read_only(
        up_weights.slice(0u, weight_bytes - sizeof(std::uint16_t)));
    auto late_failure_command = context.make_command_buffer(
        "Window head mixer discards earlier stages on late failure", error);
    if (!late_failure_command) {
        check(false, "Window head mixer late-failure command buffer", error);
        return false;
    }
    const bool rejects_late_failure =
        !encode_window_head_mixer(context, *late_failure_command, max_rows,
                                  late_failure, error) &&
        error.find("undersized") != std::string::npos;
    check(rejects_late_failure,
          "Window head mixer rejects a short up-projection after prior stages",
          error);
    const bool discarded_late_failure =
        late_failure_command->unsafe_native_handle() == nullptr;
    check(discarded_late_failure,
          "Window head mixer late failure discards the whole command buffer");
    passed &= rejects_late_failure && discarded_late_failure;
    return passed;
}

bool test_gr_cvec_encoder_validation(Context& context) {
    std::string error;
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            std::uint32_t layer, GRControlMode mode,
                            const GRControlVectorBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !encode_gr_cvec(context, *command, rows, layer, mode, false,
                            buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    const GRControlVectorBuffers empty{};
    bool passed = true;
    passed &= reject("GR cvec encoder rejects T=0", 0u, 0u,
                     GRControlMode::project, empty, "1..8");
    passed &= reject("GR cvec encoder rejects T>8", 9u, 0u,
                     GRControlMode::project, empty, "1..8");
    passed &= reject("GR cvec encoder rejects layer 48", 1u, 48u,
                     GRControlMode::project, empty, "0..47");
    passed &= reject("GR cvec encoder rejects invalid mode", 1u, 0u,
                     static_cast<GRControlMode>(2u), empty, "project or add");

    BufferRef residual = context.buffers().allocate(
        "GR cvec validation residual", HC_DIM * sizeof(float), error);
    BufferRef direction = context.buffers().allocate(
        "GR cvec validation direction",
        static_cast<std::size_t>(N_LAYER) * N_EMBD * sizeof(float), error);
    BufferRef scale = context.buffers().allocate(
        "GR cvec validation layer scale", N_LAYER * sizeof(float), error);
    BufferRef enabled = context.buffers().allocate(
        "GR cvec validation enabled", sizeof(std::int32_t), error);
    BufferRef block_out = context.buffers().allocate(
        "GR cvec validation block output", N_EMBD * sizeof(float), error);
    BufferRef inject_prev = context.buffers().allocate(
        "GR cvec validation injection", HC * sizeof(float), error);
    if (!residual || !direction || !scale || !enabled || !block_out ||
        !inject_prev) {
        check(false, "GR cvec encoder validation buffers", error);
        return false;
    }
    const GRControlVectorBuffers valid{
        residual, direction, scale, enabled, block_out, inject_prev};
    const auto reject_short = [&](const std::string& label,
                                  GRControlVectorBuffers buffers,
                                  BufferRef GRControlVectorBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, 0u, GRControlMode::project, buffers,
                      "undersized");
    };
    passed &= reject_short("GR cvec encoder rejects short residual",
                           valid, &GRControlVectorBuffers::residual);
    passed &= reject_short("GR cvec encoder rejects short direction",
                           valid, &GRControlVectorBuffers::direction);
    passed &= reject_short("GR cvec encoder rejects short layer scale",
                           valid, &GRControlVectorBuffers::layer_scale);
    passed &= reject_short("GR cvec encoder rejects short enabled flag",
                           valid, &GRControlVectorBuffers::enabled);
    passed &= reject_short("GR cvec encoder rejects short block output",
                           valid, &GRControlVectorBuffers::block_out);
    passed &= reject_short("GR cvec encoder rejects short injection",
                           valid, &GRControlVectorBuffers::inject_prev);

    GRControlVectorBuffers misaligned = valid;
    misaligned.direction =
        direction.slice(1u, direction.length - 1u);
    passed &= reject("GR cvec encoder rejects misaligned direction",
                     1u, 0u, GRControlMode::project, misaligned, "misaligned");

    GRControlVectorBuffers overlapping = valid;
    overlapping.block_out = overlapping.residual;
    passed &= reject("GR cvec encoder rejects overlapping buffers",
                     1u, 0u, GRControlMode::project, overlapping, "overlap");

    const long page_size = getpagesize();
    void* alias_memory = nullptr;
    if (page_size <= 0 ||
        posix_memalign(&alias_memory, static_cast<std::size_t>(page_size),
                       static_cast<std::size_t>(page_size) * 3u) != 0) {
        check(false, "allocate GR cvec no-copy alias fixture");
        return false;
    }
    std::shared_ptr<void> alias_lifetime(alias_memory, std::free);
    BufferRef alias_residual = context.buffers().wrap_no_copy(
        "GR cvec alias residual", alias_memory,
        static_cast<std::size_t>(page_size) * 3u, alias_lifetime, error);
    BufferRef alias_block_out = context.buffers().wrap_no_copy(
        "GR cvec alias block output", alias_memory,
        static_cast<std::size_t>(page_size) * 3u, alias_lifetime, error);
    if (!alias_residual || !alias_block_out) {
        check(false, "create GR cvec no-copy alias fixtures", error);
        return false;
    }
    check(alias_residual.handle != alias_block_out.handle,
          "GR cvec no-copy aliases use distinct Metal handles");
    GRControlVectorBuffers distinct_handle_aliases = valid;
    distinct_handle_aliases.residual = alias_residual;
    distinct_handle_aliases.block_out = alias_block_out;
    passed &= reject("GR cvec encoder rejects distinct-handle aliases",
                     1u, 0u, GRControlMode::project, distinct_handle_aliases,
                     "overlap");

    auto submitted = context.make_command_buffer("GR cvec submitted-buffer validation",
                                                 error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "GR cvec validation command-buffer submission", error);
        return false;
    }
    const bool rejected_submitted =
        !encode_gr_cvec(context, *submitted, 1u, 0u, GRControlMode::project,
                        false, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "GR cvec encoder rejects a submitted command buffer", error);
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (native != nil) [native waitUntilCompleted];
    const bool completed =
        native != nil && native.status == MTLCommandBufferStatusCompleted;
    check(completed, "GR cvec validation command buffer completes");
    passed &= rejected_submitted && completed;

    auto enqueued = context.make_command_buffer(
        "GR cvec enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GR cvec validation command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    const bool rejected_enqueued =
        !encode_gr_cvec(context, *enqueued, 1u, 0u, GRControlMode::project,
                        false, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "GR cvec encoder rejects an enqueued command buffer", error);
    const bool drained = submit_and_wait(enqueued, "GR cvec enqueued validation",
                                         error);
    check(drained, "GR cvec enqueued validation command completes", error);
    passed &= rejected_enqueued && drained;
    return passed;
}

bool test_gr_cvec_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GR cvec shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    const std::size_t residual_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    const std::size_t direction_bytes =
        static_cast<std::size_t>(N_LAYER) * N_EMBD * sizeof(float);
    const std::size_t scale_bytes =
        static_cast<std::size_t>(N_LAYER) * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    BufferRef residual = context.buffers().allocate(
        "GR cvec shared validation residual", residual_bytes, error);
    BufferRef direction = context.buffers().allocate(
        "GR cvec shared validation direction", direction_bytes, error);
    BufferRef scale = context.buffers().allocate(
        "GR cvec shared validation layer scale", scale_bytes, error);
    BufferRef enabled = context.buffers().allocate(
        "GR cvec shared validation enabled", sizeof(std::int32_t), error);
    BufferRef block_out = context.buffers().allocate(
        "GR cvec shared validation block output", block_bytes, error);
    BufferRef inject_prev = context.buffers().allocate(
        "GR cvec shared validation injection", inject_bytes, error);
    if (!residual || !direction || !scale || !enabled || !block_out ||
        !inject_prev) {
        check(false, "GR cvec shared Encoder validation buffers", error);
        return false;
    }
    const GRControlVectorBuffers valid{
        residual, direction, scale, enabled, block_out, inject_prev};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            std::uint32_t layer, GRControlMode mode,
                            const GRControlVectorBuffers& buffers,
                            const std::string& expected_error) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gr_cvec(context, encoder, rows, layer, mode,
                                          false, buffers, encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const GRControlVectorBuffers empty{};
    passed &= reject("GR cvec shared Encoder rejects T=0", 0u, 0u,
                     GRControlMode::project, empty, "1..8");
    passed &= reject("GR cvec shared Encoder rejects T>8", 9u, 0u,
                     GRControlMode::project, empty, "1..8");
    passed &= reject("GR cvec shared Encoder rejects layer 48", 1u, 48u,
                     GRControlMode::project, empty, "0..47");
    passed &= reject("GR cvec shared Encoder rejects invalid mode", 1u, 0u,
                     static_cast<GRControlMode>(2u), empty, "project or add");

    const auto reject_short = [&](const std::string& label,
                                  BufferRef GRControlVectorBuffers::*member) {
        GRControlVectorBuffers short_view = valid;
        (short_view.*member).length = 1u;
        return reject(label, 1u, 0u, GRControlMode::project, short_view,
                      "undersized");
    };
    passed &= reject_short("GR cvec shared Encoder rejects short residual",
                           &GRControlVectorBuffers::residual);
    passed &= reject_short("GR cvec shared Encoder rejects short direction",
                           &GRControlVectorBuffers::direction);
    passed &= reject_short("GR cvec shared Encoder rejects short layer scale",
                           &GRControlVectorBuffers::layer_scale);
    passed &= reject_short("GR cvec shared Encoder rejects short enabled flag",
                           &GRControlVectorBuffers::enabled);
    passed &= reject_short("GR cvec shared Encoder rejects short block output",
                           &GRControlVectorBuffers::block_out);
    passed &= reject_short("GR cvec shared Encoder rejects short injection",
                           &GRControlVectorBuffers::inject_prev);

    GRControlVectorBuffers misaligned = valid;
    misaligned.direction = direction.slice(1u, direction.length - 1u);
    passed &= reject("GR cvec shared Encoder rejects misaligned direction",
                     1u, 0u, GRControlMode::project, misaligned, "misaligned");
    misaligned = valid;
    misaligned.enabled = enabled.slice(1u, enabled.length - 1u);
    passed &= reject("GR cvec shared Encoder rejects misaligned enabled flag",
                     1u, 0u, GRControlMode::project, misaligned, "misaligned");

    GRControlVectorBuffers out_of_range = valid;
    out_of_range.direction =
        direction.slice(direction.length, sizeof(float));
    passed &= reject("GR cvec shared Encoder rejects out-of-range view",
                     1u, 0u, GRControlMode::project, out_of_range,
                     "not owned by this Context");

    GRControlVectorBuffers exact_alias = valid;
    exact_alias.block_out = exact_alias.residual;
    passed &= reject("GR cvec shared Encoder rejects exact output alias",
                     1u, 0u, GRControlMode::project, exact_alias, "overlap");
    BufferRef partial_storage = context.buffers().allocate(
        "GR cvec shared validation partial-alias storage",
        residual_bytes + block_bytes, error);
    if (!partial_storage) {
        check(false, "GR cvec shared partial-alias buffer", error);
        return false;
    }
    GRControlVectorBuffers partial_alias = valid;
    partial_alias.residual = partial_storage;
    partial_alias.block_out =
        partial_storage.slice(sizeof(float), block_bytes);
    passed &= reject("GR cvec shared Encoder rejects partial output alias",
                     1u, 0u, GRControlMode::project, partial_alias, "overlap");

    BufferRef foreign_residual = foreign_context.buffers().allocate(
        "GR cvec shared foreign residual", residual_bytes, error);
    if (!foreign_residual) {
        check(false, "GR cvec shared foreign buffer", error);
        return false;
    }
    GRControlVectorBuffers foreign_view = valid;
    foreign_view.residual = foreign_residual;
    passed &= reject("GR cvec shared Encoder rejects foreign Context view",
                     1u, 0u, GRControlMode::project, foreign_view,
                     "not owned by this Context");
    return passed;
}

bool run_set(Context& context, twin::InputSet set, const twin::Options& options) {
    bool passed = true;
    FusedSensitiveValues fused_sensitive{};
    if (set == twin::InputSet::edge) {
        float sigmoid = 0.0f;
        const bool probed = probe_sigmoid(context, 1.0f, sigmoid);
        check(probed, "gr_pre_gated fused-control sigmoid probe succeeded");
        const bool selected =
            probed && choose_fused_sensitive_values(sigmoid, fused_sensitive);
        check(selected, "gr_pre_gated fused-sensitive input selected",
              "sigmoid=" + std::to_string(sigmoid));
        if (!selected) return false;
    }
    for (const RMSNormCase shape : kCases) {
        passed &= run_case(context, shape, set, options, false);
    }
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        passed &= run_gr_down_case(context, rows, set, options);
        passed &= run_gr_norm_case(context, rows, false, set, options);
        passed &= run_gr_norm_case(context, rows, true, set, options);
        passed &= run_gr_up_case(context, rows, false, set, options);
        passed &= run_gr_up_case(context, rows, true, set, options);
        struct CvecCase {
            std::uint32_t mode;
            std::uint32_t layer;
            bool on;
            bool write;
            bool zero_scale;
        };
        constexpr std::array cvec_cases{
            CvecCase{0u, 7u, false, false, false},
            CvecCase{0u, 7u, false, true, false},
            CvecCase{0u, 7u, true, false, false},
            CvecCase{0u, 7u, true, true, false},
            CvecCase{1u, 7u, true, false, false},
            CvecCase{1u, 7u, true, true, false},
            CvecCase{1u, 7u, false, false, false},
            CvecCase{0u, 3u, true, false, false},
            CvecCase{1u, 3u, true, false, false},
            CvecCase{0u, 7u, true, false, true},
            CvecCase{0u, 7u, true, true, true},
            CvecCase{1u, 3u, true, false, true},
            CvecCase{1u, 3u, true, true, true},
        };
        for (const CvecCase test : cvec_cases) {
            passed &= run_gr_cvec_case(context, rows, test.mode, test.layer,
                                       test.on, test.write, test.zero_scale,
                                       nullptr, set, options);
        }
    }
    if (set == twin::InputSet::fixture) {
        passed &= run_gr_cvec_case(
            context, 1u, 0u, 3u, true, false, false, nullptr, set, options,
            true);
        passed &= run_gr_cvec_case(
            context, 1u, 1u, 3u, true, false, false, nullptr, set, options,
            true);
        passed &= run_gr_cvec_case(
            context, 8u, 0u, 3u, true, true, false, nullptr, set, options,
            true);
        passed &= run_gr_cvec_case(
            context, 8u, 1u, 3u, true, true, false, nullptr, set, options,
            true);
        passed &= run_gr_norm_case(context, 1u, false, set, options, true);
        passed &= run_gr_norm_case(context, 1u, true, set, options, true);
        passed &= run_gr_norm_case(context, 8u, false, set, options, true);
        passed &= run_gr_norm_case(context, 8u, true, set, options, true);
        passed &= run_gr_up_case(context, 1u, false, set, options, true);
        passed &= run_gr_up_case(context, 1u, true, set, options, true);
        passed &= run_gr_up_case(context, 8u, false, set, options, true);
        passed &= run_gr_up_case(context, 8u, true, set, options, true);
        passed &= run_gr_down_case(context, 1u, set, options, true);
        passed &= run_gr_down_case(context, 8u, set, options, true);
        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            passed &= run_gr_read_pipeline_case(context, rows, false, options);
        }
        passed &= run_gr_read_pipeline_case(context, 8u, true, options);
    }
    if (set == twin::InputSet::fixture) {
        float sigmoid = 0.0f;
        const bool probed = probe_sigmoid(context, 1.0f, sigmoid);
        check(probed, "gr_cvec fused-write sigmoid probe succeeded");
        FusedSensitiveValues cvec_fused_sensitive{};
        const bool selected =
            probed && choose_fused_sensitive_values(2.0f * sigmoid,
                                                    cvec_fused_sensitive);
        check(selected, "gr_cvec fused-write-sensitive input selected",
              "sigmoid=" + std::to_string(sigmoid));
        if (selected) {
            passed &= run_gr_cvec_case(context, 1u, 0u, 7u, false, true,
                                       false, &cvec_fused_sensitive, set,
                                       options, true);
        } else {
            passed = false;
        }
    }
    passed &= run_gr_post_case(context, 1u, set, options, false);
    passed &= run_gr_post_case(context, 8u, set, options, false);
    if (set == twin::InputSet::fixture) {
        passed &= run_gr_post_case(context, 1u, set, options, false, true);
    }
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        passed &= run_gr_down_silu_case(context, rows, set, options);
        passed &= run_gr_pre_gated_case(context, rows, set, options,
                                        fused_sensitive);
    }
    if (set == twin::InputSet::fixture) {
        passed &= run_gr_down_silu_case(
            context, 1u, set, options, true);
        passed &= run_gr_down_silu_case(
            context, 8u, set, options, true);
        passed &= run_gr_pre_gated_case(
            context, 1u, set, options, fused_sensitive, true);
        passed &= run_gr_pre_gated_case(
            context, 8u, set, options, fused_sensitive, true);
        passed &= run_case(context, {2560u, 4u, 8u}, set, options, true);
        passed &= run_gr_post_case(context, 8u, set, options, true);
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_gr_test strata.metallib "
                     "[--seed N] [--input-set random|edge|fixture]\n");
        return 2;
    }
    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    if (options.input_set == twin::InputSet::real) {
        std::fprintf(stderr, "metal_gr_test has no model-backed real input set\n");
        return 77;
    }

    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;

    ContextOptions foreign_context_options = context_options;
    foreign_context_options.pipelines = false;
    auto foreign_context = Context::open(foreign_context_options, err);
    check(foreign_context != nullptr,
          "open second Metal context for ownership validation", err);
    if (!foreign_context) return 1;

    bool passed = test_gr_cvec_encoder_validation(*context);
    passed &= test_gr_norm_encoder_validation(*context);
    passed &= test_gr_down_encoder_validation(*context, *foreign_context);
    passed &= test_gr_up_encoder_validation(*context);
    passed &= test_gr_read_encoder_validation(*context);
    passed &= test_gr_post_encoder_validation(*context, *foreign_context);
    passed &= test_gr_down_silu_encoder_validation(*context, *foreign_context);
    passed &= test_gr_pre_gated_encoder_validation(*context, *foreign_context);
    passed &= test_weighted_rms_norm_encoder_validation(
        *context, *foreign_context);
    passed &= test_window_head_mixer(*context, options);
    if (options.input_set_explicit) {
        passed &= run_set(*context, options.input_set, options);
    } else {
        constexpr std::array input_sets{
            twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
        for (const twin::InputSet set : input_sets) {
            twin::Options case_options = options;
            case_options.input_set = set;
            case_options.input_set_explicit = true;
            std::printf("GR input set: %s\n", twin::input_set_name(set));
            passed &= run_set(*context, set, case_options);
        }
    }
    passed &= test_gr_down_shared_encoder_validation(
        context_options, *context);
    passed &= test_gr_up_shared_encoder_validation(
        context_options, *context);
    passed &= test_gr_down_silu_shared_encoder_validation(
        context_options, *context);
    passed &= test_gr_pre_gated_shared_encoder_validation(
        context_options, *context);
    passed &= test_gr_cvec_shared_encoder_validation(
        context_options, *context);
    passed &= test_gr_norm_shared_encoder_validation(
        context_options, *context);
    check(passed && g_failures == 0,
          "Weighted RMS norm and GR down/norm/up/cvec/post/down-SiLU/pre-gated encoders and parity");
    std::printf("metal_gr_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
