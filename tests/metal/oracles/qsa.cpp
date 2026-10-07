#include "oracles/qsa.hpp"

#include "../../../src/metal/kernels/common/params.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace oracle {
namespace {

using twin::tf;

tf rsqrt_accurate(tf value) {
    return tf{static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(value.value)))};
}

std::array<tf, 32> butterfly(std::array<tf, 32> values) {
    for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0u; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values;
}

std::uint16_t float_to_half_bits(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign =
        static_cast<std::uint16_t>((bits >> 16u) & 0x8000u);
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<std::uint16_t>(
            sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        const std::uint32_t significand = mantissa | 0x800000u;
        const std::uint32_t shift =
            static_cast<std::uint32_t>(14 - half_exponent);
        const std::uint32_t truncated = significand >> shift;
        const std::uint32_t remainder =
            significand & ((std::uint32_t{1} << shift) - 1u);
        const std::uint32_t halfway = std::uint32_t{1} << (shift - 1u);
        const std::uint32_t rounded = truncated +
            static_cast<std::uint32_t>(
                remainder > halfway ||
                (remainder == halfway && (truncated & 1u) != 0u));
        return static_cast<std::uint16_t>(sign | rounded);
    }

    std::uint32_t rounded_mantissa =
        mantissa + 0xfffu + ((mantissa >> 13u) & 1u);
    if ((rounded_mantissa & 0x800000u) != 0u) {
        rounded_mantissa = 0u;
        ++half_exponent;
        if (half_exponent >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    } else {
        rounded_mantissa >>= 13u;
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint16_t>(half_exponent) << 10u) |
        static_cast<std::uint16_t>(rounded_mantissa));
}

float half_bits_to_float(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0u;
    const std::uint32_t exponent = (bits >> 10u) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    float value = 0.0f;
    if (exponent == 0u) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 0x1fu) {
        value = mantissa == 0u
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0f + static_cast<float>(mantissa) / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

std::uint32_t selection_order_key(float score) {
    const float rounded = score + 0.0f;
    if (std::isnan(rounded)) return 0u;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(rounded);
    return (bits & 0x80000000u) != 0u
        ? ~bits
        : bits ^ 0x80000000u;
}

}  // namespace

void qsa_norm_rope(std::span<const float> input,
                   std::span<const float> gamma,
                   std::span<const std::int32_t> positions,
                   std::span<const float> rope,
                   std::uint32_t token_rows,
                   strata::metal::QsaNormRopeMode mode,
                   std::vector<tf>& output) {
    std::uint32_t heads_per_token = 0u;
    std::uint32_t width = 0u;
    std::uint32_t input_stride = 0u;
    switch (mode) {
        case strata::metal::QsaNormRopeMode::key:
            heads_per_token = N_HEAD_KV;
            width = HEAD_DIM;
            input_stride = HEAD_DIM;
            break;
        case strata::metal::QsaNormRopeMode::query:
            heads_per_token = N_HEAD_Q;
            width = HEAD_DIM;
            input_stride = 2u * HEAD_DIM;
            break;
        case strata::metal::QsaNormRopeMode::indexer_query:
            heads_per_token = INDEXER_HEADS;
            width = INDEXER_HEAD_DIM;
            input_stride = INDEXER_HEAD_DIM;
            break;
    }

    const std::size_t norm_rows =
        static_cast<std::size_t>(token_rows) * heads_per_token;
    output.assign(norm_rows * width, tf{});
    const std::uint32_t warps = HEAD_DIM / 32u;
    for (std::size_t row = 0u; row < norm_rows; ++row) {
        const std::size_t input_base = row * input_stride;
        std::array<tf, 8> warp_sums{};
        for (std::uint32_t warp = 0u; warp < warps; ++warp) {
            std::array<tf, 32> squares{};
            for (std::uint32_t lane = 0u; lane < squares.size(); ++lane) {
                const std::uint32_t dimension = warp * 32u + lane;
                const tf value = dimension < width
                    ? tf{input[input_base + dimension]}
                    : tf{};
                squares[lane] = value * value;
            }
            warp_sums[warp] = butterfly(squares)[0];
        }
        std::array<tf, 32> partials{};
        for (std::uint32_t warp = 0u; warp < warp_sums.size(); ++warp) {
            partials[warp] = warp_sums[warp];
        }
        const tf sum = butterfly(partials)[0];
        const tf scale =
            rsqrt_accurate(sum / tf{static_cast<float>(width)} +
                           tf{RMS_EPSILON});
        std::array<tf, HEAD_DIM> normalized{};
        for (std::uint32_t dimension = 0u; dimension < width; ++dimension) {
            normalized[dimension] =
                (scale * tf{input[input_base + dimension]}) *
                tf{gamma[dimension]};
        }

        const std::uint32_t token =
            static_cast<std::uint32_t>(row / heads_per_token);
        const std::size_t rope_base =
            static_cast<std::size_t>(positions[token]) * 32u * 2u;
        const std::size_t output_base = row * width;
        for (std::uint32_t pair = 0u; pair < 32u; ++pair) {
            const tf cosine{rope[rope_base + pair * 2u]};
            const tf sine{rope[rope_base + pair * 2u + 1u]};
            output[output_base + pair] =
                twin::fma(-normalized[pair + 32u], sine,
                         normalized[pair] * cosine);
            output[output_base + pair + 32u] =
                twin::fma(normalized[pair + 32u], cosine,
                         normalized[pair] * sine);
        }
        for (std::uint32_t dimension = 64u; dimension < width; ++dimension) {
            output[output_base + dimension] = normalized[dimension];
        }
    }
}

void qsa_fwht256(std::span<const float> input, std::uint32_t rows,
                 std::vector<tf>& output) {
    output.resize(static_cast<std::size_t>(rows) * 256u);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row) * 256u;
        std::array<tf, 256> values{};
        for (std::uint32_t index = 0u; index < values.size(); ++index) {
            values[index] =
                tf{input[row_base + index]} * tf{0.0625f};
        }
        for (std::uint32_t span = 1u; span < values.size(); span <<= 1u) {
            for (std::uint32_t base = 0u; base < values.size();
                 base += 2u * span) {
                for (std::uint32_t index = 0u; index < span; ++index) {
                    const tf lower = values[base + index];
                    const tf upper = values[base + index + span];
                    values[base + index] = lower + upper;
                    values[base + index + span] = lower - upper;
                }
            }
        }
        std::copy(values.begin(), values.end(), output.begin() + row_base);
    }
}

void qsa_kv_append_f16(std::span<const float> keys,
                       std::span<const float> values,
                       std::span<const std::int32_t> positions,
                       std::vector<std::uint16_t>& key_pool,
                       std::vector<std::uint16_t>& value_pool) {
    for (std::size_t token = 0u; token < positions.size(); ++token) {
        const std::uint32_t cell =
            static_cast<std::uint32_t>(positions[token]);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::uint32_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + (cell % 4u);
            const std::size_t source =
                (token * N_HEAD_KV + kvh) * HEAD_DIM;
            const std::size_t destination =
                static_cast<std::size_t>(pool_row) * HEAD_DIM;
            for (std::uint32_t dimension = 0u; dimension < HEAD_DIM;
                 ++dimension) {
                key_pool[destination + dimension] =
                    float_to_half_bits(keys[source + dimension]);
                value_pool[destination + dimension] =
                    float_to_half_bits(values[source + dimension]);
            }
        }
    }
}

void qsa_kv_append_i8(std::span<const float> keys,
                      std::span<const float> values,
                      std::span<const std::int32_t> positions,
                      std::vector<std::int8_t>& key_codes,
                      std::vector<std::uint16_t>& key_scales,
                      std::vector<std::int8_t>& value_codes,
                      std::vector<std::uint16_t>& value_scales) {
    for (std::size_t token = 0u; token < positions.size(); ++token) {
        const std::uint32_t cell =
            static_cast<std::uint32_t>(positions[token]);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::uint32_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + (cell % 4u);
            const std::size_t source =
                (token * N_HEAD_KV + kvh) * HEAD_DIM;
            const std::size_t destination =
                static_cast<std::size_t>(pool_row) * HEAD_DIM;
            const std::size_t scale_destination =
                static_cast<std::size_t>(pool_row) * 4u;
            for (std::uint32_t group = 0u; group < 4u; ++group) {
                float key_amax = 0.0f;
                float value_amax = 0.0f;
                for (std::uint32_t lane = 0u; lane < 64u; ++lane) {
                    const std::size_t index =
                        source + static_cast<std::size_t>(group) * 64u + lane;
                    key_amax = std::max(key_amax, std::fabs(keys[index]));
                    value_amax =
                        std::max(value_amax, std::fabs(values[index]));
                }
                const std::uint16_t key_scale_bits =
                    float_to_half_bits(key_amax / 127.0f);
                const std::uint16_t value_scale_bits =
                    float_to_half_bits(value_amax / 127.0f);
                key_scales[scale_destination + group] = key_scale_bits;
                value_scales[scale_destination + group] = value_scale_bits;
                const float key_scale = half_bits_to_float(key_scale_bits);
                const float value_scale = half_bits_to_float(value_scale_bits);
                for (std::uint32_t lane = 0u; lane < 64u; ++lane) {
                    const std::size_t index =
                        source + static_cast<std::size_t>(group) * 64u + lane;
                    const auto quantize = [](float value, float scale) {
                        if (!(scale > 0.0f)) return std::int8_t{0};
                        const int rounded = static_cast<int>(
                            std::nearbyint(value / scale));
                        return static_cast<std::int8_t>(
                            std::clamp(rounded, -127, 127));
                    };
                    key_codes[destination +
                              static_cast<std::size_t>(group) * 64u + lane] =
                        quantize(keys[index], key_scale);
                    value_codes[destination +
                                static_cast<std::size_t>(group) * 64u + lane] =
                        quantize(values[index], value_scale);
                }
            }
        }
    }
}

void qsa_kv_append_q4(std::span<const float> keys,
                      std::span<const float> values,
                      std::span<const std::int32_t> positions,
                      std::vector<std::uint8_t>& key_pool,
                      std::vector<std::uint8_t>& value_pool) {
    constexpr std::size_t kGroup = 32u;
    constexpr std::size_t kBlockBytes = 18u;
    constexpr std::size_t kBlocksPerRow = HEAD_DIM / kGroup;
    for (std::size_t token = 0u; token < positions.size(); ++token) {
        const std::uint32_t cell =
            static_cast<std::uint32_t>(positions[token]);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            const std::uint32_t pool_row =
                ((cell / 4u) * N_HEAD_KV + kvh) * 4u + (cell % 4u);
            const std::size_t source =
                (token * N_HEAD_KV + kvh) * HEAD_DIM;
            for (std::size_t block = 0u; block < kBlocksPerRow; ++block) {
                const std::size_t group_base =
                    source + block * kGroup;
                float magnitude = std::fabs(keys[group_base]);
                float maximum = keys[group_base];
                float value_magnitude = std::fabs(values[group_base]);
                float value_maximum = values[group_base];
                for (std::size_t lane = 1u; lane < kGroup; ++lane) {
                    const float key = keys[group_base + lane];
                    const float key_abs = std::fabs(key);
                    if (key_abs > magnitude ||
                        (key_abs == magnitude && key > maximum)) {
                        magnitude = key_abs;
                        maximum = key;
                    }
                    const float value = values[group_base + lane];
                    const float value_abs = std::fabs(value);
                    if (value_abs > value_magnitude ||
                        (value_abs == value_magnitude &&
                         value > value_maximum)) {
                        value_magnitude = value_abs;
                        value_maximum = value;
                    }
                }

                const float key_scale = maximum / -8.0f;
                const float value_scale = value_maximum / -8.0f;
                const float key_inverse =
                    key_scale != 0.0f ? 1.0f / key_scale : 0.0f;
                const float value_inverse =
                    value_scale != 0.0f ? 1.0f / value_scale : 0.0f;
                std::array<std::uint8_t, kGroup> key_codes{};
                std::array<std::uint8_t, kGroup> value_codes{};
                for (std::size_t lane = 0u; lane < kGroup; ++lane) {
                    const float key_quantized = std::fma(
                        keys[group_base + lane], key_inverse, 8.5f);
                    const float value_quantized = std::fma(
                        values[group_base + lane], value_inverse, 8.5f);
                    key_codes[lane] = static_cast<std::uint8_t>(
                        std::clamp(static_cast<int>(std::trunc(key_quantized)),
                                   0, 15));
                    value_codes[lane] = static_cast<std::uint8_t>(
                        std::clamp(
                            static_cast<int>(std::trunc(value_quantized)),
                            0, 15));
                }

                const std::size_t destination =
                    (static_cast<std::size_t>(pool_row) * kBlocksPerRow +
                     block) *
                    kBlockBytes;
                const std::uint16_t key_scale_bits =
                    float_to_half_bits(key_scale);
                const std::uint16_t value_scale_bits =
                    float_to_half_bits(value_scale);
                key_pool[destination] =
                    static_cast<std::uint8_t>(key_scale_bits & 0xffu);
                key_pool[destination + 1u] =
                    static_cast<std::uint8_t>(key_scale_bits >> 8u);
                value_pool[destination] =
                    static_cast<std::uint8_t>(value_scale_bits & 0xffu);
                value_pool[destination + 1u] =
                    static_cast<std::uint8_t>(value_scale_bits >> 8u);
                for (std::size_t lane = 0u; lane < 16u; ++lane) {
                    key_pool[destination + 2u + lane] =
                        static_cast<std::uint8_t>(
                            key_codes[lane] | (key_codes[lane + 16u] << 4u));
                    value_pool[destination + 2u + lane] =
                        static_cast<std::uint8_t>(
                            value_codes[lane] |
                            (value_codes[lane + 16u] << 4u));
                }
            }
        }
    }
}

void qsa_indexer_append(std::span<const float> raw_keys,
                        std::span<const std::int32_t> positions,
                        std::span<const float> gamma,
                        std::span<const float> rope,
                        float rope_zero_cos, float rope_zero_sin,
                        QsaIndexerState& state) {
    constexpr std::size_t kWidth = INDEXER_HEAD_DIM;
    constexpr std::size_t kRopePairs = 32u;
    for (std::size_t row = 0u; row < positions.size(); ++row) {
        const std::uint32_t position =
            static_cast<std::uint32_t>(positions[row]);
        const std::uint32_t slot = position % INDEXER_BLOCK_CELLS;
        std::array<tf, kWidth> incoming{};
        for (std::size_t dimension = 0u; dimension < kWidth; ++dimension) {
            incoming[dimension] = tf{half_bits_to_float(float_to_half_bits(
                raw_keys[row * kWidth + dimension]))};
            if (slot < 3u) {
                state.tail[static_cast<std::size_t>(slot) * kWidth +
                           dimension] = incoming[dimension].value;
            }
        }
        if (position != 0u && slot != INDEXER_BLOCK_CELLS - 1u) continue;

        std::array<tf, kWidth> mean{};
        for (std::size_t dimension = 0u; dimension < kWidth; ++dimension) {
            tf sum = position == 0u
                ? incoming[dimension]
                : tf{state.tail[dimension]};
            for (std::uint32_t cell = 1u;
                 cell < INDEXER_BLOCK_CELLS; ++cell) {
                const tf value =
                    position == 0u || cell == INDEXER_BLOCK_CELLS - 1u
                        ? incoming[dimension]
                        : tf{state.tail[static_cast<std::size_t>(cell) *
                                            kWidth +
                                        dimension]};
                sum = sum + value;
            }
            mean[dimension] = twin::fma(tf{0.25f}, sum, tf{});
        }

        std::array<tf, 8> simdgroup_sums{};
        for (std::uint32_t simdgroup = 0u; simdgroup < 4u; ++simdgroup) {
            std::array<tf, 32> squares{};
            for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                const tf value = mean[simdgroup * 32u + lane];
                squares[lane] = value * value;
            }
            simdgroup_sums[simdgroup] = butterfly(squares)[0];
        }
        std::array<tf, 32> partials{};
        std::copy(simdgroup_sums.begin(), simdgroup_sums.end(),
                  partials.begin());
        const tf square_sum = butterfly(partials)[0];
        const tf scale = rsqrt_accurate(
            square_sum / tf{static_cast<float>(kWidth)} +
            tf{RMS_EPSILON});
        std::array<tf, kWidth> normalized{};
        for (std::size_t dimension = 0u; dimension < kWidth; ++dimension) {
            normalized[dimension] =
                (scale * mean[dimension]) * tf{gamma[dimension]};
        }

        const std::uint32_t block = position / INDEXER_BLOCK_CELLS;
        const std::size_t rope_base =
            position == 0u
                ? 0u
                : static_cast<std::size_t>(block) *
                      INDEXER_BLOCK_CELLS * kRopePairs * 2u;
        for (std::size_t dimension = 0u; dimension < kWidth; ++dimension) {
            float output = normalized[dimension].value;
            if (dimension < ROPE_DIM) {
                const std::size_t pair = dimension % kRopePairs;
                const tf cosine{position == 0u
                    ? rope_zero_cos
                    : rope[rope_base + pair * 2u]};
                const tf sine{position == 0u
                    ? rope_zero_sin
                    : rope[rope_base + pair * 2u + 1u]};
                const tf first = normalized[pair];
                const tf second = normalized[pair + ROPE_DIM / 2u];
                output = dimension < ROPE_DIM / 2u
                    ? twin::fma(-second, sine, first * cosine).value
                    : twin::fma(second, cosine, first * sine).value;
            }
            state.pooled[static_cast<std::size_t>(block) * kWidth +
                         dimension] = output;
            if (position == 0u) {
                state.dead[dimension] = output;
            } else {
                state.pooled[static_cast<std::size_t>(block + 1u) * kWidth +
                             dimension] = state.dead[dimension];
            }
        }
        if (position != 0u) {
            state.block_position =
                static_cast<std::int32_t>(block * INDEXER_BLOCK_CELLS);
        }
    }
}

void qsa_selection_scores(
    std::span<const float> pooled, std::span<const float> dead,
    std::span<const float> indexer_queries,
    std::span<const std::int32_t> n_kv,
    std::span<const std::int32_t> n_bid, std::uint32_t max_blocks,
    float untouched_score, std::vector<tf>& output) {
    constexpr std::size_t kWidth = INDEXER_HEAD_DIM;
    const std::size_t rows = n_kv.size();
    output.assign(rows * max_blocks, tf{untouched_score});
    for (std::size_t row = 0u; row < rows; ++row) {
        const std::int32_t last_block = n_bid[row];
        for (std::int32_t block = 0; block <= last_block; ++block) {
            const std::size_t key_base =
                block == last_block
                    ? 0u
                    : static_cast<std::size_t>(block) * kWidth;
            tf score{};
            for (std::uint32_t head = 0u; head < INDEXER_HEADS; ++head) {
                std::array<tf, 32> lane_dots{};
                for (std::uint32_t lane = 0u; lane < lane_dots.size();
                     ++lane) {
                    const std::size_t key_index =
                        key_base + static_cast<std::size_t>(lane) * 4u;
                    const std::size_t query_base =
                        (row * INDEXER_HEADS + head) * kWidth +
                        static_cast<std::size_t>(lane) * 4u;
                    const tf x{block == last_block
                        ? dead[key_index]
                        : pooled[key_index]};
                    const tf qx{indexer_queries[query_base]};
                    const tf xy{block == last_block
                        ? dead[key_index + 1u]
                        : pooled[key_index + 1u]};
                    const tf qy{indexer_queries[query_base + 1u]};
                    const tf xz{block == last_block
                        ? dead[key_index + 2u]
                        : pooled[key_index + 2u]};
                    const tf qz{indexer_queries[query_base + 2u]};
                    const tf xw{block == last_block
                        ? dead[key_index + 3u]
                        : pooled[key_index + 3u]};
                    const tf qw{indexer_queries[query_base + 3u]};
                    lane_dots[lane] =
                        twin::fma(xw, qw,
                                  twin::fma(xz, qz,
                                            twin::fma(xy, qy, x * qx)));
                }
                const tf head_score = butterfly(lane_dots)[0];
                if (head_score.value > 0.0f) score = score + head_score;
            }
            if (block == last_block &&
                n_kv[row] % INDEXER_BLOCK_CELLS != 0) {
                score = score + tf{1.0e9f};
            }
            output[row * max_blocks +
                   static_cast<std::size_t>(block)] = score;
        }
    }
}

void qsa_selection_topk(
    std::span<const float> scores,
    std::span<const std::int32_t> n_kv,
    std::span<const std::int32_t> n_bid, std::uint32_t max_blocks,
    std::int32_t untouched_id, std::vector<std::int32_t>& output) {
    struct Candidate {
        std::uint32_t key;
        std::int32_t id;
    };
    const std::size_t rows = n_kv.size();
    output.assign(rows * INDEXER_SELECTION_WIDTH_MAX, untouched_id);
    for (std::size_t row = 0u; row < rows; ++row) {
        const std::int32_t width = std::min<std::int32_t>(
            n_kv[row], static_cast<std::int32_t>(
                           INDEXER_SELECTION_WIDTH_MAX));
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<std::size_t>(n_kv[row]));
        for (std::int32_t block = 0; block <= n_bid[row]; ++block) {
            const std::int32_t block_weight =
                block < n_bid[row]
                    ? static_cast<std::int32_t>(INDEXER_BLOCK_CELLS)
                    : n_kv[row] -
                          n_bid[row] *
                              static_cast<std::int32_t>(
                                  INDEXER_BLOCK_CELLS);
            if (block_weight == 0) continue;
            const float score =
                scores[row * max_blocks +
                       static_cast<std::size_t>(block)] + 0.0f;
            const std::uint32_t key = selection_order_key(score);
            for (std::int32_t cell = 0; cell < block_weight; ++cell) {
                candidates.push_back(Candidate{
                    key,
                    block * static_cast<std::int32_t>(
                                INDEXER_BLOCK_CELLS) +
                        cell});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& left, const Candidate& right) {
                      if (left.key > right.key) return true;
                      if (right.key > left.key) return false;
                      return left.id < right.id;
                  });
        candidates.resize(static_cast<std::size_t>(width));
        std::vector<std::int32_t> ids;
        ids.reserve(candidates.size());
        for (const Candidate& candidate : candidates) {
            ids.push_back(candidate.id);
        }
        std::sort(ids.begin(), ids.end());
        std::copy(ids.begin(), ids.end(),
                  output.begin() +
                      row * INDEXER_SELECTION_WIDTH_MAX);
    }
}

void qsa_gate(std::span<const float> attn,
              std::span<const float> q_full,
              std::uint32_t token_rows,
              std::vector<tf>& output) {
    constexpr std::size_t kHeadCount = N_HEAD_Q;
    constexpr std::size_t kHeadWidth = HEAD_DIM;
    constexpr std::size_t kAttnRow = kHeadCount * kHeadWidth;
    constexpr std::size_t kQFullRow = kHeadCount * 2u * kHeadWidth;
    output.resize(static_cast<std::size_t>(token_rows) * kAttnRow);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::size_t attn_row =
            static_cast<std::size_t>(row) * kAttnRow;
        const std::size_t q_row =
            static_cast<std::size_t>(row) * kQFullRow;
        for (std::size_t head = 0u; head < kHeadCount; ++head) {
            const std::size_t attn_base =
                attn_row + head * kHeadWidth;
            const std::size_t q_base =
                q_row + head * (2u * kHeadWidth) + kHeadWidth;
            for (std::size_t dimension = 0u;
                 dimension < kHeadWidth; ++dimension) {
                const tf gate{q_full[q_base + dimension]};
                const tf sigmoid =
                    tf{1.0f} /
                    (tf{1.0f} + twin::accurate_exp(-gate));
                output[attn_base + dimension] =
                    tf{attn[attn_base + dimension]} * sigmoid;
            }
        }
    }
}

}  // namespace oracle
