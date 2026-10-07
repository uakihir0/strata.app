#include "attention.hpp"

#include "../../../src/metal/kernels/common/params.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace oracle {
namespace {

using twin::tf;

constexpr std::uint32_t kHeadsPerKv = N_HEAD_Q / N_HEAD_KV;
constexpr std::uint32_t kChunkCells = 64u;
constexpr std::size_t kQ4BlockBytes = sizeof(block_q4_0);
constexpr float kNegativeMax = -0x1.fffffep127f;

std::uint16_t read_u16(std::span<const std::uint8_t> bytes,
                       std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset]) |
           static_cast<std::uint16_t>(
               static_cast<std::uint16_t>(bytes[offset + 1u]) << 8u);
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

float q4_value(std::span<const std::uint8_t> data, std::uint32_t pool_row,
               std::uint32_t dimension) {
    const std::size_t block =
        static_cast<std::size_t>(pool_row) * 8u + dimension / 32u;
    const std::size_t block_offset = block * kQ4BlockBytes;
    const std::uint32_t within = dimension % 32u;
    const std::uint8_t packed =
        data[block_offset + 2u + within % 16u];
    const std::uint32_t code =
        within < 16u ? packed & 0x0fu : packed >> 4u;
    const float scale = half_bits_to_float(read_u16(data, block_offset));
    return (static_cast<float>(code) - 8.0f) * scale;
}

float cache_value(strata::metal::QsaAttentionMode mode, bool key,
                  std::span<const std::uint8_t> data,
                  std::span<const std::uint16_t> scales,
                  std::uint32_t pool_row, std::uint32_t dimension) {
    if (mode == strata::metal::QsaAttentionMode::f16) {
        const std::size_t index =
            (static_cast<std::size_t>(pool_row) * HEAD_DIM + dimension) * 2u;
        return half_bits_to_float(read_u16(data, index));
    }
    if (mode == strata::metal::QsaAttentionMode::i8 ||
        (key && mode == strata::metal::QsaAttentionMode::k8v4)) {
        const std::size_t index =
            static_cast<std::size_t>(pool_row) * HEAD_DIM + dimension;
        const std::int8_t code = std::bit_cast<std::int8_t>(data[index]);
        const std::size_t scale_index =
            static_cast<std::size_t>(pool_row) * 4u + dimension / 64u;
        const float scale = half_bits_to_float(scales[scale_index]);
        return static_cast<float>(code) * scale;
    }
    return q4_value(data, pool_row, dimension);
}

std::array<tf, 32> butterfly_sum(std::array<tf, 32> values) {
    for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0u; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values;
}

std::array<tf, 32> butterfly_max(std::array<tf, 32> values) {
    for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0u; lane < values.size(); ++lane) {
            values[lane] =
                tf{std::fmax(previous[lane].value,
                             previous[lane ^ offset].value)};
        }
    }
    return values;
}

}  // namespace

void qsa_attention(
    strata::metal::QsaAttentionMode mode,
    std::span<const float> query,
    std::span<const StepRow> steps,
    std::span<const std::int32_t> ids,
    std::span<const std::uint8_t> key_data,
    std::span<const std::uint16_t> key_scales,
    std::span<const std::uint8_t> value_data,
    std::span<const std::uint16_t> value_scales,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride, std::vector<tf>& output) {
    if (steps.size() < token_rows ||
        ids.size() < static_cast<std::size_t>(token_rows) * id_row_stride ||
        query.size() <
            static_cast<std::size_t>(token_rows) * N_HEAD_Q * HEAD_DIM) {
        throw std::invalid_argument("QSA attention oracle input extent mismatch");
    }
    if (cell_capacity == 0u) {
        throw std::invalid_argument("QSA attention oracle cell capacity is zero");
    }

    const std::uint32_t chunk_count =
        (id_row_stride - 1u) / kChunkCells + 1u;
    const std::size_t slots =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV * chunk_count;
    std::vector<tf> partial_m(slots * kHeadsPerKv, tf{kNegativeMax});
    std::vector<tf> partial_l(slots * kHeadsPerKv);
    std::vector<tf> partial_acc(
        slots * kHeadsPerKv * HEAD_DIM);

    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::uint32_t width =
            static_cast<std::uint32_t>(steps[row].width);
        for (std::uint32_t kvh = 0u; kvh < N_HEAD_KV; ++kvh) {
            for (std::uint32_t chunk = 0u; chunk < chunk_count; ++chunk) {
                const std::uint32_t chunk_start = chunk * kChunkCells;
                const std::uint32_t n_here = chunk_start < width
                    ? std::min(kChunkCells, width - chunk_start)
                    : 0u;
                const auto valid_cell = [&](std::uint32_t cell) {
                    if (cell >= n_here) return false;
                    const std::int32_t signed_cell =
                        ids[static_cast<std::size_t>(row) *
                                id_row_stride +
                            chunk_start + cell];
                    return signed_cell >= 0 &&
                           static_cast<std::uint32_t>(signed_cell) <
                               cell_capacity &&
                           signed_cell < steps[row].n_kv;
                };
                const std::size_t slot =
                    (static_cast<std::size_t>(row) * N_HEAD_KV + kvh) *
                        chunk_count +
                    chunk;
                const std::size_t stats_base = slot * kHeadsPerKv;
                if (n_here == 0u) continue;

                std::array<tf, kHeadsPerKv * kChunkCells> scores{};
                std::fill(scores.begin(), scores.end(), tf{kNegativeMax});
                for (std::uint32_t head = 0u;
                     head < kHeadsPerKv; ++head) {
                    const std::size_t query_base =
                        ((static_cast<std::size_t>(row) * N_HEAD_Q +
                          kvh * kHeadsPerKv + head) *
                         HEAD_DIM);
                    for (std::uint32_t cell = 0u; cell < n_here; ++cell) {
                        if (!valid_cell(cell)) continue;
                        const std::int32_t signed_cell =
                            ids[static_cast<std::size_t>(row) *
                                    id_row_stride +
                                chunk_start + cell];
                        const std::uint32_t cache_cell =
                            static_cast<std::uint32_t>(signed_cell);
                        const std::uint32_t pool_row =
                            ((cache_cell / INDEXER_BLOCK_CELLS) *
                                 N_HEAD_KV +
                             kvh) *
                                INDEXER_BLOCK_CELLS +
                            cache_cell % INDEXER_BLOCK_CELLS;
                        std::array<tf, 32> lane_scores{};
                        for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                            const std::uint32_t dimension = lane * 8u;
                            tf score =
                                tf{cache_value(mode, true, key_data,
                                               key_scales, pool_row,
                                               dimension)} *
                                tf{query[query_base + dimension]};
                            for (std::uint32_t offset = 1u; offset < 8u;
                                 ++offset) {
                                score = twin::fma(
                                    tf{cache_value(
                                        mode, true, key_data, key_scales,
                                        pool_row, dimension + offset)},
                                    tf{query[query_base + dimension + offset]},
                                    score);
                            }
                            lane_scores[lane] = score;
                        }
                        const auto reduced = butterfly_sum(lane_scores);
                        scores[head * kChunkCells + cell] =
                            reduced[0] * tf{0.0625f};
                    }

                    std::array<tf, 32> lane_maxima{};
                    for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                        const float first = scores[
                            head * kChunkCells + lane].value;
                        const float second = scores[
                            head * kChunkCells + lane + 32u].value;
                        lane_maxima[lane] =
                            tf{std::fmax(first, second)};
                    }
                    const tf maximum = butterfly_max(lane_maxima)[0];
                    partial_m[stats_base + head] = maximum;

                    std::array<tf, 32> lane_sums{};
                    std::array<tf, kChunkCells> weights{};
                    for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                        const tf exp_a = valid_cell(lane)
                            ? twin::accurate_exp(
                                  scores[head * kChunkCells + lane] -
                                  maximum)
                            : tf{};
                        const tf exp_b = valid_cell(lane + 32u)
                            ? twin::accurate_exp(
                                  scores[head * kChunkCells + lane + 32u] -
                                  maximum)
                            : tf{};
                        weights[lane] = exp_a;
                        weights[lane + 32u] = exp_b;
                        lane_sums[lane] = exp_a + exp_b;
                    }
                    const tf denominator = butterfly_sum(lane_sums)[0];
                    partial_l[stats_base + head] = denominator;

                    const std::size_t acc_base =
                        (slot * kHeadsPerKv + head) * HEAD_DIM;
                    for (std::uint32_t dimension = 0u;
                         dimension < HEAD_DIM; ++dimension) {
                        tf accumulator{};
                        for (std::uint32_t cell = 0u; cell < n_here; ++cell) {
                            if (!valid_cell(cell)) continue;
                            const std::int32_t signed_cell =
                                ids[static_cast<std::size_t>(row) *
                                        id_row_stride +
                                    chunk_start + cell];
                            const std::uint32_t cache_cell =
                                static_cast<std::uint32_t>(signed_cell);
                            const std::uint32_t pool_row =
                                ((cache_cell / INDEXER_BLOCK_CELLS) *
                                     N_HEAD_KV +
                                 kvh) *
                                    INDEXER_BLOCK_CELLS +
                                cache_cell % INDEXER_BLOCK_CELLS;
                            const tf value = tf{cache_value(
                                mode, false, value_data, value_scales,
                                pool_row, dimension)};
                            accumulator = twin::fma(
                                weights[cell], value, accumulator);
                        }
                        partial_acc[acc_base + dimension] = accumulator;
                    }
                }
            }
        }
    }

    output.assign(
        static_cast<std::size_t>(token_rows) * N_HEAD_Q * HEAD_DIM, tf{});
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        for (std::uint32_t head = 0u; head < N_HEAD_Q; ++head) {
            const std::uint32_t kvh = head / kHeadsPerKv;
            const std::uint32_t head_local = head % kHeadsPerKv;
            const std::size_t slot_base =
                (static_cast<std::size_t>(row) * N_HEAD_KV + kvh) *
                chunk_count;
            tf maximum{kNegativeMax};
            for (std::uint32_t chunk = 0u; chunk < chunk_count; ++chunk) {
                maximum = tf{std::fmax(
                    maximum.value,
                    partial_m[(slot_base + chunk) * kHeadsPerKv +
                              head_local]
                        .value)};
            }

            tf denominator{};
            for (std::uint32_t chunk = 0u; chunk < chunk_count; ++chunk) {
                const std::size_t slot = slot_base + chunk;
                const tf chunk_maximum =
                    partial_m[slot * kHeadsPerKv + head_local];
                if (chunk_maximum.value == kNegativeMax) continue;
                const tf weight =
                    twin::accurate_exp(chunk_maximum - maximum);
                denominator = twin::fma(
                    partial_l[slot * kHeadsPerKv + head_local], weight,
                    denominator);
            }
            for (std::uint32_t dimension = 0u;
                 dimension < HEAD_DIM; ++dimension) {
                tf accumulator{};
                for (std::uint32_t chunk = 0u; chunk < chunk_count; ++chunk) {
                    const std::size_t slot = slot_base + chunk;
                    const tf chunk_maximum =
                        partial_m[slot * kHeadsPerKv + head_local];
                    if (chunk_maximum.value == kNegativeMax) continue;
                    const tf weight =
                        twin::accurate_exp(chunk_maximum - maximum);
                    accumulator = twin::fma(
                        partial_acc[(slot * kHeadsPerKv + head_local) *
                                        HEAD_DIM +
                                    dimension],
                        weight, accumulator);
                }
                output[(static_cast<std::size_t>(row) * N_HEAD_Q + head) *
                                   HEAD_DIM +
                       dimension] = denominator.value > 0.0f
                    ? accumulator / denominator
                    : tf{};
            }
        }
    }
}

}  // namespace oracle
