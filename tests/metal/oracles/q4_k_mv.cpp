#include "oracles/q4_k_mv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <limits>

namespace twin {
namespace {

std::array<std::uint32_t, 2> scale_min(const block_q4_K& weight,
                                       std::uint32_t group) {
    if (group < 4u) {
        return {
            static_cast<std::uint32_t>(weight.scales[group] & 0x3fu),
            static_cast<std::uint32_t>(weight.scales[group + 4u] & 0x3fu)};
    }
    return {
        static_cast<std::uint32_t>(weight.scales[group + 4u] & 0x0fu) |
            (static_cast<std::uint32_t>(weight.scales[group - 4u] >> 6u) << 4u),
        static_cast<std::uint32_t>(weight.scales[group + 4u] >> 4u) |
            (static_cast<std::uint32_t>(weight.scales[group] >> 6u) << 4u)};
}

float accumulate_block(const block_q4_K& weight,
                       std::span<const Q81> activations,
                       std::uint32_t blocks_per_row,
                       std::uint32_t kbx, std::uint32_t iqs,
                       std::uint32_t input_column, float accumulator) {
    const std::uint32_t bq8_offset = 2u * ((iqs / 2u) / 4u);
    const std::uint32_t q8_word = (iqs / 2u) % 4u;
    float scaled_dot = 0.0f;
    float scaled_min = 0.0f;
    for (std::uint32_t pair = 0; pair < 2u; ++pair) {
        const std::uint32_t group = bq8_offset + pair;
        const std::size_t activation_index =
            static_cast<std::size_t>(input_column) * (8u * blocks_per_row) +
            8u * kbx + group;
        const Q81& activation = activations[activation_index];
        const auto [scale, min] = scale_min(weight, group);
        const std::uint32_t qbyte_base =
            32u * (group / 2u) + 4u * q8_word;
        const std::uint32_t nibble_shift = 4u * (group % 2u);
        int dot = 0;
        int sum_q8 = 0;
        for (std::uint32_t half = 0; half < 2u; ++half) {
            const std::uint32_t offset = 16u * half;
            for (std::uint32_t byte = 0; byte < 4u; ++byte) {
                const std::uint32_t index = offset + byte;
                const int q4 = static_cast<int>(
                    (weight.qs[qbyte_base + index] >> nibble_shift) & 0x0fu);
                const int q8 = static_cast<int>(
                    activation.qs[4u * q8_word + index]);
                dot += q4 * q8;
                sum_q8 += q8;
            }
        }
        const std::uint32_t ds = static_cast<std::uint32_t>(activation.ds);
        const float d8 = strata::kernels::f32_from_f16(
            static_cast<std::uint16_t>(ds & 0xffffu));
        scaled_dot = std::fma(
            d8, static_cast<float>(dot * static_cast<int>(scale)), scaled_dot);
        scaled_min = std::fma(
            d8, static_cast<float>(sum_q8 * static_cast<int>(min)), scaled_min);
    }
    const std::uint32_t dm = static_cast<std::uint32_t>(weight.dm);
    const float d = strata::kernels::f32_from_f16(
        static_cast<std::uint16_t>(dm & 0xffffu));
    const float dmin = strata::kernels::f32_from_f16(
        static_cast<std::uint16_t>(dm >> 16u));
    const float contribution = std::fma(-dmin, scaled_min, d * scaled_dot);
    return accumulator + contribution;
}

float butterfly_sum(std::array<float, 32> values) {
    for (std::size_t offset = 16; offset != 0; offset >>= 1) {
        const auto old = values;
        for (std::size_t lane = 0; lane < values.size(); ++lane) {
            values[lane] = old[lane] + old[lane ^ offset];
        }
    }
    return values[0];
}

}  // namespace

bool q4_k_matvec_reference(std::span<const block_q4_K> weights,
                           std::span<const Q81> activations,
                           std::uint32_t n_in, std::uint32_t n_out,
                           std::uint32_t nc, std::uint32_t live_rows,
                           std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1 && nc != 2 && nc != 4 && nc != 8) ||
        live_rows == 0 || live_rows > nc) {
        err = "Q4_K mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(blocks_per_row) * 8u * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "Q4_K mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "Q4_K mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    const std::uint32_t rows_per_group = blocks_per_row < 8u ? 4u : 1u;
    const std::uint32_t groups = (n_out - 1u) / rows_per_group + 1u;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t row_base = group * rows_per_group;
        std::array<std::array<std::array<float, 4>, 8>, 128> accumulators{};
        for (std::uint32_t tid = 0; tid < 128u; ++tid) {
            const std::uint32_t iqs = 2u * (tid % 16u);
            for (std::uint32_t kbx = tid / 16u; kbx < blocks_per_row;
                 kbx += 8u) {
                for (std::uint32_t i = 0; i < rows_per_group; ++i) {
                    const std::uint32_t row = row_base + i;
                    if (row >= n_out) continue;
                    const block_q4_K& weight =
                        weights[static_cast<std::size_t>(row) * blocks_per_row +
                                kbx];
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const std::uint32_t input_column =
                            column < live_rows ? column : 0u;
                        accumulators[tid][column][i] = accumulate_block(
                            weight, activations, blocks_per_row, kbx, iqs,
                            input_column, accumulators[tid][column][i]);
                    }
                }
            }
        }

        for (std::uint32_t column = 0; column < nc; ++column) {
            for (std::uint32_t i = 0; i < rows_per_group; ++i) {
                const std::uint32_t row = row_base + i;
                if (row >= n_out) continue;
                std::array<float, 32> lane_values{};
                for (std::uint32_t lane = 0; lane < lane_values.size(); ++lane) {
                    float value = accumulators[lane][column][i];
                    value += accumulators[32u + lane][column][i];
                    value += accumulators[64u + lane][column][i];
                    value += accumulators[96u + lane][column][i];
                    lane_values[lane] = value;
                }
                output[static_cast<std::size_t>(column) * n_out + row] =
                    butterfly_sum(lane_values);
            }
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
