#include "oracles/q6_k_mv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <limits>

namespace twin {
namespace {

std::uint32_t load_u32_b2(const std::uint8_t* bytes, std::uint32_t word_index) {
    const std::uint32_t offset = 4u * word_index;
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1u]) << 8u) |
           (static_cast<std::uint32_t>(bytes[offset + 2u]) << 16u) |
           (static_cast<std::uint32_t>(bytes[offset + 3u]) << 24u);
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

float accumulate_block(const block_q6_K& weight,
                       std::span<const Q81> activations,
                       std::uint32_t blocks_per_row,
                       std::uint32_t kbx, std::uint32_t subblock,
                       std::uint32_t input_column, float accumulator) {
    const std::uint32_t bq8_offset =
        4u * (subblock / 16u) + (subblock % 16u) / 8u;
    const std::uint32_t scale_offset =
        8u * (subblock / 16u) + (subblock % 16u) / 4u;
    const std::uint32_t qh_shift = 2u * ((subblock % 16u) / 8u);
    const std::uint32_t vl = load_u32_b2(weight.ql, subblock);
    const std::uint32_t vh = load_u32_b2(
        weight.qh, 8u * (subblock / 16u) + subblock % 8u);
    const std::uint32_t q8_word = subblock % 8u;
    float scaled_sum = 0.0f;
    for (std::uint32_t half = 0; half < 2; ++half) {
        int dot = 0;
        const std::uint32_t nibble_shift = 4u * half;
        for (std::uint32_t byte = 0; byte < 4; ++byte) {
            const std::uint32_t shift = nibble_shift + 8u * byte;
            const int low = static_cast<int>((vl >> shift) & 0x0fu);
            const int high = static_cast<int>((vh >> (qh_shift + shift)) & 0x03u);
            const int q = (low | (high << 4u)) - 32;
            const std::size_t activation_index =
                static_cast<std::size_t>(input_column) * (8u * blocks_per_row) +
                8u * kbx + bq8_offset + 2u * half;
            const Q81& activation = activations[activation_index];
            dot += q * static_cast<int>(activation.qs[4u * q8_word + byte]);
        }
        const std::size_t activation_index =
            static_cast<std::size_t>(input_column) * (8u * blocks_per_row) +
            8u * kbx + bq8_offset + 2u * half;
        const Q81& activation = activations[activation_index];
        const int scale = static_cast<int>(weight.scales[scale_offset + 4u * half]);
        const std::uint32_t ds = static_cast<std::uint32_t>(activation.ds);
        const float activation_scale = strata::kernels::f32_from_f16(
            static_cast<std::uint16_t>(ds & 0xffffu));
        scaled_sum = std::fma(activation_scale, static_cast<float>(dot * scale),
                              scaled_sum);
    }
    const float weight_scale = strata::kernels::f32_from_f16(weight.d);
    return std::fma(weight_scale, scaled_sum, accumulator);
}

}  // namespace

bool q6_k_matvec_reference(std::span<const block_q6_K> weights,
                           std::span<const Q81> activations,
                           std::uint32_t n_in, std::uint32_t n_out,
                           std::uint32_t nc, std::uint32_t live_rows,
                           std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1 && nc != 2 && nc != 4 && nc != 8) ||
        live_rows == 0 || live_rows > nc) {
        err = "Q6_K mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(blocks_per_row) * 8u * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "Q6_K mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "Q6_K mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    const std::uint32_t rows_per_group = blocks_per_row < 4u ? 4u : 1u;
    const std::uint32_t groups = (n_out - 1u) / rows_per_group + 1u;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t row_base = group * rows_per_group;
        std::array<std::array<std::array<float, 4>, 8>, 128> accumulators{};
        for (std::uint32_t tid = 0; tid < 128u; ++tid) {
            const std::uint32_t subblock = tid % 32u;
            for (std::uint32_t kbx = tid / 32u; kbx < blocks_per_row; kbx += 4u) {
                for (std::uint32_t i = 0; i < rows_per_group; ++i) {
                    const std::uint32_t row = row_base + i;
                    if (row >= n_out) continue;
                    const block_q6_K& weight =
                        weights[static_cast<std::size_t>(row) * blocks_per_row + kbx];
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const std::uint32_t input_column =
                            column < live_rows ? column : 0u;
                        accumulators[tid][column][i] = accumulate_block(
                            weight, activations, blocks_per_row, kbx,
                            subblock, input_column,
                            accumulators[tid][column][i]);
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
