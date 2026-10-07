#include "oracles/q2_0_mv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <limits>

namespace twin {
namespace {

float butterfly_sum(std::array<float, 32> values) {
    for (std::size_t offset = 16; offset != 0; offset >>= 1) {
        const auto old = values;
        for (std::size_t lane = 0; lane < values.size(); ++lane) {
            values[lane] = old[lane] + old[lane ^ offset];
        }
    }
    return values[0];
}

int integer_dot(const block_q2_0& weight, const Q81& activation,
                std::uint32_t qs_half) {
    int sum = 0;
    const std::uint32_t byte_base = qs_half * 8u;
    for (std::uint32_t word = 0; word < 4; ++word) {
        for (std::uint32_t byte_in_word = 0; byte_in_word < 2; ++byte_in_word) {
            const std::uint32_t packed =
                weight.qs[byte_base + 2u * word + byte_in_word];
            const std::uint32_t activation_base = 8u * word + 4u * byte_in_word;
            for (std::uint32_t lane = 0; lane < 4; ++lane) {
                const int code = static_cast<int>((packed >> (2u * lane)) & 0x03u) - 1;
                sum += code * static_cast<int>(activation.qs[activation_base + lane]);
            }
        }
    }
    return sum;
}

}  // namespace

bool q2_0_matvec_reference(std::span<const block_q2_0> weights,
                           std::span<const Q81> activations,
                           std::uint32_t n_in, std::uint32_t n_out,
                           std::uint32_t nc, std::uint32_t live_rows,
                           std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 64u != 0u || n_out == 0 ||
        (nc != 1 && nc != 2 && nc != 4 && nc != 8) ||
        live_rows == 0 || live_rows > nc) {
        err = "Q2_0 mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 64u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(blocks_per_row) * 2u * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "Q2_0 mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "Q2_0 mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    const bool small_k = blocks_per_row < 64u;
    const std::uint32_t rows_per_group = small_k ? 4u : 1u;
    const std::uint32_t groups = (n_out - 1u) / rows_per_group + 1u;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t row_base = group * rows_per_group;
        std::array<std::array<std::array<float, 4>, 8>, 128> accumulators{};
        for (std::uint32_t tid = 0; tid < 128u; ++tid) {
            const std::uint32_t qs_half = tid % 2u;
            for (std::uint32_t kbx = tid / 2u; kbx < blocks_per_row; kbx += 64u) {
                const std::uint32_t kby = 2u * kbx;
                for (std::uint32_t i = 0; i < rows_per_group; ++i) {
                    const std::uint32_t row = row_base + i;
                    if (row >= n_out) continue;
                    const block_q2_0& weight =
                        weights[static_cast<std::size_t>(row) * blocks_per_row + kbx];
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const std::uint32_t input_column = column < live_rows ? column : 0u;
                        const Q81& activation = activations[
                            static_cast<std::size_t>(input_column) * (2u * blocks_per_row) +
                            kby + qs_half];
                        const int dot = integer_dot(weight, activation, qs_half);
                        const std::uint32_t ds = static_cast<std::uint32_t>(activation.ds);
                        const float activation_scale = strata::kernels::f32_from_f16(
                            static_cast<std::uint16_t>(ds & 0xffffu));
                        const float scale =
                            strata::kernels::f32_from_f16(weight.d) * activation_scale;
                        accumulators[tid][column][i] =
                            std::fma(scale, static_cast<float>(dot),
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
