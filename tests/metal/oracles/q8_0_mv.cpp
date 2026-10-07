#include "oracles/q8_0_mv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
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

int integer_dot(const block_q8_0& weight, const Q81& activation,
                std::uint32_t first_word) {
    int sum = 0;
    for (std::uint32_t word = 0; word < 2; ++word) {
        const std::uint32_t offset = 4u * (first_word + word);
        for (std::uint32_t lane = 0; lane < 4; ++lane) {
            sum += static_cast<int>(weight.qs[offset + lane]) *
                   static_cast<int>(activation.qs[offset + lane]);
        }
    }
    return sum;
}

}  // namespace

bool q8_0_matvec_reference(std::span<const block_q8_0> weights,
                           std::span<const Q81> activations,
                           std::uint32_t n_in, std::uint32_t n_out,
                           std::uint32_t nc, std::uint32_t live_rows,
                           std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 32u != 0u || n_out == 0 ||
        (nc != 1 && nc != 2 && nc != 4 && nc != 8) ||
        live_rows == 0 || live_rows > nc) {
        err = "Q8_0 mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 32u;
    if (static_cast<std::uint64_t>(blocks_per_row) * n_out != weights.size() ||
        static_cast<std::uint64_t>(blocks_per_row) * nc != activations.size()) {
        err = "Q8_0 mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "Q8_0 mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    const bool small_k = blocks_per_row < 32u;
    const std::uint32_t rows_per_group = small_k ? 4u : 1u;
    const std::uint32_t groups =
        (n_out + rows_per_group - 1u) / rows_per_group;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t row_base = group * rows_per_group;
        std::array<std::array<std::array<float, 4>, 8>, 128> accumulators{};
        for (std::uint32_t tid = 0; tid < 128u; ++tid) {
            const std::uint32_t first_word = 2u * (tid % 4u);
            for (std::uint32_t kbx = tid / 4u; kbx < blocks_per_row; kbx += 32u) {
                for (std::uint32_t i = 0; i < rows_per_group; ++i) {
                    const std::uint32_t row = row_base + i;
                    if (row >= n_out) continue;
                    const block_q8_0& weight =
                        weights[static_cast<std::size_t>(row) * blocks_per_row + kbx];
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const std::uint32_t input_column = column < live_rows ? column : 0u;
                        const Q81& activation =
                            activations[static_cast<std::size_t>(input_column) * blocks_per_row + kbx];
                        const int dot = integer_dot(weight, activation, first_word);
                        const std::uint32_t ds = static_cast<std::uint32_t>(activation.ds);
                        const float scale =
                            strata::kernels::f32_from_f16(weight.d) *
                            strata::kernels::f32_from_f16(static_cast<std::uint16_t>(ds & 0xffffu));
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
