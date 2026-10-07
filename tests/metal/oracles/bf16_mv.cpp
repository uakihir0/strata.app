#include "oracles/bf16_mv.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace twin {
namespace {

std::uint32_t block_size_for(std::uint32_t n_in) {
    std::uint32_t best = 32;
    std::uint64_t best_iterations = (static_cast<std::uint64_t>(n_in) + 63u) / 64u;
    for (std::uint32_t candidate = 64; candidate <= 256; candidate += 32) {
        const std::uint64_t iterations =
            (static_cast<std::uint64_t>(n_in) + 2u * candidate - 1u) / (2u * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
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

bool bf16_matvec_reference(std::span<const std::uint16_t> weights,
                           std::span<const float> inputs,
                           std::uint32_t n_in, std::uint32_t n_out,
                           std::uint32_t nc, std::uint32_t live_rows,
                           std::uint32_t block_size,
                           std::vector<float>& output, std::string& err) {
    if (n_in == 0 || (n_in & 1u) != 0u || n_out == 0 ||
        (nc != 1 && nc != 2 && nc != 4 && nc != 8) ||
        live_rows == 0 || live_rows > nc ||
        block_size != block_size_for(n_in) ||
        static_cast<std::uint64_t>(n_in) * n_out != weights.size() ||
        static_cast<std::uint64_t>(n_in) * nc != inputs.size()) {
        err = "BF16 mat-vec oracle shape or dispatch parameters are invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "BF16 mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    for (std::uint32_t column = 0; column < nc; ++column) {
        const std::uint32_t input_column = column < live_rows ? column : 0u;
        for (std::uint32_t row = 0; row < n_out; ++row) {
            std::array<float, 256> accumulators{};
            for (std::uint32_t tid = 0; tid < block_size; ++tid) {
                for (std::uint32_t pair = tid; pair < n_in / 2u; pair += block_size) {
                    const std::size_t w_index =
                        static_cast<std::size_t>(row) * n_in + static_cast<std::size_t>(pair) * 2u;
                    const std::size_t x_index =
                        static_cast<std::size_t>(input_column) * n_in + static_cast<std::size_t>(pair) * 2u;
                    const float w0 = strata::kernels::f32_from_bf16(weights[w_index]);
                    const float w1 = strata::kernels::f32_from_bf16(weights[w_index + 1]);
                    accumulators[tid] = std::fma(w0, inputs[x_index], accumulators[tid]);
                    accumulators[tid] = std::fma(w1, inputs[x_index + 1], accumulators[tid]);
                }
            }

            std::array<float, 256> thread_sums = accumulators;
            for (std::uint32_t warp = 0; warp < block_size / 32u; ++warp) {
                std::array<float, 32> lanes{};
                std::copy_n(thread_sums.begin() + warp * 32u, 32, lanes.begin());
                thread_sums[warp * 32u] = butterfly_sum(lanes);
            }

            float sum = thread_sums[0];
            if (block_size > 32u) {
                std::array<float, 32> partials{};
                for (std::uint32_t warp = 0; warp < block_size / 32u; ++warp) {
                    partials[warp] = thread_sums[warp * 32u];
                }
                sum = butterfly_sum(partials);
            }
            output[static_cast<std::size_t>(column) * n_out + row] = sum;
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
