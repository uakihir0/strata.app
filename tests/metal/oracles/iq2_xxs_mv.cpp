#include "oracles/iq2_xxs_mv.hpp"

#include "../../../src/metal/kernels/common/codebooks.h"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <bit>
#include <limits>

namespace twin {
namespace {

int iq2_xxs_value(std::uint32_t index, std::uint32_t aux32,
                  std::uint32_t indices) {
    const std::uint32_t code_slot = (index % 32u) / 8u;
    const std::uint32_t component = index % 8u;
    const std::uint32_t code =
        (indices >> (8u * code_slot)) & 0xffu;
    const std::uint32_t raw_signs =
        (aux32 >> (7u * code_slot)) & 0xffu;
    const std::uint32_t signs =
        raw_signs ^ ((std::popcount(raw_signs) & 1u) << 7u);
    const int magnitude = static_cast<int>(
        (iq2xxs_grid[code] >> (8u * component)) & 0xffu);
    const bool negative = ((signs >> component) & 1u) != 0u;
    return negative ? -magnitude : magnitude;
}

std::uint32_t indices_for_part(const block_iq2_xxs& weight,
                               std::uint32_t part) {
    const std::uint32_t first = 4u * part;
    return static_cast<std::uint32_t>(weight.qs[first]) |
           (static_cast<std::uint32_t>(weight.qs[first + 1u]) << 16u);
}

std::uint32_t aux_for_part(const block_iq2_xxs& weight,
                           std::uint32_t part) {
    const std::uint32_t first = 4u * part + 2u;
    return static_cast<std::uint32_t>(weight.qs[first]) |
           (static_cast<std::uint32_t>(weight.qs[first + 1u]) << 16u);
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

tf butterfly_sum_tf(std::array<tf, 32> values) {
    for (std::size_t offset = 16; offset != 0; offset >>= 1) {
        const auto old = values;
        for (std::size_t lane = 0; lane < values.size(); ++lane) {
            values[lane] = old[lane] + old[lane ^ offset];
        }
    }
    return values[0];
}

int iq2_xxs_integer_dot(const block_iq2_xxs& weight, std::uint32_t part,
                        const Q81& activation) {
    const std::uint32_t indices = indices_for_part(weight, part);
    const std::uint32_t aux32 = aux_for_part(weight, part);
    int sumi = 0;
    for (std::uint32_t i = 0; i < 32u; ++i) {
        sumi += iq2_xxs_value(i, aux32, indices) *
                static_cast<int>(activation.qs[i]);
    }
    const int odd_scale = static_cast<int>((aux32 >> 27u) | 1u);
    return (sumi * odd_scale) / 8;
}

}  // namespace

bool iq2_xxs_matvec_reference(std::span<const block_iq2_xxs> weights,
                              std::span<const Q81> activations,
                              std::uint32_t n_in, std::uint32_t n_out,
                              std::uint32_t nc, std::uint32_t live_rows,
                              std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ2_XXS mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ2_XXS mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ2_XXS mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<float, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq2_xxs& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + kbx];
                for (std::uint32_t column = 0; column < nc; ++column) {
                    const std::uint32_t input_column =
                        column < live_rows ? column : 0u;
                    const Q81& activation = activations[
                        static_cast<std::size_t>(input_column) * (n_in / 32u) +
                        kbx * 8u + part];
                    const int sumi =
                        iq2_xxs_integer_dot(weight, part, activation);
                    const std::uint32_t ds = activation.ds;
                    const float d8 = strata::kernels::f32_from_f16(
                        static_cast<std::uint16_t>(ds & 0xffffu));
                    const float d = strata::kernels::f32_from_f16(weight.d);
                    accumulators[column][lane] = std::fma(
                        d * d8, static_cast<float>(sumi),
                        accumulators[column][lane]);
                }
            }
        }
        for (std::uint32_t column = 0; column < nc; ++column) {
            output[static_cast<std::size_t>(column) * n_out + row] =
                butterfly_sum(accumulators[column]);
        }
    }
    err.clear();
    return true;
}

bool iq2_xxs_matvec_tf_reference(std::span<const block_iq2_xxs> weights,
                                 std::span<const Q81> activations,
                                 std::uint32_t n_in, std::uint32_t n_out,
                                 std::uint32_t nc, std::uint32_t live_rows,
                                 std::vector<tf>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ2_XXS mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ2_XXS mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ2_XXS mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, tf{});
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<tf, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq2_xxs& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row +
                            kbx];
                const float d = strata::kernels::f32_from_f16(weight.d);
                for (std::uint32_t column = 0; column < nc; ++column) {
                    const std::uint32_t input_column =
                        column < live_rows ? column : 0u;
                    const Q81& activation = activations[
                        static_cast<std::size_t>(input_column) * (n_in / 32u) +
                        kbx * 8u + part];
                    const std::uint32_t ds = activation.ds;
                    const float d8 = strata::kernels::f32_from_f16(
                        static_cast<std::uint16_t>(ds & 0xffffu));
                    const int sumi =
                        iq2_xxs_integer_dot(weight, part, activation);
                    accumulators[column][lane] = fma(
                        tf(d) * tf(d8), tf(static_cast<float>(sumi)),
                        accumulators[column][lane]);
                }
            }
        }
        for (std::uint32_t column = 0; column < nc; ++column) {
            output[static_cast<std::size_t>(column) * n_out + row] =
                butterfly_sum_tf(accumulators[column]);
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
