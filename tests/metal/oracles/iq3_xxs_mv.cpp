#include "oracles/iq3_xxs_mv.hpp"

#include "../../../src/metal/kernels/common/codebooks.h"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <bit>
#include <limits>

namespace twin {
namespace {

std::uint32_t unpack_ksigns(std::uint32_t sign_index) {
    sign_index &= 0x7fu;
    return sign_index ^ ((std::popcount(sign_index) & 1u) << 7u);
}

std::uint32_t part_aux(const block_iq3_xxs& weight, std::uint32_t part) {
    const std::uint32_t offset = 64u + 4u * part;
    return std::uint32_t(weight.qs[offset + 0u]) |
           (std::uint32_t(weight.qs[offset + 1u]) << 8u) |
           (std::uint32_t(weight.qs[offset + 2u]) << 16u) |
           (std::uint32_t(weight.qs[offset + 3u]) << 24u);
}

int iq3_xxs_value(const block_iq3_xxs& weight, std::uint32_t part,
                  std::uint32_t index) {
    const std::uint32_t code_slot = (index % 32u) / 8u;
    const std::uint32_t component = index % 8u;
    const std::uint32_t grid_slot = component / 4u;
    const std::uint32_t grid_component = component % 4u;
    const std::uint32_t grid_index =
        weight.qs[8u * part + 2u * code_slot + grid_slot];
    const std::uint32_t aux32 = part_aux(weight, part);
    const std::uint32_t sign_index =
        (aux32 >> (7u * code_slot)) & 0x7fu;
    const std::uint32_t signs = unpack_ksigns(sign_index);
    const int magnitude = static_cast<int>(
        (iq3xxs_grid[grid_index] >> (8u * grid_component)) & 0xffu);
    return (signs & (1u << component)) != 0u ? -magnitude : magnitude;
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

int iq3_xxs_integer_dot(const block_iq3_xxs& weight, std::uint32_t part,
                        const Q81& activation) {
    const std::uint32_t scale = part_aux(weight, part) >> 28u;
    int sumi = 0;
    for (std::uint32_t i = 0; i < 32u; ++i) {
        sumi += iq3_xxs_value(weight, part, i) *
                static_cast<int>(activation.qs[i]);
    }
    return (static_cast<int>(scale) * sumi + sumi / 2) / 2;
}

}  // namespace

void iq3_xxs_dequantize_reference(const block_iq3_xxs& weight,
                                  std::span<float, 256> output) {
    const float d = strata::kernels::f32_from_f16(weight.d);
    for (std::uint32_t part = 0u; part < 8u; ++part) {
        const std::uint32_t aux = part_aux(weight, part);
        const std::uint32_t scale_code = aux >> 28u;
        const float scale =
            d * (0.5f + static_cast<float>(scale_code)) * 0.5f;
        for (std::uint32_t code_slot = 0u; code_slot < 4u; ++code_slot) {
            const std::uint32_t sign_index =
                (aux >> (7u * code_slot)) & 0x7fu;
            const std::uint32_t signs = unpack_ksigns(sign_index);
            for (std::uint32_t component = 0u; component < 8u;
                 ++component) {
                const std::uint32_t grid_index =
                    weight.qs[8u * part + 2u * code_slot + component / 4u];
                const std::uint32_t packed =
                    iq3xxs_grid[grid_index];
                const std::uint32_t magnitude =
                    (packed >> (8u * (component % 4u))) & 0xffu;
                const float sign =
                    (signs & (1u << component)) != 0u ? -1.0f : 1.0f;
                const std::uint32_t index =
                    part * 32u + code_slot * 8u + component;
                output[index] =
                    scale * static_cast<float>(magnitude) * sign;
            }
        }
    }
}

bool iq3_xxs_matvec_reference(std::span<const block_iq3_xxs> weights,
                              std::span<const Q81> activations,
                              std::uint32_t n_in, std::uint32_t n_out,
                              std::uint32_t nc, std::uint32_t live_rows,
                              std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ3_XXS mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ3_XXS mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ3_XXS mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<float, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq3_xxs& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + kbx];
                for (std::uint32_t column = 0; column < nc; ++column) {
                    const std::uint32_t input_column =
                        column < live_rows ? column : 0u;
                    const Q81& activation = activations[
                        static_cast<std::size_t>(input_column) * (n_in / 32u) +
                        kbx * 8u + part];
                    const int sumi =
                        iq3_xxs_integer_dot(weight, part, activation);
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

bool iq3_xxs_matvec_tf_reference(
    std::span<const block_iq3_xxs> weights,
    std::span<const Q81> activations, std::uint32_t n_in,
    std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
    std::vector<tf>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ3_XXS mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ3_XXS mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ3_XXS mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, tf{});
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<tf, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq3_xxs& weight =
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
                        iq3_xxs_integer_dot(weight, part, activation);
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
