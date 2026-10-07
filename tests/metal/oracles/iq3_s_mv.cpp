#include "oracles/iq3_s_mv.hpp"

#include "../../../src/metal/kernels/common/codebooks.h"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <limits>

namespace twin {
namespace {

int iq3_s_value(const block_iq3_s& weight, std::uint32_t index) {
    const std::uint32_t ib = index / 32u;
    const std::uint32_t il = (index % 32u) / 8u;
    const std::uint32_t pos = index % 8u;
    const std::uint32_t g = pos / 4u;
    const std::uint32_t component = pos % 4u;
    const std::uint32_t qs_index = 8u * ib + 2u * il + g;
    const std::uint32_t high_bit =
        (static_cast<std::uint32_t>(weight.qh[ib]) >> (2u * il + g)) & 1u;
    const std::uint32_t code =
        static_cast<std::uint32_t>(weight.qs[qs_index]) | (high_bit << 8u);
    const std::uint32_t magnitude =
        (iq3s_grid[code] >> (8u * component)) & 0xffu;
    const bool negative =
        ((static_cast<std::uint32_t>(weight.signs[4u * ib + il]) >> pos) & 1u) != 0u;
    return negative ? -static_cast<int>(magnitude) : static_cast<int>(magnitude);
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

int iq3_s_integer_dot(const block_iq3_s& weight, std::uint32_t part,
                      const Q81& activation) {
    int sumi = 0;
    const std::uint32_t value_base = 32u * part;
    for (std::uint32_t i = 0; i < 32u; ++i) {
        sumi += iq3_s_value(weight, value_base + i) *
                static_cast<int>(activation.qs[i]);
    }
    const std::uint32_t packed_scale = weight.scales[part / 2u];
    const std::uint32_t nibble =
        (packed_scale >> (4u * (part % 2u))) & 0x0fu;
    return sumi * (1 + 2 * static_cast<int>(nibble));
}

}  // namespace

bool iq3_s_matvec_reference(std::span<const block_iq3_s> weights,
                            std::span<const Q81> activations,
                            std::uint32_t n_in, std::uint32_t n_out,
                            std::uint32_t nc, std::uint32_t live_rows,
                            std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0 || live_rows > nc) {
        err = "IQ3_S mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ3_S mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ3_S mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<float, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq3_s& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + kbx];
                const std::uint32_t activation_index = kbx * 8u + part;
                for (std::uint32_t column = 0; column < nc; ++column) {
                    const std::uint32_t input_column =
                        column < live_rows ? column : 0u;
                    const Q81& activation = activations[
                        static_cast<std::size_t>(input_column) * (n_in / 32u) +
                        activation_index];
                    const int sumi =
                        iq3_s_integer_dot(weight, part, activation);
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

bool iq3_s_matvec_tf_reference(
    std::span<const block_iq3_s> weights,
    std::span<const Q81> activations, std::uint32_t n_in,
    std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
    std::vector<tf>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ3_S mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ3_S mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ3_S mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, tf{});
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<tf, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq3_s& weight =
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
                        iq3_s_integer_dot(weight, part, activation);
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
