#include "oracles/iq1_m_mv.hpp"

#include "../../../src/metal/kernels/common/codebooks.h"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace twin {
namespace {

static_assert(sizeof(block_iq1_m) == 56u);
static_assert(offsetof(block_iq1_m, qs) == 0u);
static_assert(offsetof(block_iq1_m, qh) == 32u);
static_assert(offsetof(block_iq1_m, scales) == 48u);

std::uint16_t iq1_m_half_scale_bits(const block_iq1_m& weight) {
    std::uint16_t bits = 0;
    for (std::uint32_t word_index = 0; word_index < 4u; ++word_index) {
        const std::uint16_t word =
            static_cast<std::uint16_t>(weight.scales[2u * word_index]) |
            (static_cast<std::uint16_t>(weight.scales[2u * word_index + 1u])
             << 8u);
        bits |= static_cast<std::uint16_t>(
            ((word >> 12u) & 0x0fu) << (4u * word_index));
    }
    return bits;
}

std::uint32_t iq1_m_packed_scale(const block_iq1_m& weight,
                                 std::uint32_t part) {
    const std::uint32_t word_index = part / 2u;
    const std::uint16_t word =
        static_cast<std::uint16_t>(weight.scales[2u * word_index]) |
        (static_cast<std::uint16_t>(weight.scales[2u * word_index + 1u])
         << 8u);
    return static_cast<std::uint32_t>(word) >> (6u * (part % 2u));
}

float iq1_m_dot(const block_iq1_m& weight, std::uint32_t part,
                const Q81& activation) {
    std::array<int, 2> sumi{};
    std::array<float, 2> sumf{};

    for (std::uint32_t codeword = 0; codeword < 4u; ++codeword) {
        const std::uint32_t qh_byte = weight.qh[2u * part + codeword / 2u];
        const std::uint32_t qhl =
            (qh_byte >> (4u * (codeword % 2u))) & 0x0fu;
        const std::uint32_t grid_index =
            static_cast<std::uint32_t>(weight.qs[4u * part + codeword]) |
            ((qhl & 0x07u) << 8u);
        const std::uint32_t grid = iq1s_grid_gpu[grid_index];
        const std::uint32_t half = codeword / 2u;
        const std::uint32_t activation_word = 2u * codeword;
        int sumy = 0;

        for (std::uint32_t i = 0; i < 4u; ++i) {
            const std::uint32_t packed_byte = (grid >> (8u * i)) & 0xffu;
            const int x0 = static_cast<int>(
                activation.qs[4u * activation_word + i]);
            const int x1 = static_cast<int>(
                activation.qs[4u * (activation_word + 1u) + i]);
            sumi[half] += static_cast<int>(packed_byte & 0x0fu) * x0;
            sumi[half] += static_cast<int>((packed_byte >> 4u) & 0x0fu) * x1;
            sumy += x0;
            sumy += x1;
        }

        const float delta = (qhl & 0x08u) != 0u ? -1.125f : -0.875f;
        sumf[half] = std::fma(delta, static_cast<float>(sumy), sumf[half]);
    }

    const std::uint32_t packed_scale = iq1_m_packed_scale(weight, part);
    const int scale0 =
        2 * static_cast<int>((packed_scale >> 0u) & 0x07u) + 1;
    const int scale1 =
        2 * static_cast<int>((packed_scale >> 3u) & 0x07u) + 1;
    const float x0 =
        (static_cast<float>(sumi[0]) + sumf[0]) * static_cast<float>(scale0);
    const float x1 =
        (static_cast<float>(sumi[1]) + sumf[1]) * static_cast<float>(scale1);
    return x0 + x1;
}

tf iq1_m_dot_tf(const block_iq1_m& weight, std::uint32_t part,
                const Q81& activation) {
    std::array<int, 2> sumi{};
    std::array<tf, 2> sumf{};

    for (std::uint32_t codeword = 0; codeword < 4u; ++codeword) {
        const std::uint32_t qh_byte = weight.qh[2u * part + codeword / 2u];
        const std::uint32_t qhl =
            (qh_byte >> (4u * (codeword % 2u))) & 0x0fu;
        const std::uint32_t grid_index =
            static_cast<std::uint32_t>(weight.qs[4u * part + codeword]) |
            ((qhl & 0x07u) << 8u);
        const std::uint32_t grid = iq1s_grid_gpu[grid_index];
        const std::uint32_t half = codeword / 2u;
        const std::uint32_t activation_word = 2u * codeword;
        int sumy = 0;

        for (std::uint32_t i = 0; i < 4u; ++i) {
            const std::uint32_t packed_byte = (grid >> (8u * i)) & 0xffu;
            const int x0 = static_cast<int>(
                activation.qs[4u * activation_word + i]);
            const int x1 = static_cast<int>(
                activation.qs[4u * (activation_word + 1u) + i]);
            sumi[half] += static_cast<int>(packed_byte & 0x0fu) * x0;
            sumi[half] += static_cast<int>((packed_byte >> 4u) & 0x0fu) * x1;
            sumy += x0;
            sumy += x1;
        }

        const float delta =
            (qhl & 0x08u) != 0u ? -1.125f : -0.875f;
        sumf[half] = fma(tf(delta), tf(static_cast<float>(sumy)), sumf[half]);
    }

    const std::uint32_t packed_scale = iq1_m_packed_scale(weight, part);
    const int scale0 =
        2 * static_cast<int>((packed_scale >> 0u) & 0x07u) + 1;
    const int scale1 =
        2 * static_cast<int>((packed_scale >> 3u) & 0x07u) + 1;
    const tf x0 = (tf(static_cast<float>(sumi[0])) + sumf[0]) *
                  tf(static_cast<float>(scale0));
    const tf x1 = (tf(static_cast<float>(sumi[1])) + sumf[1]) *
                  tf(static_cast<float>(scale1));
    return x0 + x1;
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

}  // namespace

void iq1_m_dequantize_reference(const block_iq1_m& weight,
                                std::span<float, 256u> output) {
    const float d = strata::kernels::f32_from_f16(
        iq1_m_half_scale_bits(weight));
    for (std::uint32_t part = 0u; part < 8u; ++part) {
        const std::uint32_t packed_scale = iq1_m_packed_scale(weight, part);
        const std::array<std::uint32_t, 2u> local_scales = {
            2u * (packed_scale & 0x07u) + 1u,
            2u * ((packed_scale >> 3u) & 0x07u) + 1u};
        for (std::uint32_t codeword = 0u; codeword < 4u; ++codeword) {
            const std::uint32_t qh =
                weight.qh[2u * part + codeword / 2u];
            const std::uint32_t selector =
                (qh >> (4u * (codeword & 1u))) & 0x0fu;
            const std::uint32_t grid_index =
                static_cast<std::uint32_t>(
                    weight.qs[4u * part + codeword]) |
                ((selector & 0x07u) << 8u);
            const std::uint32_t packed_grid = iq1s_grid_gpu[grid_index];
            const float delta =
                (selector & 0x08u) != 0u ? -0.125f : 0.125f;
            const std::uint32_t scale =
                local_scales[codeword / 2u];
            const float scaled_d = d * static_cast<float>(scale);
            for (std::uint32_t lane = 0u; lane < 8u; ++lane) {
                const std::uint32_t code =
                    (packed_grid >> (4u * lane)) & 0x0fu;
                output[32u * part + 8u * codeword + lane] =
                    scaled_d *
                    (static_cast<float>(code) - 1.0f + delta);
            }
        }
    }
}

bool iq1_m_matvec_reference(std::span<const block_iq1_m> weights,
                            std::span<const Q81> activations,
                            std::uint32_t n_in, std::uint32_t n_out,
                            std::uint32_t nc, std::uint32_t live_rows,
                            std::vector<float>& output, std::string& err) {
    if (n_in == 0u || n_in % 256u != 0u || n_out == 0u ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ1_M mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ1_M mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ1_M mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<float, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq1_m& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row +
                            kbx];
                const float d = strata::kernels::f32_from_f16(
                    iq1_m_half_scale_bits(weight));
                for (std::uint32_t column = 0; column < nc; ++column) {
                    const std::uint32_t input_column =
                        column < live_rows ? column : 0u;
                    const Q81& activation = activations[
                        static_cast<std::size_t>(input_column) * (n_in / 32u) +
                        kbx * 8u + part];
                    const float d8 = strata::kernels::f32_from_f16(
                        static_cast<std::uint16_t>(activation.ds & 0xffffu));
                    const float x = iq1_m_dot(weight, part, activation);
                    accumulators[column][lane] = std::fma(
                        d * d8, x, accumulators[column][lane]);
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

bool iq1_m_matvec_tf_reference(std::span<const block_iq1_m> weights,
                               std::span<const Q81> activations,
                               std::uint32_t n_in, std::uint32_t n_out,
                               std::uint32_t nc, std::uint32_t live_rows,
                               std::vector<tf>& output, std::string& err) {
    if (n_in == 0u || n_in % 256u != 0u || n_out == 0u ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ1_M mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ1_M mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ1_M mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, tf{});
    for (std::uint32_t row = 0; row < n_out; ++row) {
        std::array<std::array<tf, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t kbx = k / 8u;
                const std::uint32_t part = k % 8u;
                const block_iq1_m& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row +
                            kbx];
                const float d = strata::kernels::f32_from_f16(
                    iq1_m_half_scale_bits(weight));
                for (std::uint32_t column = 0; column < nc; ++column) {
                    const std::uint32_t input_column =
                        column < live_rows ? column : 0u;
                    const Q81& activation = activations[
                        static_cast<std::size_t>(input_column) * (n_in / 32u) +
                        kbx * 8u + part];
                    const std::uint32_t ds = activation.ds;
                    const float d8 = strata::kernels::f32_from_f16(
                        static_cast<std::uint16_t>(ds & 0xffffu));
                    const tf dot = iq1_m_dot_tf(weight, part, activation);
                    accumulators[column][lane] = fma(
                        tf(d) * tf(d8), dot, accumulators[column][lane]);
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
