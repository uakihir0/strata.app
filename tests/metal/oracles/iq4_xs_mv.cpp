#include "oracles/iq4_xs_mv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <limits>

namespace twin {
namespace {

constexpr std::array<std::int8_t, 16> k_iq4_xs_values = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};

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

std::uint32_t scale_code(const block_iq4_xs& weight, std::uint32_t subblock) {
    const std::uint32_t low =
        (static_cast<std::uint32_t>(weight.scales_l[subblock / 2u]) >>
         (4u * (subblock % 2u))) &
        0x0fu;
    const std::uint32_t high =
        (static_cast<std::uint32_t>(weight.scales_h) >> (2u * subblock)) & 0x03u;
    return low | (high << 4u);
}

int integer_dot(const block_iq4_xs& weight, const Q81& activation,
                std::uint32_t subblock) {
    int sum = 0;
    const std::uint32_t byte_base = 16u * subblock;
    for (std::uint32_t word = 0; word < 4; ++word) {
        const std::uint32_t offset = byte_base + 4u * word;
        for (std::uint32_t lane = 0; lane < 4; ++lane) {
            const std::uint32_t packed = weight.qs[offset + lane];
            const int low = k_iq4_xs_values[packed & 0x0fu];
            const int high = k_iq4_xs_values[packed >> 4u];
            sum += low * static_cast<int>(activation.qs[4u * word + lane]);
            sum += high * static_cast<int>(activation.qs[16u + 4u * word + lane]);
        }
    }
    return sum;
}

}  // namespace

bool iq4_xs_matvec_reference(std::span<const block_iq4_xs> weights,
                             std::span<const Q81> activations,
                             std::uint32_t n_in, std::uint32_t n_out,
                             std::uint32_t nc, std::uint32_t live_rows,
                             std::vector<float>& output, std::string& err) {
    if (n_in == 0 || n_in % 256u != 0u || n_out == 0 ||
        (nc != 1 && nc != 2 && nc != 4 && nc != 8) ||
        live_rows == 0 || live_rows > nc) {
        err = "IQ4_XS mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(blocks_per_row) * 8u * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ4_XS mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ4_XS mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, 0.0f);
    const bool small_k = blocks_per_row < 16u;
    const std::uint32_t rows_per_group = small_k ? 4u : 1u;
    const std::uint32_t groups = (n_out - 1u) / rows_per_group + 1u;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t row_base = group * rows_per_group;
        std::array<std::array<std::array<float, 4>, 8>, 128> accumulators{};
        for (std::uint32_t tid = 0; tid < 128u; ++tid) {
            const std::uint32_t subblock = tid % 8u;
            for (std::uint32_t kbx = tid / 8u; kbx < blocks_per_row; kbx += 16u) {
                for (std::uint32_t i = 0; i < rows_per_group; ++i) {
                    const std::uint32_t row = row_base + i;
                    if (row >= n_out) continue;
                    const block_iq4_xs& weight =
                        weights[static_cast<std::size_t>(row) * blocks_per_row + kbx];
                    const std::uint32_t scale =
                        scale_code(weight, subblock);
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const std::uint32_t input_column = column < live_rows ? column : 0u;
                        const Q81& activation = activations[
                            static_cast<std::size_t>(input_column) *
                                (8u * blocks_per_row) +
                            8u * kbx + subblock];
                        const int dot = integer_dot(weight, activation, subblock);
                        const int scaled_dot = dot * (static_cast<int>(scale) - 32);
                        const std::uint32_t ds = static_cast<std::uint32_t>(activation.ds);
                        const float activation_scale = strata::kernels::f32_from_f16(
                            static_cast<std::uint16_t>(ds & 0xffffu));
                        const float scale_product =
                            strata::kernels::f32_from_f16(weight.d) * activation_scale;
                        accumulators[tid][column][i] =
                            std::fma(scale_product, static_cast<float>(scaled_dot),
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

bool iq4_xs_matvec_tf_reference(
    std::span<const block_iq4_xs> weights,
    std::span<const Q81> activations, std::uint32_t n_in,
    std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
    std::vector<tf>& output, std::string& err) {
    if (n_in == 0u || n_in % 256u != 0u || n_out == 0u ||
        (nc != 1u && nc != 2u && nc != 4u && nc != 8u) ||
        live_rows == 0u || live_rows > nc) {
        err = "IQ4_XS mat-vec oracle shape is invalid";
        return false;
    }
    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::uint64_t expected_weights =
        static_cast<std::uint64_t>(blocks_per_row) * n_out;
    const std::uint64_t expected_activations =
        static_cast<std::uint64_t>(n_in / 32u) * nc;
    if (expected_weights != weights.size() ||
        expected_activations != activations.size()) {
        err = "IQ4_XS mat-vec oracle buffer size is invalid";
        return false;
    }
    if (n_out > std::numeric_limits<std::size_t>::max() / nc) {
        err = "IQ4_XS mat-vec output size overflows size_t";
        return false;
    }

    output.assign(static_cast<std::size_t>(n_out) * nc, tf{});
    for (std::uint32_t row = 0u; row < n_out; ++row) {
        std::array<std::array<tf, 32>, 8> accumulators{};
        for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
            for (std::uint32_t k = lane; k < blocks_per_row * 8u; k += 32u) {
                const std::uint32_t block = k / 8u;
                const std::uint32_t subblock = k % 8u;
                const block_iq4_xs& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row +
                            block];
                const float weight_scale =
                    strata::kernels::f32_from_f16(weight.d);
                const int scale =
                    static_cast<int>(scale_code(weight, subblock)) - 32;
                for (std::uint32_t column = 0u; column < nc; ++column) {
                    const std::uint32_t input_column =
                        column < live_rows ? column : 0u;
                    const Q81& activation = activations[
                        static_cast<std::size_t>(input_column) * (n_in / 32u) +
                        block * 8u + subblock];
                    const std::uint32_t ds =
                        static_cast<std::uint32_t>(activation.ds);
                    const float activation_scale =
                        strata::kernels::f32_from_f16(
                            static_cast<std::uint16_t>(ds & 0xffffu));
                    const int scaled_dot =
                        integer_dot(weight, activation, subblock) * scale;
                    accumulators[column][lane] = fma(
                        tf(weight_scale) * tf(activation_scale),
                        tf(static_cast<float>(scaled_dot)),
                        accumulators[column][lane]);
                }
            }
        }
        for (std::uint32_t column = 0u; column < nc; ++column) {
            output[static_cast<std::size_t>(column) * n_out + row] =
                butterfly_sum_tf(accumulators[column]);
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
