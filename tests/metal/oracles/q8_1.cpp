#include "oracles/q8_1.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

namespace twin {
namespace {

std::int32_t round_ratio(std::uint32_t value_bits, std::uint32_t denominator_bits,
                         std::uint32_t multiplier) {
    const std::uint32_t magnitude_bits = value_bits & 0x7fffffffu;
    const std::uint32_t value_exponent = (magnitude_bits >> 23u) & 0xffu;
    const std::uint32_t denominator_exponent = (denominator_bits >> 23u) & 0xffu;
    const std::uint32_t value_significand = value_exponent == 0u
                                               ? magnitude_bits & 0x007fffffu
                                               : (magnitude_bits & 0x007fffffu) | 0x00800000u;
    const std::uint32_t denominator_significand = denominator_exponent == 0u
                                                      ? denominator_bits & 0x007fffffu
                                                      : (denominator_bits & 0x007fffffu) | 0x00800000u;
    const int value_power = value_exponent == 0u ? -149 : static_cast<int>(value_exponent) - 150;
    const int denominator_power = denominator_exponent == 0u
                                      ? -149
                                      : static_cast<int>(denominator_exponent) - 150;
    const int shift = denominator_power - value_power;
    if (denominator_significand == 0u) return 0;

    std::uint64_t numerator = static_cast<std::uint64_t>(value_significand) * multiplier;
    std::uint64_t denominator = denominator_significand;
    if (shift >= 0) {
        if (shift >= 32) return 0;
        denominator <<= shift;
    } else {
        const int left_shift = -shift;
        if (left_shift >= 32) {
            return (value_bits & 0x80000000u) != 0u ? -127 : 127;
        }
        numerator <<= left_shift;
    }
    const std::uint64_t quotient = numerator / denominator;
    const std::uint64_t remainder = numerator % denominator;
    const std::uint64_t rounded = std::min<std::uint64_t>(
        quotient + (remainder * 2u >= denominator ? 1u : 0u), 127u);
    const auto signed_value = static_cast<std::int32_t>(rounded);
    return (value_bits & 0x80000000u) != 0u ? -signed_value : signed_value;
}

}  // namespace

bool quantize_q8_1_reference(std::span<const float> input, std::uint32_t n,
                             std::uint32_t rows, std::vector<Q81>& output,
                             std::string& err) {
    if (n == 0 || n % 32 != 0 || rows == 0 ||
        static_cast<std::uint64_t>(n) * rows != input.size()) {
        err = "Q8_1 oracle input shape is invalid";
        return false;
    }
    for (const float value : input) {
        if (!std::isfinite(value)) {
            err = "Q8_1 oracle accepts finite activations only";
            return false;
        }
    }

    const std::size_t blocks_per_row = n / 32;
    if (blocks_per_row > std::numeric_limits<std::size_t>::max() / rows) {
        err = "Q8_1 oracle output size overflows size_t";
        return false;
    }
    output.assign(blocks_per_row * rows, Q81{});

    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const std::size_t base = static_cast<std::size_t>(row) * n + block_index * 32;
            std::array<float, 32> maxima{};
            std::array<float, 32> sums{};
            for (std::size_t lane = 0; lane < 32; ++lane) {
                const float x = input[base + lane];
                maxima[lane] = std::fabs(x);
                sums[lane] = x;
            }
            for (std::size_t offset = 16; offset != 0; offset >>= 1) {
                const auto old_maxima = maxima;
                const auto old_sums = sums;
                for (std::size_t lane = 0; lane < 32; ++lane) {
                    maxima[lane] = std::fmax(old_maxima[lane], old_maxima[lane ^ offset]);
                    sums[lane] = old_sums[lane] + old_sums[lane ^ offset];
                }
            }

            const float amax = maxima[0];
            const float scale = amax / 127.0f;
            const std::uint16_t scale_half_bits = strata::kernels::f16_from_f32(scale);
            const std::uint16_t sum_bits = strata::kernels::f16_from_f32(sums[0]);
            Q81& result = output[static_cast<std::size_t>(row) * blocks_per_row + block_index];
            result.ds = static_cast<std::uint32_t>(scale_half_bits) |
                        (static_cast<std::uint32_t>(sum_bits) << 16);
            const std::uint32_t scale_f32_bits = std::bit_cast<std::uint32_t>(scale);
            const std::uint32_t scale_exponent = scale_f32_bits & 0x7f800000u;
            for (std::size_t lane = 0; lane < 32; ++lane) {
                const float x = input[base + lane];
                const std::uint32_t x_bits = std::bit_cast<std::uint32_t>(x);
                std::int32_t quantized = 0;
                if (amax != 0.0f) {
                    if (scale_f32_bits == 0u) {
                        quantized = round_ratio(x_bits, std::bit_cast<std::uint32_t>(amax), 127u);
                    } else {
                        quantized = static_cast<std::int32_t>(std::round(x / scale));
                        if (scale_exponent == 0u) {
                            quantized = std::clamp(quantized, -127, 127);
                        }
                    }
                }
                result.qs[lane] = static_cast<std::int8_t>(quantized);
            }
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
