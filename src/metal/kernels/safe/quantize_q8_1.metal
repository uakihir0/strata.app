// The precise Q8_1 activation quantizer (docs/mac/04-metal-kernels.md §5.8).
#include "types.h"
#include "math.h"
#include "reduce.h"

using namespace metal;

struct Q81Params {
    uint n;
    uint rows;
};

static inline uint q8_1_bfly_max_abs_bits(uint value_bits) {
    uint magnitude = value_bits & 0x7fffffffu;
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        magnitude = max(magnitude, simd_shuffle_xor(magnitude, offset));
    }
    return magnitude;
}

static inline uint q8_1_subnormal_scale_bits(uint amax_bits) {
    const uint exponent = (amax_bits >> 23u) & 0xffu;
    const uint significand = exponent == 0u
                                 ? amax_bits & 0x007fffffu
                                 : (amax_bits & 0x007fffffu) | 0x00800000u;
    const uint shift = exponent == 0u ? 0u : exponent - 1u;
    const uint units = significand << shift;
    uint rounded = units / 127u;
    const uint remainder = units % 127u;
    const uint twice_remainder = remainder * 2u;
    if (twice_remainder > 127u ||
        (twice_remainder == 127u && (rounded & 1u) != 0u)) {
        ++rounded;
    }
    return rounded;
}

static inline int q8_1_round_ratio(uint value_bits, uint denominator_bits, uint multiplier) {
    const uint magnitude_bits = value_bits & 0x7fffffffu;
    const uint value_exponent = (magnitude_bits >> 23u) & 0xffu;
    const uint denominator_exponent = (denominator_bits >> 23u) & 0xffu;
    const uint value_significand = value_exponent == 0u
                                       ? magnitude_bits & 0x007fffffu
                                       : (magnitude_bits & 0x007fffffu) | 0x00800000u;
    const uint denominator_significand = denominator_exponent == 0u
                                             ? denominator_bits & 0x007fffffu
                                             : (denominator_bits & 0x007fffffu) | 0x00800000u;
    const int value_power = value_exponent == 0u ? -149 : int(value_exponent) - 150;
    const int denominator_power =
        denominator_exponent == 0u ? -149 : int(denominator_exponent) - 150;
    const int shift = denominator_power - value_power;
    if (denominator_significand == 0u) return 0;

    ulong numerator = ulong(value_significand) * ulong(multiplier);
    ulong denominator = ulong(denominator_significand);
    if (shift >= 0) {
        if (shift >= 32) return 0;
        denominator <<= uint(shift);
    } else {
        const int left_shift = -shift;
        if (left_shift >= 32) {
            const int sign = (value_bits & 0x80000000u) != 0u ? -1 : 1;
            return 127 * sign;
        }
        numerator <<= uint(left_shift);
    }
    const ulong quotient = numerator / denominator;
    const ulong remainder = numerator % denominator;
    ulong rounded = quotient + (remainder * 2ul >= denominator ? 1ul : 0ul);
    rounded = min(rounded, 127ul);
    const int signed_value = int(rounded);
    return (value_bits & 0x80000000u) != 0u ? -signed_value : signed_value;
}

static inline int q8_1_round_subnormal_division(uint value_bits, uint d_bits) {
    const uint magnitude_bits = value_bits & 0x7fffffffu;
    const uint value_exponent = (magnitude_bits >> 23u) & 0xffu;
    const uint value_significand = value_exponent == 0u
                                       ? magnitude_bits & 0x007fffffu
                                       : (magnitude_bits & 0x007fffffu) | 0x00800000u;
    const uint d_significand = d_bits & 0x007fffffu;
    if (d_significand == 0u) return 0;

    const uint value_shift = value_exponent == 0u ? 0u : value_exponent - 1u;
    const float scaled_value = float(value_significand) * float(1u << value_shift);
    const float scaled_d = float(d_significand);
    const int magnitude = min(round_half_away(scaled_value / scaled_d), 127);
    return (value_bits & 0x80000000u) != 0u ? -magnitude : magnitude;
}

static inline int q8_1_round_subnormal_input(uint value_bits, uint d_bits) {
    const uint magnitude_bits = value_bits & 0x7fffffffu;
    const uint d_exponent = (d_bits >> 23u) & 0xffu;
    if (magnitude_bits == 0u || d_exponent > 1u) return 0;

    const uint value_significand = magnitude_bits & 0x007fffffu;
    const uint d_significand = (d_bits & 0x007fffffu) | 0x00800000u;
    const int magnitude =
        round_half_away(float(value_significand) / float(d_significand));
    return (value_bits & 0x80000000u) != 0u ? -magnitude : magnitude;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_q_q8_1(device const uint* x [[buffer(0)]],
                          device Q81* y [[buffer(1)]],
                          constant Q81Params& params [[buffer(2)]],
                          uint3 tid [[thread_position_in_threadgroup]],
                          uint3 group [[threadgroup_position_in_grid]]) {
    const uint i = group.x * 256u + tid.x;
    const uint row = group.y;
    if (row >= params.rows || i >= params.n) return;

    const uint value_bits = x[row * params.n + i];
    const float xi = as_type<float>(value_bits);
    const uint amax_bits = q8_1_bfly_max_abs_bits(value_bits);
    const float amax = as_type<float>(amax_bits);
    const float sum = bfly_sum(xi);
    const uint amax_exponent = (amax_bits >> 23u) & 0xffu;
    const uint d_bits = amax_exponent <= 7u
                            ? q8_1_subnormal_scale_bits(amax_bits)
                            : as_type<uint>(amax / 127.0f);
    const float d = as_type<float>(d_bits);
    const uint d_exponent = (d_bits >> 23u) & 0xffu;
    const uint value_exponent = (value_bits >> 23u) & 0xffu;
    const int q = amax_bits == 0u
                      ? 0
                      : (d_bits == 0u
                             ? q8_1_round_ratio(value_bits, amax_bits, 127u)
                             : (d_exponent == 0u
                                    ? q8_1_round_subnormal_division(value_bits, d_bits)
                                    : (value_exponent == 0u
                                           ? q8_1_round_subnormal_input(value_bits, d_bits)
                                           : round_half_away(xi / d))));

    device Q81* block = y + row * (params.n / 32u) + i / 32u;
    block->qs[i & 31u] = char(q);
    if ((i & 31u) == 0u) block->ds = half2(half(d), half(sum));
}
