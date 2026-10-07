// Numerics shared by the safe Metal kernels. See docs/mac/04-metal-kernels.md §3.
#pragma once
#include <metal_stdlib>
using namespace metal;

static inline float sigmoid_k(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float silu_k(float x) {
    return x / (1.0f + exp(-x));
}

static inline float log1p_k(float x) {
    const float u = 1.0f + x;
    return u == 1.0f ? x : log(u) * (x / (u - 1.0f));
}

static inline float softplus_k(float x) {
    return x > 20.0f ? x : log1p_k(exp(x));
}

static inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort f32_to_bf16_rne(float x) {
    const uint bits = as_type<uint>(x);
    return isnan(x) ? ushort((bits >> 16) | 64u)
                    : ushort((bits + ((bits >> 16) & 1u) + 0x7fffu) >> 16);
}

static inline half half_sat(float x) {
    const uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u) {
        return as_type<half>(ushort(0u));
    }
    if (isnan(x)) return as_type<half>(ushort(0x7e00u));
    return half(clamp(x, -65504.0f, 65504.0f));
}

static inline int round_half_away(float x) {
    return int(round(x));
}
