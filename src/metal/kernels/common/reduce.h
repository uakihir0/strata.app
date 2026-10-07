// Explicit reduction orders shared by the safe Metal kernels.
#pragma once
#include <metal_stdlib>
using namespace metal;

static inline float bfly_sum(float value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        value += simd_shuffle_xor(value, offset);
    }
    return value;
}

static inline float bfly_max(float value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        value = fmax(value, simd_shuffle_xor(value, offset));
    }
    return value;
}

static inline int bfly_max(int value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        value = max(value, simd_shuffle_xor(value, offset));
    }
    return value;
}

static inline float sum_partials_in_order(threadgroup const float* partials, uint count) {
    if (count == 0) return 0.0f;
    float value = partials[0];
    for (uint i = 1; i < count; ++i) value += partials[i];
    return value;
}
