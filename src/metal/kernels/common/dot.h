// Exact integer dot helpers shared by quantized kernels.
#pragma once
#include <metal_stdlib>
using namespace metal;

static inline int dot4_i8(int a, int b) {
    const char4 x = as_type<char4>(a);
    const char4 y = as_type<char4>(b);
    return int(x.x) * int(y.x) + int(x.y) * int(y.y) +
           int(x.z) * int(y.z) + int(x.w) * int(y.w);
}
