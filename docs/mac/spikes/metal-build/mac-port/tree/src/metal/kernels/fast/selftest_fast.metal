// src/metal/kernels/fast/selftest_fast.metal - skeleton kernel compiled with fast math.
#include "strata_common.h"
using namespace metal;

kernel void strata_selftest_exp(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] = exp(x[i]);
}
