// src/metal/kernels/safe/selftest.metal - skeleton kernels for the build and the context test.
#include "strata_common.h"
using namespace metal;

kernel void strata_selftest_add(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] = strata_twice_plus_one(x[i]);
}
