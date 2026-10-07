#include <metal_stdlib>
#include "h.h"
using namespace metal;
kernel void e(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = twice(x[i]); }
