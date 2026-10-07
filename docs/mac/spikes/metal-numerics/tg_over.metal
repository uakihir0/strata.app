// 32 KiB + 16 B of static threadgroup memory: does the compiler, the library or the pipeline refuse it?
#include <metal_stdlib>
using namespace metal;

kernel void k_tg_over(device float *o [[buffer(0)]], uint i [[thread_position_in_threadgroup]]) {
    threadgroup float t[8196];
    for (uint j = i; j < 8196; j += 1024) t[j] = float(j);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = t[8195 - i];
}
