// src/metal/kernels/safe/window_warm.metal - force one 16-byte read from each file-backed VM page.
#include <metal_stdlib>
using namespace metal;

kernel void strata_window_warm(device const uchar* bytes [[buffer(0)]],
                               device atomic_uint* checksum [[buffer(1)]],
                               constant uint& page_count [[buffer(2)]],
                               uint page [[thread_position_in_grid]]) {
    if (page >= page_count) return;
    constexpr uint words_per_page = 16 * 1024 / sizeof(uint4);
    const device uint4* word =
        reinterpret_cast<device const uint4*>(bytes + ulong(page) * words_per_page * sizeof(uint4));
    const uint4 value = *word;
    const uint folded = value.x ^ value.y ^ value.z ^ value.w;
    atomic_fetch_xor_explicit(checksum, folded, memory_order_relaxed);
}
