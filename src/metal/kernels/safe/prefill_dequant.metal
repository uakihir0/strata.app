// Native-weight dequantization for the Metal Prefill path (docs/mac/04 §6.2).
#include "types.h"
#include "codebooks.h"

using namespace metal;

struct PrefillDequantParams {
    uint n_out;
    uint k;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q2_0(
    device const block_q2_0* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 64u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q2_0* row =
        weights + ulong(column) * blocks_per_row + group.y;
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint packed = uint(row->qs[value >> 2u]);
        const uint shift = (value & 3u) * 2u;
        const int code = int((packed >> shift) & 0x03u) - 1;
        const float expanded = float(code) * float(row->d);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_iq4_nl(
    device const block_iq4_nl* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 32u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_iq4_nl* row =
        weights + ulong(column) * blocks_per_row + group.y;
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint packed = uint(row->qs[value & 15u]);
        const uint shift = (value >> 4u) * 4u;
        const uint code = (packed >> shift) & 0x0fu;
        const float expanded =
            float(row->d) * float(kvalues_iq4nl[code]);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q8_0(
    device const block_q8_0* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 32u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q8_0* row =
        weights + ulong(column) * blocks_per_row + group.y;
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const float expanded =
            float(row->d) * float(static_cast<int>(row->qs[value]));
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q4_0(
    device const block_q4_0* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 32u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q4_0* row =
        weights + ulong(column) * blocks_per_row + group.y;
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint packed = uint(row->qs[value & 15u]);
        const uint shift = (value >> 4u) * 4u;
        const int code = int((packed >> shift) & 0x0fu) - 8;
        const float expanded = float(code) * float(row->d);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q5_0(
    device const block_q5_0* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 32u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q5_0* row =
        weights + ulong(column) * blocks_per_row + group.y;
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint packed = uint(row->qs[value & 15u]);
        const uint shift = (value >> 4u) * 4u;
        const uint low_bits = (packed >> shift) & 0x0fu;
        const uint high_bit =
            (uint(row->qh[value >> 3u]) >> (value & 7u)) & 1u;
        const int code = int(low_bits | (high_bit << 4u)) - 16;
        const float expanded = float(code) * float(row->d);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}
