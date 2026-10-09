// Native-weight dequantization for the Metal Prefill path (docs/mac/04 §6.2).
#include "types.h"
#include "codebooks.h"

using namespace metal;

struct PrefillDequantParams {
    uint n_out;
    uint k;
};

static inline uint2 pf_q4_k_scale_min(device const block_q4_K* weight,
                                     uint group) {
    if (group < 4u) {
        return uint2(uint(weight->scales[group] & 0x3fu),
                     uint(weight->scales[group + 4u] & 0x3fu));
    }
    return uint2(
        uint(weight->scales[group + 4u] & 0x0fu) |
            (uint(weight->scales[group - 4u] >> 6u) << 4u),
        uint(weight->scales[group + 4u] >> 4u) |
            (uint(weight->scales[group] >> 6u) << 4u));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q4_k(
    device const block_q4_K* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 256u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q4_K* row =
        weights + ulong(column) * blocks_per_row + group.y;
    const half2 dm = as_type<half2>(row->dm);
    const float d = float(dm.x);
    const float dmin = float(dm.y);
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint scale_group = value / 32u;
        const uint2 scale_min = pf_q4_k_scale_min(row, scale_group);
        const uint within_group = value % 32u;
        const uint qbyte = 32u * (scale_group / 2u) + within_group;
        const uint nibble_shift = 4u * (scale_group % 2u);
        const uint code =
            (uint(row->qs[qbyte]) >> nibble_shift) & 0x0fu;
        const float d1 = d * float(scale_min.x);
        const float m1 = dmin * float(scale_min.y);
        const float expanded = fma(d1, float(code), -m1);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

static inline uint2 pf_q5_k_scale_min(device const block_q5_K* weight,
                                     uint group) {
    if (group < 4u) {
        return uint2(uint(weight->scales[group] & 0x3fu),
                     uint(weight->scales[group + 4u] & 0x3fu));
    }
    return uint2(
        uint(weight->scales[group + 4u] & 0x0fu) |
            (uint(weight->scales[group - 4u] >> 6u) << 4u),
        uint(weight->scales[group + 4u] >> 4u) |
            (uint(weight->scales[group] >> 6u) << 4u));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q5_k(
    device const block_q5_K* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 256u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q5_K* row =
        weights + ulong(column) * blocks_per_row + group.y;
    const half2 dm = as_type<half2>(row->dm);
    const float d = float(dm.x);
    const float dmin = float(dm.y);
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint scale_group = value / 32u;
        const uint2 scale_min = pf_q5_k_scale_min(row, scale_group);
        const uint within_group = value % 32u;
        const uint qbyte = 32u * (scale_group / 2u) + within_group;
        const uint nibble_shift = 4u * (scale_group % 2u);
        const uint low = (uint(row->qs[qbyte]) >> nibble_shift) & 0x0fu;
        const uint high =
            (uint(row->qh[within_group]) >> scale_group) & 1u;
        const uint code = low | (high << 4u);
        const float d1 = d * float(scale_min.x);
        const float m1 = dmin * float(scale_min.y);
        const float expanded = fma(d1, float(code), -m1);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

static_assert(sizeof(block_q6_K) == 210,
              "Q6_K Prefill dequantization requires 210-byte blocks");

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q6_k(
    device const block_q6_K* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 256u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q6_K* row =
        weights + ulong(column) * blocks_per_row + group.y;
    const float d = float(row->d);
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint group32 = value / 32u;
        const uint within_group = value % 32u;
        const uint ql_half = group32 % 4u;
        const uint ql_chunk = group32 / 4u;
        const uint ql_byte = 64u * ql_chunk + 32u * (ql_half % 2u) +
                             within_group;
        const uint ql_shift = ql_half < 2u ? 0u : 4u;
        const uint qh_byte = 32u * ql_chunk + within_group;
        const uint qh_shift = 2u * ql_half;
        const uint low =
            (uint(row->ql[ql_byte]) >> ql_shift) & 0x0fu;
        const uint high =
            (uint(row->qh[qh_byte]) >> qh_shift) & 0x03u;
        const int code = int(low | (high << 4u)) - 32;
        const uint scale_index = 8u * ql_chunk + 2u * ql_half +
                                 within_group / 16u;
        const float scaled_d =
            d * float(int(row->scales[scale_index]));
        const float expanded = scaled_d * float(code);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

static_assert(sizeof(block_q3_K) == 110,
              "Q3_K Prefill dequantization requires 110-byte blocks");

static inline uint pf_q3_k_scale(device const block_q3_K* weight,
                                 uint scale_index) {
    if (scale_index < 4u) {
        return (uint(weight->scales[scale_index]) & 0x0fu) |
               ((uint(weight->scales[scale_index + 8u]) & 0x03u) << 4u);
    }
    if (scale_index < 8u) {
        return (uint(weight->scales[scale_index]) & 0x0fu) |
               (((uint(weight->scales[scale_index + 4u]) >> 2u) & 0x03u)
                << 4u);
    }
    if (scale_index < 12u) {
        return (uint(weight->scales[scale_index - 8u]) >> 4u) |
               (((uint(weight->scales[scale_index]) >> 4u) & 0x03u) << 4u);
    }
    return (uint(weight->scales[scale_index - 8u]) >> 4u) |
           (((uint(weight->scales[scale_index - 4u]) >> 6u) & 0x03u)
            << 4u);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_q3_k(
    device const block_q3_K* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 256u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_q3_K* row =
        weights + ulong(column) * blocks_per_row + group.y;
    const float d = float(row->d);
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint group32 = value / 32u;
        const uint half_index = group32 / 4u;
        const uint quarter = group32 % 4u;
        const uint within_group = value % 32u;
        const uint scale_index =
            8u * half_index + 2u * quarter + within_group / 16u;
        const uint shift = 2u * quarter;
        const uint mask = 1u << (4u * half_index + quarter);
        const uint q2 =
            (uint(row->qs[32u * half_index + within_group]) >> shift) & 0x03u;
        const int code =
            int(q2) -
            ((uint(row->hmask[within_group]) & mask) != 0u ? 0 : 4);
        const int scale = int(pf_q3_k_scale(row, scale_index)) - 32;
        const float scaled_d = d * float(scale);
        const float expanded = scaled_d * float(code);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

static_assert(sizeof(block_iq4_xs) == 136,
              "IQ4_XS Prefill dequantization requires 136-byte blocks");

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_iq4_xs(
    device const block_iq4_xs* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 256u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_iq4_xs* row =
        weights + ulong(column) * blocks_per_row + group.y;
    const float d = float(row->d);
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint scale_group = value / 32u;
        const uint within_group = value % 32u;
        const uint qs_index =
            16u * scale_group + (within_group < 16u ? within_group
                                                     : within_group - 16u);
        const uint packed = uint(row->qs[qs_index]);
        const uint code =
            within_group < 16u ? packed & 0x0fu : packed >> 4u;
        const uint scale_low =
            (uint(row->scales_l[scale_group / 2u]) >>
             (4u * (scale_group % 2u))) &
            0x0fu;
        const uint scale_high =
            (uint(row->scales_h) >> (2u * scale_group)) & 0x03u;
        const int scale = int(scale_low | (scale_high << 4u)) - 32;
        const float scaled_d = d * float(scale);
        const float expanded = scaled_d * float(kvalues_iq4nl[code]);
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

static_assert(sizeof(block_iq3_s) == 110,
              "IQ3_S Prefill dequantization requires 110-byte blocks");

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_dequant_iq3_s(
    device const block_iq3_s* weights [[buffer(0)]],
    device half* k_major_f16 [[buffer(1)]],
    constant PrefillDequantParams& params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint column = 256u * group.x + tid;
    if (column >= params.n_out) return;

    constexpr uint values_per_block = 256u;
    const uint blocks_per_row = params.k / values_per_block;
    const device block_iq3_s* row =
        weights + ulong(column) * blocks_per_row + group.y;
    const float d = float(row->d);
    for (uint value = 0u; value < values_per_block; ++value) {
        const uint k = values_per_block * group.y + value;
        const uint ib = value / 32u;
        const uint il = (value % 32u) / 8u;
        const uint pos = value % 8u;
        const uint grid_group = pos / 4u;
        const uint lane = pos % 4u;
        const uint qh_shift = 8u - 2u * il - grid_group;
        const uint grid_index =
            uint(row->qs[8u * ib + 2u * il + grid_group]) |
            ((uint(row->qh[ib]) << qh_shift) & 0x100u);
        const uint grid_word = iq3s_grid[grid_index];
        const uint magnitude = (grid_word >> (8u * lane)) & 0xffu;
        const uint scale_nibble =
            (uint(row->scales[ib / 2u]) >> (4u * (ib % 2u))) & 0x0fu;
        const float scale = d * float(1u + 2u * scale_nibble);
        const float scaled = scale * float(magnitude);
        const bool negative =
            (uint(row->signs[4u * ib + il]) & (1u << pos)) != 0u;
        const float expanded = negative ? -scaled : scaled;
        k_major_f16[ulong(k) * params.n_out + column] = half(expanded);
    }
}

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
