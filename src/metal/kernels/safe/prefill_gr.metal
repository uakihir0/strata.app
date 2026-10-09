// Prefill hyper-connection read kernels (docs/mac/04 §6.3).
#include "../common/math.h"
#include "../common/reduce.h"
#include "../common/params.h"

using namespace metal;

static inline float prefill_block_sum(float value,
                                     threadgroup float* partials,
                                     uint lane, uint simdgroup) {
    value = bfly_sum(value);
    if (lane == 0u) partials[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = lane < 8u ? partials[lane] : 0.0f;
    return bfly_sum(value);
}

static inline void prefill_gr_norm(device const float* residual,
                                  device const float* norm,
                                  device float* rs_out,
                                  device ushort* xn16,
                                  device ushort* xn16_lo,
                                  threadgroup float* partials,
                                  bool write_low_remainder,
                                  uint tid, uint lane, uint simdgroup,
                                  uint row) {
    constexpr uint kWidth = N_EMBD;
    const ulong row_offset = ulong(row) * kWidth;
    const uint stream = row % HC;

    float sum = 0.0f;
    for (uint column = tid; column < kWidth; column += 256u) {
        const float value = residual[row_offset + column];
        sum = fma(value, value, sum);
    }
    sum = prefill_block_sum(sum, partials, lane, simdgroup);
    const float scale = rsqrt(sum / float(kWidth) + 1.0e-6f);
    if (tid == 0u) rs_out[row] = scale;

    const ulong gamma_offset = ulong(stream) * kWidth;
    for (uint column = tid; column < kWidth; column += 256u) {
        const float value =
            (residual[row_offset + column] * scale) *
            norm[gamma_offset + column];
        const ushort high = f32_to_bf16_rne(value);
        xn16[row_offset + column] = high;
        if (write_low_remainder) {
            xn16_lo[row_offset + column] =
                f32_to_bf16_rne(value - bf16_to_f32(high));
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_gr_norm(device const float* residual [[buffer(0)]],
                              device const float* norm [[buffer(1)]],
                              device float* rs_out [[buffer(2)]],
                              device ushort* xn16 [[buffer(3)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simdgroup [[simdgroup_index_in_threadgroup]],
                              uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partials[8];
    prefill_gr_norm(residual, norm, rs_out, xn16, nullptr, partials, false,
                    tid, lane, simdgroup, row);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_gr_norm_lo(device const float* residual [[buffer(0)]],
                                 device const float* norm [[buffer(1)]],
                                 device float* rs_out [[buffer(2)]],
                                 device ushort* xn16 [[buffer(3)]],
                                 device ushort* xn16_lo [[buffer(4)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simdgroup [[simdgroup_index_in_threadgroup]],
                                 uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partials[8];
    prefill_gr_norm(residual, norm, rs_out, xn16, xn16_lo, partials, true,
                    tid, lane, simdgroup, row);
}

static inline void prefill_gr_silu(device const float* low_rank,
                                   device ushort* high_bf16,
                                   device ushort* low_bf16,
                                   uint rows, uint index,
                                   bool write_low_remainder) {
    const uint count = rows * 320u;
    if (index >= count) return;
    const float x = low_rank[index] / 4.0f;
    const float value = silu_k(x);
    const ushort high = f32_to_bf16_rne(value);
    high_bf16[index] = high;
    if (write_low_remainder) {
        low_bf16[index] = f32_to_bf16_rne(value - bf16_to_f32(high));
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_gr_silu(device const float* low_rank [[buffer(0)]],
                              device ushort* high_bf16 [[buffer(1)]],
                              constant uint& rows [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    prefill_gr_silu(low_rank, high_bf16, nullptr, rows, index, false);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_gr_silu_lo(device const float* low_rank [[buffer(0)]],
                                 device ushort* high_bf16 [[buffer(1)]],
                                 device ushort* low_bf16 [[buffer(2)]],
                                 constant uint& rows [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    prefill_gr_silu(low_rank, high_bf16, low_bf16, rows, index, true);
}
