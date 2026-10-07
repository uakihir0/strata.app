// src/metal/kernels/safe/selftest.metal - skeleton kernels for the build and the context test.
#include "strata_common.h"
#include "types.h"
#include "params.h"
#include "math.h"
#include "reduce.h"
#include "dot.h"
#include "codebooks.h"
using namespace metal;

struct SelftestMathParams {
    uint count;
    uint operation;
    float lower;
    float upper;
};

kernel void strata_selftest_add(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] = strata_twice_plus_one(x[i]);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_copy_u32(device const uint* input [[buffer(0)]],
                                     device uint* output [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
    output[i] = input[i];
}

[[max_total_threads_per_threadgroup(1)]]
kernel void strata_selftest_layout(device uint* out [[buffer(0)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    out[0] = sizeof(StepRow);
    out[1] = sizeof(SampRow);
    out[2] = sizeof(WindowParams);
    out[3] = sizeof(PrefillParams);
    out[4] = sizeof(MtpParams);
    out[5] = __builtin_offsetof(StepRow, pos);
    out[6] = __builtin_offsetof(StepRow, n_kv);
    out[7] = __builtin_offsetof(StepRow, n_bid);
    out[8] = __builtin_offsetof(StepRow, width);
    out[9] = __builtin_offsetof(SampRow, temperature);
    out[10] = __builtin_offsetof(SampRow, top_p);
    out[11] = __builtin_offsetof(SampRow, min_p);
    out[12] = __builtin_offsetof(SampRow, penalty_repeat);
    out[13] = __builtin_offsetof(SampRow, penalty_freq);
    out[14] = __builtin_offsetof(SampRow, penalty_present);
    out[15] = __builtin_offsetof(SampRow, top_k);
    out[16] = __builtin_offsetof(SampRow, min_keep);
    out[17] = __builtin_offsetof(SampRow, greedy);
    out[18] = __builtin_offsetof(SampRow, counter);
    out[19] = __builtin_offsetof(SampRow, hist_off);
    out[20] = __builtin_offsetof(SampRow, hist_len);
    out[21] = __builtin_offsetof(WindowParams, T);
    out[22] = __builtin_offsetof(WindowParams, nc);
    out[23] = __builtin_offsetof(WindowParams, pos0);
    out[24] = __builtin_offsetof(WindowParams, n_keep);
    out[25] = __builtin_offsetof(WindowParams, seed_lo);
    out[26] = __builtin_offsetof(WindowParams, seed_hi);
    out[27] = __builtin_offsetof(WindowParams, tokens);
    out[28] = __builtin_offsetof(WindowParams, step);
    out[29] = __builtin_offsetof(WindowParams, samp);
    out[30] = __builtin_offsetof(PrefillParams, T);
    out[31] = __builtin_offsetof(PrefillParams, p0);
    out[32] = __builtin_offsetof(PrefillParams, idx_f);
    out[33] = __builtin_offsetof(PrefillParams, idx_e);
    out[34] = __builtin_offsetof(MtpParams, T);
    out[35] = __builtin_offsetof(MtpParams, hid);
    out[36] = __builtin_offsetof(MtpParams, a);
    out[37] = __builtin_offsetof(MtpParams, j);
    out[38] = __builtin_offsetof(MtpParams, tokens);
    out[39] = __builtin_offsetof(MtpParams, step);
    out[40] = __builtin_offsetof(MtpParams, samp);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_math(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant SelftestMathParams& params [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= params.count) return;
    const float x = input[i];
    switch (params.operation) {
        case 0: output[i] = exp(x); break;
        case 1: output[i] = sigmoid_k(x); break;
        case 2: output[i] = silu_k(x); break;
        case 3: output[i] = softplus_k(x); break;
        case 4: output[i] = log1p_k(x); break;
        default: output[i] = as_type<float>(0x7fc00000u); break;
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_bf16_to_f32(device const ushort* input [[buffer(0)]],
                                        device uint* output [[buffer(1)]],
                                        constant SelftestMathParams& params [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i < params.count) output[i] = as_type<uint>(bf16_to_f32(input[i]));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_f32_to_bf16(device const float* input [[buffer(0)]],
                                        device ushort* output [[buffer(1)]],
                                        constant SelftestMathParams& params [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i < params.count) output[i] = f32_to_bf16_rne(input[i]);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_half_sat(device const float* input [[buffer(0)]],
                                     device ushort* output [[buffer(1)]],
                                     constant SelftestMathParams& params [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i < params.count) output[i] = as_type<ushort>(half_sat(input[i]));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_round_half_away(device const float* input [[buffer(0)]],
                                            device int* output [[buffer(1)]],
                                            constant SelftestMathParams& params [[buffer(2)]],
                                            uint i [[thread_position_in_grid]]) {
    if (i < params.count) output[i] = round_half_away(input[i]);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_bfly(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant SelftestMathParams& params [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= params.count) return;
    output[i] = params.operation == 0 ? bfly_sum(input[i]) : bfly_max(input[i]);
}

[[max_total_threads_per_threadgroup(8)]]
kernel void strata_selftest_partials(device const float* input [[buffer(0)]],
                                     device float* output [[buffer(1)]],
                                     constant SelftestMathParams& params [[buffer(2)]],
                                     threadgroup float* partials [[threadgroup(0)]],
                                     uint lane [[thread_index_in_threadgroup]],
                                     uint group [[threadgroup_position_in_grid]]) {
    partials[lane] = input[group * 8 + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) output[group] = sum_partials_in_order(partials, min(params.operation, 8u));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_bfly_int_max(device const int* input [[buffer(0)]],
                                         device int* output [[buffer(1)]],
                                         constant SelftestMathParams& params [[buffer(2)]],
                                         uint i [[thread_position_in_grid]]) {
    if (i < params.count) output[i] = bfly_max(input[i]);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_selftest_dot4_i8(device const int2* input [[buffer(0)]],
                                    device int* output [[buffer(1)]],
                                    constant SelftestMathParams& params [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i < params.count) output[i] = dot4_i8(input[i].x, input[i].y);
}
