// Prefill GDN post-projection kernels (docs/mac/04 §6.4).
#include "../common/math.h"
#include "../common/params.h"

using namespace metal;

static inline void prefill_gdn_gates(device const float* alpha_beta,
                                     device const float* dt_bias,
                                     device const float* a_values,
                                     device float* gate,
                                     device float* beta,
                                     uint rows, uint index) {
    const uint count = rows * GDN_V_HEADS;
    if (index >= count) return;

    const uint row = index / GDN_V_HEADS;
    const uint head = index % GDN_V_HEADS;
    const ulong ab_offset = ulong(row) * (2u * GDN_V_HEADS);
    const float shifted_alpha =
        alpha_beta[ab_offset + head] + dt_bias[head];
    gate[index] = softplus_k(shifted_alpha) * a_values[head];
    beta[index] =
        sigmoid_k(alpha_beta[ab_offset + GDN_V_HEADS + head]);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_gdn_gates(
    device const float* alpha_beta [[buffer(0)]],
    device const float* dt_bias [[buffer(1)]],
    device const float* a_values [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* beta [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    prefill_gdn_gates(alpha_beta, dt_bias, a_values, gate, beta, rows, index);
}
