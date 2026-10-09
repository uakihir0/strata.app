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

static inline float prefill_gdn_conv_input(device const float* history,
                                           device const float* qkv,
                                           int row, uint channel) {
    if (row < 0) {
        return history[ulong(channel) * 3ul + ulong(row + 3)];
    }
    return qkv[ulong(row) * GDN_CONV_CHANNELS + channel];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gdn_conv_tiled(
    device const float* history [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const float* conv_weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint channel = 128u * group.x + tid;
    const uint tile_start = 64u * group.y;
    const uint tile_end = min(tile_start + 64u, rows);
    const int first_row = int(tile_start);
    float v0 = prefill_gdn_conv_input(history, qkv, first_row - 3,
                                      channel);
    float v1 = prefill_gdn_conv_input(history, qkv, first_row - 2,
                                      channel);
    float v2 = prefill_gdn_conv_input(history, qkv, first_row - 1,
                                      channel);
    const ulong weight_base = ulong(channel) * GDN_CONV_KERNEL;
    const float w0 = conv_weights[weight_base];
    const float w1 = conv_weights[weight_base + 1ul];
    const float w2 = conv_weights[weight_base + 2ul];
    const float w3 = conv_weights[weight_base + 3ul];

    for (uint row = tile_start; row < tile_end; ++row) {
        const ulong index = ulong(row) * GDN_CONV_CHANNELS + channel;
        const float current = qkv[index];
        const float convolved =
            fma(current, w3,
                fma(v2, w2, fma(v1, w1, v0 * w0)));
        output[index] = silu_k(convolved);
        v0 = v1;
        v1 = v2;
        v2 = current;
    }
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
