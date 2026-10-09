// Prefill GDN post-projection kernels (docs/mac/04 §6.4).
#include "../common/math.h"
#include "../common/params.h"
#include "../common/reduce.h"

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

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gdn_conv_hist(
    device float* history [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    const uint channel = 128u * group + tid;
    const ulong history_base = ulong(channel) * (GDN_CONV_KERNEL - 1u);
    float next[GDN_CONV_KERNEL - 1u];
    for (uint k = 0u; k < GDN_CONV_KERNEL - 1u; ++k) {
        const uint source = rows + k;
        next[k] = source < GDN_CONV_KERNEL - 1u
            ? history[history_base + source]
            : qkv[ulong(source - (GDN_CONV_KERNEL - 1u)) *
                      GDN_CONV_CHANNELS +
                  channel];
    }
    for (uint k = 0u; k < GDN_CONV_KERNEL - 1u; ++k) {
        history[history_base + k] = next[k];
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gdn_l2(
    device float* h [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const ulong index =
        ulong(group.y) * GDN_CONV_CHANNELS +
        ulong(group.x) * GDN_HEAD_DIM + tid;
    const float value = h[index];
    threadgroup float partials[4];
    const float partial = bfly_sum(value * value);
    if (lane == 0u) partials[warp] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sum =
        ((partials[0] + partials[1]) + partials[2]) + partials[3];
    h[index] = value * rsqrt(sum + RMS_EPSILON);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gdn_rec(
    device float* state [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    threadgroup float* sk [[threadgroup(0)]],
    threadgroup float* sq [[threadgroup(1)]],
    threadgroup float* red [[threadgroup(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint head = group.x / 4u;
    const uint column_group = group.x % 4u;
    const uint column = 32u * column_group + tid % 32u;
    const uint row_group = tid / 32u;
    const uint lane = tid % 32u;
    const uint qk_head = head % GDN_QK_HEADS;
    const ulong state_stride = ulong(GDN_V_HEADS) * GDN_HEAD_DIM;
    device float* base =
        state + (ulong(32u * row_group) * GDN_V_HEADS + head) *
                    GDN_HEAD_DIM +
                column;
    float s[32];
    for (uint r = 0u; r < 32u; ++r) {
        s[r] = base[ulong(r) * state_stride];
    }

    for (uint t = 0u; t < rows; ++t) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sk[tid] = h[ulong(t) * GDN_CONV_CHANNELS +
                    GDN_QK_HEADS * GDN_HEAD_DIM +
                    qk_head * GDN_HEAD_DIM + tid];
        sq[tid] = h[ulong(t) * GDN_CONV_CHANNELS +
                    qk_head * GDN_HEAD_DIM + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = exp(gate[ulong(t) * GDN_V_HEADS + head]);
        float kv = 0.0f;
        for (uint r = 0u; r < 32u; ++r) {
            kv = fma(s[r], sk[32u * row_group + r], kv);
        }
        red[row_group * 32u + lane] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float kv_column =
            ((red[lane] + red[32u + lane]) + red[64u + lane]) +
            red[96u + lane];
        const float value =
            h[ulong(t) * GDN_CONV_CHANNELS +
              2u * GDN_QK_HEADS * GDN_HEAD_DIM +
              head * GDN_HEAD_DIM + column];
        const float delta =
            fma(-decay, kv_column, value) *
            beta[ulong(t) * GDN_V_HEADS + head];

        float o = 0.0f;
        for (uint r = 0u; r < 32u; ++r) {
            const float k_value = sk[32u * row_group + r];
            const float q_value = sq[32u * row_group + r];
            s[r] = fma(decay, s[r], k_value * delta);
            o = fma(s[r], q_value, o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[row_group * 32u + lane] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_group == 0u) {
            const float recurrent_output =
                (((red[lane] + red[32u + lane]) + red[64u + lane]) +
                 red[96u + lane]) *
                rsqrt(float(GDN_HEAD_DIM));
            output[ulong(t) * GDN_VALUE_DIM +
                   head * GDN_HEAD_DIM + column] = recurrent_output;
        }
    }

    for (uint r = 0u; r < 32u; ++r) {
        base[ulong(r) * state_stride] = s[r];
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
