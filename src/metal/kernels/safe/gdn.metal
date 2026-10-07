// Gated DeltaNet decode, verify and commit kernels (docs/mac/04-metal-kernels.md §5.5).
#include "types.h"
#include "math.h"
#include "reduce.h"
#include "params.h"

using namespace metal;

struct GDNRowsParams {
    uint rows;
};

struct GDNStepParams {
    uint rows;
    uint mode;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_gdn_conv_l2(device const float* history [[buffer(0)]],
                               device const float* qkv [[buffer(1)]],
                               device const float* conv_weights [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant GDNRowsParams& params [[buffer(4)]],
                               uint3 tid3 [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint warp [[simdgroup_index_in_threadgroup]],
                               uint3 group [[threadgroup_position_in_grid]]) {
    const uint channel = 128u * group.x + tid3.x;
    const uint row = group.y;
    const ulong row_base = ulong(row) * GDN_CONV_CHANNELS;
    const ulong channel_base = ulong(channel) * 3ul;

    float taps[3];
    for (uint j = 0u; j < 3u; ++j) {
        const uint source = row + j;
        taps[j] = source < 3u
            ? history[channel_base + source]
            : qkv[ulong(source - 3u) * GDN_CONV_CHANNELS + channel];
    }

    const float current = qkv[row_base + channel];
    const ulong weight_base = ulong(channel) * GDN_CONV_KERNEL;
    const float convolved =
        fma(current, conv_weights[weight_base + 3u],
            fma(taps[2], conv_weights[weight_base + 2u],
                fma(taps[1], conv_weights[weight_base + 1u],
                    taps[0] * conv_weights[weight_base])));
    float value = silu_k(convolved);

    if (group.x < 32u) {
        threadgroup float partials[4];
        float partial = bfly_sum(value * value);
        if (lane == 0u) partials[warp] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sum =
            ((partials[0] + partials[1]) + partials[2]) + partials[3];
        value = value * rsqrt(sum + RMS_EPSILON);
    }
    output[row_base + channel] = value;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gdn_ab(device const float* input [[buffer(0)]],
                          device const ushort* alpha [[buffer(1)]],
                          device const ushort* beta_weights [[buffer(2)]],
                          device const float* dt_bias [[buffer(3)]],
                          device const float* a_values [[buffer(4)]],
                          device float* gate [[buffer(5)]],
                          device float* beta [[buffer(6)]],
                          constant GDNRowsParams& params [[buffer(7)]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]],
                          uint3 group [[threadgroup_position_in_grid]]) {
    const uint row8 = 8u * group.x + warp;
    const bool beta_row = row8 >= 48u;
    const uint head = row8 % 48u;
    const device ushort* weights =
        (beta_row ? beta_weights : alpha) + ulong(head) * N_EMBD;
    float accum[VERIFY_MAX_T];
    for (uint t = 0u; t < VERIFY_MAX_T; ++t) accum[t] = 0.0f;

    for (uint j = lane; j < 320u; j += 32u) {
        const device uint4* packed =
            reinterpret_cast<device const uint4*>(weights) + j;
        const uint4 w = *packed;
        for (uint t = 0u; t < params.rows; ++t) {
            const device float* x = input + ulong(t) * N_EMBD + 8ul * j;
            float value = accum[t];
            value = fma(as_type<float>(w.x << 16u), x[0], value);
            value = fma(as_type<float>(w.x & 0xffff0000u), x[1], value);
            value = fma(as_type<float>(w.y << 16u), x[2], value);
            value = fma(as_type<float>(w.y & 0xffff0000u), x[3], value);
            value = fma(as_type<float>(w.z << 16u), x[4], value);
            value = fma(as_type<float>(w.z & 0xffff0000u), x[5], value);
            value = fma(as_type<float>(w.w << 16u), x[6], value);
            value = fma(as_type<float>(w.w & 0xffff0000u), x[7], value);
            accum[t] = value;
        }
    }

    for (uint t = 0u; t < params.rows; ++t) {
        const float projected = bfly_sum(accum[t]);
        if (lane != 0u) continue;
        const ulong output_index = ulong(t) * 48ul + head;
        if (beta_row) {
            beta[output_index] = sigmoid_k(projected);
        } else {
            const float shifted = projected + dt_bias[head];
            gate[output_index] = softplus_k(shifted) * a_values[head];
        }
    }
}

[[max_total_threads_per_threadgroup(512)]]
kernel void strata_gdn_step_norm(device float* state [[buffer(0)]],
                                device const float* h [[buffer(1)]],
                                device const float* gate [[buffer(2)]],
                                device const float* beta [[buffer(3)]],
                                device const float* z [[buffer(4)]],
                                device const float* gamma [[buffer(5)]],
                                device float* output [[buffer(6)]],
                                constant GDNStepParams& params [[buffer(7)]],
                                threadgroup float* sk [[threadgroup(0)]],
                                threadgroup float* sq [[threadgroup(1)]],
                                threadgroup float* red [[threadgroup(2)]],
                                threadgroup float* wsum [[threadgroup(3)]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint warp [[simdgroup_index_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    const uint head = group.x;
    const uint column = tid % GDN_HEAD_DIM;
    const uint row_group = tid / GDN_HEAD_DIM;
    const uint qk_head = head % GDN_QK_HEADS;
    const ulong state_stride = ulong(GDN_V_HEADS) * GDN_HEAD_DIM;
    device float* base =
        state + (ulong(32u * row_group) * GDN_V_HEADS + head) *
                    GDN_HEAD_DIM + column;
    float s[32];
    for (uint r = 0u; r < 32u; ++r) s[r] = base[ulong(r) * state_stride];

    const bool emit = params.mode != 1u;
    for (uint t = 0u; t < params.rows; ++t) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < GDN_HEAD_DIM) {
            sk[tid] = h[ulong(t) * GDN_CONV_CHANNELS +
                        GDN_QK_HEADS * GDN_HEAD_DIM +
                        GDN_HEAD_DIM * qk_head + tid];
            sq[tid] = h[ulong(t) * GDN_CONV_CHANNELS +
                        GDN_HEAD_DIM * qk_head + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = exp(gate[ulong(t) * GDN_V_HEADS + head]);
        float kv = 0.0f;
        for (uint r = 0u; r < 32u; ++r) {
            kv = fma(s[r], sk[32u * row_group + r], kv);
        }
        red[row_group * GDN_HEAD_DIM + column] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float kv_column =
            ((red[column] + red[GDN_HEAD_DIM + column]) +
             red[2u * GDN_HEAD_DIM + column]) +
             red[3u * GDN_HEAD_DIM + column];
        const float value =
            h[ulong(t) * GDN_CONV_CHANNELS +
              2ul * GDN_QK_HEADS * GDN_HEAD_DIM +
              ulong(head) * GDN_HEAD_DIM + column];
        const float delta =
            fma(-decay, kv_column, value) *
            beta[ulong(t) * GDN_V_HEADS + head];

        float o = 0.0f;
        for (uint r = 0u; r < 32u; ++r) {
            const float k_value = sk[32u * row_group + r];
            s[r] = fma(decay, s[r], k_value * delta);
            o = fma(s[r], sq[32u * row_group + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[row_group * GDN_HEAD_DIM + column] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float output_value = 0.0f;
        float square = 0.0f;
        if (row_group == 0u) {
            output_value =
                (((red[column] + red[GDN_HEAD_DIM + column]) +
                   red[2u * GDN_HEAD_DIM + column]) +
                  red[3u * GDN_HEAD_DIM + column]) *
                rsqrt(float(GDN_HEAD_DIM));
            square = output_value * output_value;
        }
        if (!emit) continue;

        square = bfly_sum(square);
        if (lane == 0u) wsum[warp] = square;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_group == 0u) {
            const float sum =
                ((wsum[0] + wsum[1]) + wsum[2]) + wsum[3];
            const float scale =
                rsqrt(sum / float(GDN_HEAD_DIM) + RMS_EPSILON);
            const float normalized = (output_value * scale) * gamma[column];
            output[ulong(t) * GDN_VALUE_DIM +
                   ulong(head) * GDN_HEAD_DIM + column] =
                normalized *
                sigmoid_k(z[ulong(t) * GDN_VALUE_DIM +
                            ulong(head) * GDN_HEAD_DIM + column]);
        }
    }

    if (params.mode != 0u && params.rows > 0u) {
        for (uint r = 0u; r < 32u; ++r) {
            base[ulong(r) * state_stride] = s[r];
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_gdn_conv_commit(device float* history [[buffer(0)]],
                                   device const float* qkv [[buffer(1)]],
                                   constant GDNRowsParams& params [[buffer(2)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint3 group [[threadgroup_position_in_grid]]) {
    if (params.rows == 0u) return;
    const uint channel = 128u * group.x + tid;
    const ulong history_base = ulong(channel) * 3ul;
    float next[3];
    for (uint j = 0u; j < 3u; ++j) {
        const uint source = params.rows + j;
        next[j] = source < 3u
            ? history[history_base + source]
            : qkv[ulong(source - 3u) * GDN_CONV_CHANNELS + channel];
    }
    for (uint j = 0u; j < 3u; ++j) {
        history[history_base + j] = next[j];
    }
}
