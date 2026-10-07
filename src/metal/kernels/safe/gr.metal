// GR operations (docs/mac/04-metal-kernels.md §5.4).
#include "reduce.h"
#include "math.h"
#include "params.h"

using namespace metal;

struct RMSNormParams {
    uint n_cols;
    uint gp;
    float eps;
};

struct GRUpParams {
    uint token_count;
    uint apply;
};

struct GRCvecParams {
    uint layer;
    uint token_count;
    uint mode;
    uint write;
};

static inline float gr_dot8(uint4 weight, threadgroup const float* values) {
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        acc = fma(as_type<float>(weight[j] << 16u), values[2 * j], acc);
        acc = fma(as_type<float>(weight[j] & 0xffff0000u),
                  values[2 * j + 1], acc);
    }
    return acc;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void strata_rms_norm_w(device const float* input [[buffer(0)]],
                              device const float* gamma [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant RMSNormParams& params [[buffer(3)]],
                              uint3 tid3 [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint warp [[simdgroup_index_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]]) {
    const uint tid = tid3.x;
    const uint row = group.x;
    const ulong row_offset = ulong(row) * params.n_cols;

    float partial = 0.0f;
    for (uint col = tid; col < params.n_cols; col += 1024u) {
        const float value = input[row_offset + col];
        partial = fma(value, value, partial);
    }

    threadgroup float sums[32];
    partial = bfly_sum(partial);
    if (lane == 0u) sums[warp] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    partial = bfly_sum(sums[lane]);
    const float mean = partial / float(params.n_cols);
    const float scale = rsqrt(mean + params.eps);
    const ulong gamma_offset = ulong(row % params.gp) * params.n_cols;
    for (uint col = tid; col < params.n_cols; col += 1024u) {
        output[row_offset + col] =
            (scale * input[row_offset + col]) * gamma[gamma_offset + col];
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_down(device const float* xn [[buffer(0)]],
                           device const ushort* hc_down [[buffer(1)]],
                           device const ushort* hc_inject [[buffer(2)]],
                           device float* lo [[buffer(3)]],
                           device float* inj_out [[buffer(4)]],
                           constant uint& token_count [[buffer(5)]],
                           threadgroup float* tile [[threadgroup(0)]],
                           uint3 tid3 [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]],
                           uint3 group [[threadgroup_position_in_grid]]) {
    constexpr uint kTileValues = 512u;
    constexpr uint kChunksPerTile = kTileValues / 256u;
    const uint block = group.x;
    const bool inject_block = block == 40u;
    const uint output_row = inject_block ? warp : 8u * block + warp;
    const bool active = !inject_block || warp < HC;
    const uint tid = tid3.x;

    float accum[8];
    for (uint token = 0u; token < 8u; ++token) accum[token] = 0.0f;

    for (uint base = 0u; base < HC_DIM; base += kTileValues) {
        uint4 lane_weights[kChunksPerTile];
        if (active) {
            const device ushort* matrix =
                inject_block ? hc_inject : hc_down;
            const device uint4* weight_vectors =
                reinterpret_cast<device const uint4*>(
                    matrix + ulong(output_row) * ulong(HC_DIM));
            for (uint chunk = 0u; chunk < kChunksPerTile; ++chunk) {
                lane_weights[chunk] =
                    weight_vectors[base / 8u + lane + 32u * chunk];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint token = 0u; token < token_count; ++token) {
            for (uint value = tid; value < kTileValues; value += 256u) {
                tile[token * kTileValues + value] =
                    xn[ulong(token) * ulong(HC_DIM) + base + value];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint chunk = 0u; chunk < kChunksPerTile; ++chunk) {
                const uint lane_chunk = lane + 32u * chunk;
                for (uint token = 0u; token < token_count; ++token) {
                    const float dot = gr_dot8(
                        lane_weights[chunk],
                        tile + token * kTileValues + lane_chunk * 8u);
                    accum[token] = accum[token] + dot;
                }
            }
        }
    }

    if (!active) return;
    for (uint token = 0u; token < token_count; ++token) {
        const float sum = bfly_sum(accum[token]);
        if (lane != token) continue;
        if (inject_block) {
            inj_out[token * HC + output_row] = sum;
        } else {
            const float scaled = sum / 4.0f;
            lo[token * HC_LOW_RANK + output_row] = silu_k(scaled);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_up(device float* residual [[buffer(0)]],
                         device const float* block_out [[buffer(1)]],
                         device const float* inject_prev [[buffer(2)]],
                         device const float* norm [[buffer(3)]],
                         device const float* rs_out [[buffer(4)]],
                         device const float* lo [[buffer(5)]],
                         device const ushort* hc_up [[buffer(6)]],
                         device float* mixed [[buffer(7)]],
                         constant GRUpParams& params [[buffer(8)]],
                         uint3 tid3 [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint warp [[simdgroup_index_in_threadgroup]],
                         uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float lo_s[VERIFY_MAX_T][HC_LOW_RANK];
    threadgroup float g_s[VERIFY_MAX_T][HC][16];

    const uint tid = tid3.x;
    const uint d0 = group.x * 16u;
    for (uint index = tid;
         index < params.token_count * HC_LOW_RANK; index += 256u) {
        lo_s[index / HC_LOW_RANK][index % HC_LOW_RANK] = lo[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = warp; r < HC * 16u; r += 8u) {
        const uint stream = r / 16u;
        const uint dd = r % 16u;
        const uint column = d0 + dd;
        const uint index = stream * N_EMBD + column;
        const device uint4* weight_vectors =
            reinterpret_cast<device const uint4*>(
                hc_up + ulong(index) * ulong(HC_LOW_RANK));
        const uint4 weight_a = weight_vectors[lane];
        uint4 weight_b = uint4(0u);
        if (lane < 8u) weight_b = weight_vectors[32u + lane];

        float mine = 0.0f;
        for (uint token = 0u; token < params.token_count; ++token) {
            float projection =
                gr_dot8(weight_a, &lo_s[token][lane * 8u]);
            if (lane < 8u) {
                projection = projection +
                    gr_dot8(weight_b, &lo_s[token][(32u + lane) * 8u]);
            }
            projection = bfly_sum(projection);
            if (lane == token) mine = projection;
        }

        if (lane < params.token_count) {
            const uint token = lane;
            const ulong residual_index =
                ulong(token) * ulong(HC_DIM) + ulong(index);
            float value = residual[residual_index];
            if (params.apply != 0u) {
                const float gate_weight =
                    2.0f * sigmoid_k(inject_prev[token * HC + stream] / 4.0f);
                value = fma(block_out[ulong(token) * ulong(N_EMBD) + column],
                            gate_weight, value);
                residual[residual_index] = value;
            }

            const float normalized =
                (value * norm[index]) * rs_out[token * HC + stream];
            g_s[token][stream][dd] = normalized * sigmoid_k(mine);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < params.token_count * 16u) {
        const uint token = tid / 16u;
        const uint column = tid % 16u;
        float sum = 0.0f;
        for (uint stream = 0u; stream < HC; ++stream) {
            sum = sum + g_s[token][stream][column];
        }
        mixed[ulong(token) * ulong(N_EMBD) + d0 + column] = sum / 4.0f;
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_cvec(device float* residual [[buffer(0)]],
                           device const float* direction [[buffer(1)]],
                           device const float* layer_scale [[buffer(2)]],
                           device const int* on [[buffer(3)]],
                           device const float* block_out [[buffer(4)]],
                           device const float* inject_prev [[buffer(5)]],
                           constant GRCvecParams& params [[buffer(6)]],
                           uint3 tid3 [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]],
                           uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float partials[8];
    const uint stream = group.x;
    const uint token = group.y;
    const uint tid = tid3.x;
    if (token >= params.token_count || stream >= HC) return;
    const ulong row_offset =
        ulong(token) * ulong(HC_DIM) + ulong(stream) * ulong(N_EMBD);
    const ulong direction_offset =
        ulong(params.layer) * ulong(N_EMBD);
    const float scale = layer_scale[params.layer];
    const bool steer = on[0] != 0 && scale != 0.0f;
    if (!steer && params.write == 0u) return;

    const float write_weight = params.write != 0u
        ? 2.0f * sigmoid_k(inject_prev[token * HC + stream] / float(HC))
        : 0.0f;
    float values[10];
    float dot = 0.0f;
    for (uint k = 0u; k < 10u; ++k) {
        const uint column = tid + 256u * k;
        float value = residual[row_offset + column];
        if (params.write != 0u) {
            value = fma(block_out[ulong(token) * ulong(N_EMBD) + column],
                        write_weight, value);
        }
        values[k] = value;
        if (steer && params.mode == 0u) {
            dot = fma(value, direction[direction_offset + column], dot);
        }
    }

    if (steer && params.mode == 0u) {
        dot = bfly_sum(dot);
        if (lane == 0u) partials[warp] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (warp == 0u) {
            float total = lane < 8u ? partials[lane] : 0.0f;
            total = bfly_sum(total);
            if (lane == 0u) partials[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = partials[0] * scale;
    }

    for (uint k = 0u; k < 10u; ++k) {
        const uint column = tid + 256u * k;
        float value = values[k];
        if (steer) {
            if (params.mode == 0u) {
                value = fma(-dot, direction[direction_offset + column], value);
            } else {
                value = value + direction[direction_offset + column];
            }
        }
        residual[row_offset + column] = value;
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_norm(device const float* residual [[buffer(0)]],
                           device const float* block_out [[buffer(1)]],
                           device const float* norm [[buffer(2)]],
                           device const float* inject_prev [[buffer(3)]],
                           device float* xn [[buffer(4)]],
                           device float* rs_out [[buffer(5)]],
                           constant uint& apply [[buffer(6)]],
                           uint3 tid3 [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]],
                           uint3 group [[threadgroup_position_in_grid]]) {
    const uint row = group.x;
    const uint stream = group.y;
    const uint tid = tid3.x;
    const uint vectors_per_stream = N_EMBD / 4u;
    const ulong row_offset = ulong(row) * ulong(HC_DIM);
    const float gate_weight = apply != 0u
        ? 2.0f * sigmoid_k(inject_prev[row * HC + stream] / 4.0f)
        : 0.0f;

    threadgroup float partials[8];
    float sum_squares = 0.0f;
    for (uint k = 0u; k < 10u; ++k) {
        const uint q = tid + 256u * k;
        if (q / vectors_per_stream != stream) continue;

        const ulong index = row_offset + ulong(q) * 4ul;
        float4 value = float4(residual[index], residual[index + 1ul],
                              residual[index + 2ul], residual[index + 3ul]);
        if (apply != 0u) {
            const ulong block_index =
                ulong(row) * ulong(N_EMBD) +
                ulong(q % vectors_per_stream) * 4ul;
            const float4 update =
                float4(block_out[block_index], block_out[block_index + 1ul],
                       block_out[block_index + 2ul],
                       block_out[block_index + 3ul]);
            value = fma(update, float4(gate_weight), value);
        }

        const float square_sum =
            fma(value.w, value.w,
                fma(value.z, value.z,
                    fma(value.y, value.y, value.x * value.x)));
        sum_squares = sum_squares + square_sum;

        const ulong norm_index = ulong(q) * 4ul;
        const float4 scale = float4(norm[norm_index], norm[norm_index + 1ul],
                                    norm[norm_index + 2ul],
                                    norm[norm_index + 3ul]);
        const float4 normalized = value * scale;
        xn[index] = normalized.x;
        xn[index + 1ul] = normalized.y;
        xn[index + 2ul] = normalized.z;
        xn[index + 3ul] = normalized.w;
    }

    sum_squares = bfly_sum(sum_squares);
    if (lane == 0u) partials[warp] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint simd = 0u; simd < 8u; ++simd) {
        total = total + partials[simd];
    }
    const float reciprocal_scale =
        rsqrt(total / float(N_EMBD) + 1.0e-6f);
    if (tid == 0u) rs_out[row * HC + stream] = reciprocal_scale;

    for (uint k = 0u; k < 10u; ++k) {
        const uint q = tid + 256u * k;
        if (q / vectors_per_stream != stream) continue;
        const ulong index = row_offset + ulong(q) * 4ul;
        xn[index] *= reciprocal_scale;
        xn[index + 1ul] *= reciprocal_scale;
        xn[index + 2ul] *= reciprocal_scale;
        xn[index + 3ul] *= reciprocal_scale;
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_post(device const float* residual [[buffer(0)]],
                           device const float* block_out [[buffer(1)]],
                           device const float* inject [[buffer(2)]],
                           device float* output [[buffer(3)]],
                           uint3 tid3 [[thread_position_in_threadgroup]],
                           uint3 group [[threadgroup_position_in_grid]]) {
    const uint index = group.x * 256u + tid3.x;
    const uint row = group.y;
    const uint stream = index / N_EMBD;
    const uint col = index % N_EMBD;
    const float scaled_inject =
        fma(0.25f, inject[row * HC + stream], 0.0f);
    const float weight = fma(2.0f, sigmoid_k(scaled_inject), 0.0f);
    const ulong residual_index = ulong(row) * HC_DIM + index;
    const ulong block_index = ulong(row) * N_EMBD + col;
    output[residual_index] =
        fma(block_out[block_index], weight, residual[residual_index]);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_down_silu(device float* lo [[buffer(0)]],
                                constant uint& n_values [[buffer(1)]],
                                uint3 tid3 [[thread_position_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    const uint index = group.x * 256u + tid3.x;
    if (index >= n_values) return;
    const float scaled = fma(0.25f, lo[index], 0.0f);
    lo[index] = silu_k(scaled);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_pre_gated(device const float* xn [[buffer(0)]],
                                device const float* gate [[buffer(1)]],
                                device float* mixed [[buffer(2)]],
                                uint3 tid3 [[thread_position_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    const uint col = group.x * 256u + tid3.x;
    const uint row = group.y;
    float sum = 0.0f;
    for (uint stream = 0u; stream < HC; ++stream) {
        const ulong index = ulong(row) * HC_DIM + ulong(stream) * N_EMBD + col;
        const float product = xn[index] * sigmoid_k(gate[index]);
        sum = stream == 0u ? product : sum + product;
    }
    mixed[ulong(row) * N_EMBD + col] = fma(0.25f, sum, 0.0f);
}
