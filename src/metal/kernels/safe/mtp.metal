// MTP drafter kernels whose arithmetic follows docs/mac/04 §5.11.
#include "types.h"
#include "params.h"
#include "math.h"
#include "dot.h"
#include "reduce.h"

using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_mtp_add_streams(
    constant MtpParams* params [[buffer(0)]],
    device const float* h2 [[buffer(1)]],
    device const float* e2 [[buffer(2)]],
    device float* R [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (group.y >= params->T) return;

    const uint value = 256u * group.x + tid3.x;
    if (value >= HC_DIM) return;

    const uint row = group.y;
    const uint output_index = row * HC_DIM + value;
    const uint embedding_index = row * N_EMBD + value % N_EMBD;
    R[output_index] = h2[output_index] + e2[embedding_index];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_mtp_window_ids(
    constant MtpParams* params [[buffer(0)]],
    device int* mtp_ids [[buffer(1)]],
    constant uint& window [[buffer(2)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint n_kv = uint(params->step[0].n_kv);
    const uint start = n_kv > window ? n_kv - window : 0u;
    const uint width = n_kv - start;

    for (uint index = 256u * group.x + tid3.x; index < width;
         index += 2048u) {
        mtp_ids[index] = int(start + index);
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_q_q8_0_scaled(
    device const float* input [[buffer(0)]],
    device char* codes [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint block = 128u * group.x + tid3.x;
    const uint blocks = n / 32u;
    if (block >= blocks) return;

    const device float* values = input + 32u * block;
    float amax = 0.0f;
    for (uint index = 0u; index < 32u; ++index) {
        amax = fmax(amax, fabs(values[index]));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    scales[block] = scale;
    for (uint index = 0u; index < 32u; ++index) {
        const float value = values[index] * inverse;
        const float rounded = value + (value >= 0.0f ? 0.5f : -0.5f);
        codes[32u * block + index] =
            char(clamp(int(rounded), -127, 127));
    }
}

static inline float mtp_s2_chunk(
    device const uchar* codes, threadgroup const int* activations,
    float weight_scale, float activation_scale, float accumulator) {
    int dot = 0;
    int activation_sum = 0;
    for (uint word = 0u; word < 8u; ++word) {
        const uint packed = uint(codes[word]);
        const int centered_codes =
            int((packed & 3u) |
                (((packed >> 2u) & 3u) << 8u) |
                (((packed >> 4u) & 3u) << 16u) |
                (((packed >> 6u) & 3u) << 24u));
        dot = dot4_i8(centered_codes, activations[word]) + dot;
        activation_sum =
            dot4_i8(0x01010101, activations[word]) + activation_sum;
    }
    return fma(weight_scale * activation_scale,
               float(dot - activation_sum), accumulator);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_mtp_gu_s2(
    device const int* ids [[buffer(0)]],
    device const uchar* expert_blob [[buffer(1)]],
    device const char* input_codes [[buffer(2)]],
    device const float* input_scales [[buffer(3)]],
    device float* gate [[buffer(4)]],
    device float* up [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint rank = group.y;
    const int expert = ids[rank];
    if (expert < 0 || expert >= 512) return;

    threadgroup int activations[640];
    threadgroup float scales[80];
    device const int* input_words =
        reinterpret_cast<device const int*>(input_codes);
    for (uint index = tid; index < 640u; index += 256u) {
        activations[index] = input_words[index];
    }
    for (uint index = tid; index < 80u; index += 256u) {
        scales[index] = input_scales[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    const device uchar* blob =
        expert_blob + uint(expert) * 1'382'400u;
    for (uint row_in_group = warp; row_in_group < 32u;
         row_in_group += 8u) {
        const uint slot = 32u * group.x + row_in_group;
        const device uchar* row_codes = blob + slot * 640u;
        device const half* row_scales =
            reinterpret_cast<device const half*>(
                blob + 1'228'800u + slot * 80u);
        float accumulator = 0.0f;
        for (uint part = 0u; part < 3u; ++part) {
            const uint chunk = lane + 32u * part;
            if (chunk >= 80u) break;
            const float weight_scale =
                float(row_scales[chunk >> 1u]);
            accumulator = mtp_s2_chunk(
                row_codes + 8u * chunk, activations + 8u * chunk,
                weight_scale, scales[chunk], accumulator);
        }
        const float result = bfly_sum(accumulator);
        if (lane == 0u) {
            const uint index = rank * 640u + slot / 2u;
            if ((slot & 1u) == 0u) {
                gate[index] = result;
            } else {
                up[index] = result;
            }
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_mtp_swiglu_s2(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint index = 256u * group.x + tid3.x;
    if (index >= 6'400u) return;
    output[index] = silu_k(gate[index]) * up[index];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_mtp_down_s2(
    device const int* ids [[buffer(0)]],
    device const uchar* expert_blob [[buffer(1)]],
    device const char* input_codes [[buffer(2)]],
    device const float* input_scales [[buffer(3)]],
    device float* parts [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint rank = group.y;
    const int expert = ids[rank];
    if (expert < 0 || expert >= 512) return;

    threadgroup int activations[160];
    threadgroup float scales[20];
    device const int* input_words =
        reinterpret_cast<device const int*>(input_codes) + rank * 160u;
    for (uint index = tid; index < 160u; index += 256u) {
        activations[index] = input_words[index];
    }
    for (uint index = tid; index < 20u; index += 256u) {
        scales[index] = input_scales[rank * 20u + index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    const device uchar* blob =
        expert_blob + uint(expert) * 1'382'400u;
    for (uint row_in_group = warp; row_in_group < 64u;
         row_in_group += 8u) {
        const uint row = 64u * group.x + row_in_group;
        float accumulator = 0.0f;
        if (lane < 20u) {
            const device uchar* row_codes =
                blob + 819'200u + row * 160u + 8u * lane;
            device const half* row_scales =
                reinterpret_cast<device const half*>(
                    blob + 1'331'200u + row * 20u);
            accumulator = mtp_s2_chunk(
                row_codes, activations + 8u * lane,
                float(row_scales[lane >> 1u]), scales[lane],
                accumulator);
        }
        const float result = bfly_sum(accumulator);
        if (lane == 0u) parts[rank * 2'560u + row] = result;
    }
}
