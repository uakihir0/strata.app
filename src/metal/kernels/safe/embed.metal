// Token embedding gather and stream broadcast (docs/mac/04-metal-kernels.md §5.2).
#include "types.h"
#include "params.h"
#include "math.h"
#include "codebooks.h"

using namespace metal;

static inline float decode_q3_k(device const block_q3_K* row, uint v) {
    const uint n = v / 128u;
    const uint j = (v % 128u) / 32u;
    const uint l = v % 32u;
    const uint is = 8u * n + 2u * j + l / 16u;
    const uint shift = 2u * j;
    const uint mask = 1u << (4u * n + j);

    uint scale;
    if (is < 4u) {
        scale = (uint(row->scales[is]) & 0x0fu) |
                ((uint(row->scales[is + 8u]) & 0x03u) << 4u);
    } else if (is < 8u) {
        scale = (uint(row->scales[is]) & 0x0fu) |
                (((uint(row->scales[is + 4u]) >> 2u) & 0x03u) << 4u);
    } else if (is < 12u) {
        scale = (uint(row->scales[is - 8u]) >> 4u) |
                (((uint(row->scales[is]) >> 4u) & 0x03u) << 4u);
    } else {
        scale = (uint(row->scales[is - 8u]) >> 4u) |
                (((uint(row->scales[is - 4u]) >> 6u) & 0x03u) << 4u);
    }

    const int q = int((uint(row->qs[32u * n + l]) >> shift) & 0x03u) -
                  ((uint(row->hmask[l]) & mask) != 0u ? 0 : 4);
    const float d = float(row->d) * float(int(scale) - 32);
    return d * float(q);
}

static inline float decode_iq3_s(device const block_iq3_s* row, uint v) {
    const uint ib = v / 32u;
    const uint il = (v % 32u) / 8u;
    const uint pos = v % 8u;
    const uint group = pos / 4u;
    const uint lane = pos % 4u;
    const uint qh_shift = 8u - 2u * il - group;
    const uint grid_index =
        uint(row->qs[8u * ib + 2u * il + group]) |
        ((uint(row->qh[ib]) << qh_shift) & 0x100u);
    const uint grid_word = iq3s_grid[grid_index];
    const float scale = float(row->d) *
        float(1u + 2u * ((uint(row->scales[ib / 2u]) >> (4u * (ib % 2u))) & 0x0fu));
    const float code = float((grid_word >> (8u * lane)) & 0xffu);
    const float sign = (uint(row->signs[4u * ib + il]) & uint(kmask_iq2xs[pos])) != 0u
        ? -1.0f : 1.0f;
    return (scale * code) * sign;
}

static inline float decode_iq4_xs(device const block_iq4_xs* row, uint v) {
    const uint ib = v / 32u;
    const uint e = v % 32u;
    const uint qs_index = 16u * ib + (e < 16u ? e : e - 16u);
    const uint packed = uint(row->qs[qs_index]);
    const uint code = e < 16u ? packed & 0x0fu : packed >> 4u;
    const uint scale_code =
        ((uint(row->scales_l[ib / 2u]) >> (4u * (ib % 2u))) & 0x0fu) |
        (((uint(row->scales_h) >> (2u * ib)) & 0x03u) << 4u);
    const float scale = float(row->d) * float(int(scale_code) - 32);
    return scale * float(kvalues_iq4nl[code]);
}

#define STRATA_EMBED_ROWS_Q3_K \
    [[max_total_threads_per_threadgroup(256)]] \
    kernel void strata_embed_rows_q3_k( \
        device const block_q3_K* table [[buffer(0)]], \
        device const uint* tokens [[buffer(1)]], \
        device float* emb [[buffer(2)]], \
        constant uint& vocabulary_size [[buffer(3)]], \
        constant uint& tokens_are_gpu_written [[buffer(4)]], \
        uint3 tid3 [[thread_position_in_threadgroup]], \
        uint3 group [[threadgroup_position_in_grid]]) { \
        const uint token = tokens[group.y]; \
        if ((token & 0x80000000u) != 0u && \
            tokens_are_gpu_written == 0u) return; \
        const uint value = 256u * group.x + tid3.x; \
        if (token >= vocabulary_size) { \
            emb[N_EMBD * group.y + value] = as_type<float>(0x7fc00000u); \
            return; \
        } \
        const device block_q3_K* row = table + ulong(token) * 10ul; \
        emb[N_EMBD * group.y + value] = \
            decode_q3_k(row + group.x, value % 256u); \
    }

#define STRATA_EMBED_ROWS_IQ3_S \
    [[max_total_threads_per_threadgroup(256)]] \
    kernel void strata_embed_rows_iq3_s( \
        device const block_iq3_s* table [[buffer(0)]], \
        device const uint* tokens [[buffer(1)]], \
        device float* emb [[buffer(2)]], \
        constant uint& vocabulary_size [[buffer(3)]], \
        constant uint& tokens_are_gpu_written [[buffer(4)]], \
        uint3 tid3 [[thread_position_in_threadgroup]], \
        uint3 group [[threadgroup_position_in_grid]]) { \
        const uint token = tokens[group.y]; \
        if ((token & 0x80000000u) != 0u && \
            tokens_are_gpu_written == 0u) return; \
        const uint value = 256u * group.x + tid3.x; \
        if (token >= vocabulary_size) { \
            emb[N_EMBD * group.y + value] = as_type<float>(0x7fc00000u); \
            return; \
        } \
        const device block_iq3_s* row = table + ulong(token) * 10ul; \
        emb[N_EMBD * group.y + value] = \
            decode_iq3_s(row + group.x, value % 256u); \
    }

#define STRATA_EMBED_ROWS_IQ4_XS \
    [[max_total_threads_per_threadgroup(256)]] \
    kernel void strata_embed_rows_iq4_xs( \
        device const block_iq4_xs* table [[buffer(0)]], \
        device const uint* tokens [[buffer(1)]], \
        device float* emb [[buffer(2)]], \
        constant uint& vocabulary_size [[buffer(3)]], \
        constant uint& tokens_are_gpu_written [[buffer(4)]], \
        uint3 tid3 [[thread_position_in_threadgroup]], \
        uint3 group [[threadgroup_position_in_grid]]) { \
        const uint token = tokens[group.y]; \
        if ((token & 0x80000000u) != 0u && \
            tokens_are_gpu_written == 0u) return; \
        const uint value = 256u * group.x + tid3.x; \
        if (token >= vocabulary_size) { \
            emb[N_EMBD * group.y + value] = as_type<float>(0x7fc00000u); \
            return; \
        } \
        const device block_iq4_xs* row = table + ulong(token) * 10ul; \
        emb[N_EMBD * group.y + value] = \
            decode_iq4_xs(row + group.x, value % 256u); \
    }

STRATA_EMBED_ROWS_Q3_K
STRATA_EMBED_ROWS_IQ3_S
STRATA_EMBED_ROWS_IQ4_XS

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_embed_rows_bf16(device const ushort* table [[buffer(0)]],
                                   device const uint* tokens [[buffer(1)]],
                                   device float* emb [[buffer(2)]],
                                   constant uint& vocabulary_size [[buffer(3)]],
                                   constant uint& tokens_are_gpu_written [[buffer(4)]],
                                   uint3 tid3 [[thread_position_in_threadgroup]],
                                   uint3 group [[threadgroup_position_in_grid]]) {
    const uint token = tokens[group.y];
    if ((token & 0x80000000u) != 0u &&
        tokens_are_gpu_written == 0u) return;
    const uint value = 256u * group.x + tid3.x;
    if (token >= vocabulary_size) {
        emb[N_EMBD * group.y + value] = as_type<float>(0x7fc00000u);
        return;
    }
    const device ushort* row = table + ulong(token) * ulong(N_EMBD);
    emb[N_EMBD * group.y + value] = bf16_to_f32(row[value]);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_gr_broadcast(device const uint* emb [[buffer(0)]],
                                device uint* streams [[buffer(1)]],
                                uint3 tid3 [[thread_position_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    const uint value = 256u * group.x + tid3.x;
    streams[HC_DIM * group.y + value] =
        emb[N_EMBD * group.y + value % N_EMBD];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_embed_image(device const uint* tokens [[buffer(0)]],
                               device const uint* img_in [[buffer(1)]],
                               device uint* emb [[buffer(2)]],
                               uint3 tid3 [[thread_position_in_threadgroup]],
                               uint3 group [[threadgroup_position_in_grid]]) {
    const uint token = tokens[group.y];
    if ((token & 0x80000000u) == 0u) return;

    const uint value = 256u * group.x + tid3.x;
    const uint index = N_EMBD * group.y + value;
    emb[index] = img_in[index];
}
