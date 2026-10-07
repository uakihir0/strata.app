// Sparse QSA attention, split into per-chunk softmax partials and an ordered merge.
#include "types.h"
#include "reduce.h"
#include "params.h"

using namespace metal;

constant constexpr uint QSA_ATTN_F16 = 0u;
constant constexpr uint QSA_ATTN_I8 = 1u;
constant constexpr uint QSA_ATTN_Q4 = 2u;
constant constexpr uint QSA_ATTN_K8V4 = 3u;
constant constexpr uint QSA_ATTN_CHUNK_CELLS = 64u;
constant constexpr uint QSA_ATTN_HEADS_PER_KV = N_HEAD_Q / N_HEAD_KV;
constant constexpr uint QSA_ATTN_INVALID_ROW = 0xffffffffu;

struct QsaAttentionParams {
    uint token_rows;
    uint cell_capacity;
    uint id_row_stride;
    uint chunk_count;
};

template <uint MODE>
static inline float qsa_attention_key_value(
    device const uchar* key_data, device const ushort* key_scales,
    uint pool_row, uint dimension) {
    if (MODE == QSA_ATTN_F16) {
        device const half* values =
            reinterpret_cast<device const half*>(key_data);
        return float(values[ulong(pool_row) * HEAD_DIM + dimension]);
    }
    if (MODE == QSA_ATTN_I8 || MODE == QSA_ATTN_K8V4) {
        device const char* codes =
            reinterpret_cast<device const char*>(key_data);
        const float scale = float(as_type<half>(
            key_scales[ulong(pool_row) * 4ul + dimension / 64u]));
        return float(codes[ulong(pool_row) * HEAD_DIM + dimension]) * scale;
    }

    device const block_q4_0* blocks =
        reinterpret_cast<device const block_q4_0*>(key_data);
    const uint block = dimension / 32u;
    const uint within = dimension % 32u;
    const uchar packed =
        blocks[ulong(pool_row) * 8ul + block].qs[within % 16u];
    const uint code = within < 16u ? uint(packed & 0x0fu)
                                   : uint(packed >> 4u);
    return (float(code) - 8.0f) *
           float(blocks[ulong(pool_row) * 8ul + block].d);
}

template <uint MODE>
static inline float qsa_attention_value(
    device const uchar* value_data, device const ushort* value_scales,
    uint pool_row, uint dimension) {
    if (MODE == QSA_ATTN_F16) {
        device const half* values =
            reinterpret_cast<device const half*>(value_data);
        return float(values[ulong(pool_row) * HEAD_DIM + dimension]);
    }
    if (MODE == QSA_ATTN_I8) {
        device const char* codes =
            reinterpret_cast<device const char*>(value_data);
        const float scale = float(as_type<half>(
            value_scales[ulong(pool_row) * 4ul + dimension / 64u]));
        return float(codes[ulong(pool_row) * HEAD_DIM + dimension]) * scale;
    }

    device const block_q4_0* blocks =
        reinterpret_cast<device const block_q4_0*>(value_data);
    const uint block = dimension / 32u;
    const uint within = dimension % 32u;
    const uchar packed =
        blocks[ulong(pool_row) * 8ul + block].qs[within % 16u];
    const uint code = within < 16u ? uint(packed & 0x0fu)
                                   : uint(packed >> 4u);
    return (float(code) - 8.0f) *
           float(blocks[ulong(pool_row) * 8ul + block].d);
}

template <uint MODE>
static inline void qsa_attention_chunk_body(
    device const float* query, device const StepRow* steps,
    device const int* ids, device const uchar* key_data,
    device const ushort* key_scales, device const uchar* value_data,
    device const ushort* value_scales, device float* partial_acc,
    device float* partial_m, device float* partial_l,
    constant QsaAttentionParams& params,
    threadgroup float* staged_query, threadgroup float* probabilities,
    threadgroup uint* selected_rows, uint tid, uint lane, uint warp,
    uint3 group) {
    const uint chunk = group.x;
    const uint kvh = group.y;
    const uint row = group.z;
    if (kvh >= N_HEAD_KV || row >= params.token_rows ||
        chunk >= params.chunk_count) {
        return;
    }

    const StepRow step = steps[row];
    const uint chunk_start = chunk * QSA_ATTN_CHUNK_CELLS;
    const uint width = uint(step.width);
    const uint n_here = chunk_start < width
        ? min(QSA_ATTN_CHUNK_CELLS, width - chunk_start)
        : 0u;
    const ulong slot =
        (ulong(row) * N_HEAD_KV + kvh) * params.chunk_count + chunk;
    const ulong stats_base = slot * QSA_ATTN_HEADS_PER_KV;
    if (n_here == 0u) {
        if (tid < QSA_ATTN_HEADS_PER_KV) {
            partial_m[stats_base + tid] = -0x1.fffffep127f;
            partial_l[stats_base + tid] = 0.0f;
        }
        return;
    }

    const uint query_head_base = kvh * QSA_ATTN_HEADS_PER_KV;
    const ulong query_base =
        (ulong(row) * N_HEAD_Q + query_head_base) * HEAD_DIM;
    for (uint index = tid;
         index < QSA_ATTN_HEADS_PER_KV * HEAD_DIM; index += 256u) {
        staged_query[index] = query[query_base + index];
    }
    if (tid < n_here) {
        const int signed_cell =
            ids[ulong(row) * params.id_row_stride + chunk_start + tid];
        const uint cell = signed_cell >= 0 ? uint(signed_cell)
                                           : QSA_ATTN_INVALID_ROW;
        selected_rows[tid] =
            cell < params.cell_capacity && cell < uint(step.n_kv)
                ? ((cell / 4u) * N_HEAD_KV + kvh) * 4u + cell % 4u
                : QSA_ATTN_INVALID_ROW;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local_cell = warp;
         local_cell < QSA_ATTN_CHUNK_CELLS;
         local_cell += 8u) {
        if (local_cell >= n_here ||
            selected_rows[local_cell] == QSA_ATTN_INVALID_ROW) {
            if (lane < QSA_ATTN_HEADS_PER_KV) {
                probabilities[lane * QSA_ATTN_CHUNK_CELLS + local_cell] =
                    -0x1.fffffep127f;
            }
            continue;
        }

        const uint pool_row = selected_rows[local_cell];
        const uint dimension_base = lane * 8u;
        const float k0 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base);
        const float k1 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base + 1u);
        const float k2 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base + 2u);
        const float k3 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base + 3u);
        const float k4 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base + 4u);
        const float k5 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base + 5u);
        const float k6 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base + 6u);
        const float k7 = qsa_attention_key_value<MODE>(
            key_data, key_scales, pool_row, dimension_base + 7u);

        for (uint head = 0u; head < QSA_ATTN_HEADS_PER_KV; ++head) {
            const ulong head_base = ulong(head) * HEAD_DIM + dimension_base;
            float score = k0 * staged_query[head_base];
            score = fma(k1, staged_query[head_base + 1u], score);
            score = fma(k2, staged_query[head_base + 2u], score);
            score = fma(k3, staged_query[head_base + 3u], score);
            score = fma(k4, staged_query[head_base + 4u], score);
            score = fma(k5, staged_query[head_base + 5u], score);
            score = fma(k6, staged_query[head_base + 6u], score);
            score = fma(k7, staged_query[head_base + 7u], score);
            score = bfly_sum(score) * 0.0625f;
            if (lane == 0u) {
                probabilities[head * QSA_ATTN_CHUNK_CELLS + local_cell] =
                    score;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = warp;
         head < QSA_ATTN_HEADS_PER_KV;
         head += 8u) {
        const ulong head_base = ulong(head) * QSA_ATTN_CHUNK_CELLS;
        const float a = probabilities[head_base + lane];
        const float b = probabilities[head_base + lane + 32u];
        const float maximum = bfly_max(fmax(a, b));
        const float exp_a =
            lane < n_here &&
                    selected_rows[lane] != QSA_ATTN_INVALID_ROW
                ? exp(a - maximum)
                : 0.0f;
        const float exp_b =
            lane + 32u < n_here &&
                    selected_rows[lane + 32u] != QSA_ATTN_INVALID_ROW
                ? exp(b - maximum)
                : 0.0f;
        probabilities[head_base + lane] = exp_a;
        probabilities[head_base + lane + 32u] = exp_b;
        const float denominator = bfly_sum(exp_a + exp_b);
        if (lane == 0u) {
            partial_m[stats_base + head] = maximum;
            partial_l[stats_base + head] = denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum[QSA_ATTN_HEADS_PER_KV];
    for (uint head = 0u; head < QSA_ATTN_HEADS_PER_KV; ++head) {
        accum[head] = 0.0f;
    }
    for (uint local_cell = 0u; local_cell < n_here; ++local_cell) {
        if (selected_rows[local_cell] == QSA_ATTN_INVALID_ROW) continue;
        const float value = qsa_attention_value<MODE>(
            value_data, value_scales, selected_rows[local_cell], tid);
        for (uint head = 0u; head < QSA_ATTN_HEADS_PER_KV; ++head) {
            const float weight =
                probabilities[head * QSA_ATTN_CHUNK_CELLS + local_cell];
            accum[head] = fma(weight, value, accum[head]);
        }
    }

    const ulong partial_base = slot *
        QSA_ATTN_HEADS_PER_KV * HEAD_DIM + tid;
    for (uint head = 0u; head < QSA_ATTN_HEADS_PER_KV; ++head) {
        partial_acc[partial_base + ulong(head) * HEAD_DIM] = accum[head];
    }
}

#define STRATA_ATTN_CHUNK_ENTRY(name, mode) \
    [[max_total_threads_per_threadgroup(256)]] \
    kernel void name( \
        device const float* query [[buffer(0)]], \
        device const StepRow* steps [[buffer(1)]], \
        device const int* ids [[buffer(2)]], \
        device const uchar* key_data [[buffer(3)]], \
        device const ushort* key_scales [[buffer(4)]], \
        device const uchar* value_data [[buffer(5)]], \
        device const ushort* value_scales [[buffer(6)]], \
        device float* partial_acc [[buffer(7)]], \
        device float* partial_m [[buffer(8)]], \
        device float* partial_l [[buffer(9)]], \
        constant QsaAttentionParams& params [[buffer(10)]], \
        threadgroup float* staged_query [[threadgroup(0)]], \
        threadgroup float* probabilities [[threadgroup(1)]], \
        threadgroup uint* selected_rows [[threadgroup(2)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]], \
        uint warp [[simdgroup_index_in_threadgroup]], \
        uint3 group [[threadgroup_position_in_grid]]) { \
        qsa_attention_chunk_body<mode>( \
            query, steps, ids, key_data, key_scales, value_data, value_scales, \
            partial_acc, partial_m, partial_l, params, staged_query, \
            probabilities, selected_rows, tid, lane, warp, group); \
    }

STRATA_ATTN_CHUNK_ENTRY(strata_attn_chunk_f16, QSA_ATTN_F16)
STRATA_ATTN_CHUNK_ENTRY(strata_attn_chunk_i8, QSA_ATTN_I8)
STRATA_ATTN_CHUNK_ENTRY(strata_attn_chunk_q4, QSA_ATTN_Q4)
STRATA_ATTN_CHUNK_ENTRY(strata_attn_chunk_k8v4, QSA_ATTN_K8V4)

#undef STRATA_ATTN_CHUNK_ENTRY

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_attn_merge(
    device const float* partial_acc [[buffer(0)]],
    device const float* partial_m [[buffer(1)]],
    device const float* partial_l [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant QsaAttentionParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint head = group.x;
    const uint row = group.y;
    if (head >= N_HEAD_Q || row >= params.token_rows) return;

    const uint kvh = head / QSA_ATTN_HEADS_PER_KV;
    const uint head_local = head % QSA_ATTN_HEADS_PER_KV;
    const ulong slot_base =
        (ulong(row) * N_HEAD_KV + kvh) * params.chunk_count;
    float maximum = -0x1.fffffep127f;
    for (uint chunk = 0u; chunk < params.chunk_count; ++chunk) {
        maximum = fmax(
            maximum,
            partial_m[(slot_base + chunk) * QSA_ATTN_HEADS_PER_KV +
                      head_local]);
    }

    float denominator = 0.0f;
    float accum = 0.0f;
    for (uint chunk = 0u; chunk < params.chunk_count; ++chunk) {
        const ulong slot = slot_base + chunk;
        const float chunk_maximum =
            partial_m[slot * QSA_ATTN_HEADS_PER_KV + head_local];
        if (chunk_maximum == -0x1.fffffep127f) continue;
        const float weight = exp(chunk_maximum - maximum);
        denominator = fma(
            partial_l[slot * QSA_ATTN_HEADS_PER_KV + head_local],
            weight, denominator);
        const ulong acc_index =
            (slot * QSA_ATTN_HEADS_PER_KV + head_local) * HEAD_DIM + tid;
        accum = fma(partial_acc[acc_index], weight, accum);
    }

    output[(ulong(row) * N_HEAD_Q + head) * HEAD_DIM + tid] =
        denominator > 0.0f ? accum / denominator : 0.0f;
}
