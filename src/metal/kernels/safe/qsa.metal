// QSA normalization and RoPE (docs/mac/04-metal-kernels.md §5.6).
#include "types.h"
#include "math.h"
#include "reduce.h"
#include "params.h"

using namespace metal;

struct QsaIndexerSnapshotParams {
    ulong tail_layer_stride_floats;
    ulong block_pos_layer_stride_ints;
};

[[max_total_threads_per_threadgroup(384)]]
kernel void strata_idx_snapshot(
    device const uint* tails [[buffer(0)]],
    device const int* block_positions [[buffer(1)]],
    device uint* tail_snapshot [[buffer(2)]],
    device int* block_pos_snapshot [[buffer(3)]],
    constant QsaIndexerSnapshotParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid < 3u * INDEXER_HEAD_DIM) {
        for (uint layer = 0u; layer < INDEXER_SNAPSHOT_LAYERS; ++layer) {
            const ulong source =
                ulong(layer) * params.tail_layer_stride_floats + tid;
            const ulong destination =
                ulong(layer) * (3u * INDEXER_HEAD_DIM) + tid;
            tail_snapshot[destination] = tails[source];
        }
    }
    if (tid == 0u) {
        for (uint layer = 0u; layer < INDEXER_SNAPSHOT_LAYERS; ++layer) {
            const ulong source =
                ulong(layer) * params.block_pos_layer_stride_ints;
            block_pos_snapshot[layer] = block_positions[source];
        }
    }
}

struct QsaIndexerAppendParams {
    uint token_rows;
    uint cell_capacity;
    uint mode;
    uint qsa_layer_index;
    int position0;
    float rope_zero_cos;
    float rope_zero_sin;
};

struct QsaSelectionScoresParams {
    uint token_rows;
    uint max_blocks;
    uint grid_count_x;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_sel_scores(
    device const float* pooled [[buffer(0)]],
    device const float* dead [[buffer(1)]],
    device const float* indexer_queries [[buffer(2)]],
    device const StepRow* steps [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant QsaSelectionScoresParams& params [[buffer(5)]],
    threadgroup float* staged_queries [[threadgroup(0)]],
    threadgroup StepRow* staged_steps [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint query_elements =
        params.token_rows * INDEXER_HEADS * INDEXER_HEAD_DIM;
    for (uint index = tid; index < query_elements; index += 256u) {
        staged_queries[index] = indexer_queries[index];
    }
    if (tid < params.token_rows) staged_steps[tid] = steps[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint top = 0u;
    for (uint row = 0u; row < params.token_rows; ++row) {
        top = max(top, uint(staged_steps[row].n_bid));
    }

    const uint first_block = group.x * 8u + simdgroup;
    for (uint block = first_block;
         block <= top && block < params.max_blocks;
         block += params.grid_count_x * 8u) {
        const ulong key_offset =
            ulong(block) * INDEXER_HEAD_DIM + lane * 4u;
        const float4 pooled_key =
            *reinterpret_cast<device const float4*>(pooled + key_offset);
        const float4 dead_key =
            *reinterpret_cast<device const float4*>(dead + lane * 4u);

        for (uint row = 0u; row < params.token_rows; ++row) {
            const StepRow step = staged_steps[row];
            if (block > uint(step.n_bid)) continue;
            const float4 key =
                block == uint(step.n_bid) ? dead_key : pooled_key;
            float score = 0.0f;
            for (uint head = 0u; head < INDEXER_HEADS; ++head) {
                const uint query_offset =
                    (row * INDEXER_HEADS + head) * INDEXER_HEAD_DIM +
                    lane * 4u;
                const float4 query =
                    *reinterpret_cast<threadgroup const float4*>(
                        staged_queries + query_offset);
                const float dot = fma(
                    key.w, query.w,
                    fma(key.z, query.z,
                        fma(key.y, query.y, key.x * query.x)));
                const float head_score = bfly_sum(dot);
                score = score +
                    (head_score > 0.0f ? head_score : 0.0f);
            }
            if (lane == 0u) {
                if (block == uint(step.n_bid) &&
                    step.n_kv % INDEXER_BLOCK_CELLS != 0) {
                    score += 1.0e9f;
                }
                scores[ulong(row) * params.max_blocks + block] = score;
            }
        }
    }
}

struct QsaSelectionTopkParams {
    uint token_rows;
    uint max_blocks;
};

struct QsaGateParams {
    uint token_rows;
};

static inline uint qsa_selection_order_key(float score) {
    const float rounded = score + 0.0f;
    if (rounded != rounded) return 0u;
    const uint bits = as_type<uint>(rounded);
    return (bits & 0x80000000u) != 0u
        ? ~bits
        : bits ^ 0x80000000u;
}

static inline uint qsa_selection_block_weight(StepRow step, uint block) {
    if (block < uint(step.n_bid)) return INDEXER_BLOCK_CELLS;
    return uint(step.n_kv) -
        uint(INDEXER_BLOCK_CELLS) * uint(step.n_bid);
}

static inline uint qsa_selection_thread_key(
    device const float* scores, ulong score_base, uint block,
    thread const uint* cached_keys, uint cache_start, bool cache_keys) {
    return cache_keys
        ? cached_keys[block - cache_start]
        : qsa_selection_order_key(scores[score_base + block]);
}

static inline void qsa_selection_topk_body(
    device const float* scores, StepRow step, uint max_blocks,
    device int* ids, uint row, thread const uint* cached_keys,
    bool cache_keys, threadgroup atomic_uint* histogram,
    threadgroup uint* scratch_a, threadgroup uint* scratch_b,
    uint tid, uint threads, uint histogram_banks) {
    const uint n_kv = uint(step.n_kv);
    const uint width = uint(step.width);
    const ulong score_base = ulong(row) * max_blocks;
    if (n_kv <= width) {
        for (uint cell = tid; cell < n_kv; cell += threads) {
            ids[ulong(row) * INDEXER_SELECTION_WIDTH_MAX + cell] =
                int(cell);
        }
        return;
    }

    const uint block_count = uint(step.n_bid) + 1u;
    const uint per = (block_count + threads - 1u) / threads;
    const uint first_block = tid * per;
    const uint end_block = min(first_block + per, block_count);
    if (tid == 0u) {
        scratch_a[0] = 0u;
        scratch_b[0] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pass = 0u; pass < 4u; ++pass) {
        const uint shift = 24u - pass * 8u;
        const uint prefix = scratch_a[0];
        const uint above = scratch_b[0];
        const uint histogram_size = histogram_banks * 256u;
        for (uint index = tid; index < histogram_size; index += threads) {
            atomic_store_explicit(&histogram[index], 0u,
                                  memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint high_prefix_shift = 32u - pass * 8u;
        const uint bank = histogram_banks == 1u ? 0u : tid / 128u;
        for (uint block = first_block; block < end_block; ++block) {
            const uint weight = qsa_selection_block_weight(step, block);
            if (weight == 0u) continue;
            const uint key = qsa_selection_thread_key(
                scores, score_base, block, cached_keys, first_block,
                cache_keys);
            if (pass != 0u && (key >> high_prefix_shift) != prefix) {
                continue;
            }
            const uint digit = (key >> shift) & 0xffu;
            atomic_fetch_add_explicit(
                &histogram[bank * 256u + digit], weight,
                memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint preceding_weight = 0u;
            uint selected_digit = 0u;
            for (int digit = 255; digit >= 0; --digit) {
                uint digit_weight = 0u;
                for (uint hist_bank = 0u;
                     hist_bank < histogram_banks; ++hist_bank) {
                    digit_weight += atomic_load_explicit(
                        &histogram[hist_bank * 256u + uint(digit)],
                        memory_order_relaxed);
                }
                if (above + preceding_weight + digit_weight >= width) {
                    selected_digit = uint(digit);
                    break;
                }
                preceding_weight += digit_weight;
            }
            scratch_a[0] = (prefix << 8u) | selected_digit;
            scratch_b[0] = above + preceding_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint threshold = scratch_a[0];
    const uint above_weight = scratch_b[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint greater_count = 0u;
    uint equal_count = 0u;
    for (uint block = first_block; block < end_block; ++block) {
        const uint weight = qsa_selection_block_weight(step, block);
        if (weight == 0u) continue;
        const uint key = qsa_selection_thread_key(
            scores, score_base, block, cached_keys, first_block, cache_keys);
        if (key > threshold) {
            greater_count += weight;
        } else if (key == threshold) {
            equal_count += weight;
        }
    }
    scratch_a[tid] = equal_count;
    scratch_b[tid] = greater_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint prefix_sum = 0u;
        for (uint worker = 0u; worker < threads; ++worker) {
            const uint count = scratch_a[worker];
            scratch_a[worker] = prefix_sum;
            prefix_sum += count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint equal_before = scratch_a[tid];
    const uint equal_budget = width - above_weight;
    const uint my_equal_count = equal_before < equal_budget
        ? min(equal_count, equal_budget - equal_before)
        : 0u;
    scratch_b[tid] = greater_count + my_equal_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint output_prefix = 0u;
        for (uint worker = 0u; worker < threads; ++worker) {
            const uint count = scratch_b[worker];
            scratch_b[worker] = output_prefix;
            output_prefix += count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output = scratch_b[tid];
    uint equal_left = my_equal_count;
    for (uint block = first_block; block < end_block; ++block) {
        const uint weight = qsa_selection_block_weight(step, block);
        if (weight == 0u) continue;
        const uint key = qsa_selection_thread_key(
            scores, score_base, block, cached_keys, first_block, cache_keys);
        uint selected = 0u;
        if (key > threshold) {
            selected = weight;
        } else if (key == threshold) {
            selected = min(weight, equal_left);
            equal_left -= selected;
        }
        for (uint cell = 0u; cell < selected; ++cell) {
            ids[ulong(row) * INDEXER_SELECTION_WIDTH_MAX + output] =
                int(block * INDEXER_BLOCK_CELLS + cell);
            ++output;
        }
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void strata_sel_topk_reg(
    device const float* scores [[buffer(0)]],
    device const StepRow* steps [[buffer(1)]],
    device int* ids [[buffer(2)]],
    constant QsaSelectionTopkParams& params [[buffer(3)]],
    threadgroup atomic_uint* histogram [[threadgroup(0)]],
    threadgroup uint* scratch_a [[threadgroup(1)]],
    threadgroup uint* scratch_b [[threadgroup(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint row = group.x;
    if (row >= params.token_rows) return;
    const StepRow step = steps[row];
    const uint n_kv = uint(step.n_kv);
    const uint width = uint(step.width);
    if (n_kv <= width) {
        for (uint cell = tid; cell < n_kv; cell += 1024u) {
            ids[ulong(row) * INDEXER_SELECTION_WIDTH_MAX + cell] =
                int(cell);
        }
        return;
    }

    const uint block_count = uint(step.n_bid) + 1u;
    const uint per = (block_count + 1023u) / 1024u;
    const uint first_block = tid * per;
    uint cached_keys[33];
    for (uint local = 0u;
         local < per && first_block + local < block_count; ++local) {
        const uint block = first_block + local;
        cached_keys[local] = qsa_selection_order_key(
            scores[ulong(row) * params.max_blocks + block]);
    }
    qsa_selection_topk_body(
        scores, step, params.max_blocks, ids, row, cached_keys, true,
        histogram, scratch_a, scratch_b, tid, 1024u, 8u);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_sel_topk_mem(
    device const float* scores [[buffer(0)]],
    device const StepRow* steps [[buffer(1)]],
    device int* ids [[buffer(2)]],
    constant QsaSelectionTopkParams& params [[buffer(3)]],
    threadgroup atomic_uint* histogram [[threadgroup(0)]],
    threadgroup uint* scratch_a [[threadgroup(1)]],
    threadgroup uint* scratch_b [[threadgroup(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint row = group.x;
    if (row >= params.token_rows) return;
    const StepRow step = steps[row];
    uint uncached_key = 0u;
    qsa_selection_topk_body(
        scores, step, params.max_blocks, ids, row, &uncached_key, false,
        histogram, scratch_a, scratch_b, tid, 256u, 1u);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_qsa_gate(
    device const float* attn [[buffer(0)]],
    device const float* q_full [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QsaGateParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint head = group.x;
    const uint row = group.y;
    if (head >= N_HEAD_Q || row >= params.token_rows) return;

    const ulong attn_base =
        (ulong(row) * N_HEAD_Q + head) * HEAD_DIM;
    const ulong q_base =
        (ulong(row) * N_HEAD_Q + head) * (2u * HEAD_DIM) + HEAD_DIM;
    output[attn_base + tid] =
        attn[attn_base + tid] * sigmoid_k(q_full[q_base + tid]);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_idx_append(
    device const float* raw_keys [[buffer(0)]],
    device const StepRow* steps [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float2* rope [[buffer(3)]],
    device uint* tail_bits [[buffer(4)]],
    device float* dead [[buffer(5)]],
    device float* pooled [[buffer(6)]],
    device int* block_position [[buffer(7)]],
    device const uint* tail_snapshot [[buffer(8)]],
    device const int* block_pos_snapshot [[buffer(9)]],
    constant QsaIndexerAppendParams& params [[buffer(10)]],
    threadgroup float* partials [[threadgroup(0)]],
    threadgroup float* values [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint layer_base =
        params.qsa_layer_index * (3u * INDEXER_HEAD_DIM);
    if (params.mode == 1u) {
        for (uint index = tid; index < 3u * INDEXER_HEAD_DIM; index += 128u) {
            tail_bits[index] = tail_snapshot[layer_base + index];
        }
        if (tid == 0u) {
            *block_position =
                block_pos_snapshot[params.qsa_layer_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint row = 0u; row < params.token_rows; ++row) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int signed_position =
            params.mode == 1u
                ? params.position0 + int(row)
                : steps[row].pos;
        const uint position = uint(signed_position);
        const uint slot = position % INDEXER_BLOCK_CELLS;
        const float incoming =
            float(half(raw_keys[ulong(row) * INDEXER_HEAD_DIM + tid]));
        if (slot < INDEXER_BLOCK_CELLS - 1u) {
            tail_bits[slot * INDEXER_HEAD_DIM + tid] = as_type<uint>(incoming);
        }
        if (position != 0u && slot != INDEXER_BLOCK_CELLS - 1u) {
            continue;
        }

        float sum = position == 0u
            ? incoming
            : as_type<float>(tail_bits[tid]);
        for (uint cell = 1u; cell < INDEXER_BLOCK_CELLS; ++cell) {
            const float value =
                position == 0u || cell == INDEXER_BLOCK_CELLS - 1u
                    ? incoming
                    : as_type<float>(
                          tail_bits[cell * INDEXER_HEAD_DIM + tid]);
            sum = sum + value;
        }
        const float mean = fma(0.25f, sum, 0.0f);

        const float square_sum = bfly_sum(mean * mean);
        if (lane == 0u) partials[warp] = square_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float all_squares =
            bfly_sum(lane < 4u ? partials[lane] : 0.0f);
        const float scale =
            rsqrt(all_squares / float(INDEXER_HEAD_DIM) + RMS_EPSILON);
        values[tid] = (scale * mean) * gamma[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint block = position / INDEXER_BLOCK_CELLS;
        const ulong rope_position =
            position == 0u
                ? 0ul
                : ulong(INDEXER_BLOCK_CELLS) * ulong(block);
        const float2 cs = position == 0u
            ? float2(params.rope_zero_cos, params.rope_zero_sin)
            : rope[rope_position * 32ul + (tid % 32u)];
        float output = values[tid];
        if (tid < ROPE_DIM) {
            const uint pair = tid % (ROPE_DIM / 2u);
            const float first = values[pair];
            const float second = values[pair + ROPE_DIM / 2u];
            output = tid < ROPE_DIM / 2u
                ? fma(-second, cs.y, first * cs.x)
                : fma(second, cs.x, first * cs.y);
        }

        pooled[ulong(block) * INDEXER_HEAD_DIM + tid] = output;
        if (position == 0u) {
            dead[tid] = output;
        } else {
            pooled[ulong(block + 1u) * INDEXER_HEAD_DIM + tid] = dead[tid];
        }
        if (tid == 0u && position != 0u) {
            *block_position = int(rope_position);
        }
    }
}

struct QsaPrefillIndexerAppendParams {
    PrefillParams prefill;
    float rope_zero_cos;
    float rope_zero_sin;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_idx_append_first(
    device const float* raw_keys [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float2* rope [[buffer(2)]],
    device uint* tail_bits [[buffer(3)]],
    device float* dead [[buffer(4)]],
    device float* pooled [[buffer(5)]],
    device int* block_position [[buffer(6)]],
    constant QsaPrefillIndexerAppendParams& params [[buffer(7)]],
    threadgroup float* partials [[threadgroup(0)]],
    threadgroup float* values [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint p0 = params.prefill.p0;
    const uint end = p0 + params.prefill.T;
    const uint rows = min(params.prefill.idx_f, end) - p0;
    for (uint row = 0u; row < rows; ++row) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint position = p0 + row;
        const uint slot = position % INDEXER_BLOCK_CELLS;
        const float incoming =
            float(half(raw_keys[ulong(row) * INDEXER_HEAD_DIM + tid]));
        if (slot < INDEXER_BLOCK_CELLS - 1u) {
            tail_bits[slot * INDEXER_HEAD_DIM + tid] = as_type<uint>(incoming);
        }
        if (position != 0u && slot != INDEXER_BLOCK_CELLS - 1u) {
            continue;
        }

        float sum = position == 0u
            ? incoming
            : as_type<float>(tail_bits[tid]);
        for (uint cell = 1u; cell < INDEXER_BLOCK_CELLS; ++cell) {
            const float value =
                position == 0u || cell == INDEXER_BLOCK_CELLS - 1u
                    ? incoming
                    : as_type<float>(
                          tail_bits[cell * INDEXER_HEAD_DIM + tid]);
            sum = sum + value;
        }
        const float mean = fma(0.25f, sum, 0.0f);

        const float square_sum = bfly_sum(mean * mean);
        if (lane == 0u) partials[warp] = square_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float all_squares =
            bfly_sum(lane < 4u ? partials[lane] : 0.0f);
        const float scale =
            rsqrt(all_squares / float(INDEXER_HEAD_DIM) + RMS_EPSILON);
        values[tid] = (scale * mean) * gamma[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint block = position / INDEXER_BLOCK_CELLS;
        const ulong rope_position = position == 0u
            ? 0ul
            : ulong(INDEXER_BLOCK_CELLS) * ulong(block);
        const float2 cs = position == 0u
            ? float2(params.rope_zero_cos, params.rope_zero_sin)
            : rope[rope_position * 32ul + (tid % 32u)];
        float output = values[tid];
        if (tid < ROPE_DIM) {
            const uint pair = tid % (ROPE_DIM / 2u);
            const float first = values[pair];
            const float second = values[pair + ROPE_DIM / 2u];
            output = tid < ROPE_DIM / 2u
                ? fma(-second, cs.y, first * cs.x)
                : fma(second, cs.x, first * cs.y);
        }

        pooled[ulong(block) * INDEXER_HEAD_DIM + tid] = output;
        if (position == 0u) {
            dead[tid] = output;
        } else {
            pooled[ulong(block + 1u) * INDEXER_HEAD_DIM + tid] = dead[tid];
        }
        if (tid == 0u && position != 0u) {
            *block_position = int(rope_position);
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_idx_append_blocks(
    device const float* raw_keys [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float2* rope [[buffer(2)]],
    device const float* dead [[buffer(3)]],
    device float* pooled [[buffer(4)]],
    device int* block_position [[buffer(5)]],
    constant QsaPrefillIndexerAppendParams& params [[buffer(6)]],
    threadgroup float* partials [[threadgroup(0)]],
    threadgroup float* values [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint block =
        params.prefill.idx_f / INDEXER_BLOCK_CELLS + group.x;
    const uint first_row =
        block * INDEXER_BLOCK_CELLS - params.prefill.p0;
    float sum = float(half(
        raw_keys[ulong(first_row) * INDEXER_HEAD_DIM + tid]));
    for (uint cell = 1u; cell < INDEXER_BLOCK_CELLS; ++cell) {
        const float value = float(half(
            raw_keys[(ulong(first_row) + cell) * INDEXER_HEAD_DIM + tid]));
        sum = sum + value;
    }
    const float mean = fma(0.25f, sum, 0.0f);

    const float square_sum = bfly_sum(mean * mean);
    if (lane == 0u) partials[warp] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float all_squares =
        bfly_sum(lane < 4u ? partials[lane] : 0.0f);
    const float scale =
        rsqrt(all_squares / float(INDEXER_HEAD_DIM) + RMS_EPSILON);
    values[tid] = (scale * mean) * gamma[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong rope_position =
        ulong(INDEXER_BLOCK_CELLS) * ulong(block);
    const float2 cs = rope[rope_position * 32ul + (tid % 32u)];
    float output = values[tid];
    if (tid < ROPE_DIM) {
        const uint pair = tid % (ROPE_DIM / 2u);
        const float first = values[pair];
        const float second = values[pair + ROPE_DIM / 2u];
        output = tid < ROPE_DIM / 2u
            ? fma(-second, cs.y, first * cs.x)
            : fma(second, cs.x, first * cs.y);
    }

    pooled[ulong(block) * INDEXER_HEAD_DIM + tid] = output;
    if (params.prefill.idx_e ==
        (block + 1u) * INDEXER_BLOCK_CELLS) {
        pooled[ulong(block + 1u) * INDEXER_HEAD_DIM + tid] = dead[tid];
        if (tid == 0u) {
            *block_position = int(rope_position);
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_idx_append_tail(
    device const float* raw_keys [[buffer(0)]],
    device uint* tail_bits [[buffer(1)]],
    constant QsaPrefillIndexerAppendParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint slot = group.x;
    const uint p0 = params.prefill.p0;
    const uint last = p0 + params.prefill.T - 1u;
    const uint back = (last % INDEXER_BLOCK_CELLS +
                       INDEXER_BLOCK_CELLS - slot) %
                      INDEXER_BLOCK_CELLS;
    if (last < back) return;
    const uint cell = last - back;
    if (cell >= p0) {
        const uint row = cell - p0;
        const float value = float(
            half(raw_keys[ulong(row) * INDEXER_HEAD_DIM + tid]));
        tail_bits[slot * INDEXER_HEAD_DIM + tid] = as_type<uint>(value);
    }
}

struct QsaNormRopeParams {
    uint token_rows;
    uint mode;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_qsa_norm_rope(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const StepRow* steps [[buffer(2)]],
    device const float2* rope [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant QsaNormRopeParams& params [[buffer(5)]],
    threadgroup float* sums [[threadgroup(0)]],
    threadgroup float* values [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint mode = params.mode;
    const uint heads_per_token =
        mode == 0u ? N_HEAD_KV
                   : (mode == 1u ? N_HEAD_Q : INDEXER_HEADS);
    const uint width = mode == 2u ? INDEXER_HEAD_DIM : HEAD_DIM;
    const uint input_stride =
        mode == 1u ? 2u * HEAD_DIM : width;
    const uint row = group.x;
    const uint token = row / heads_per_token;
    const uint position = uint(steps[token].pos);
    const ulong input_base = ulong(row) * input_stride;
    const ulong output_base = ulong(row) * width;

    const float x = tid < width ? input[input_base + tid] : 0.0f;
    float partial = bfly_sum(x * x);
    if (lane == 0u) sums[warp] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial = bfly_sum(lane < 8u ? sums[lane] : 0.0f);

    const float scale =
        rsqrt(partial / float(width) + RMS_EPSILON);
    const float normalized =
        tid < width ? (scale * x) * gamma[tid] : 0.0f;
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        const float2 cs = rope[ulong(position) * 32ul + tid];
        const float a = values[tid];
        const float b = values[tid + 32u];
        output[output_base + tid] = fma(-b, cs.y, a * cs.x);
        output[output_base + tid + 32u] = fma(b, cs.x, a * cs.y);
    } else if (tid >= 64u && tid < width) {
        output[output_base + tid] = normalized;
    }
}

struct QsaFwht256Params {
    uint rows;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_qsa_fwht256(
    device float* values [[buffer(0)]],
    constant QsaFwht256Params& params [[buffer(1)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint row = 8u * group.x + simdgroup;
    if (row >= params.rows) return;

    const ulong row_base = ulong(row) * 256ul;
    float registers[8];
    for (uint i = 0u; i < 8u; ++i) {
        registers[i] = values[row_base + i * 32u + lane] * 0.0625f;
    }

    for (ushort h = 1u; h < 32u; h <<= 1u) {
        for (uint i = 0u; i < 8u; ++i) {
            const float a = registers[i];
            const float b = simd_shuffle_xor(a, h);
            registers[i] = (lane & h) == 0u ? a + b : b - a;
        }
    }
    for (uint h = 32u; h < 256u; h <<= 1u) {
        const uint span = h / 32u;
        for (uint j = 0u; j < 8u; j += 2u * span) {
            for (uint k = 0u; k < span; ++k) {
                const float a = registers[j + k];
                const float b = registers[j + k + span];
                registers[j + k] = a + b;
                registers[j + k + span] = a - b;
            }
        }
    }

    for (uint i = 0u; i < 8u; ++i) {
        values[row_base + i * 32u + lane] = registers[i];
    }
}

struct QsaKvAppendF16Params {
    uint token_rows;
    uint cell_capacity;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_qsa_kv_append_f16(
    device const float* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const StepRow* steps [[buffer(2)]],
    device half* key_pool [[buffer(3)]],
    device half* value_pool [[buffer(4)]],
    constant QsaKvAppendF16Params& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint head_row = group.x;
    if (head_row >= params.token_rows * N_HEAD_KV) return;

    const uint token = head_row / N_HEAD_KV;
    const uint kvh = head_row % N_HEAD_KV;
    const int signed_cell = steps[token].pos;
    if (signed_cell < 0 || uint(signed_cell) >= params.cell_capacity) return;

    const uint cell = uint(signed_cell);
    const uint pool_row =
        ((cell / 4u) * N_HEAD_KV + kvh) * 4u + (cell % 4u);
    const ulong source = ulong(head_row) * HEAD_DIM + tid;
    const ulong destination = ulong(pool_row) * HEAD_DIM + tid;
    key_pool[destination] = half(keys[source]);
    value_pool[destination] = half(values[source]);
}

struct QsaKvAppendI8Params {
    uint token_rows;
    uint cell_capacity;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_qsa_kv_append_i8(
    device const float* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const StepRow* steps [[buffer(2)]],
    device char* key_codes [[buffer(3)]],
    device ushort* key_scales [[buffer(4)]],
    device char* value_codes [[buffer(5)]],
    device ushort* value_scales [[buffer(6)]],
    constant QsaKvAppendI8Params& params [[buffer(7)]],
    threadgroup float* key_maxima [[threadgroup(0)]],
    threadgroup float* value_maxima [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint head_row = group.x;
    if (head_row >= params.token_rows * N_HEAD_KV) return;

    const uint token = head_row / N_HEAD_KV;
    const uint kvh = head_row % N_HEAD_KV;
    const int signed_cell = steps[token].pos;
    if (signed_cell < 0 || uint(signed_cell) >= params.cell_capacity) return;

    const uint cell = uint(signed_cell);
    const uint pool_row =
        ((cell / 4u) * N_HEAD_KV + kvh) * 4u + (cell % 4u);
    const ulong source = ulong(head_row) * HEAD_DIM + tid;

    const float key = keys[source];
    const float value = values[source];
    const float key_partial = bfly_max(fabs(key));
    const float value_partial = bfly_max(fabs(value));
    if (lane == 0u) {
        key_maxima[warp] = key_partial;
        value_maxima[warp] = value_partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint scale_group = warp / 2u;
    const uint pair = 2u * scale_group;
    const float key_amax =
        fmax(key_maxima[pair], key_maxima[pair + 1u]);
    const float value_amax =
        fmax(value_maxima[pair], value_maxima[pair + 1u]);
    const ushort key_scale_bits =
        as_type<ushort>(half(key_amax / 127.0f));
    const ushort value_scale_bits =
        as_type<ushort>(half(value_amax / 127.0f));
    const float key_scale = float(as_type<half>(key_scale_bits));
    const float value_scale = float(as_type<half>(value_scale_bits));
    const int key_code = key_scale > 0.0f
        ? clamp(int(rint(key / key_scale)), -127, 127)
        : 0;
    const int value_code = value_scale > 0.0f
        ? clamp(int(rint(value / value_scale)), -127, 127)
        : 0;

    const ulong destination = ulong(pool_row) * HEAD_DIM + tid;
    key_codes[destination] = char(key_code);
    value_codes[destination] = char(value_code);
    if (lane == 0u && (warp & 1u) == 0u) {
        const ulong scale_index = ulong(pool_row) * 4ul + scale_group;
        key_scales[scale_index] = key_scale_bits;
        value_scales[scale_index] = value_scale_bits;
    }
}

struct QsaKvAppendQ4Params {
    uint token_rows;
    uint cell_capacity;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_qsa_kv_append_q4(
    device const float* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const StepRow* steps [[buffer(2)]],
    device block_q4_0* key_pool [[buffer(3)]],
    device block_q4_0* value_pool [[buffer(4)]],
    constant QsaKvAppendQ4Params& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint head_row = group.x;
    if (head_row >= params.token_rows * N_HEAD_KV) return;

    const uint token = head_row / N_HEAD_KV;
    const uint kvh = head_row % N_HEAD_KV;
    const int signed_cell = steps[token].pos;
    if (signed_cell < 0 || uint(signed_cell) >= params.cell_capacity) return;

    const uint cell = uint(signed_cell);
    const uint pool_row =
        ((cell / 4u) * N_HEAD_KV + kvh) * 4u + (cell % 4u);
    const ulong source = ulong(head_row) * HEAD_DIM + tid;
    const float key = keys[source];
    const float value = values[source];

    float key_amax = fabs(key);
    float key_magnitude = key;
    float value_amax = fabs(value);
    float value_magnitude = value;
    for (ushort offset = 16u; offset != 0u; offset >>= 1u) {
        const float key_other_amax =
            simd_shuffle_xor(key_amax, offset);
        const float key_other_magnitude =
            simd_shuffle_xor(key_magnitude, offset);
        if (key_other_amax > key_amax ||
            (key_other_amax == key_amax &&
             key_other_magnitude > key_magnitude)) {
            key_amax = key_other_amax;
            key_magnitude = key_other_magnitude;
        }

        const float value_other_amax =
            simd_shuffle_xor(value_amax, offset);
        const float value_other_magnitude =
            simd_shuffle_xor(value_magnitude, offset);
        if (value_other_amax > value_amax ||
            (value_other_amax == value_amax &&
             value_other_magnitude > value_magnitude)) {
            value_amax = value_other_amax;
            value_magnitude = value_other_magnitude;
        }
    }

    const float key_scale = key_magnitude / -8.0f;
    const float value_scale = value_magnitude / -8.0f;
    const float key_inverse = key_scale != 0.0f
        ? 1.0f / key_scale
        : 0.0f;
    const float value_inverse = value_scale != 0.0f
        ? 1.0f / value_scale
        : 0.0f;
    const int key_code =
        clamp(int(trunc(fma(key, key_inverse, 8.5f))), 0, 15);
    const int value_code =
        clamp(int(trunc(fma(value, value_inverse, 8.5f))), 0, 15);

    const ulong block_index = ulong(pool_row) * 8ul + warp;
    if (lane == 0u) {
        key_pool[block_index].d = half(key_scale);
        value_pool[block_index].d = half(value_scale);
    }
    const int key_high = simd_shuffle_down(key_code, 16u);
    const int value_high = simd_shuffle_down(value_code, 16u);
    if (lane < 16u) {
        key_pool[block_index].qs[lane] =
            uchar(key_code | (key_high << 4u));
        value_pool[block_index].qs[lane] =
            uchar(value_code | (value_high << 4u));
    }
}

struct QsaKvAppendK8V4Params {
    uint token_rows;
    uint cell_capacity;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_qsa_kv_append_k8v4(
    device const float* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const StepRow* steps [[buffer(2)]],
    device char* key_codes [[buffer(3)]],
    device ushort* key_scales [[buffer(4)]],
    device block_q4_0* value_pool [[buffer(5)]],
    constant QsaKvAppendK8V4Params& params [[buffer(6)]],
    threadgroup float* key_maxima [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint head_row = group.x;
    if (head_row >= params.token_rows * N_HEAD_KV) return;

    const uint token = head_row / N_HEAD_KV;
    const uint kvh = head_row % N_HEAD_KV;
    const int signed_cell = steps[token].pos;
    if (signed_cell < 0 || uint(signed_cell) >= params.cell_capacity) return;

    const uint cell = uint(signed_cell);
    const uint pool_row =
        ((cell / 4u) * N_HEAD_KV + kvh) * 4u + (cell % 4u);
    const ulong source = ulong(head_row) * HEAD_DIM + tid;
    const float key = keys[source];
    const float value = values[source];

    const float key_partial = bfly_max(fabs(key));
    if (lane == 0u) key_maxima[warp] = key_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint scale_group = warp / 2u;
    const uint pair = 2u * scale_group;
    const float key_amax =
        fmax(key_maxima[pair], key_maxima[pair + 1u]);
    const ushort key_scale_bits =
        as_type<ushort>(half(key_amax / 127.0f));
    const float key_scale = float(as_type<half>(key_scale_bits));
    const int key_code = key_scale > 0.0f
        ? clamp(int(rint(key / key_scale)), -127, 127)
        : 0;
    const ulong key_destination = ulong(pool_row) * HEAD_DIM + tid;
    key_codes[key_destination] = char(key_code);
    if (lane == 0u && (warp & 1u) == 0u) {
        key_scales[ulong(pool_row) * 4ul + scale_group] = key_scale_bits;
    }

    float value_amax = fabs(value);
    float value_magnitude = value;
    for (ushort offset = 16u; offset != 0u; offset >>= 1u) {
        const float other_amax = simd_shuffle_xor(value_amax, offset);
        const float other_magnitude =
            simd_shuffle_xor(value_magnitude, offset);
        if (other_amax > value_amax ||
            (other_amax == value_amax &&
             other_magnitude > value_magnitude)) {
            value_amax = other_amax;
            value_magnitude = other_magnitude;
        }
    }

    const float value_scale = value_magnitude / -8.0f;
    const float value_inverse = value_scale != 0.0f
        ? 1.0f / value_scale
        : 0.0f;
    const int value_code =
        clamp(int(trunc(fma(value, value_inverse, 8.5f))), 0, 15);
    const int value_high = simd_shuffle_down(value_code, 16u);
    const ulong block_index = ulong(pool_row) * 8ul + warp;
    if (lane == 0u) value_pool[block_index].d = half(value_scale);
    if (lane < 16u) {
        value_pool[block_index].qs[lane] =
            uchar(value_code | (value_high << 4u));
    }
}
