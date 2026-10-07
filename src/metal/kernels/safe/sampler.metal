// The Metal sampler's GPU selection stages (docs/mac/04-metal-kernels.md §5.12).
#include <metal_stdlib>
#include "types.h"

using namespace metal;

static inline uint sampled_k(int top_k, uint n_vocab) {
    const uint k = top_k > 0 && top_k < 64 ? uint(top_k) : 64u;
    return min(n_vocab, k);
}

static inline void take_first(thread float& best_value, thread int& best_id,
                              float other_value, int other_id) {
    if (other_value > best_value ||
        (other_value == best_value && other_id < best_id)) {
        best_value = other_value;
        best_id = other_id;
    }
}

static inline void simd_first(thread float& best_value, thread int& best_id) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        const float other_value = simd_shuffle_xor(best_value, offset);
        const int other_id = simd_shuffle_xor(best_id, offset);
        take_first(best_value, best_id, other_value, other_id);
    }
}

static inline int history_count(device const int* history, uint count, int token) {
    int matches = 0;
    for (uint i = 0; i < count; ++i) matches += history[i] == token;
    return matches;
}

static inline float penalize(float logit, int count, float repeat,
                             float frequency, float presence) {
    if (count <= 0) return logit;
    logit = logit <= 0.0f ? logit * repeat : logit / repeat;
    return logit - fma(float(count), frequency, presence);
}

// Merge already sorted, disjoint lists. Each lane owns list lane and lane+32;
// every lane sees the same next item after the SIMD butterfly.
static inline void warp_merge_lists(threadgroup const int2* lists, uint list_count,
                                    uint stride, uint list_length, uint n_take, uint n_vocab,
                                    bool map_sentinel, device int2* output,
                                    uint output_offset,
                                    uint lane [[thread_index_in_simdgroup]]) {
    float head_value[2] = {-INFINITY, -INFINITY};
    int head_id[2] = {int(n_vocab), int(n_vocab)};
    uint head_position[2] = {0u, 0u};

    for (uint slot = 0; slot < 2; ++slot) {
        const uint list = lane + 32u * slot;
        if (list < list_count) {
            const int2 item = lists[list * stride];
            head_id[slot] = item.x;
            head_value[slot] = as_type<float>(item.y);
        }
    }

    for (uint rank = 0; rank < n_take; ++rank) {
        float best_value = head_value[0];
        int best_id = head_id[0];
        take_first(best_value, best_id, head_value[1], head_id[1]);
        simd_first(best_value, best_id);

        if (lane == 0) {
            const int stored_id =
                map_sentinel && uint(best_id) >= n_vocab ? 0 : best_id;
            output[output_offset + rank] = int2(stored_id, as_type<int>(best_value));
        }

        if (uint(best_id) < n_vocab) {
            for (uint slot = 0; slot < 2; ++slot) {
                if (head_id[slot] != best_id) continue;
                ++head_position[slot];
                const uint list = lane + 32u * slot;
                if (list < list_count && head_position[slot] < list_length) {
                    const int2 item = lists[list * stride + head_position[slot]];
                    head_id[slot] = item.x;
                    head_value[slot] = as_type<float>(item.y);
                } else {
                    head_id[slot] = int(n_vocab);
                    head_value[slot] = -INFINITY;
                }
            }
        }
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void strata_samp_greedy(
    device const float* logits [[buffer(0)]],
    device const int* history [[buffer(1)]],
    constant SampRow* rows [[buffer(2)]],
    device int* output [[buffer(3)]],
    constant SampArgs& args [[buffer(4)]],
    threadgroup atomic_uint* bits [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint row_index [[threadgroup_position_in_grid]]) {
    constexpr uint kGreedyChunkVocab = 131072u;
    constexpr uint kGreedyGroupCount = 32u;

    const SampRow row = rows[row_index];
    const device int* row_history = history + row.hist_off;
    const device float* row_logits =
        logits + row_index * args.n_vocab;
    float best_value = -INFINITY;
    int best_id = int(args.n_vocab);
    const uint bitmap_words =
        (min(args.n_vocab, kGreedyChunkVocab) + 31u) / 32u;
    const uint chunk_count =
        (args.n_vocab + kGreedyChunkVocab - 1u) / kGreedyChunkVocab;
    for (uint chunk = 0u; chunk < chunk_count; ++chunk) {
        const uint chunk_base = chunk * kGreedyChunkVocab;
        const uint chunk_size = min(kGreedyChunkVocab, args.n_vocab - chunk_base);
        const uint bit_words = (chunk_size + 31u) / 32u;

        if (row.hist_len > 0u) {
            for (uint word = tid; word < bit_words; word += 1024u) {
                atomic_store_explicit(&bits[word], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < row.hist_len; i += 1024u) {
                const int token = row_history[i];
                if (token >= 0 && uint(token) >= chunk_base &&
                    uint(token) - chunk_base < chunk_size) {
                    const uint local = uint(token) - chunk_base;
                    atomic_fetch_or_explicit(&bits[local >> 5u],
                                             1u << (local & 31u),
                                             memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint local = tid; local < chunk_size; local += 1024u) {
            const uint token = chunk_base + local;
            float value = row_logits[token];
            if (row.hist_len > 0u &&
                (atomic_load_explicit(&bits[local >> 5u], memory_order_relaxed) &
                 (1u << (local & 31u))) != 0u) {
                value = penalize(value,
                                 history_count(row_history, row.hist_len, int(token)),
                                 row.penalty_repeat, row.penalty_freq,
                                 row.penalty_present);
            }
            if (value > best_value ||
                (value == best_value && best_id < int(args.n_vocab) &&
                 int(token) < best_id)) {
                best_value = value;
                best_id = int(token);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simd_first(best_value, best_id);

    if (lane == 0u) {
        atomic_store_explicit(&bits[bitmap_words + simdgroup],
                              as_type<uint>(best_value), memory_order_relaxed);
        atomic_store_explicit(&bits[bitmap_words + kGreedyGroupCount + simdgroup],
                              as_type<uint>(best_id), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        best_value = as_type<float>(
            atomic_load_explicit(&bits[bitmap_words + lane], memory_order_relaxed));
        best_id = as_type<int>(atomic_load_explicit(
            &bits[bitmap_words + kGreedyGroupCount + lane], memory_order_relaxed));
        simd_first(best_value, best_id);
        if (lane == 0u) output[row_index] =
            uint(best_id) < args.n_vocab ? best_id : 0;
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_samp_split_part(
    device const float* logits [[buffer(0)]],
    device const int* history [[buffer(1)]],
    constant SampRow* rows [[buffer(2)]],
    device int2* candidates [[buffer(3)]],
    constant SampArgs& args [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    constexpr uint kPerLane = 32u;
    constexpr uint kWarpSpan = 32u * kPerLane;
    constexpr uint kBlockSpan = 4u * kWarpSpan;
    constexpr uint kMaxSelected = 64u;

    const uint block = group.x;
    const uint row_index = group.y;
    if (block >= args.n_blocks) return;
    const SampRow row = rows[row_index];
    const uint kpart = args.kpart != 0u
        ? args.kpart
        : sampled_k(row.top_k, args.n_vocab);
    const uint base = block * kBlockSpan;
    const uint warp_base = base + simdgroup * kWarpSpan;
    const device int* row_history = history + row.hist_off;
    const device float* row_logits = logits + row_index * args.n_vocab;

    threadgroup atomic_uint seen[128];
    const bool use_penalties = args.penalize != 0u && row.hist_len > 0u;
    if (use_penalties) {
        for (uint word = tid; word < 128u; word += 128u) {
            atomic_store_explicit(&seen[word], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < row.hist_len; i += 128u) {
            const int token = row_history[i];
            if (token >= 0 && uint(token) < args.n_vocab &&
                uint(token) >= base && uint(token) - base < kBlockSpan) {
                const uint local = uint(token) - base;
                atomic_fetch_or_explicit(&seen[local >> 5u],
                                         1u << (local & 31u),
                                         memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float values[kPerLane];
    for (uint j = 0; j < kPerLane; ++j) {
        const uint token = warp_base + 32u * j + lane;
        values[j] = token < args.n_vocab ? row_logits[token] : -INFINITY;
        if (use_penalties && token < args.n_vocab) {
            const uint local = token - base;
            if ((atomic_load_explicit(&seen[local >> 5u], memory_order_relaxed) &
                 (1u << (local & 31u))) != 0u) {
                values[j] = penalize(values[j],
                                     history_count(row_history, row.hist_len, int(token)),
                                     row.penalty_repeat, row.penalty_freq,
                                     row.penalty_present);
            }
        }
    }

    threadgroup int2 warp_lists[4][kMaxSelected];
    float previous_value = INFINITY;
    int previous_id = -1;
    uint written = 0u;
    for (; written < kpart; ++written) {
        float even_value = -INFINITY;
        float odd_value = -INFINITY;
        int even_id = int(args.n_vocab);
        int odd_id = int(args.n_vocab);
        for (uint j = 0; j < kPerLane; j += 2u) {
            const uint even_token = warp_base + 32u * j + lane;
            const uint odd_token = even_token + 32u;
            const float even_candidate = values[j];
            const float odd_candidate = values[j + 1u];
            if ((even_candidate < previous_value ||
                 (even_candidate == previous_value && int(even_token) > previous_id)) &&
                even_candidate > even_value) {
                even_value = even_candidate;
                even_id = int(even_token);
            }
            if ((odd_candidate < previous_value ||
                 (odd_candidate == previous_value && int(odd_token) > previous_id)) &&
                odd_candidate > odd_value) {
                odd_value = odd_candidate;
                odd_id = int(odd_token);
            }
        }
        take_first(even_value, even_id, odd_value, odd_id);
        simd_first(even_value, even_id);
        if (uint(even_id) >= args.n_vocab) break;
        if (lane == 0u) {
            warp_lists[simdgroup][written] =
                int2(even_id, as_type<int>(even_value));
        }
        previous_value = even_value;
        previous_id = even_id;
    }

    for (uint rank = written + lane; rank < kpart; rank += 32u) {
        warp_lists[simdgroup][rank] =
            int2(int(args.n_vocab), as_type<int>(-INFINITY));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        const uint destination =
            (row_index * args.n_blocks + block) * kMaxSelected;
        warp_merge_lists(&warp_lists[0][0], 4u, kMaxSelected, kpart, kpart,
                         args.n_vocab, false, candidates, destination, lane);
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void strata_samp_split_merge(
    device const int2* candidates [[buffer(0)]],
    constant SampRow* rows [[buffer(1)]],
    device int2* selected [[buffer(2)]],
    constant SampArgs& args [[buffer(3)]],
    threadgroup int2* lists [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row_index [[threadgroup_position_in_grid]]) {
    const SampRow row = rows[row_index];
    const uint k = sampled_k(row.top_k, args.n_vocab);
    const uint kpart = args.kpart != 0u ? args.kpart : k;
    for (uint item = lane; item < args.n_blocks * kpart; item += 32u) {
        const uint block = item / kpart;
        const uint rank = item - block * kpart;
        lists[item] = candidates[
            (row_index * args.n_blocks + block) * 64u + rank];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    warp_merge_lists(lists, args.n_blocks, kpart, kpart, k, args.n_vocab, true,
                     selected, row_index * 64u, lane);
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void strata_samp_coupled_penalize(
    device float* logits [[buffer(0)]],
    device const int* id_to_sub [[buffer(1)]],
    device const int* ring [[buffer(2)]],
    constant SampRow& row [[buffer(3)]],
    constant SampPenalizeArgs& args [[buffer(4)]],
    threadgroup atomic_uint* seen [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (row.hist_len == 0u) return;
    const device int* history = ring + row.hist_off;
    const uint words = (args.nv + 31u) / 32u;
    for (uint word = tid; word < words; word += 1024u) {
        atomic_store_explicit(&seen[word], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < row.hist_len; i += 1024u) {
        const int token = history[i];
        if (token < 0 || uint(token) >= args.n_vocab) continue;
        const int subset = args.has_map != 0u ? id_to_sub[uint(token)] : token;
        if (subset < 0 || uint(subset) >= args.nv) continue;
        const uint bit = 1u << (uint(subset) & 31u);
        const uint prior = atomic_fetch_or_explicit(
            &seen[uint(subset) >> 5u], bit, memory_order_relaxed);
        if ((prior & bit) != 0u) continue;
        logits[uint(subset)] =
            penalize(logits[uint(subset)],
                     history_count(history, row.hist_len, token),
                     row.penalty_repeat, row.penalty_freq,
                     row.penalty_present);
    }
}
