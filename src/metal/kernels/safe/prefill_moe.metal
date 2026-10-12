// Prefill MoE route grouping and F16 row gather (docs/mac/04 §6.6).
#include <metal_stdlib>
#include "codebooks.h"
#include "../common/math.h"
#include "../common/reduce.h"
#include "../common/types.h"

using namespace metal;

struct PrefillMoeGroupingParams {
    uint rows;
    uint n_expert;
    uint layer;
    uint tiered;
};

struct PrefillMoeTileParams {
    uint rows;
    uint n_expert;
    uint tile_capacity;
};

struct PfPlan {
    uint n_groups;
    uint n_entries;
    uint n_miss;
    uint skip;
    uint n_tiles;
    uint bad;
    uint pad[2];
    int grp_e[512];
    int grp_res[512];
    int grp_start[513];
};

static_assert(sizeof(PrefillMoeGroupingParams) == 16,
              "PrefillMoeGroupingParams host ABI");
static_assert(sizeof(PrefillMoeTileParams) == 12,
              "PrefillMoeTileParams host ABI");
static_assert(sizeof(PfPlan) == 6180, "PfPlan host ABI");

struct PrefillMoeExpertAddr {
    device const uchar* role[3];
};

static_assert(sizeof(PrefillMoeExpertAddr) == 24,
              "PrefillMoeExpertAddr host ABI");

enum : uint {
    kPfMoeRoutesPerRow = 10u,
    kPfMoeExperts = 512u,
    kPfMoeTileEntries = 256u,
    kPfMoeLayerCount = 48u,
    kPfMoeMaxRows = 419430u,
};

static inline bool pf_moe_rows_valid(uint rows) {
    return rows > 0u && rows <= kPfMoeMaxRows;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_sort_count(
    device const int* ids [[buffer(0)]],
    device uint* hist [[buffer(1)]],
    constant PrefillMoeGroupingParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (!pf_moe_rows_valid(params.rows)) return;

    threadgroup int experts[256];
    const ulong total_entries =
        ulong(params.rows) * ulong(kPfMoeRoutesPerRow);
    const ulong n_tiles =
        total_entries / ulong(kPfMoeTileEntries) +
        (total_entries % ulong(kPfMoeTileEntries) != 0ul ? 1ul : 0ul);
    const ulong tile = ulong(group.x);
    if (tile >= n_tiles) return;

    const ulong entry = tile * ulong(kPfMoeTileEntries) + ulong(tid);
    experts[tid] = entry < total_entries ? ids[entry] : -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong hist_capacity = n_tiles * ulong(kPfMoeExperts);
    for (uint expert = tid; expert < kPfMoeExperts; expert += 256u) {
        uint count = 0u;
        for (uint j = 0u; j < kPfMoeTileEntries; ++j) {
            count += experts[j] == int(expert) ? 1u : 0u;
        }
        const ulong hist_index = tile * ulong(kPfMoeExperts) + ulong(expert);
        if (hist_index < hist_capacity) hist[hist_index] = count;
    }
}

[[max_total_threads_per_threadgroup(512)]]
kernel void strata_pf_sort_scan(
    device uint* hist [[buffer(0)]],
    device const int* resident [[buffer(1)]],
    device PfPlan* plans [[buffer(2)]],
    device uint2* tiles [[buffer(3)]],
    constant PrefillMoeGroupingParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (!pf_moe_rows_valid(params.rows) || params.n_expert == 0u ||
        params.n_expert > kPfMoeExperts || params.layer >= kPfMoeLayerCount) {
        return;
    }

    threadgroup uint counts[512];
    threadgroup uint offsets[512];
    threadgroup int group_for_expert[512];
    threadgroup uint tile_starts[512];

    const ulong total_entries =
        ulong(params.rows) * ulong(kPfMoeRoutesPerRow);
    const ulong n_tiles_64 =
        total_entries / ulong(kPfMoeTileEntries) +
        (total_entries % ulong(kPfMoeTileEntries) != 0ul ? 1ul : 0ul);
    const uint n_tiles = uint(n_tiles_64);
    const ulong hist_capacity = n_tiles_64 * ulong(kPfMoeExperts);
    const ulong tile_capacity =
        total_entries / 64ul + (total_entries % 64ul != 0ul ? 1ul : 0ul) +
        ulong(kPfMoeExperts);

    // The host binds plans to the selected layer's 6,400-byte slot.
    device PfPlan* plan = &plans[0];
    device uint* plan_words = reinterpret_cast<device uint*>(plans);
    plan->grp_e[tid] = -1;
    plan->grp_res[tid] = -1;
    plan->grp_start[tid] = 0;
    if (tid < 55u) plan_words[1545u + tid] = 0u;
    if (tid == 0u) {
        plan->n_groups = 0u;
        plan->n_entries = 0u;
        plan->n_miss = 0u;
        plan->skip = 0u;
        plan->n_tiles = 0u;
        plan->bad = 0u;
        plan->pad[0] = 0u;
        plan->pad[1] = 0u;
        plan->grp_start[512] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint count = 0u;
    if (tid < params.n_expert) {
        for (uint tile = 0u; tile < n_tiles; ++tile) {
            const ulong hist_index =
                ulong(tile) * ulong(kPfMoeExperts) + ulong(tid);
            if (hist_index < hist_capacity) {
                const uint tile_count = hist[hist_index];
                hist[hist_index] = count;
                count += tile_count;
            }
        }
    }
    counts[tid] = count;
    offsets[tid] = 0u;
    group_for_expert[tid] = -1;
    tile_starts[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint group_count = 0u;
        uint entry_offset = 0u;
        uint tile_offset = 0u;
        uint miss_count = 0u;
        for (uint expert = 0u; expert < params.n_expert; ++expert) {
            offsets[expert] = entry_offset;
            tile_starts[expert] = tile_offset;
            const uint expert_count = counts[expert];
            if (expert_count != 0u) {
                const bool is_resident =
                    params.tiered == 0u || resident[expert] >= 0;
                plan->grp_e[group_count] = int(expert);
                plan->grp_res[group_count] = is_resident ? 1 : 0;
                plan->grp_start[group_count] = int(entry_offset);
                group_for_expert[expert] = int(group_count);
                miss_count += is_resident ? 0u : 1u;
                entry_offset += expert_count;
                tile_offset += expert_count / 64u +
                               (expert_count % 64u != 0u ? 1u : 0u);
                ++group_count;
            }
        }
        plan->grp_start[group_count] = int(entry_offset);
        plan->n_groups = group_count;
        plan->n_entries = uint(total_entries);
        plan->n_miss = miss_count;
        plan->skip = miss_count == 0u ? 1u : 0u;
        plan->n_tiles = tile_offset;
        plan->bad = uint(total_entries) - entry_offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (tid < params.n_expert) {
        for (uint tile = 0u; tile < n_tiles; ++tile) {
            const ulong hist_index =
                ulong(tile) * ulong(kPfMoeExperts) + ulong(tid);
            if (hist_index < hist_capacity) {
                hist[hist_index] += offsets[tid];
            }
        }

        const uint group_index =
            group_for_expert[tid] >= 0
                ? uint(group_for_expert[tid])
                : 0u;
        const uint group_count = counts[tid] / 64u +
                                 (counts[tid] % 64u != 0u ? 1u : 0u);
        if (group_for_expert[tid] >= 0) {
            for (uint tile = 0u; tile < group_count; ++tile) {
                const ulong tile_index =
                    ulong(tile_starts[tid]) + ulong(tile);
                if (tile_index < tile_capacity) {
                    tiles[tile_index] = uint2(
                        group_index, offsets[tid] + tile * 64u);
                }
            }
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_sort_scatter(
    device const int* ids [[buffer(0)]],
    device const uint* hist [[buffer(1)]],
    device uint* src [[buffer(2)]],
    device uint* dst [[buffer(3)]],
    constant PrefillMoeGroupingParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (!pf_moe_rows_valid(params.rows) || params.n_expert == 0u ||
        params.n_expert > kPfMoeExperts) {
        return;
    }

    threadgroup int experts[256];
    const ulong total_entries =
        ulong(params.rows) * ulong(kPfMoeRoutesPerRow);
    const ulong n_tiles =
        total_entries / ulong(kPfMoeTileEntries) +
        (total_entries % ulong(kPfMoeTileEntries) != 0ul ? 1ul : 0ul);
    const ulong tile = ulong(group.x);
    if (tile >= n_tiles) return;

    const ulong entry =
        tile * ulong(kPfMoeTileEntries) + ulong(tid);
    experts[tid] = entry < total_entries ? ids[entry] : -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int expert = experts[tid];
    if (entry >= total_entries || expert < 0 ||
        uint(expert) >= params.n_expert) {
        return;
    }

    uint rank = 0u;
    for (uint j = 0u; j < tid; ++j) {
        rank += experts[j] == expert ? 1u : 0u;
    }
    const ulong hist_index =
        tile * ulong(kPfMoeExperts) + ulong(uint(expert));
    const ulong hist_capacity = n_tiles * ulong(kPfMoeExperts);
    if (hist_index >= hist_capacity) return;

    const ulong sorted_entry = ulong(hist[hist_index]) + ulong(rank);
    if (sorted_entry >= total_entries) return;
    src[sorted_entry] = uint(entry / ulong(kPfMoeRoutesPerRow));
    dst[sorted_entry] = uint(entry);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_gather_f16(
    device const half* mixed_h [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device const PfPlan* plans [[buffer(2)]],
    device uint4* Xs [[buffer(3)]],
    constant PrefillMoeGroupingParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (!pf_moe_rows_valid(params.rows)) return;

    // The host binds plans to the selected layer's 6,400-byte slot.
    const device PfPlan* plan = &plans[0];
    if (plan->n_groups > kPfMoeExperts ||
        plan->n_entries != params.rows * kPfMoeRoutesPerRow ||
        plan->bad > plan->n_entries) {
        return;
    }

    const int sorted_entries_signed = plan->grp_start[plan->n_groups];
    if (sorted_entries_signed < 0) return;
    const ulong sorted_entries = ulong(uint(sorted_entries_signed));
    const ulong total_entries =
        ulong(params.rows) * ulong(kPfMoeRoutesPerRow);
    if (sorted_entries > total_entries ||
        sorted_entries != total_entries - ulong(plan->bad)) {
        return;
    }

    // One thread copies one uint4 (eight halves), matching gather_rows16.
    const ulong vector_index =
        ulong(group.x) * 256ul + ulong(tid);
    const ulong output_vectors = sorted_entries * 320ul;
    const ulong maximum_output_vectors = total_entries * 320ul;
    if (vector_index >= output_vectors ||
        vector_index >= maximum_output_vectors) {
        return;
    }

    const ulong entry = vector_index / 320ul;
    const ulong vector_in_row = vector_index % 320ul;
    const uint source_row = src[entry];
    if (source_row >= params.rows) return;

    const ulong input_vector =
        ulong(source_row) * 320ul + vector_in_row;
    const ulong input_capacity = ulong(params.rows) * 320ul;
    if (input_vector >= input_capacity) return;

    const device uint4* mixed_vectors =
        reinterpret_cast<device const uint4*>(mixed_h);
    Xs[vector_index] = mixed_vectors[input_vector];
}

static inline half pf_moe_q2_0_value(
    device const block_q2_0* row_weights, uint k) {
    const device block_q2_0* block = row_weights + k / 64u;
    const uint within_block = k % 64u;
    const uint packed = uint(block->qs[within_block / 4u]);
    const int code =
        int((packed >> (2u * (within_block % 4u))) & 0x03u) - 1;
    return half(float(block->d) * float(code));
}

static inline half pf_moe_iq4_nl_value(
    device const block_iq4_nl* row_weights, uint k) {
    const device block_iq4_nl* block = row_weights + k / 32u;
    const uint within_block = k % 32u;
    const uint packed = uint(block->qs[within_block & 15u]);
    const uint code = (packed >> ((within_block >> 4u) * 4u)) & 0x0fu;
    return half(float(block->d) * float(kvalues_iq4nl[code]));
}

static inline half pf_moe_iq4_xs_value(
    device const block_iq4_xs* row_weights, uint k) {
    const device block_iq4_xs* block = row_weights + k / 256u;
    const uint within_block = k % 256u;
    const uint scale_group = within_block / 32u;
    const uint within_group = within_block % 32u;
    const uint qs_index =
        16u * scale_group +
        (within_group < 16u ? within_group : within_group - 16u);
    const uint packed = uint(block->qs[qs_index]);
    const uint code =
        within_group < 16u ? packed & 0x0fu : packed >> 4u;
    const uint scale_low =
        (uint(block->scales_l[scale_group / 2u]) >>
         (4u * (scale_group % 2u))) &
        0x0fu;
    const uint scale_high =
        (uint(block->scales_h) >> (2u * scale_group)) & 0x03u;
    const int scale = int(scale_low | (scale_high << 4u)) - 32;
    const float scaled_d = float(block->d) * float(scale);
    const float expanded = scaled_d * float(kvalues_iq4nl[code]);
    return half(expanded);
}

static inline half pf_moe_iq2_xxs_value(
    device const block_iq2_xxs* row_weights, uint k) {
    const device block_iq2_xxs* block = row_weights + k / 256u;
    const uint within_block = k % 256u;
    const uint part = within_block / 32u;
    const uint within_part = within_block % 32u;
    const uint code_slot = within_part / 8u;
    const uint component = within_part % 8u;
    const device ushort* packed_words = block->qs + 4u * part;
    const uint indices =
        uint(packed_words[0]) | (uint(packed_words[1]) << 16u);
    const uint auxiliary =
        uint(packed_words[2]) | (uint(packed_words[3]) << 16u);
    const uint code = (indices >> (8u * code_slot)) & 0xffu;
    const uint sign_index =
        (auxiliary >> (7u * code_slot)) & 0x7fu;
    const uint signs = uint(ksigns_iq2xs[sign_index]);
    const ulong grid = iq2xxs_grid[code];
    const uint magnitude =
        uint((grid >> (8u * component)) & 0xfful);
    const uint scale_code = auxiliary >> 28u;
    const float scale =
        float(block->d) * (0.5f + float(scale_code)) * 0.25f;
    const float sign =
        (signs & uint(kmask_iq2xs[component])) != 0u ? -1.0f : 1.0f;
    return half(scale * float(magnitude) * sign);
}

static inline half pf_moe_iq2_xs_value(
    device const block_iq2_xs* row_weights, uint k) {
    const device block_iq2_xs* block = row_weights + k / 256u;
    const uint within_block = k % 256u;
    const uint part = within_block / 32u;
    const uint within_part = within_block % 32u;
    const uint code_slot = within_part / 8u;
    const uint component = within_part % 8u;
    const ushort codeword = block->qs[4u * part + code_slot];
    const uint grid_index = uint(codeword & 0x1ffu);
    const uint sign_index = uint(codeword >> 9u);
    const uint signs = uint(ksigns_iq2xs[sign_index]);
    const ulong grid = iq2xs_grid[grid_index];
    const uint magnitude =
        uint((grid >> (8u * component)) & 0xfful);
    const uint scale_pair = uint(block->scales[part]);
    const uint scale_code =
        code_slot < 2u ? (scale_pair & 0x0fu) : (scale_pair >> 4u);
    const float scale =
        float(block->d) * (0.5f + float(scale_code)) * 0.25f;
    const float sign =
        (signs & uint(kmask_iq2xs[component])) != 0u ? -1.0f : 1.0f;
    return half(scale * float(magnitude) * sign);
}

static inline half pf_moe_iq2_s_value(
    device const block_iq2_s* row_weights, uint k) {
    const device block_iq2_s* block = row_weights + k / 256u;
    const uint within_block = k % 256u;
    const uint part = within_block / 32u;
    const uint within_part = within_block % 32u;
    const uint code_slot = within_part / 8u;
    const uint component = within_part % 8u;
    const uint high_codes = uint(block->qh[part]);
    const uint grid_index =
        uint(block->qs[4u * part + code_slot]) |
        (((high_codes >> (2u * code_slot)) & 0x03u) << 8u);
    const uint signs = uint(block->qs[32u + 4u * part + code_slot]);
    const ulong grid = iq2s_grid[grid_index];
    const uint magnitude =
        uint((grid >> (8u * component)) & 0xfful);
    const uint scale_pair = uint(block->scales[part]);
    const uint scale_code =
        code_slot < 2u ? (scale_pair & 0x0fu) : (scale_pair >> 4u);
    const float scale =
        float(block->d) * (0.5f + float(scale_code)) * 0.25f;
    const float sign =
        (signs & uint(kmask_iq2xs[component])) != 0u ? -1.0f : 1.0f;
    return half(scale * float(magnitude) * sign);
}

static inline uint pf_moe_iq3_xxs_unpack_signs(uint sign_index) {
    const uint low_signs = sign_index & 0x7fu;
    return low_signs ^ ((popcount(low_signs) & 1u) << 7u);
}

static inline half pf_moe_iq3_xxs_value(
    device const block_iq3_xxs* row_weights, uint k) {
    const device block_iq3_xxs* block = row_weights + k / 256u;
    const uint within_block = k % 256u;
    const uint part = within_block / 32u;
    const uint within_part = within_block % 32u;
    const uint code_slot = within_part / 8u;
    const uint component = within_part % 8u;
    const device uchar* indices = block->qs + 8u * part;
    const device uchar* aux = block->qs + 64u + 4u * part;
    const uint aux32 = uint(aux[0]) | (uint(aux[1]) << 8u) |
                       (uint(aux[2]) << 16u) | (uint(aux[3]) << 24u);
    const uint sign_index = (aux32 >> (7u * code_slot)) & 0x7fu;
    const uint signs = pf_moe_iq3_xxs_unpack_signs(sign_index);
    const uint grid_index = indices[2u * code_slot + component / 4u];
    const uint packed_grid = iq3xxs_grid[grid_index];
    const uint magnitude =
        (packed_grid >> (8u * (component % 4u))) & 0xffu;
    const uint scale_code = aux32 >> 28u;
    const float scale =
        float(block->d) * (0.5f + float(scale_code)) * 0.5f;
    const float sign =
        (signs & (1u << component)) != 0u ? -1.0f : 1.0f;
    return half(scale * float(magnitude) * sign);
}

static inline half pf_moe_iq3_s_value(
    device const block_iq3_s* row_weights, uint k) {
    const device block_iq3_s* block = row_weights + k / 256u;
    const uint within_block = k % 256u;
    const uint part = within_block / 32u;
    const uint within_part = within_block % 32u;
    const uint code_slot = within_part / 8u;
    const uint position = within_part % 8u;
    const uint grid_half = position / 4u;
    const uint component = position % 4u;
    const uint qs_index = 8u * part + 2u * code_slot + grid_half;
    const uint high_bit =
        (uint(block->qh[part]) >> (2u * code_slot + grid_half)) & 1u;
    const uint grid_index = uint(block->qs[qs_index]) | (high_bit << 8u);
    const uint packed_grid = iq3s_grid[grid_index];
    const uint magnitude =
        (packed_grid >> (8u * component)) & 0xffu;
    const bool negative =
        ((uint(block->signs[4u * part + code_slot]) >> position) & 1u) !=
        0u;
    const uint packed_scales = uint(block->scales[part / 2u]);
    const uint scale_code =
        (packed_scales >> (4u * (part % 2u))) & 0x0fu;
    const float sign = negative ? -1.0f : 1.0f;
    const float weight =
        float(block->d) * float(1u + 2u * scale_code) *
        float(magnitude) * sign;
    return half(weight);
}

static inline uint pf_moe_iq1_m_half_scale_bits(
    device const block_iq1_m* block) {
    uint result = 0u;
    for (uint word = 0u; word < 4u; ++word) {
        const uint byte_offset = 2u * word;
        const uint packed = uint(block->scales[byte_offset]) |
                            (uint(block->scales[byte_offset + 1u]) << 8u);
        result |= ((packed >> 12u) & 0x0fu) << (4u * word);
    }
    return result;
}

static inline uint pf_moe_iq1_m_packed_scale(
    device const block_iq1_m* block, uint part) {
    const uint word_index = part / 2u;
    const uint byte_offset = 2u * word_index;
    const uint packed = uint(block->scales[byte_offset]) |
                        (uint(block->scales[byte_offset + 1u]) << 8u);
    return packed >> (6u * (part & 1u));
}

static inline half pf_moe_iq1_m_value(
    device const block_iq1_m* row_weights, uint k) {
    const device block_iq1_m* block = row_weights + k / 256u;
    const uint within_block = k % 256u;
    const uint part = within_block / 32u;
    const uint within_part = within_block % 32u;
    const uint codeword = within_part / 8u;
    const uint lane = within_part % 8u;
    const uint qh_byte = uint(block->qh[2u * part + codeword / 2u]);
    const uint selector =
        (qh_byte >> (4u * (codeword & 1u))) & 0x0fu;
    const uint grid_index =
        uint(block->qs[4u * part + codeword]) |
        ((selector & 0x07u) << 8u);
    const uint packed_grid = iq1s_grid_gpu[grid_index];
    const uint code = (packed_grid >> (4u * lane)) & 0x0fu;
    const uint packed_scale = pf_moe_iq1_m_packed_scale(block, part);
    const uint local_scale =
        2u * ((packed_scale >> (3u * (codeword / 2u))) & 0x07u) + 1u;
    const float delta =
        (selector & 0x08u) != 0u ? -0.125f : 0.125f;
    const float d = float(as_type<half>(
        ushort(pf_moe_iq1_m_half_scale_bits(block))));
    const float scaled_d = d * float(local_scale);
    return half(scaled_d * (float(code) - 1.0f + delta));
}

static inline void pf_moe_zero_output_tile(
    device half* output, uint total_entries, uint first_entry,
    uint end_entry, uint column_group, uint tid) {
    for (uint i = tid; i < 64u * 32u; i += 128u) {
        const ulong entry = ulong(first_entry) + ulong(i / 32u);
        const uint column = i % 32u;
        if (entry < ulong(total_entries) && entry < ulong(end_entry)) {
            output[entry * 640ul +
                   ulong(column_group * 32u + column)] = half(0.0f);
        }
    }
}

static inline void pf_moe_zero_down_tile(
    device float* output, uint total_entries, uint first_entry,
    uint column_group, uint tid) {
    for (uint i = tid; i < 64u * 64u; i += 128u) {
        const ulong entry = ulong(first_entry) + ulong(i / 64u);
        const uint column = i % 64u;
        if (entry < ulong(total_entries)) {
            output[entry * 2560ul +
                   ulong(column_group * 64u + column)] = 0.0f;
        }
    }
}

static inline void pf_moe_zero_down_group_tile(
    device float* output, device const uint* destinations,
    uint total_entries, uint first_entry, uint end_entry,
    uint column_group, uint tid, device atomic_uint* status) {
    for (uint i = tid; i < 64u * 64u; i += 128u) {
        const uint entry = first_entry + i / 64u;
        const uint column = i % 64u;
        if (entry >= end_entry) continue;
        const uint destination = destinations[entry];
        if (destination >= total_entries) {
            if (tid == 0u) {
                atomic_fetch_or_explicit(status, 2u,
                                         memory_order_relaxed);
            }
            continue;
        }
        output[ulong(destination) * 2560ul +
               ulong(column_group * 64u + column)] = 0.0f;
    }
}

static inline bool pf_moe_plan_header_valid(
    const device PfPlan* plan,
    constant PrefillMoeTileParams& params) {
    const uint entries = params.rows * kPfMoeRoutesPerRow;
    if (plan->bad != 0u || plan->n_groups == 0u ||
        plan->n_groups > params.n_expert ||
        plan->n_groups > kPfMoeExperts || plan->n_entries != entries ||
        plan->n_tiles > params.tile_capacity ||
        plan->n_tiles > plan->n_entries || plan->grp_start[0] != 0 ||
        plan->grp_start[plan->n_groups] != int(entries)) {
        return false;
    }

    uint expected_tiles = 0u;
    int previous_end = 0;
    for (uint group = 0u; group < plan->n_groups; ++group) {
        const int first = plan->grp_start[group];
        const int end = plan->grp_start[group + 1u];
        const int expert = plan->grp_e[group];
        if (first != previous_end || end <= first || uint(end) > entries ||
            expert < 0 || uint(expert) >= params.n_expert) {
            return false;
        }
        if (group != 0u && expert <= plan->grp_e[group - 1u]) {
            return false;
        }
        previous_end = end;
        const uint count = uint(end - first);
        expected_tiles += count / 64u + (count % 64u != 0u ? 1u : 0u);
        if (expected_tiles > params.tile_capacity) return false;
    }
    return previous_end == int(entries) &&
           expected_tiles == plan->n_tiles;
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_moe_plan_validate(
    device const PfPlan* plans [[buffer(0)]],
    device const uint2* tiles [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant PrefillMoeTileParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    const device PfPlan* plan = &plans[0];
    threadgroup uint header_valid = 0u;
    if (tid == 0u) {
        atomic_store_explicit(status, 0u, memory_order_relaxed);
        header_valid =
            pf_moe_plan_header_valid(plan, params) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (header_valid == 0u) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        return;
    }

    for (uint index = tid; index < plan->n_tiles; index += 128u) {
        const uint2 tile = tiles[index];
        bool valid = tile.x < plan->n_groups;
        if (valid) {
            const uint first = uint(plan->grp_start[tile.x]);
            const uint end = uint(plan->grp_start[tile.x + 1u]);
            valid = tile.y >= first && tile.y < end &&
                    (tile.y - first) % 64u == 0u;
            if (index == 0u) {
                valid &= tile.x == 0u && tile.y == 0u;
            } else {
                const uint2 previous = tiles[index - 1u];
                if (previous.x >= plan->n_groups) {
                    valid = false;
                } else if (previous.x == tile.x) {
                    valid &= tile.y == previous.y + 64u;
                } else if (previous.x + 1u == tile.x) {
                    const uint previous_end =
                        uint(plan->grp_start[previous.x + 1u]);
                    valid &= previous.y + 64u >= previous_end &&
                             tile.y == first;
                } else {
                    valid = false;
                }
            }
            if (index + 1u == plan->n_tiles) {
                valid &= tile.x + 1u == plan->n_groups &&
                         tile.y + 64u >= end;
            }
        }
        if (!valid) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_moe_destination_clear(
    device atomic_uint* seen [[buffer(0)]],
    constant PrefillMoeTileParams& params [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint entries = params.rows * kPfMoeRoutesPerRow;
    const uint words = entries / 32u + (entries % 32u != 0u ? 1u : 0u);
    const uint word = group.x * 128u + tid;
    if (word < words) {
        atomic_store_explicit(&seen[word], 0u, memory_order_relaxed);
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_moe_destination_validate(
    device const uint* destinations [[buffer(0)]],
    device atomic_uint* seen [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant PrefillMoeTileParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint entries = params.rows * kPfMoeRoutesPerRow;
    const uint entry = group.x * 128u + tid;
    if (entry >= entries) return;

    const uint destination = destinations[entry];
    if (destination >= entries) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        return;
    }
    const uint mask = 1u << (destination % 32u);
    const uint previous = atomic_fetch_or_explicit(
        &seen[destination / 32u], mask, memory_order_relaxed);
    if ((previous & mask) != 0u) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
}

// Direct bitwise twin probe for the in-place Q2_0 weight decoder.
[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_q2_0_decode_probe(
    device const block_q2_0* weights [[buffer(0)]],
    device half* decoded [[buffer(1)]],
    constant uint& values [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint index = group.x * 256u + tid;
    if (index < values) decoded[index] = pf_moe_q2_0_value(weights, index);
}

// Exhaustive packed-field twin probe for the IQ3_S in-place decoder.
[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_iq3_s_decode_probe(
    device const block_iq3_s* weights [[buffer(0)]],
    device half* decoded [[buffer(1)]],
    constant uint& values [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint index = group.x * 256u + tid;
    if (index < values) decoded[index] = pf_moe_iq3_s_value(weights, index);
}

// Exhaustive packed-field twin probe for the IQ1_M in-place decoder.
[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_iq1_m_decode_probe(
    device const block_iq1_m* weights [[buffer(0)]],
    device half* decoded [[buffer(1)]],
    constant uint& values [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint index = group.x * 256u + tid;
    if (index < values) decoded[index] = pf_moe_iq1_m_value(weights, index);
}

// One 64-entry x 64-column simdgroup matrix tile. Each 64-column tile
// interleaves 32 gate and 32 up columns; the epilogue applies SwiGLU and
// writes the resulting hidden row directly.
[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_moe_gu_f16(
    device const PrefillMoeExpertAddr* addresses [[buffer(0)]],
    device const half* Xs [[buffer(1)]],
    device const PfPlan* plans [[buffer(2)]],
    device const uint2* tiles [[buffer(3)]],
    device half* eh [[buffer(4)]],
    constant PrefillMoeTileParams& params [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    constant uchar* tiered_lease_mask [[buffer(7)]],
    constant uint& weight_format [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    if (!pf_moe_rows_valid(params.rows) ||
        (params.n_expert != 256u && params.n_expert != 512u)) {
        return;
    }

    const device PfPlan* plan = &plans[0];
    threadgroup uint plan_is_invalid = 0u;
    if (tid == 0u) {
        plan_is_invalid =
            (atomic_load_explicit(status, memory_order_relaxed) & 2u) != 0u
                ? 1u
                : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (plan_is_invalid != 0u) {
        pf_moe_zero_output_tile(
            eh, params.rows * kPfMoeRoutesPerRow, group.y * 64u,
            params.rows * kPfMoeRoutesPerRow, group.x, tid);
        return;
    }
    if (weight_format > 7u) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        pf_moe_zero_output_tile(
            eh, params.rows * kPfMoeRoutesPerRow, group.y * 64u,
            params.rows * kPfMoeRoutesPerRow, group.x, tid);
        return;
    }
    const uint tile_index = group.y;
    if (tile_index >= plan->n_tiles) return;
    const uint2 tile = tiles[tile_index];
    const uint group_index = tile.x;
    const uint a0 = tile.y;
    if (group_index >= plan->n_groups) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        pf_moe_zero_output_tile(eh, params.rows * kPfMoeRoutesPerRow, a0,
                                params.rows * kPfMoeRoutesPerRow, group.x,
                                tid);
        return;
    }

    const int first_signed = plan->grp_start[group_index];
    const int end_signed = plan->grp_start[group_index + 1u];
    const int expert_signed = plan->grp_e[group_index];
    if (first_signed < 0 || end_signed <= first_signed ||
        uint(end_signed) > plan->n_entries || expert_signed < 0 ||
        uint(expert_signed) >= params.n_expert) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        pf_moe_zero_output_tile(eh, params.rows * kPfMoeRoutesPerRow, a0,
                                params.rows * kPfMoeRoutesPerRow, group.x,
                                tid);
        return;
    }
    const uint group_first = uint(first_signed);
    const uint group_end = uint(end_signed);
    if (a0 < group_first || a0 >= group_end ||
        (a0 - group_first) % 64u != 0u) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        const uint safe_first = min(max(a0, group_first), group_end);
        const uint safe_end = min(group_end, safe_first + 64u);
        pf_moe_zero_output_tile(eh, params.rows * kPfMoeRoutesPerRow,
                                safe_first, safe_end, group.x, tid);
        return;
    }
    const uint a1 = min(a0 + 64u, group_end);

    const PrefillMoeExpertAddr address = addresses[uint(expert_signed)];
    const bool missing_weights =
        address.role[0] == nullptr || address.role[1] == nullptr;
    const bool missing_tiered_lease =
        !missing_weights && tiered_lease_mask[uint(expert_signed)] == 0u;
    if (missing_weights || missing_tiered_lease) {
        if (tid == 0u) {
            const uint failure_bit = missing_weights ? 1u : 4u;
            atomic_fetch_or_explicit(status, failure_bit,
                                     memory_order_relaxed);
        }
        pf_moe_zero_output_tile(eh, params.rows * kPfMoeRoutesPerRow, a0,
                                a1, group.x, tid);
        return;
    }

    threadgroup uchar staging[16384];
    threadgroup half* sx =
        reinterpret_cast<threadgroup half*>(staging);
    threadgroup half* sw =
        reinterpret_cast<threadgroup half*>(staging + 8192u);
    threadgroup float* so =
        reinterpret_cast<threadgroup float*>(staging);
    simdgroup_float8x8 acc[2][8];
    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 8u; ++c) {
            acc[i][c] = simdgroup_float8x8(0.0f);
        }
    }

    const uint row_base = 16u * uint(sg);
    for (uint k0 = 0u; k0 < 2560u; k0 += 32u) {
        for (uint i = tid; i < 64u * 32u; i += 128u) {
            const uint tile_row = i / 32u;
            const uint tile_k = i % 32u;
            const uint entry = min(a0 + tile_row, a1 - 1u);
            sx[i] = Xs[ulong(entry) * 2560ul + ulong(k0 + tile_k)];
        }
        for (uint i = tid; i < 32u * 64u; i += 128u) {
            const uint tile_k = i / 64u;
            const uint tile_column = i % 64u;
            const uint output_column =
                group.x * 32u + (tile_column % 32u);
            const uint role = tile_column < 32u ? 0u : 1u;
            if (weight_format == 0u) {
                const device block_q2_0* row_weights =
                    reinterpret_cast<device const block_q2_0*>(
                        address.role[role]) +
                    output_column * 40u;
                sw[i] = pf_moe_q2_0_value(row_weights, k0 + tile_k);
            } else if (weight_format == 1u) {
                const device block_iq4_xs* row_weights =
                    reinterpret_cast<device const block_iq4_xs*>(
                        address.role[role]) +
                    output_column * 10u;
                sw[i] = pf_moe_iq4_xs_value(row_weights, k0 + tile_k);
            } else if (weight_format == 2u) {
                const device block_iq2_xxs* row_weights =
                    reinterpret_cast<device const block_iq2_xxs*>(
                        address.role[role]) +
                    output_column * 10u;
                sw[i] = pf_moe_iq2_xxs_value(row_weights, k0 + tile_k);
            } else if (weight_format == 3u) {
                const device block_iq2_xs* row_weights =
                    reinterpret_cast<device const block_iq2_xs*>(
                        address.role[role]) +
                    output_column * 10u;
                sw[i] = pf_moe_iq2_xs_value(row_weights, k0 + tile_k);
            } else if (weight_format == 4u) {
                const device block_iq2_s* row_weights =
                    reinterpret_cast<device const block_iq2_s*>(
                        address.role[role]) +
                    output_column * 10u;
                sw[i] = pf_moe_iq2_s_value(row_weights, k0 + tile_k);
            } else if (weight_format == 5u) {
                const device block_iq3_xxs* row_weights =
                    reinterpret_cast<device const block_iq3_xxs*>(
                        address.role[role]) +
                    output_column * 10u;
                sw[i] = pf_moe_iq3_xxs_value(row_weights, k0 + tile_k);
            } else if (weight_format == 6u) {
                const device block_iq3_s* row_weights =
                    reinterpret_cast<device const block_iq3_s*>(
                        address.role[role]) +
                    output_column * 10u;
                sw[i] = pf_moe_iq3_s_value(row_weights, k0 + tile_k);
            } else {
                const device block_iq1_m* row_weights =
                    reinterpret_cast<device const block_iq1_m*>(
                        address.role[role]) +
                    output_column * 10u;
                sw[i] = pf_moe_iq1_m_value(row_weights, k0 + tile_k);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 x[2];
            simdgroup_half8x8 w[8];
            for (uint i = 0u; i < 2u; ++i) {
                simdgroup_load(
                    x[i], &sx[(row_base + 8u * i) * 32u + kk], 32u);
            }
            for (uint c = 0u; c < 8u; ++c) {
                simdgroup_load(w[c], &sw[kk * 64u + 8u * c], 64u);
            }
            for (uint i = 0u; i < 2u; ++i) {
                for (uint c = 0u; c < 8u; ++c) {
                    simdgroup_multiply_accumulate(acc[i][c], x[i], w[c],
                                                  acc[i][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 8u; ++c) {
            simdgroup_store(acc[i][c],
                            &so[(row_base + 8u * i) * 64u + 8u * c], 64u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 64u * 64u; i += 128u) {
        const uint row = i / 64u;
        const uint column = i % 64u;
        const uint entry = a0 + row;
        if (entry < a1 && column < 32u) {
            const ulong output_index =
                ulong(entry) * 640ul +
                ulong(group.x * 32u + column);
            eh[output_index] =
                half_sat(silu_k(so[i]) * so[i + 32u]);
        }
    }
}

// A 64-entry x 64-output-column SIMD-group matrix tile for F16-activation
// down projection. Inputs stay grouped by expert; dst restores route order.
[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_moe_down_f16(
    device const PrefillMoeExpertAddr* addresses [[buffer(0)]],
    device const half* eh [[buffer(1)]],
    device const PfPlan* plans [[buffer(2)]],
    device const uint2* tiles [[buffer(3)]],
    device const uint* destinations [[buffer(4)]],
    device float* parts [[buffer(5)]],
    constant PrefillMoeTileParams& params [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],
    constant uchar* tiered_lease_mask [[buffer(8)]],
    constant uint& weight_format [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    if (!pf_moe_rows_valid(params.rows) ||
        (params.n_expert != 256u && params.n_expert != 512u)) {
        return;
    }
    const uint total_entries = params.rows * kPfMoeRoutesPerRow;
    const uint tile_index = group.y;
    if (tile_index >= params.tile_capacity) return;

    const device PfPlan* plan = &plans[0];
    const uint status_bits =
        atomic_load_explicit(status, memory_order_relaxed);
    if ((status_bits & 2u) != 0u) {
        pf_moe_zero_down_tile(parts, total_entries, tile_index * 64u,
                              group.x, tid);
        return;
    }
    if (weight_format > 1u) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        pf_moe_zero_down_tile(parts, total_entries, tile_index * 64u,
                              group.x, tid);
        return;
    }
    if (tile_index >= plan->n_tiles) return;

    const uint2 tile = tiles[tile_index];
    const uint group_index = tile.x;
    const uint first_entry = tile.y;
    if (group_index >= plan->n_groups) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        pf_moe_zero_down_tile(parts, total_entries, tile_index * 64u,
                              group.x, tid);
        return;
    }
    const int group_first_signed = plan->grp_start[group_index];
    const int group_end_signed = plan->grp_start[group_index + 1u];
    const int expert_signed = plan->grp_e[group_index];
    if (group_first_signed < 0 ||
        group_end_signed <= group_first_signed ||
        uint(group_end_signed) > plan->n_entries ||
        expert_signed < 0 || uint(expert_signed) >= params.n_expert) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        pf_moe_zero_down_tile(parts, total_entries, tile_index * 64u,
                              group.x, tid);
        return;
    }
    const uint group_first = uint(group_first_signed);
    const uint group_end = uint(group_end_signed);
    if (first_entry < group_first || first_entry >= group_end ||
        (first_entry - group_first) % 64u != 0u) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        pf_moe_zero_down_tile(parts, total_entries, tile_index * 64u,
                              group.x, tid);
        return;
    }
    const uint end_entry = min(first_entry + 64u, group_end);
    const uint expert = uint(expert_signed);
    const PrefillMoeExpertAddr address = addresses[expert];
    const bool missing_weights = address.role[2] == nullptr;
    const bool missing_tiered_lease =
        !missing_weights && tiered_lease_mask[expert] == 0u;
    if (missing_weights || missing_tiered_lease) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(status,
                                     missing_weights ? 1u : 4u,
                                     memory_order_relaxed);
        }
        pf_moe_zero_down_group_tile(
            parts, destinations, total_entries, first_entry, end_entry,
            group.x, tid, status);
        return;
    }

    threadgroup uchar staging[16384];
    threadgroup half* sx =
        reinterpret_cast<threadgroup half*>(staging);
    threadgroup half* sw =
        reinterpret_cast<threadgroup half*>(staging + 8192u);
    threadgroup float* so =
        reinterpret_cast<threadgroup float*>(staging);
    simdgroup_float8x8 acc[2][8];
    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 8u; ++c) {
            acc[i][c] = simdgroup_float8x8(0.0f);
        }
    }

    const uint row_base = 16u * uint(sg);
    for (uint k0 = 0u; k0 < 640u; k0 += 32u) {
        for (uint i = tid; i < 64u * 32u; i += 128u) {
            const uint tile_row = i / 32u;
            const uint tile_k = i % 32u;
            const uint entry = min(first_entry + tile_row, end_entry - 1u);
            sx[i] = eh[ulong(entry) * 640ul + ulong(k0 + tile_k)];
        }
        for (uint i = tid; i < 32u * 64u; i += 128u) {
            const uint tile_k = i / 64u;
            const uint tile_column = i % 64u;
            const uint output_row = group.x * 64u + tile_column;
            if (weight_format == 0u) {
                const device block_q2_0* row_weights =
                    reinterpret_cast<device const block_q2_0*>(
                        address.role[2]) +
                    output_row * 10u;
                sw[i] = pf_moe_q2_0_value(row_weights, k0 + tile_k);
            } else {
                const device block_iq4_nl* row_weights =
                    reinterpret_cast<device const block_iq4_nl*>(
                        address.role[2]) +
                    output_row * 20u;
                sw[i] = pf_moe_iq4_nl_value(row_weights, k0 + tile_k);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 x[2];
            simdgroup_half8x8 w[8];
            for (uint i = 0u; i < 2u; ++i) {
                simdgroup_load(
                    x[i], &sx[(row_base + 8u * i) * 32u + kk], 32u);
            }
            for (uint c = 0u; c < 8u; ++c) {
                simdgroup_load(w[c], &sw[kk * 64u + 8u * c], 64u);
            }
            for (uint i = 0u; i < 2u; ++i) {
                for (uint c = 0u; c < 8u; ++c) {
                    simdgroup_multiply_accumulate(acc[i][c], x[i], w[c],
                                                  acc[i][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 8u; ++c) {
            simdgroup_store(acc[i][c],
                            &so[(row_base + 8u * i) * 64u + 8u * c], 64u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 64u * 64u; i += 128u) {
        const uint row = i / 64u;
        const uint column = i % 64u;
        const uint entry = first_entry + row;
        if (entry < end_entry) {
            const uint destination = destinations[entry];
            if (destination < total_entries) {
                parts[ulong(destination) * 2560ul +
                      ulong(group.x * 64u + column)] = so[i];
            } else {
                atomic_fetch_or_explicit(status, 2u,
                                         memory_order_relaxed);
                parts[ulong(entry) * 2560ul +
                      ulong(group.x * 64u + column)] = 0.0f;
            }
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_swiglu_pair(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (!pf_moe_rows_valid(rows)) return;
    const ulong index = ulong(group.x) * 256ul + ulong(tid);
    const ulong values = ulong(rows) * 640ul;
    if (index < values) {
        output[index] = half_sat(silu_k(gate[index]) * up[index]);
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_pf_combine(
    device const float* parts [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* shared [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint row = group.y;
    const uint column = group.x * 256u + tid;
    const ulong row_index = ulong(row);
    const ulong value_index = row_index * 2560ul + ulong(column);
    float sum = 0.0f;
    for (uint rank = 0u; rank < 10u; ++rank) {
        const ulong route_index = row_index * 10ul + ulong(rank);
        sum = fma(weights[route_index],
                  parts[route_index * 2560ul + ulong(column)], sum);
    }
    output[value_index] =
        fma(shared[value_index], sigmoid_k(gate[row_index]), sum);
}
