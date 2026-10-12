// Prefill MoE route grouping and F16 row gather (docs/mac/04 §6.6).
#include <metal_stdlib>

using namespace metal;

struct PrefillMoeGroupingParams {
    uint rows;
    uint n_expert;
    uint layer;
    uint tiered;
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
static_assert(sizeof(PfPlan) == 6180, "PfPlan host ABI");

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
                const ulong resident_index =
                    ulong(params.layer) * ulong(params.n_expert) +
                    ulong(expert);
                const bool is_resident =
                    params.tiered == 0u || resident[resident_index] >= 0;
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

    // The specified dispatch is sized in half elements. One active lane copies
    // one uint4 (eight halves), so seven of each eight lanes are idle.
    const ulong dispatch_thread =
        ulong(group.x) * 256ul + ulong(tid);
    if ((dispatch_thread & 7ul) != 0ul) return;
    const ulong vector_index = dispatch_thread >> 3u;
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
