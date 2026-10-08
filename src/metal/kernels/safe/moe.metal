// Fused native MoE router (docs/mac/04-metal-kernels.md §5.10).
#include "math.h"
#include "dot.h"
#include "reduce.h"
#include "types.h"
#include "codebooks.h"

using namespace metal;

static inline float moe_route_weight(float selected, float selected_sum,
                                     float inverse_selected_sum,
                                     uint divide_weights) {
    return divide_weights != 0u ? selected / selected_sum
                                : selected * inverse_selected_sum;
}

#define STRATA_MOE_ROUTE(NV, NAME, EXTRA_PARAMS, MODE_INPUT, PROBE_MODE)                \
    [[max_total_threads_per_threadgroup(32)]]                                          \
    kernel void NAME(                                                                 \
        device const float* logits [[buffer(0)]],                                     \
        device int* ids [[buffer(1)]],                                                \
        device float* weights [[buffer(2)]],                                          \
        constant uint& divide_weights [[buffer(3)]],                                 \
        uint lane [[thread_index_in_simdgroup]],                                      \
        uint3 group [[threadgroup_position_in_grid]] EXTRA_PARAMS) {                  \
        constexpr uint vector_count = NV;                                             \
        constexpr uint expert_count = 32u * vector_count;                             \
        const uint row = group.x;                                                     \
        const device float* row_logits = logits + row * expert_count;                \
        device int* row_ids = ids + row * 10u;                                        \
        device float* row_weights = weights + row * 10u;                              \
        float values[vector_count];                                                   \
        float maximum = -INFINITY;                                                    \
        for (uint i = 0u; i < vector_count; ++i) {                                    \
            values[i] = row_logits[lane + 32u * i];                                  \
            maximum = fmax(maximum, values[i]);                                      \
        }                                                                             \
        maximum = bfly_max(maximum);                                                  \
        float sum = 0.0f;                                                             \
        for (uint i = 0u; i < vector_count; ++i) {                                   \
            values[i] = exp(values[i] - maximum);                                    \
            sum = sum + values[i];                                                   \
        }                                                                             \
        const float reciprocal = 1.0f / bfly_sum(sum);                               \
        for (uint i = 0u; i < vector_count; ++i) {                                   \
            values[i] = values[i] * reciprocal;                                      \
            if (isnan(values[i])) values[i] = -0x1.fffffep127f; /* -FLT_MAX */       \
        }                                                                             \
        float selected = 0.0f;                                                       \
        float selected_sum = 0.0f;                                                   \
        for (uint rank = 0u; rank < 10u; ++rank) {                                   \
            float best = values[0];                                                  \
            int expert = int(lane);                                                  \
            for (uint i = 1u; i < vector_count; ++i) {                               \
                if (values[i] > best) {                                              \
                    best = values[i];                                                \
                    expert = int(lane + 32u * i);                                    \
                }                                                                     \
            }                                                                         \
            for (ushort offset = 16u; offset > 0u; offset >>= 1u) {                   \
                const float other = simd_shuffle_xor(best, offset);                  \
                const int other_expert = simd_shuffle_xor(expert, offset);            \
                if (other > best ||                                                  \
                    (other == best && other_expert < expert)) {                      \
                    best = other;                                                    \
                    expert = other_expert;                                            \
                }                                                                     \
            }                                                                         \
            if ((uint(expert) & 31u) == lane) {                                       \
                values[uint(expert) / 32u] = -INFINITY;                              \
                row_ids[rank] = expert;                                               \
                selected_sum = selected_sum + best;                                  \
            }                                                                         \
            if (rank == lane) selected = best;                                        \
        }                                                                             \
        if (PROBE_MODE != 0u) {                                                       \
            selected = MODE_INPUT[0];                                                \
            selected_sum = MODE_INPUT[1];                                            \
        } else {                                                                      \
            selected_sum = fmax(bfly_sum(selected_sum), 6.103515625e-5f);            \
        }                                                                             \
        const float inverse_selected_sum = 1.0f / selected_sum;                      \
        if (lane < 10u) {                                                             \
            row_weights[lane] = moe_route_weight(                                    \
                selected, selected_sum, inverse_selected_sum, divide_weights);       \
        }                                                                             \
    }

#define STRATA_MOE_ROUTE_NO_EXTRA
#define STRATA_MOE_ROUTE_PROBE_EXTRA , device const float* mode_values [[buffer(4)]]

STRATA_MOE_ROUTE(16, strata_moe_route_r16, STRATA_MOE_ROUTE_NO_EXTRA, logits, 0u)
STRATA_MOE_ROUTE(8, strata_moe_route_r8, STRATA_MOE_ROUTE_NO_EXTRA, logits, 0u)
STRATA_MOE_ROUTE(16, strata_moe_route_weight_mode_probe,                            \
                 STRATA_MOE_ROUTE_PROBE_EXTRA, mode_values, 1u)

#undef STRATA_MOE_ROUTE
#undef STRATA_MOE_ROUTE_NO_EXTRA
#undef STRATA_MOE_ROUTE_PROBE_EXTRA

struct MoePlan {
    uint n_groups;
    uint n_entries;
    uint n_miss;
    uint skip;
    int grp_e[80];
    int grp_res[80];
    int grp_start[81];
    int ent_tok[80];
    int ent_dst[80];
};

struct MoePlanParams {
    uint rows;
    uint n_expert;
    uint tiered;
    uint layer;
};

static_assert(sizeof(MoePlan) == 1620, "MoePlan host ABI");
static_assert(sizeof(MoePlanParams) == 16, "MoePlanParams host ABI");

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_moe_plan(
    device const int* ids_flat [[buffer(0)]],
    device const int* resident [[buffer(1)]],
    device float* usage [[buffer(2)]],
    device MoePlan* plans [[buffer(3)]],
    constant MoePlanParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int e_s[128];
    threadgroup int first_s[128];
    threadgroup int size_s[128];
    threadgroup int gidx_s[128];
    threadgroup int gstart_s[129];

    const uint n = params.rows * 10u;
    const int expert = tid < n ? ids_flat[tid] : -1;
    e_s[tid] = expert;
    if (tid < 80u) {
        plans[params.layer].grp_e[tid] = 0;
        plans[params.layer].grp_res[tid] = 0;
        plans[params.layer].ent_tok[tid] = 0;
        plans[params.layer].ent_dst[tid] = 0;
    }
    if (tid < 81u) {
        plans[params.layer].grp_start[tid] = 0;
    }
    if (tid == 0u) {
        plans[params.layer].n_groups = 0u;
        plans[params.layer].n_entries = 0u;
        plans[params.layer].n_miss = 0u;
        plans[params.layer].skip = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    int first = int(tid);
    int rank = 0;
    int size = 0;
    if (tid < n) {
        for (uint j = 0u; j < tid; ++j) {
            if (e_s[j] == expert) {
                if (first == int(tid)) first = int(j);
                ++rank;
            }
        }
        if (first == int(tid)) {
            for (uint j = tid; j < n; ++j) {
                size += e_s[j] == expert;
            }
        }
    }
    first_s[tid] = first;
    size_s[tid] = tid < n && first == int(tid) ? size : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint group = 0u;
        uint entry = 0u;
        uint misses = 0u;
        for (uint j = 0u; j < n; ++j) {
            if (first_s[j] == int(j)) {
                gidx_s[j] = int(group);
                gstart_s[group] = int(entry);
                plans[params.layer].grp_e[group] = e_s[j];
                plans[params.layer].grp_start[group] = int(entry);
                const bool is_resident =
                    params.tiered == 0u ||
                    resident[params.layer * params.n_expert +
                             uint(e_s[j])] >= 0;
                plans[params.layer].grp_res[group] = is_resident ? 1 : 0;
                misses += is_resident ? 0u : 1u;
                entry += uint(size_s[j]);
                ++group;
            }
        }
        gstart_s[group] = int(entry);
        plans[params.layer].grp_start[group] = int(entry);
        plans[params.layer].n_groups = group;
        plans[params.layer].n_entries = n;
        plans[params.layer].n_miss = misses;
        plans[params.layer].skip = misses == 0u ? 1u : 0u;
        if (params.tiered != 0u) {
            const uint usage_offset = params.layer * params.n_expert;
            for (uint j = 0u; j < n; ++j) {
                usage[usage_offset + uint(e_s[j])] += 1.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < n) {
        const uint group = uint(gidx_s[first_s[tid]]);
        const uint destination =
            uint(gstart_s[group] + rank);
        plans[params.layer].ent_dst[destination] = int(tid);
        plans[params.layer].ent_tok[destination] = int(tid / 10u);
    }
}

static inline void moe_swiglu_entry(device const float* gate,
                                    device const float* up,
                                    device float* output, uint entry,
                                    uint column) {
    if (column < 640u) {
        const uint index = entry * 640u + column;
        output[index] = silu_k(gate[index]) * up[index];
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_swiglu(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    moe_swiglu_entry(gate, up, output, group.y,
                     group.x * 256u + tid.x);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_shx_swiglu(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    moe_swiglu_entry(gate, up, output, group.y,
                     group.x * 256u + tid.x);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_combine(
    device const float* parts [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* shared [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint row = group.y;
    const uint column = group.x * 256u + tid.x;
    const device float* row_parts = parts + row * 10u * 2560u;
    const device float* row_weights = weights + row * 10u;
    float sum = row_parts[column] * row_weights[0];
    for (uint rank = 1u; rank < 10u; ++rank) {
        sum = fma(row_parts[rank * 2560u + column], row_weights[rank],
                  sum);
    }
    const float gate_scale = sigmoid_k(gate[row]);
    const float shared_term =
        shared[row * 2560u + column] * gate_scale;
    output[row * 2560u + column] = sum + shared_term;
}

struct MoeExpertAddr {
    device const uchar* role[3];
};

struct MoeGroupedParams {
    uint n_expert;
    uint layer;
    uint n_rows;
};

struct MoeGroupedGateUpParams {
    uint n_expert;
    uint layer;
    uint n_rows;
};

static_assert(sizeof(MoeExpertAddr) == 24, "MoeExpertAddr host ABI");
static_assert(sizeof(MoeGroupedParams) == 12, "MoeGroupedParams host ABI");
static_assert(sizeof(MoeGroupedGateUpParams) == 12,
              "MoeGroupedGateUpParams host ABI");

static inline bool moe_group_header_valid(
    device const MoePlan* plan, uint n_rows, uint n_expert,
    uint group_index) {
    if (n_rows == 0u || n_rows > 8u ||
        plan->n_groups == 0u || plan->n_groups > 80u ||
        plan->n_entries != n_rows * 10u ||
        plan->n_entries > 80u || group_index >= plan->n_groups) {
        return false;
    }
    const int entry_begin = plan->grp_start[group_index];
    const int entry_end = plan->grp_start[group_index + 1u];
    const int expert = plan->grp_e[group_index];
    return entry_begin >= 0 && entry_end > entry_begin &&
           uint(entry_end) <= plan->n_entries && expert >= 0 &&
           uint(expert) < n_expert;
}

static inline uint moe_group_entry_tokens_valid(
    device const MoePlan* plan, uint entry_begin, uint entry_end,
    uint n_rows, uint lane) {
    if (lane != 0u) return 1u;
    for (uint entry = entry_begin; entry < entry_end; ++entry) {
        const int token = plan->ent_tok[entry];
        if (token < 0 || uint(token) >= n_rows) return 0u;
    }
    return 1u;
}

static inline uint moe_group_entry_destinations_valid(
    device const MoePlan* plan, uint entry_begin, uint entry_end,
    uint n_entries, uint lane) {
    if (lane != 0u) return 1u;
    for (uint entry = entry_begin; entry < entry_end; ++entry) {
        const int destination = plan->ent_dst[entry];
        if (destination < 0 || uint(destination) >= n_entries) return 0u;
    }
    return 1u;
}

struct MoeQ2Part {
    int qx[4];
    int qy[4];
    float d2;
};

static inline int moe_q2_0_pack_codes(uint packed) {
    const char4 codes = char4(
        char(int(packed & 0x03u) - 1),
        char(int((packed >> 2u) & 0x03u) - 1),
        char(int((packed >> 4u) & 0x03u) - 1),
        char(int((packed >> 6u) & 0x03u) - 1));
    return as_type<int>(codes);
}

static inline MoeQ2Part moe_q2_0_load(device const block_q2_0* weight,
                                     uint qs_half) {
    const device ushort* qs =
        reinterpret_cast<device const ushort*>(weight->qs) + qs_half * 4u;
    MoeQ2Part result;
    result.d2 = float(weight->d);
    for (uint word = 0u; word < 4u; ++word) {
        const uint packed = uint(qs[word]);
        result.qx[word] = moe_q2_0_pack_codes(packed & 0xffu);
        result.qy[word] = moe_q2_0_pack_codes(packed >> 8u);
    }
    return result;
}

static inline float moe_q2_0_apply(MoeQ2Part weight,
                                   device const Q81* activation) {
    const device int* qs =
        reinterpret_cast<device const int*>(activation->qs);
    int sum = 0;
    for (uint word = 0u; word < 4u; ++word) {
        sum += dot4_i8(weight.qx[word], qs[2u * word]);
        sum += dot4_i8(weight.qy[word], qs[2u * word + 1u]);
    }
    return weight.d2 * float(activation->ds.x) * float(sum);
}

static inline int moe_q2_0_integer_dot(MoeQ2Part weight,
                                      device const Q81* activation) {
    const device int* qs =
        reinterpret_cast<device const int*>(activation->qs);
    int sum = 0;
    for (uint word = 0u; word < 4u; ++word) {
        sum += dot4_i8(weight.qx[word], qs[2u * word]);
        sum += dot4_i8(weight.qy[word], qs[2u * word + 1u]);
    }
    return sum;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_q2_0(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_q2_0* row_weights =
        reinterpret_cast<device const block_q2_0*>(address.role[role]) +
        expert_row * 40u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 2u;
            const uint qs_half = k % 2u;
            const MoeQ2Part weight = moe_q2_0_load(
                row_weights + block, qs_half);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 2u + qs_half;
                sums[column] += moe_q2_0_apply(weight, activation);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_down_q2_0(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* ehq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* parts [[buffer(3)]],
    constant MoeGroupedParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                                group_index)) {
        return;
    }

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    const uint row = group.x * 8u + warp;
    if (row >= 2560u) return;
    const uint entry_begin = uint(plan->grp_start[group_index]);
    const uint entry_end = uint(plan->grp_start[group_index + 1u]);
    if (simd_broadcast_first(moe_group_entry_destinations_valid(
            plan, entry_begin, entry_end, plan->n_entries, lane)) == 0u) {
        return;
    }
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert +
                  uint(plan->grp_e[group_index])];
    if (address.role[2] == nullptr) return;
    const device block_q2_0* row_weights =
        reinterpret_cast<device const block_q2_0*>(address.role[2]) +
        row * 10u;

    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        uint destinations[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] = entry * 20u;
            destinations[column] = uint(plan->ent_dst[entry]) * 2560u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 20u; k += 32u) {
            const uint block = k / 2u;
            const uint qs_half = k % 2u;
            const MoeQ2Part weight =
                moe_q2_0_load(row_weights + block, qs_half);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    ehq + activation_offsets[column] + block * 2u + qs_half;
                const int dot = moe_q2_0_integer_dot(weight, activation);
                sums[column] =
                    fma(weight.d2 * float(activation->ds.x),
                        float(dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                parts[destinations[column] + row] = sums[column];
            }
        }
    }
}

static inline int moe_iq4_nl_integer_dot(
    device const block_iq4_nl* weight, uint first_word,
    device const Q81* activation) {
    int sum = 0;
    for (uint word = 0u; word < 2u; ++word) {
        const uint offset = 4u * (first_word + word);
        for (uint component = 0u; component < 4u; ++component) {
            const uint packed = uint(weight->qs[offset + component]);
            const int low = int(kvalues_iq4nl[packed & 0x0fu]);
            const int high = int(kvalues_iq4nl[packed >> 4u]);
            sum += low * int(activation->qs[offset + component]);
            sum += high * int(activation->qs[16u + offset + component]);
        }
    }
    return sum;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_down_iq4_nl(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* ehq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* parts [[buffer(3)]],
    constant MoeGroupedParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                                group_index)) {
        return;
    }

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    const uint row = group.x * 8u + warp;
    if (row >= 2560u) return;
    const uint entry_begin = uint(plan->grp_start[group_index]);
    const uint entry_end = uint(plan->grp_start[group_index + 1u]);
    if (simd_broadcast_first(moe_group_entry_destinations_valid(
            plan, entry_begin, entry_end, plan->n_entries, lane)) == 0u) {
        return;
    }
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert +
                  uint(plan->grp_e[group_index])];
    if (address.role[2] == nullptr) return;
    const device block_iq4_nl* row_weights =
        reinterpret_cast<device const block_iq4_nl*>(address.role[2]) +
        row * 20u;

    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        uint destinations[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] = entry * 20u;
            destinations[column] = uint(plan->ent_dst[entry]) * 2560u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 40u; k += 32u) {
            const uint block = k / 2u;
            const uint part = k % 2u;
            const device block_iq4_nl* weight = row_weights + block;
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    ehq + activation_offsets[column] + block;
                const int dot = moe_iq4_nl_integer_dot(
                    weight, 2u * part, activation);
                sums[column] =
                    fma(float(weight->d) * float(activation->ds.x),
                        float(dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                parts[destinations[column] + row] = sums[column];
            }
        }
    }
}

static inline uint moe_iq1_m_half_scale_bits(
    device const block_iq1_m* weight) {
    uint result = 0u;
    for (uint word = 0u; word < 4u; ++word) {
        const uint byte_offset = 2u * word;
        const uint packed = uint(weight->scales[byte_offset]) |
                            (uint(weight->scales[byte_offset + 1u]) << 8u);
        result |= ((packed >> 12u) & 0x0fu) << (4u * word);
    }
    return result;
}

static inline uint moe_iq1_m_packed_scale(
    device const block_iq1_m* weight, uint part) {
    const uint word_index = part / 2u;
    const uint byte_offset = 2u * word_index;
    const uint packed = uint(weight->scales[byte_offset]) |
                        (uint(weight->scales[byte_offset + 1u]) << 8u);
    return packed >> (6u * (part & 1u));
}

static inline float moe_iq1_m_dot(device const block_iq1_m* weight,
                                  uint part,
                                  device const Q81* activation) {
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};

    for (uint codeword = 0u; codeword < 4u; ++codeword) {
        const uint qh_byte = uint(weight->qh[2u * part + codeword / 2u]);
        const uint qhl =
            (qh_byte >> (4u * (codeword & 1u))) & 0x0fu;
        const uint grid_index =
            uint(weight->qs[4u * part + codeword]) | ((qhl & 0x07u) << 8u);
        const uint grid = iq1s_grid_gpu[grid_index];
        const int grid0 = int((grid >> 0u) & 0x0f0f0f0fu);
        const int grid1 = int((grid >> 4u) & 0x0f0f0f0fu);
        const uint word = 2u * codeword;
        const int u0 = activation_words[word];
        const int u1 = activation_words[word + 1u];
        const uint partial = codeword / 2u;
        sumi[partial] = dot4_i8(grid0, u0) + sumi[partial];
        sumi[partial] = dot4_i8(grid1, u1) + sumi[partial];

        const int sumy = dot4_i8(u0, 0x01010101) +
                         dot4_i8(u1, 0x01010101);
        const float delta = (qhl & 0x08u) != 0u ? -1.125f : -0.875f;
        sumf[partial] = fma(delta, float(sumy), sumf[partial]);
    }

    const uint packed_scale = moe_iq1_m_packed_scale(weight, part);
    const int scale0 = 2 * int((packed_scale >> 0u) & 0x07u) + 1;
    const int scale1 = 2 * int((packed_scale >> 3u) & 0x07u) + 1;
    const float x0 = (float(sumi[0]) + sumf[0]) * float(scale0);
    const float x1 = (float(sumi[1]) + sumf[1]) * float(scale1);
    return x0 + x1;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_iq1_m(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_iq1_m* row_weights =
        reinterpret_cast<device const block_iq1_m*>(address.role[role]) +
        expert_row * 10u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 8u;
            const uint part = k % 8u;
            const device block_iq1_m* weight = row_weights + block;
            const uint half_scale_bits = moe_iq1_m_half_scale_bits(weight);
            const float weight_scale =
                float(as_type<half>(ushort(half_scale_bits)));
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 8u + part;
                const float dot = moe_iq1_m_dot(weight, part, activation);
                sums[column] =
                    fma(weight_scale * float(activation->ds.x), dot,
                        sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}

static inline char4 moe_iq2_xxs_signed_quartet(uint packed, uint signs,
                                               uint sign_shift) {
    const char4 magnitude = as_type<char4>(packed);
    return char4(
        (signs & (1u << (sign_shift + 0u))) != 0u
            ? -int(magnitude.x) : int(magnitude.x),
        (signs & (1u << (sign_shift + 1u))) != 0u
            ? -int(magnitude.y) : int(magnitude.y),
        (signs & (1u << (sign_shift + 2u))) != 0u
            ? -int(magnitude.z) : int(magnitude.z),
        (signs & (1u << (sign_shift + 3u))) != 0u
            ? -int(magnitude.w) : int(magnitude.w));
}

static inline int moe_iq2_xxs_integer_dot(
    device const block_iq2_xxs* weight, uint part,
    device const Q81* activation) {
    const device ushort* packed_words = weight->qs + 4u * part;
    const uint indices = uint(packed_words[0]) |
                         (uint(packed_words[1]) << 16u);
    const uint aux32 = uint(packed_words[2]) |
                       (uint(packed_words[3]) << 16u);
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    int sumi = 0;
    for (uint code_slot = 0u; code_slot < 4u; ++code_slot) {
        const uint code = (indices >> (8u * code_slot)) & 0xffu;
        const uint raw_signs = (aux32 >> (7u * code_slot)) & 0xffu;
        const uint signs =
            raw_signs ^ ((popcount(raw_signs) & 1u) << 7u);
        const ulong grid = iq2xxs_grid[code];
        const char4 first =
            moe_iq2_xxs_signed_quartet(uint(grid), signs, 0u);
        const char4 second =
            moe_iq2_xxs_signed_quartet(uint(grid >> 32u), signs, 4u);
        sumi = dot4_i8(as_type<int>(first),
                       activation_words[2u * code_slot]) + sumi;
        sumi = dot4_i8(as_type<int>(second),
                       activation_words[2u * code_slot + 1u]) + sumi;
    }
    const int odd_scale = int((aux32 >> 27u) | 1u);
    return (sumi * odd_scale) / 8;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_iq2_xxs(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_iq2_xxs* row_weights =
        reinterpret_cast<device const block_iq2_xxs*>(address.role[role]) +
        expert_row * 10u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 8u;
            const uint part = k % 8u;
            const device block_iq2_xxs* weight = row_weights + block;
            const float weight_scale = float(weight->d);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 8u + part;
                const int dot =
                    moe_iq2_xxs_integer_dot(weight, part, activation);
                sums[column] =
                    fma(weight_scale * float(activation->ds.x),
                        float(dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}

static inline char4 moe_iq2_xs_signed_quartet(uint packed, uint signs,
                                               uint component) {
    const char4 magnitude = as_type<char4>(packed);
    return char4(
        (signs & uint(kmask_iq2xs[component + 0u])) != 0u
            ? -int(magnitude.x) : int(magnitude.x),
        (signs & uint(kmask_iq2xs[component + 1u])) != 0u
            ? -int(magnitude.y) : int(magnitude.y),
        (signs & uint(kmask_iq2xs[component + 2u])) != 0u
            ? -int(magnitude.z) : int(magnitude.z),
        (signs & uint(kmask_iq2xs[component + 3u])) != 0u
            ? -int(magnitude.w) : int(magnitude.w));
}

static inline int moe_iq2_xs_integer_dot(
    device const block_iq2_xs* weight, uint part,
    device const Q81* activation) {
    const device ushort* codes = weight->qs + 4u * part;
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    const int scale_pair = int(weight->scales[part]);
    const int ls0 = scale_pair & 0x0f;
    const int ls1 = scale_pair >> 4u;

    int sumi0 = 0;
    int sumi1 = 0;
    for (uint code_slot = 0u; code_slot < 4u; ++code_slot) {
        const uint codeword = uint(codes[code_slot]);
        const uint grid_index = codeword & 0x1ffu;
        const uint sign_index = codeword >> 9u;
        const uint signs = uint(ksigns_iq2xs[sign_index]);
        const ulong grid = iq2xs_grid[grid_index];
        const char4 low = moe_iq2_xs_signed_quartet(
            uint(grid), signs, 0u);
        const char4 high = moe_iq2_xs_signed_quartet(
            uint(grid >> 32u), signs, 4u);
        int partial = code_slot < 2u ? sumi0 : sumi1;
        partial = dot4_i8(as_type<int>(low),
                          activation_words[2u * code_slot]) + partial;
        partial = dot4_i8(as_type<int>(high),
                          activation_words[2u * code_slot + 1u]) + partial;
        if (code_slot < 2u) {
            sumi0 = partial;
        } else {
            sumi1 = partial;
        }
    }
    return (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_iq2_xs(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_iq2_xs* row_weights =
        reinterpret_cast<device const block_iq2_xs*>(address.role[role]) +
        expert_row * 10u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 8u;
            const uint part = k % 8u;
            const device block_iq2_xs* weight = row_weights + block;
            const float weight_scale = float(weight->d);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 8u + part;
                const int dot =
                    moe_iq2_xs_integer_dot(weight, part, activation);
                sums[column] =
                    fma(weight_scale * float(activation->ds.x),
                        float(dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}

static inline char4 moe_iq2_s_signed_quartet(uint packed, uint signs,
                                              uint component) {
    const char4 magnitude = as_type<char4>(packed);
    return char4(
        (signs & uint(kmask_iq2xs[component + 0u])) != 0u
            ? -int(magnitude.x) : int(magnitude.x),
        (signs & uint(kmask_iq2xs[component + 1u])) != 0u
            ? -int(magnitude.y) : int(magnitude.y),
        (signs & uint(kmask_iq2xs[component + 2u])) != 0u
            ? -int(magnitude.z) : int(magnitude.z),
        (signs & uint(kmask_iq2xs[component + 3u])) != 0u
            ? -int(magnitude.w) : int(magnitude.w));
}

static inline int moe_iq2_s_integer_dot(
    device const block_iq2_s* weight, uint part,
    device const Q81* activation) {
    const device uchar* low_codes = weight->qs + 4u * part;
    const device uchar* sign_codes = weight->qs + 32u + 4u * part;
    const uint high_codes = uint(weight->qh[part]);
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    const int scale_pair = int(weight->scales[part]);
    const int ls0 = scale_pair & 0x0f;
    const int ls1 = scale_pair >> 4u;

    int sumi0 = 0;
    int sumi1 = 0;
    for (uint code_slot = 0u; code_slot < 4u; ++code_slot) {
        const uint grid_index =
            uint(low_codes[code_slot]) |
            (((high_codes >> (2u * code_slot)) & 0x03u) << 8u);
        const uint signs = uint(sign_codes[code_slot]);
        const ulong grid = iq2s_grid[grid_index];
        const char4 low = moe_iq2_s_signed_quartet(
            uint(grid), signs, 0u);
        const char4 high = moe_iq2_s_signed_quartet(
            uint(grid >> 32u), signs, 4u);
        int partial = code_slot < 2u ? sumi0 : sumi1;
        partial = dot4_i8(as_type<int>(low),
                          activation_words[2u * code_slot]) + partial;
        partial = dot4_i8(as_type<int>(high),
                          activation_words[2u * code_slot + 1u]) + partial;
        if (code_slot < 2u) {
            sumi0 = partial;
        } else {
            sumi1 = partial;
        }
    }
    return (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_iq2_s(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_iq2_s* row_weights =
        reinterpret_cast<device const block_iq2_s*>(address.role[role]) +
        expert_row * 10u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 8u;
            const uint part = k % 8u;
            const device block_iq2_s* weight = row_weights + block;
            const float weight_scale = float(weight->d);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 8u + part;
                const int dot =
                    moe_iq2_s_integer_dot(weight, part, activation);
                sums[column] =
                    fma(weight_scale * float(activation->ds.x),
                        float(dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}

static inline uint moe_iq3_xxs_unpack_signs(uint sign_index) {
    const uint signs = sign_index & 0x7fu;
    return signs ^ ((popcount(signs) & 1u) << 7u);
}

static inline char4 moe_iq3_xxs_signed_quartet(uint packed, uint signs,
                                               uint component_base) {
    const char4 magnitude = as_type<char4>(packed);
    return char4(
        (signs & (1u << (component_base + 0u))) != 0u
            ? -int(magnitude.x) : int(magnitude.x),
        (signs & (1u << (component_base + 1u))) != 0u
            ? -int(magnitude.y) : int(magnitude.y),
        (signs & (1u << (component_base + 2u))) != 0u
            ? -int(magnitude.z) : int(magnitude.z),
        (signs & (1u << (component_base + 3u))) != 0u
            ? -int(magnitude.w) : int(magnitude.w));
}

static inline int moe_iq3_xxs_integer_dot(
    device const block_iq3_xxs* weight, uint part,
    device const Q81* activation) {
    const device uchar* indices = weight->qs + 8u * part;
    const device uchar* aux = weight->qs + 64u + 4u * part;
    const uint aux32 = uint(aux[0]) | (uint(aux[1]) << 8u) |
                       (uint(aux[2]) << 16u) | (uint(aux[3]) << 24u);
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);

    int sumi = 0;
    for (uint code_slot = 0u; code_slot < 4u; ++code_slot) {
        const uint sign_index = (aux32 >> (7u * code_slot)) & 0x7fu;
        const uint signs = moe_iq3_xxs_unpack_signs(sign_index);
        const char4 low = moe_iq3_xxs_signed_quartet(
            iq3xxs_grid[indices[2u * code_slot]], signs, 0u);
        const char4 high = moe_iq3_xxs_signed_quartet(
            iq3xxs_grid[indices[2u * code_slot + 1u]], signs, 4u);
        sumi = dot4_i8(as_type<int>(low),
                       activation_words[2u * code_slot]) + sumi;
        sumi = dot4_i8(as_type<int>(high),
                       activation_words[2u * code_slot + 1u]) + sumi;
    }
    const int scale = int(aux32 >> 28u);
    return (scale * sumi + sumi / 2) / 2;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_iq3_xxs(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_iq3_xxs* row_weights =
        reinterpret_cast<device const block_iq3_xxs*>(address.role[role]) +
        expert_row * 10u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 8u;
            const uint part = k % 8u;
            const device block_iq3_xxs* weight = row_weights + block;
            const float weight_scale = float(weight->d);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 8u + part;
                const int dot =
                    moe_iq3_xxs_integer_dot(weight, part, activation);
                sums[column] =
                    fma(weight_scale * float(activation->ds.x),
                        float(dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}

static inline char4 moe_iq3_s_signed_grid(uint index, uint signs) {
    const char4 magnitude = as_type<char4>(iq3s_grid[index]);
    return char4(
        (signs & 0x01u) != 0u ? -int(magnitude.x) : int(magnitude.x),
        (signs & 0x02u) != 0u ? -int(magnitude.y) : int(magnitude.y),
        (signs & 0x04u) != 0u ? -int(magnitude.z) : int(magnitude.z),
        (signs & 0x08u) != 0u ? -int(magnitude.w) : int(magnitude.w));
}

static inline int moe_iq3_s_integer_dot(
    device const block_iq3_s* weight, uint part,
    device const Q81* activation) {
    const uint qh = uint(weight->qh[part]);
    device const int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    int sumi = 0;
    for (uint pair = 0u; pair < 4u; ++pair) {
        const uint l0 = 2u * pair;
        const uint qs_offset = 8u * part + l0;
        const uint low_index =
            uint(weight->qs[qs_offset]) | (((qh >> l0) & 1u) << 8u);
        const uint high_index =
            uint(weight->qs[qs_offset + 1u]) |
            (((qh >> (l0 + 1u)) & 1u) << 8u);
        const uint packed_signs = uint(weight->signs[4u * part + pair]);
        const char4 low =
            moe_iq3_s_signed_grid(low_index, packed_signs & 0x0fu);
        const char4 high =
            moe_iq3_s_signed_grid(high_index, packed_signs >> 4u);
        sumi = dot4_i8(as_type<int>(low), activation_words[l0]) + sumi;
        sumi = dot4_i8(as_type<int>(high), activation_words[l0 + 1u]) + sumi;
    }
    const uint scale_byte = uint(weight->scales[part / 2u]);
    const uint scale_nibble = (scale_byte >> (4u * (part % 2u))) & 0x0fu;
    return sumi * (1 + 2 * int(scale_nibble));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_iq3_s(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_iq3_s* row_weights =
        reinterpret_cast<device const block_iq3_s*>(address.role[role]) +
        expert_row * 10u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 8u;
            const uint part = k % 8u;
            const device block_iq3_s* weight = row_weights + block;
            const float weight_scale = float(weight->d);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 8u + part;
                const int dot =
                    moe_iq3_s_integer_dot(weight, part, activation);
                sums[column] =
                    fma(weight_scale * float(activation->ds.x),
                        float(dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}

static inline int moe_iq4_xs_scale_code(
    device const block_iq4_xs* weight, uint part) {
    const uint low =
        (uint(weight->scales_l[part / 2u]) >> (4u * (part % 2u))) & 0x0fu;
    const uint high = (uint(weight->scales_h) >> (2u * part)) & 0x03u;
    return int(low | (high << 4u));
}

static inline int moe_iq4_xs_integer_dot(
    device const block_iq4_xs* weight, uint part,
    device const Q81* activation) {
    int sumi = 0;
    const uint byte_base = 16u * part;
    for (uint word = 0u; word < 4u; ++word) {
        const uint offset = byte_base + 4u * word;
        for (uint lane = 0u; lane < 4u; ++lane) {
            const uint packed = uint(weight->qs[offset + lane]);
            const int low = int(kvalues_iq4nl[packed & 0x0fu]);
            const int high = int(kvalues_iq4nl[packed >> 4u]);
            sumi += low * int(activation->qs[4u * word + lane]);
            sumi += high * int(activation->qs[16u + 4u * word + lane]);
        }
    }
    return sumi;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_moe_gu_iq4_xs(
    device const MoeExpertAddr* addresses [[buffer(0)]],
    device const Q81* xq [[buffer(1)]],
    device const MoePlan* plans [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant MoeGroupedGateUpParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    device const MoePlan* plan = plans + params.layer;
    const uint group_index = group.y;
    if (!moe_group_header_valid(plan, params.n_rows, params.n_expert,
                               group_index)) {
        return;
    }
    const int group_entry_begin = plan->grp_start[group_index];
    const int group_entry_end = plan->grp_start[group_index + 1u];
    const int group_expert = plan->grp_e[group_index];

    const uint warp = tid / 32u;
    const uint lane = tid % 32u;
    if (simd_broadcast_first(moe_group_entry_tokens_valid(
            plan, uint(group_entry_begin), uint(group_entry_end),
            params.n_rows, lane)) == 0u) {
        return;
    }
    const uint row = group.x * 8u + warp;
    const bool is_up = row >= 640u;
    const uint expert_row = is_up ? row - 640u : row;
    const uint role = is_up ? 1u : 0u;
    const MoeExpertAddr address =
        addresses[params.layer * params.n_expert + uint(group_expert)];
    if (address.role[role] == nullptr) return;
    const device block_iq4_xs* row_weights =
        reinterpret_cast<device const block_iq4_xs*>(address.role[role]) +
        expert_row * 10u;

    const uint entry_begin = uint(group_entry_begin);
    const uint entry_end = uint(group_entry_end);
    device float* destination = is_up ? up : gate;
    for (uint first = entry_begin; first < entry_end; first += 4u) {
        const uint count = min(4u, entry_end - first);
        uint activation_offsets[4];
        for (uint column = 0u; column < 4u; ++column) {
            const uint entry = first + min(column, count - 1u);
            activation_offsets[column] =
                uint(plan->ent_tok[entry]) * 80u;
        }

        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = lane; k < 80u; k += 32u) {
            const uint block = k / 8u;
            const uint part = k % 8u;
            const device block_iq4_xs* weight = row_weights + block;
            const int scale = moe_iq4_xs_scale_code(weight, part) - 32;
            const float weight_scale = float(weight->d);
            for (uint column = 0u; column < count; ++column) {
                const device Q81* activation =
                    xq + activation_offsets[column] + block * 8u + part;
                const int dot =
                    moe_iq4_xs_integer_dot(weight, part, activation);
                const int scaled_dot = dot * scale;
                sums[column] =
                    fma(weight_scale * float(activation->ds.x),
                        float(scaled_dot), sums[column]);
            }
        }
        for (uint column = 0u; column < count; ++column) {
            sums[column] = bfly_sum(sums[column]);
            if (lane == column) {
                destination[(first + column) * 640u + expert_row] =
                    sums[column];
            }
        }
    }
}
