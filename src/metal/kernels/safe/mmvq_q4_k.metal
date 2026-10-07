// Native Q4_K weights multiplied by Q8_1 activations.
#include "types.h"
#include "reduce.h"

using namespace metal;

struct Q4MVParams {
    uint n_in;
    uint n_out;
};

static inline uint2 q4_k_scale_min(device const block_q4_K* weight, uint group) {
    if (group < 4u) {
        return uint2(uint(weight->scales[group] & 0x3fu),
                     uint(weight->scales[group + 4u] & 0x3fu));
    }
    return uint2(
        uint(weight->scales[group + 4u] & 0x0fu) |
            (uint(weight->scales[group - 4u] >> 6u) << 4u),
        uint(weight->scales[group + 4u] >> 4u) |
            (uint(weight->scales[group] >> 6u) << 4u));
}

static inline float q4_k_accumulate(device const block_q4_K* weight,
                                    device const Q81* xq,
                                    uint blocks_per_row, uint kbx,
                                    uint iqs, uint input_column,
                                    float accumulator) {
    const uint bq8_offset = 2u * ((iqs / 2u) / 4u);
    const uint q8_word = (iqs / 2u) % 4u;
    float scaled_dot = 0.0f;
    float scaled_min = 0.0f;
    for (uint pair = 0; pair < 2u; ++pair) {
        const uint group = bq8_offset + pair;
        const device Q81* activation =
            xq + input_column * (8u * blocks_per_row) + 8u * kbx + group;
        const uint2 scale_min = q4_k_scale_min(weight, group);
        const uint qbyte_base = 32u * (group / 2u) + 4u * q8_word;
        const uint nibble_shift = 4u * (group % 2u);
        int dot = 0;
        int sum_q8 = 0;
        for (uint segment = 0; segment < 2u; ++segment) {
            const uint offset = 16u * segment;
            for (uint byte = 0; byte < 4u; ++byte) {
                const uint index = offset + byte;
                const int q4 = int(
                    (weight->qs[qbyte_base + index] >> nibble_shift) & 0x0fu);
                const int q8 = int(activation->qs[4u * q8_word + index]);
                dot += q4 * q8;
                sum_q8 += q8;
            }
        }
        const float d8 = float(activation->ds.x);
        scaled_dot = fma(d8, float(dot * int(scale_min.x)), scaled_dot);
        scaled_min = fma(d8, float(sum_q8 * int(scale_min.y)), scaled_min);
    }
    const half2 dm = as_type<half2>(weight->dm);
    const float contribution =
        fma(-float(dm.y), scaled_min, float(dm.x) * scaled_dot);
    return accumulator + contribution;
}

#define STRATA_MV_Q4_K(NC)                                                                      \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void strata_mv_q4_k_n##NC(                                                          \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_q4_K* weights [[buffer(1)]],                                         \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                  \
        constant Q4MVParams& params [[buffer(5)]],                                              \
        uint3 tid3 [[thread_position_in_threadgroup]],                                          \
        uint3 group [[threadgroup_position_in_grid]]) {                                        \
        constexpr uint column_count = NC;                                                       \
        constexpr uint row_slots = 4u;                                                          \
        const uint tid = tid3.y * 32u + tid3.x;                                                 \
        const uint lane = tid3.x;                                                               \
        const uint warp = tid3.y;                                                               \
        const uint blocks_per_row = params.n_in / 256u;                                         \
        const bool small_k = blocks_per_row < 8u;                                               \
        const uint rows_per_group = small_k ? row_slots : 1u;                                   \
        const uint row_base = group.x * rows_per_group;                                        \
        const uint live_rows = *rows;                                                           \
        threadgroup float partials[3][column_count][row_slots][32];                            \
        float accum[column_count][row_slots];                                                   \
        for (uint j = 0; j < column_count; ++j) {                                               \
            for (uint i = 0; i < row_slots; ++i) accum[j][i] = 0.0f;                           \
        }                                                                                       \
        for (uint kbx = tid / 16u; kbx < blocks_per_row; kbx += 8u) {                          \
            const uint iqs = 2u * (tid % 16u);                                                 \
            for (uint i = 0; i < rows_per_group; ++i) {                                        \
                const uint row = row_base + i;                                                 \
                if (row < params.n_out) {                                                       \
                    const device block_q4_K* weight =                                          \
                        weights + row * blocks_per_row + kbx;                                   \
                    for (uint j = 0; j < column_count; ++j) {                                    \
                        const uint input_column = j < live_rows ? j : 0u;                       \
                        accum[j][i] = q4_k_accumulate(                                          \
                            weight, xq, blocks_per_row, kbx, iqs, input_column, accum[j][i]);    \
                    }                                                                           \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        if (warp > 0u) {                                                                        \
            for (uint j = 0; j < column_count; ++j) {                                           \
                for (uint i = 0; i < row_slots; ++i) {                                         \
                    partials[warp - 1u][j][i][lane] = accum[j][i];                             \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
        if (warp == 0u) {                                                                       \
            for (uint j = 0; j < column_count; ++j) {                                           \
                for (uint i = 0; i < rows_per_group; ++i) {                                     \
                    const uint row = row_base + i;                                              \
                    if (row < params.n_out) {                                                   \
                        float value = accum[j][i];                                              \
                        value += partials[0][j][i][lane];                                        \
                        value += partials[1][j][i][lane];                                        \
                        value += partials[2][j][i][lane];                                        \
                        value = bfly_sum(value);                                                \
                        if (lane == i) {                                                        \
                            device float* destination =                                        \
                                j < live_rows ? y + j * params.n_out                            \
                                              : sink + (j - live_rows) * params.n_out;           \
                            destination[row] = value;                                           \
                        }                                                                        \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }

STRATA_MV_Q4_K(1)
STRATA_MV_Q4_K(2)
STRATA_MV_Q4_K(4)
STRATA_MV_Q4_K(8)

#undef STRATA_MV_Q4_K
