// Native Q5_K weights multiplied by Q8_1 activations.
#include "types.h"
#include "reduce.h"

using namespace metal;

struct Q5MVParams {
    uint n_in;
    uint n_out;
};

static inline uint2 q5_k_scale_min(device const block_q5_K* weight, uint group) {
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

static inline float q5_k_accumulate(device const block_q5_K* weight,
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
        const uint2 scale_min = q5_k_scale_min(weight, group);
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
                const int qh =
                    int((weight->qh[4u * q8_word + index] >> group) & 1u);
                const int q5 = q4 | (qh << 4u);
                const int q8 = int(activation->qs[4u * q8_word + index]);
                dot += q5 * q8;
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

#define STRATA_MV_NO_RMAP
#define STRATA_MV_RMAP device const int* rmap [[buffer(6)]],

#define STRATA_MV_Q5_K_IMPL(NC, KERNEL_NAME, RMAP_DECL, WEIGHT_ROW)                            \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void KERNEL_NAME(                                                                    \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_q5_K* weights [[buffer(1)]],                                        \
        device float* y [[buffer(2)]],                                                         \
        device float* sink [[buffer(3)]],                                                      \
        device const uint* rows [[buffer(4)]],                                                 \
        constant Q5MVParams& params [[buffer(5)]],                                             \
        RMAP_DECL                                                                               \
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
                    const uint weight_row = WEIGHT_ROW;                                        \
                    const device block_q5_K* weight =                                         \
                        weights + weight_row * blocks_per_row + kbx;                            \
                    for (uint j = 0; j < column_count; ++j) {                                    \
                        const uint input_column = j < live_rows ? j : 0u;                       \
                        accum[j][i] = q5_k_accumulate(                                          \
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

#define STRATA_MV_Q5_K(NC) \
    STRATA_MV_Q5_K_IMPL(NC, strata_mv_q5_k_n##NC, STRATA_MV_NO_RMAP, row)
#define STRATA_MV_Q5_K_INDEXED(NAME) \
    STRATA_MV_Q5_K_IMPL(1, NAME, STRATA_MV_RMAP, uint(rmap[row]))

STRATA_MV_Q5_K(1)
STRATA_MV_Q5_K(2)
STRATA_MV_Q5_K(4)
STRATA_MV_Q5_K(8)
STRATA_MV_Q5_K_INDEXED(strata_mv_q5_k_n1_x)

#undef STRATA_MV_Q5_K
#undef STRATA_MV_Q5_K_INDEXED
#undef STRATA_MV_Q5_K_IMPL
#undef STRATA_MV_NO_RMAP
#undef STRATA_MV_RMAP
