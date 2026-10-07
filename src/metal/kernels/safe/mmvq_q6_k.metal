// Native Q6_K weights multiplied by Q8_1 activations.
#include "types.h"
#include "reduce.h"

using namespace metal;

struct Q6MVParams {
    uint n_in;
    uint n_out;
};

static inline uint q6_load_u32_b2(device const uchar* bytes, uint word_index) {
    device const ushort* words = reinterpret_cast<device const ushort*>(bytes);
    return uint(words[2u * word_index]) | (uint(words[2u * word_index + 1u]) << 16u);
}

static inline float q6_k_accumulate(device const block_q6_K* weight,
                                    device const Q81* xq,
                                    uint blocks_per_row, uint kbx,
                                    uint subblock, uint input_column,
                                    float accumulator) {
    const uint bq8_offset =
        4u * (subblock / 16u) + (subblock % 16u) / 8u;
    const uint scale_offset =
        8u * (subblock / 16u) + (subblock % 16u) / 4u;
    const uint qh_shift = 2u * ((subblock % 16u) / 8u);
    const uint vl = q6_load_u32_b2(weight->ql, subblock);
    const uint vh = q6_load_u32_b2(
        weight->qh, 8u * (subblock / 16u) + subblock % 8u);
    const uint q8_word = subblock % 8u;
    float scaled_sum = 0.0f;
    for (uint pair_index = 0; pair_index < 2; ++pair_index) {
        int dot = 0;
        const uint nibble_shift = 4u * pair_index;
        for (uint byte = 0; byte < 4; ++byte) {
            const uint shift = nibble_shift + 8u * byte;
            const int low = int((vl >> shift) & 0x0fu);
            const int high = int((vh >> (qh_shift + shift)) & 0x03u);
            const int q = (low | (high << 4u)) - 32;
            const device Q81* activation =
                xq + input_column * (8u * blocks_per_row) +
                8u * kbx + bq8_offset + 2u * pair_index;
            dot += q * int(activation->qs[4u * q8_word + byte]);
        }
        const device Q81* activation =
            xq + input_column * (8u * blocks_per_row) +
            8u * kbx + bq8_offset + 2u * pair_index;
        const int scale = int(weight->scales[scale_offset + 4u * pair_index]);
        scaled_sum = fma(float(activation->ds.x),
                         float(dot * scale), scaled_sum);
    }
    return fma(float(weight->d), scaled_sum, accumulator);
}

#define STRATA_MV_Q6_K_IMPL(NC, KERNEL_NAME, RMAP_DECL, WEIGHT_ROW)                            \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void KERNEL_NAME(                                                                    \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_q6_K* weights [[buffer(1)]],                                         \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                  \
        constant Q6MVParams& params [[buffer(5)]],                                              \
        RMAP_DECL                                                                               \
        uint3 tid3 [[thread_position_in_threadgroup]],                                          \
        uint3 group [[threadgroup_position_in_grid]]) {                                        \
        constexpr uint column_count = NC;                                                       \
        constexpr uint row_slots = 4u;                                                          \
        const uint tid = tid3.y * 32u + tid3.x;                                                 \
        const uint lane = tid3.x;                                                               \
        const uint warp = tid3.y;                                                               \
        const uint blocks_per_row = params.n_in / 256u;                                         \
        const bool small_k = blocks_per_row < 4u;                                               \
        const uint rows_per_group = small_k ? row_slots : 1u;                                   \
        const uint row_base = group.x * rows_per_group;                                        \
        const uint live_rows = *rows;                                                           \
        threadgroup float partials[3][column_count][row_slots][32];                            \
        float accum[column_count][row_slots];                                                   \
        for (uint j = 0; j < column_count; ++j) {                                               \
            for (uint i = 0; i < row_slots; ++i) accum[j][i] = 0.0f;                           \
        }                                                                                       \
        for (uint kbx = tid / 32u; kbx < blocks_per_row; kbx += 4u) {                          \
            const uint subblock = tid % 32u;                                                   \
            for (uint i = 0; i < rows_per_group; ++i) {                                        \
                const uint row = row_base + i;                                                 \
                if (row < params.n_out) {                                                       \
                    const uint weight_row = WEIGHT_ROW;                                        \
                    const device block_q6_K* weight =                                          \
                        weights + weight_row * blocks_per_row + kbx;                            \
                    for (uint j = 0; j < column_count; ++j) {                                    \
                        const uint input_column = j < live_rows ? j : 0u;                       \
                        accum[j][i] = q6_k_accumulate(                                          \
                            weight, xq, blocks_per_row, kbx, subblock,                          \
                            input_column, accum[j][i]);                                         \
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

#define STRATA_MV_NO_RMAP
#define STRATA_MV_RMAP device const int* rmap [[buffer(6)]],
#define STRATA_MV_Q6_K(NC) \
    STRATA_MV_Q6_K_IMPL(NC, strata_mv_q6_k_n##NC, STRATA_MV_NO_RMAP, row)
#define STRATA_MV_Q6_K_INDEXED(NAME) \
    STRATA_MV_Q6_K_IMPL(1, NAME, STRATA_MV_RMAP, uint(rmap[row]))

STRATA_MV_Q6_K(1)
STRATA_MV_Q6_K(2)
STRATA_MV_Q6_K(4)
STRATA_MV_Q6_K(8)
STRATA_MV_Q6_K_INDEXED(strata_mv_q6_k_n1_x)

#undef STRATA_MV_Q6_K
#undef STRATA_MV_Q6_K_INDEXED
#undef STRATA_MV_Q6_K_IMPL
#undef STRATA_MV_NO_RMAP
#undef STRATA_MV_RMAP
