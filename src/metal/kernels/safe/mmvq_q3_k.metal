// Native Q3_K weights multiplied by Q8_1 activations.
#include "types.h"
#include "reduce.h"

using namespace metal;

struct Q3MVParams {
    uint n_in;
    uint n_out;
};

static inline uint q3_load_u32_b2(device const uchar* bytes, uint word_index) {
    device const ushort* words = reinterpret_cast<device const ushort*>(bytes);
    return uint(words[2u * word_index]) | (uint(words[2u * word_index + 1u]) << 16u);
}

static inline float q3_k_accumulate(device const block_q3_K* weight,
                                    device const Q81* xq,
                                    uint blocks_per_row, uint kbx,
                                    uint iqs, uint input_column,
                                    float accumulator) {
    const uint bq8_offset = 4u * (iqs / 8u);
    const uint scale_offset = iqs - iqs % 8u + (iqs % 8u) / 4u;
    const uint q8_word = iqs % 8u;
    const uint vl = q3_load_u32_b2(weight->qs, iqs);
    const uint vh = ~q3_load_u32_b2(weight->hmask, iqs % 8u) >> bq8_offset;
    float scaled_sum = 0.0f;
    for (uint group = 0; group < 4u; ++group) {
        const device Q81* activation =
            xq + input_column * (8u * blocks_per_row) +
            8u * kbx + bq8_offset + group;
        int dot = 0;
        for (uint byte = 0; byte < 4u; ++byte) {
            const uint shift = 2u * group + 8u * byte;
            const int low = int((vl >> shift) & 0x03u);
            const int negative = int((vh >> (group + 8u * byte)) & 0x01u);
            const int q = low - (negative != 0 ? 4 : 0);
            dot += q * int(activation->qs[4u * q8_word + byte]);
        }
        const uint scale_index = scale_offset + 2u * group;
        const uint scale_low =
            (uint(weight->scales[scale_index % 8u]) >>
             (4u * (scale_index / 8u))) & 0x0fu;
        const uint scale_high =
            (uint(weight->scales[8u + scale_index % 4u]) >>
             (2u * (scale_index / 4u))) & 0x03u;
        const int scale = int(scale_low | (scale_high << 4u)) - 32;
        scaled_sum = fma(float(activation->ds.x), float(dot * scale), scaled_sum);
    }
    return fma(float(weight->d), scaled_sum, accumulator);
}

#define STRATA_MV_Q3_K(NC)                                                                      \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void strata_mv_q3_k_n##NC(                                                          \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_q3_K* weights [[buffer(1)]],                                         \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                  \
        constant Q3MVParams& params [[buffer(5)]],                                              \
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
            const uint iqs = tid % 16u;                                                        \
            for (uint i = 0; i < rows_per_group; ++i) {                                        \
                const uint row = row_base + i;                                                 \
                if (row < params.n_out) {                                                       \
                    const device block_q3_K* weight =                                          \
                        weights + row * blocks_per_row + kbx;                                   \
                    for (uint j = 0; j < column_count; ++j) {                                    \
                        const uint input_column = j < live_rows ? j : 0u;                       \
                        accum[j][i] = q3_k_accumulate(                                          \
                            weight, xq, blocks_per_row, kbx, iqs, input_column,                 \
                            accum[j][i]);                                                       \
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

STRATA_MV_Q3_K(1)
STRATA_MV_Q3_K(2)
STRATA_MV_Q3_K(4)
STRATA_MV_Q3_K(8)

#undef STRATA_MV_Q3_K
