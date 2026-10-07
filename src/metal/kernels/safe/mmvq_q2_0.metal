// Native Q2_0 weights multiplied by Q8_1 activations.
#include "types.h"
#include "reduce.h"

using namespace metal;

struct Q2MVParams {
    uint n_in;
    uint n_out;
};

static inline int q2_0_integer_dot(device const block_q2_0* weight,
                                   device const Q81* activation,
                                   uint qs_half) {
    int sum = 0;
    const uint byte_base = qs_half * 8u;
    for (uint word = 0; word < 4; ++word) {
        for (uint byte_in_word = 0; byte_in_word < 2; ++byte_in_word) {
            const uint packed =
                uint(weight->qs[byte_base + 2u * word + byte_in_word]);
            const uint activation_base = 8u * word + 4u * byte_in_word;
            for (uint lane = 0; lane < 4; ++lane) {
                const int code = int((packed >> (2u * lane)) & 0x03u) - 1;
                sum += code * int(activation->qs[activation_base + lane]);
            }
        }
    }
    return sum;
}

#define STRATA_MV_Q2_0(NC)                                                                       \
    [[max_total_threads_per_threadgroup(128)]]                                                   \
    kernel void strata_mv_q2_0_n##NC(                                                            \
        device const Q81* xq [[buffer(0)]],                                                       \
        device const block_q2_0* weights [[buffer(1)]],                                          \
        device float* y [[buffer(2)]],                                                           \
        device float* sink [[buffer(3)]],                                                        \
        device const uint* rows [[buffer(4)]],                                                    \
        constant Q2MVParams& params [[buffer(5)]],                                                \
        uint3 tid3 [[thread_position_in_threadgroup]],                                            \
        uint3 group [[threadgroup_position_in_grid]]) {                                          \
        constexpr uint column_count = NC;                                                        \
        constexpr uint row_slots = 4u;                                                           \
        const uint tid = tid3.y * 32u + tid3.x;                                                   \
        const uint lane = tid3.x;                                                                \
        const uint warp = tid3.y;                                                                \
        const uint blocks_per_row = params.n_in / 64u;                                           \
        const bool small_k = blocks_per_row < 64u;                                                \
        const uint rows_per_group = small_k ? row_slots : 1u;                                    \
        const uint row_base = group.x * rows_per_group;                                          \
        const uint live_rows = *rows;                                                            \
        threadgroup float partials[3][column_count][row_slots][32];                              \
        float accum[column_count][row_slots];                                                     \
        for (uint j = 0; j < column_count; ++j) {                                                 \
            for (uint i = 0; i < row_slots; ++i) accum[j][i] = 0.0f;                             \
        }                                                                                         \
        for (uint kbx = tid / 2u; kbx < blocks_per_row; kbx += 64u) {                            \
            const uint qs_half = tid % 2u;                                                       \
            const uint kby = 2u * kbx;                                                           \
            for (uint i = 0; i < rows_per_group; ++i) {                                          \
                const uint row = row_base + i;                                                   \
                if (row < params.n_out) {                                                        \
                    const device block_q2_0* weight =                                            \
                        weights + row * blocks_per_row + kbx;                                    \
                    for (uint j = 0; j < column_count; ++j) {                                     \
                        const uint input_column = j < live_rows ? j : 0u;                        \
                        const device Q81* activation =                                            \
                            xq + input_column * (2u * blocks_per_row) + kby + qs_half;            \
                        const int integer_dot =                                                   \
                            q2_0_integer_dot(weight, activation, qs_half);                        \
                        const float scale =                                                       \
                            float(weight->d) * float(activation->ds.x);                           \
                        accum[j][i] = fma(scale, float(integer_dot), accum[j][i]);                \
                    }                                                                             \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        if (warp > 0u) {                                                                          \
            for (uint j = 0; j < column_count; ++j) {                                             \
                for (uint i = 0; i < row_slots; ++i) {                                           \
                    partials[warp - 1u][j][i][lane] = accum[j][i];                               \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                         \
        if (warp == 0u) {                                                                         \
            for (uint j = 0; j < column_count; ++j) {                                             \
                for (uint i = 0; i < rows_per_group; ++i) {                                      \
                    const uint row = row_base + i;                                               \
                    if (row < params.n_out) {                                                    \
                        float value = accum[j][i];                                               \
                        value += partials[0][j][i][lane];                                         \
                        value += partials[1][j][i][lane];                                         \
                        value += partials[2][j][i][lane];                                         \
                        value = bfly_sum(value);                                                 \
                        if (lane == i) {                                                         \
                            device float* destination =                                         \
                                j < live_rows ? y + j * params.n_out                             \
                                              : sink + (j - live_rows) * params.n_out;            \
                            destination[row] = value;                                            \
                        }                                                                         \
                    }                                                                             \
                }                                                                             \
            }                                                                                     \
        }                                                                                         \
    }

STRATA_MV_Q2_0(1)
STRATA_MV_Q2_0(2)
STRATA_MV_Q2_0(4)
STRATA_MV_Q2_0(8)

#undef STRATA_MV_Q2_0
