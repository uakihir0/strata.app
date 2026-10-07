// Native Q4_0 weights multiplied by Q8_1 activations (docs/mac/04-metal-kernels.md §5.8).
#include "types.h"
#include "reduce.h"

using namespace metal;

struct Q4MVParams {
    uint n_in;
    uint n_out;
};

static inline int q4_0_integer_dot(device const block_q4_0* weight,
                                   device const Q81* activation,
                                   uint first_word) {
    int sum = 0;
    for (uint word = 0; word < 2; ++word) {
        const uint offset = 4u * (first_word + word);
        for (uint lane = 0; lane < 4; ++lane) {
            const uint packed = uint(weight->qs[offset + lane]);
            const int low = int(packed & 0x0fu);
            const int high = int(packed >> 4u);
            sum += low * int(activation->qs[offset + lane]);
            sum += high * int(activation->qs[offset + 16u + lane]);
        }
    }
    return sum;
}

#define STRATA_MV_Q4_0(NC)                                                                       \
    [[max_total_threads_per_threadgroup(128)]]                                                   \
    kernel void strata_mv_q4_0_n##NC(                                                            \
        device const Q81* xq [[buffer(0)]],                                                       \
        device const block_q4_0* weights [[buffer(1)]],                                           \
        device float* y [[buffer(2)]],                                                           \
        device float* sink [[buffer(3)]],                                                        \
        device const uint* rows [[buffer(4)]],                                                    \
        constant Q4MVParams& params [[buffer(5)]],                                                \
        uint3 tid3 [[thread_position_in_threadgroup]],                                            \
        uint3 group [[threadgroup_position_in_grid]]) {                                          \
        constexpr uint column_count = NC;                                                        \
        constexpr uint row_slots = 4u;                                                           \
        const uint tid = tid3.y * 32u + tid3.x;                                                   \
        const uint lane = tid3.x;                                                                \
        const uint warp = tid3.y;                                                                \
        const uint blocks_per_row = params.n_in / 32u;                                           \
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
            const uint first_word = 2u * (tid % 2u);                                             \
            for (uint i = 0; i < rows_per_group; ++i) {                                          \
                const uint row = row_base + i;                                                   \
                if (row < params.n_out) {                                                        \
                    const device block_q4_0* weight =                                            \
                        weights + row * blocks_per_row + kbx;                                    \
                    for (uint j = 0; j < column_count; ++j) {                                     \
                        const uint input_column = j < live_rows ? j : 0u;                        \
                        const device Q81* activation =                                            \
                            xq + input_column * blocks_per_row + kbx;                            \
                        const int integer_dot =                                                   \
                            q4_0_integer_dot(weight, activation, first_word);                    \
                        const float activation_scale = float(activation->ds.x);                   \
                        const float activation_sum = float(activation->ds.y);                     \
                        const float scaled_dot = float(integer_dot) * activation_scale;            \
                        const float centered_dot = fma(-4.0f, activation_sum, scaled_dot);          \
                        accum[j][i] = fma(float(weight->d), centered_dot, accum[j][i]);             \
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
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    }

STRATA_MV_Q4_0(1)
STRATA_MV_Q4_0(2)
STRATA_MV_Q4_0(4)
STRATA_MV_Q4_0(8)

#undef STRATA_MV_Q4_0
