// Native IQ4_XS weights multiplied by Q8_1 activations.
#include "types.h"
#include "reduce.h"
#include "codebooks.h"

using namespace metal;

struct IQ4XSParams {
    uint n_in;
    uint n_out;
};

static inline int iq4_xs_integer_dot(device const block_iq4_xs* weight,
                                     device const Q81* activation,
                                     uint subblock) {
    int sum = 0;
    const uint byte_base = 16u * subblock;
    for (uint word = 0; word < 4; ++word) {
        const uint offset = byte_base + 4u * word;
        for (uint lane = 0; lane < 4; ++lane) {
            const uint packed = uint(weight->qs[offset + lane]);
            const int low = int(kvalues_iq4nl[packed & 0x0fu]);
            const int high = int(kvalues_iq4nl[packed >> 4u]);
            sum += low * int(activation->qs[4u * word + lane]);
            sum += high * int(activation->qs[16u + 4u * word + lane]);
        }
    }
    return sum;
}

static inline int iq4_xs_scale_code(device const block_iq4_xs* weight,
                                    uint subblock) {
    const uint low =
        (uint(weight->scales_l[subblock / 2u]) >> (4u * (subblock % 2u))) & 0x0fu;
    const uint high = (uint(weight->scales_h) >> (2u * subblock)) & 0x03u;
    return int(low | (high << 4u));
}

#define STRATA_MV_IQ4_XS_IMPL(NC, KERNEL_NAME, RMAP_DECL, WEIGHT_ROW)                           \
    [[max_total_threads_per_threadgroup(128)]]                                                   \
    kernel void KERNEL_NAME(                                                                     \
        device const Q81* xq [[buffer(0)]],                                                      \
        device const block_iq4_xs* weights [[buffer(1)]],                                       \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                   \
        constant IQ4XSParams& params [[buffer(5)]],                                              \
        RMAP_DECL                                                                                \
        uint3 tid3 [[thread_position_in_threadgroup]],                                           \
        uint3 group [[threadgroup_position_in_grid]]) {                                         \
        constexpr uint column_count = NC;                                                       \
        constexpr uint row_slots = 4u;                                                          \
        const uint tid = tid3.y * 32u + tid3.x;                                                  \
        const uint lane = tid3.x;                                                               \
        const uint warp = tid3.y;                                                               \
        const uint blocks_per_row = params.n_in / 256u;                                         \
        const bool small_k = blocks_per_row < 16u;                                              \
        const uint rows_per_group = small_k ? row_slots : 1u;                                   \
        const uint row_base = group.x * rows_per_group;                                         \
        const uint live_rows = *rows;                                                           \
        threadgroup float partials[3][column_count][row_slots][32];                             \
        float accum[column_count][row_slots];                                                    \
        for (uint j = 0; j < column_count; ++j) {                                                \
            for (uint i = 0; i < row_slots; ++i) accum[j][i] = 0.0f;                            \
        }                                                                                        \
        for (uint kbx = tid / 8u; kbx < blocks_per_row; kbx += 16u) {                           \
            const uint subblock = tid % 8u;                                                     \
            for (uint i = 0; i < rows_per_group; ++i) {                                         \
                const uint row = row_base + i;                                                  \
                if (row < params.n_out) {                                                       \
                    const uint weight_row = WEIGHT_ROW;                                         \
                    const device block_iq4_xs* weight =                                         \
                        weights + weight_row * blocks_per_row + kbx;                            \
                    const uint row_scale_code =                                                 \
                        uint(iq4_xs_scale_code(weight, subblock));                               \
                    for (uint j = 0; j < column_count; ++j) {                                    \
                        const uint input_column = j < live_rows ? j : 0u;                       \
                        const device Q81* activation =                                           \
                            xq + input_column * (8u * blocks_per_row) + 8u * kbx + subblock;   \
                        const int integer_dot =                                                  \
                            iq4_xs_integer_dot(weight, activation, subblock);                    \
                        const int scaled_dot =                                                   \
                            integer_dot * (int(row_scale_code) - 32);                           \
                        const float scale = float(weight->d) * float(activation->ds.x);           \
                        accum[j][i] = fma(scale, float(scaled_dot), accum[j][i]);                \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
        if (warp > 0u) {                                                                         \
            for (uint j = 0; j < column_count; ++j) {                                            \
                for (uint i = 0; i < row_slots; ++i) {                                          \
                    partials[warp - 1u][j][i][lane] = accum[j][i];                              \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
        if (warp == 0u) {                                                                        \
            for (uint j = 0; j < column_count; ++j) {                                            \
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
#define STRATA_MV_IQ4_XS(NC) \
    STRATA_MV_IQ4_XS_IMPL(NC, strata_mv_iq4_xs_n##NC, STRATA_MV_NO_RMAP, row)
#define STRATA_MV_IQ4_XS_INDEXED(NAME) \
    STRATA_MV_IQ4_XS_IMPL(1, NAME, STRATA_MV_RMAP, uint(rmap[row]))

STRATA_MV_IQ4_XS(1)
STRATA_MV_IQ4_XS(2)
STRATA_MV_IQ4_XS(4)
STRATA_MV_IQ4_XS(8)
STRATA_MV_IQ4_XS_INDEXED(strata_mv_iq4_xs_n1_x)

#undef STRATA_MV_IQ4_XS
#undef STRATA_MV_IQ4_XS_INDEXED
#undef STRATA_MV_IQ4_XS_IMPL
#undef STRATA_MV_NO_RMAP
#undef STRATA_MV_RMAP
