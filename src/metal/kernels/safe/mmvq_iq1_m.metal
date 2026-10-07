// Standalone IQ1_M row-dot twin for the grouped expert math in §5.10.
#include "types.h"
#include "reduce.h"
#include "dot.h"
#include "codebooks.h"

using namespace metal;

struct IQ1MMVParams {
    uint n_in;
    uint n_out;
};

static inline uint iq1_m_half_scale_bits(device const block_iq1_m* weight) {
    uint result = 0u;
    for (uint word = 0u; word < 4u; ++word) {
        const uint byte_offset = 2u * word;
        const uint packed = uint(weight->scales[byte_offset]) |
                            (uint(weight->scales[byte_offset + 1u]) << 8u);
        result |= ((packed >> 12u) & 0x0fu) << (4u * word);
    }
    return result;
}

static inline uint iq1_m_packed_scale(device const block_iq1_m* weight,
                                       uint part) {
    const uint word_index = part / 2u;
    const uint byte_offset = 2u * word_index;
    const uint packed = uint(weight->scales[byte_offset]) |
                        (uint(weight->scales[byte_offset + 1u]) << 8u);
    return packed >> (6u * (part & 1u));
}

static inline float iq1_m_dot(device const block_iq1_m* weight,
                               uint part,
                               device const Q81* activation) {
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};

    for (uint codeword = 0u; codeword < 4u; ++codeword) {
        const uint qh_byte = weight->qh[2u * part + codeword / 2u];
        const uint qhl =
            (qh_byte >> (4u * (codeword & 1u))) & 0x0fu;
        const uint grid_index =
            uint(weight->qs[4u * part + codeword]) | ((qhl & 0x07u) << 8u);
        const uint grid = iq1s_grid_gpu[grid_index];
        const int grid0 = int((grid >> 0u) & 0x0f0f0f0fu);
        const int grid1 = int((grid >> 4u) & 0x0f0f0f0fu);

        const uint word = 2u * codeword;
        const int u0 = activation_words[word + 0u];
        const int u1 = activation_words[word + 1u];
        const uint partial_index = codeword / 2u;
        sumi[partial_index] =
            dot4_i8(grid0, u0) + sumi[partial_index];
        sumi[partial_index] =
            dot4_i8(grid1, u1) + sumi[partial_index];

        const int sumy = dot4_i8(u0, 0x01010101) +
                         dot4_i8(u1, 0x01010101);
        const float delta = (qhl & 0x08u) != 0u ? -1.125f : -0.875f;
        sumf[partial_index] =
            fma(delta, float(sumy), sumf[partial_index]);
    }

    const uint packed_scale = iq1_m_packed_scale(weight, part);
    const int scale0 = 2 * int((packed_scale >> 0u) & 0x07u) + 1;
    const int scale1 = 2 * int((packed_scale >> 3u) & 0x07u) + 1;
    const float x0 = (float(sumi[0]) + sumf[0]) * float(scale0);
    const float x1 = (float(sumi[1]) + sumf[1]) * float(scale1);
    return x0 + x1;
}

#define STRATA_MV_IQ1_M(NC)                                                                   \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void strata_mv_iq1_m_n##NC(                                                         \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_iq1_m* weights [[buffer(1)]],                                       \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                   \
        constant IQ1MMVParams& params [[buffer(5)]],                                           \
        uint3 tid3 [[thread_position_in_threadgroup]],                                           \
        uint3 group [[threadgroup_position_in_grid]]) {                                         \
        constexpr uint column_count = NC;                                                       \
        const uint lane = tid3.x;                                                               \
        const uint warp = tid3.y;                                                               \
        const uint row = group.x * 4u + warp;                                                   \
        if (row >= params.n_out) return;                                                         \
        const uint blocks_per_row = params.n_in / 256u;                                        \
        const uint calls_per_row = blocks_per_row * 8u;                                        \
        const uint live_rows = *rows;                                                           \
        const device block_iq1_m* row_weights = weights + row * blocks_per_row;                \
        float accum[column_count];                                                              \
        for (uint j = 0u; j < column_count; ++j) accum[j] = 0.0f;                               \
        for (uint k = lane; k < calls_per_row; k += 32u) {                                      \
            const uint kbx = k / 8u;                                                            \
            const uint part = k % 8u;                                                           \
            const device block_iq1_m* weight = row_weights + kbx;                              \
            for (uint j = 0u; j < column_count; ++j) {                                          \
                const uint input_column = j < live_rows ? j : 0u;                               \
                const device Q81* activation =                                                   \
                    xq + input_column * (params.n_in / 32u) + kbx * 8u + part;                  \
                const float x = iq1_m_dot(weight, part, activation);                           \
                const half weight_scale = as_type<half>(                                       \
                    ushort(iq1_m_half_scale_bits(weight)));                                     \
                const float d8 = float(activation->ds.x);                                       \
                accum[j] = fma(float(weight_scale) * d8, x, accum[j]);                         \
            }                                                                                    \
        }                                                                                        \
        for (uint j = 0u; j < column_count; ++j) {                                               \
            const float value = bfly_sum(accum[j]);                                             \
            if (lane == j) {                                                                     \
                device float* destination = j < live_rows                                       \
                    ? y + j * params.n_out                                                      \
                    : sink + (j - live_rows) * params.n_out;                                    \
                destination[row] = value;                                                        \
            }                                                                                    \
        }                                                                                        \
    }

STRATA_MV_IQ1_M(1)
STRATA_MV_IQ1_M(2)
STRATA_MV_IQ1_M(4)
STRATA_MV_IQ1_M(8)

#undef STRATA_MV_IQ1_M
