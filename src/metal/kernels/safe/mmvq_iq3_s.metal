// Native IQ3_S weights multiplied by Q8_1 activations.
#include "types.h"
#include "reduce.h"
#include "dot.h"
#include "codebooks.h"

using namespace metal;

struct IQ3SMVParams {
    uint n_in;
    uint n_out;
};

struct IQ3SPart {
    char4 grid[8];
    int scale;
    float d;
};

static inline char4 iq3_s_signed_grid(uint index, uint signs) {
    const char4 magnitude = as_type<char4>(iq3s_grid[index]);
    return char4(
        (signs & 0x01u) != 0u ? -int(magnitude.x) : int(magnitude.x),
        (signs & 0x02u) != 0u ? -int(magnitude.y) : int(magnitude.y),
        (signs & 0x04u) != 0u ? -int(magnitude.z) : int(magnitude.z),
        (signs & 0x08u) != 0u ? -int(magnitude.w) : int(magnitude.w));
}

static inline IQ3SPart iq3_s_load(device const block_iq3_s* weight,
                                  uint iqs) {
    IQ3SPart result;
    const uint qh = uint(weight->qh[iqs / 2u]);
    for (uint l0 = 0u; l0 < 8u; l0 += 2u) {
        const uint qs_offset = 4u * iqs + l0;
        const uint low_index = uint(weight->qs[qs_offset]) |
            (((qh >> l0) & 1u) << 8u);
        const uint high_index = uint(weight->qs[qs_offset + 1u]) |
            (((qh >> (l0 + 1u)) & 1u) << 8u);
        const uint packed_signs =
            uint(weight->signs[4u * (iqs / 2u) + l0 / 2u]);
        result.grid[l0] = iq3_s_signed_grid(low_index, packed_signs & 0x0fu);
        result.grid[l0 + 1u] =
            iq3_s_signed_grid(high_index, packed_signs >> 4u);
    }
    const uint packed_scale = uint(weight->scales[iqs / 4u]);
    const uint scale_shift = (iqs << 1u) & 4u;
    result.scale = 1 + 2 * int((packed_scale >> scale_shift) & 0x0fu);
    result.d = float(weight->d);
    return result;
}

static inline int iq3_s_apply(thread const IQ3SPart& weight,
                              device const Q81* activation) {
    int sumi = 0;
    device const int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    for (uint word = 0u; word < 8u; ++word) {
        sumi = dot4_i8(as_type<int>(weight.grid[word]),
                       activation_words[word]) + sumi;
    }
    return sumi * weight.scale;
}

#define STRATA_MV_IQ3_S(NC)                                                                    \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void strata_mv_iq3_s_n##NC(                                                         \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_iq3_s* weights [[buffer(1)]],                                       \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                   \
        constant IQ3SMVParams& params [[buffer(5)]],                                             \
        uint3 tid3 [[thread_position_in_threadgroup]],                                           \
        uint3 group [[threadgroup_position_in_grid]]) {                                         \
        constexpr uint column_count = NC;                                                       \
        const uint lane = tid3.x;                                                               \
        const uint warp = tid3.y;                                                               \
        const uint row = group.x * 4u + warp;                                                   \
        if (row >= params.n_out) return;                                                         \
        const uint blocks_per_row = params.n_in / 256u;                                         \
        const uint calls_per_row = blocks_per_row * 8u;                                         \
        const uint live_rows = *rows;                                                           \
        const device block_iq3_s* row_weights = weights + row * blocks_per_row;                 \
        float accum[column_count];                                                              \
        for (uint j = 0u; j < column_count; ++j) accum[j] = 0.0f;                               \
        for (uint k = lane; k < calls_per_row; k += 32u) {                                      \
            const uint kbx = k / 8u;                                                            \
            const uint iqs = 2u * (k % 8u);                                                     \
            const IQ3SPart weight = iq3_s_load(row_weights + kbx, iqs);                         \
            for (uint j = 0u; j < column_count; ++j) {                                          \
                const uint input_column = j < live_rows ? j : 0u;                               \
                const device Q81* activation =                                                   \
                    xq + input_column * (params.n_in / 32u) + kbx * 8u + iqs / 2u;              \
                const int sumi = iq3_s_apply(weight, activation);                               \
                const float d8 = float(activation->ds.x);                                       \
                accum[j] = fma(weight.d * d8, float(sumi), accum[j]);                           \
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

STRATA_MV_IQ3_S(1)
STRATA_MV_IQ3_S(2)
STRATA_MV_IQ3_S(4)
STRATA_MV_IQ3_S(8)

#undef STRATA_MV_IQ3_S
