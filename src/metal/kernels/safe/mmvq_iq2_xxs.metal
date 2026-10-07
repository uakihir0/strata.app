// Standalone IQ2_XXS row-dot twin for the grouped expert math in §5.10.
#include "types.h"
#include "reduce.h"
#include "dot.h"
#include "codebooks.h"

using namespace metal;

struct IQ2XXSMVParams {
    uint n_in;
    uint n_out;
};

static inline char4 iq2_xxs_signed_quartet(uint packed, uint signs,
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

static inline int iq2_xxs_integer_dot(device const block_iq2_xxs* weight,
                                       uint iqs,
                                       device const Q81* activation) {
    const device ushort* packed = weight->qs + 2u * iqs;
    const uint indices = uint(packed[0]) | (uint(packed[1]) << 16u);
    const uint aux32 = uint(packed[2]) | (uint(packed[3]) << 16u);
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);

    int sumi = 0;
    for (uint code_slot = 0u; code_slot < 4u; ++code_slot) {
        const uint code = (indices >> (8u * code_slot)) & 0xffu;
        const uint raw_signs = (aux32 >> (7u * code_slot)) & 0xffu;
        const uint signs =
            raw_signs ^ ((popcount(raw_signs) & 1u) << 7u);
        const ulong grid = iq2xxs_grid[code];
        const char4 first = iq2_xxs_signed_quartet(
            uint(grid), signs, 0u);
        const char4 second = iq2_xxs_signed_quartet(
            uint(grid >> 32u), signs, 4u);
        sumi = dot4_i8(as_type<int>(first),
                       activation_words[2u * code_slot]) + sumi;
        sumi = dot4_i8(as_type<int>(second),
                       activation_words[2u * code_slot + 1u]) + sumi;
    }
    const int odd_scale = int((aux32 >> 27u) | 1u);
    return (sumi * odd_scale) / 8;
}

#define STRATA_MV_IQ2_XXS(NC)                                                                  \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void strata_mv_iq2_xxs_n##NC(                                                       \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_iq2_xxs* weights [[buffer(1)]],                                     \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                   \
        constant IQ2XXSMVParams& params [[buffer(5)]],                                          \
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
        const device block_iq2_xxs* row_weights = weights + row * blocks_per_row;               \
        float accum[column_count];                                                              \
        for (uint j = 0u; j < column_count; ++j) accum[j] = 0.0f;                               \
        for (uint k = lane; k < calls_per_row; k += 32u) {                                      \
            const uint kbx = k / 8u;                                                            \
            const uint iqs = 2u * (k % 8u);                                                     \
            const device block_iq2_xxs* weight = row_weights + kbx;                             \
            for (uint j = 0u; j < column_count; ++j) {                                          \
                const uint input_column = j < live_rows ? j : 0u;                               \
                const device Q81* activation =                                                   \
                    xq + input_column * (params.n_in / 32u) + kbx * 8u + iqs / 2u;              \
                const int sumi = iq2_xxs_integer_dot(weight, iqs, activation);                   \
                const float d8 = float(activation->ds.x);                                       \
                accum[j] = fma(float(weight->d) * d8, float(sumi), accum[j]);                   \
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

STRATA_MV_IQ2_XXS(1)
STRATA_MV_IQ2_XXS(2)
STRATA_MV_IQ2_XXS(4)
STRATA_MV_IQ2_XXS(8)

#undef STRATA_MV_IQ2_XXS
