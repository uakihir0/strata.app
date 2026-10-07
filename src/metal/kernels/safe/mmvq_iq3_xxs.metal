// Standalone IQ3_XXS row-dot twin for the grouped expert math in §5.10.
#include "types.h"
#include "reduce.h"
#include "dot.h"
#include "codebooks.h"

using namespace metal;

struct IQ3XXSMVParams {
    uint n_in;
    uint n_out;
};

static inline uint iq3_xxs_unpack_signs(uint sign_index) {
    const uint signs = sign_index & 0x7fu;
    return signs ^ ((popcount(signs) & 1u) << 7u);
}

static inline char4 iq3_xxs_signed_quartet(uint packed, uint signs,
                                            uint component_base) {
    const char4 magnitude = as_type<char4>(packed);
    return char4(
        (signs & (1u << (component_base + 0u))) != 0u
            ? -int(magnitude.x) : int(magnitude.x),
        (signs & (1u << (component_base + 1u))) != 0u
            ? -int(magnitude.y) : int(magnitude.y),
        (signs & (1u << (component_base + 2u))) != 0u
            ? -int(magnitude.z) : int(magnitude.z),
        (signs & (1u << (component_base + 3u))) != 0u
            ? -int(magnitude.w) : int(magnitude.w));
}

static inline int iq3_xxs_integer_dot(device const block_iq3_xxs* weight,
                                      uint part,
                                      device const Q81* activation) {
    const device uchar* indices = weight->qs + 8u * part;
    const device uchar* aux = weight->qs + 64u + 4u * part;
    const uint aux32 = uint(aux[0]) | (uint(aux[1]) << 8u) |
                       (uint(aux[2]) << 16u) | (uint(aux[3]) << 24u);
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);

    int sumi = 0;
    for (uint code_slot = 0u; code_slot < 4u; ++code_slot) {
        const uint sign_index = (aux32 >> (7u * code_slot)) & 0x7fu;
        const uint signs = iq3_xxs_unpack_signs(sign_index);
        const char4 low = iq3_xxs_signed_quartet(
            iq3xxs_grid[indices[2u * code_slot]], signs, 0u);
        const char4 high = iq3_xxs_signed_quartet(
            iq3xxs_grid[indices[2u * code_slot + 1u]], signs, 4u);
        sumi = dot4_i8(as_type<int>(low),
                       activation_words[2u * code_slot]) + sumi;
        sumi = dot4_i8(as_type<int>(high),
                       activation_words[2u * code_slot + 1u]) + sumi;
    }
    const int scale = int(aux32 >> 28u);
    return (scale * sumi + sumi / 2) / 2;
}

#define STRATA_MV_IQ3_XXS(NC)                                                                 \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void strata_mv_iq3_xxs_n##NC(                                                       \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_iq3_xxs* weights [[buffer(1)]],                                     \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                   \
        constant IQ3XXSMVParams& params [[buffer(5)]],                                          \
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
        const device block_iq3_xxs* row_weights = weights + row * blocks_per_row;               \
        float accum[column_count];                                                              \
        for (uint j = 0u; j < column_count; ++j) accum[j] = 0.0f;                               \
        for (uint k = lane; k < calls_per_row; k += 32u) {                                      \
            const uint kbx = k / 8u;                                                            \
            const uint part = k % 8u;                                                           \
            const device block_iq3_xxs* weight = row_weights + kbx;                             \
            for (uint j = 0u; j < column_count; ++j) {                                          \
                const uint input_column = j < live_rows ? j : 0u;                               \
                const device Q81* activation =                                                   \
                    xq + input_column * (params.n_in / 32u) + kbx * 8u + part;                  \
                const int sumi = iq3_xxs_integer_dot(weight, part, activation);                  \
                const float d8 = float(activation->ds.x);                                       \
                accum[j] = fma(float(weight->d) * d8, float(sumi), accum[j]);                  \
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

STRATA_MV_IQ3_XXS(1)
STRATA_MV_IQ3_XXS(2)
STRATA_MV_IQ3_XXS(4)
STRATA_MV_IQ3_XXS(8)

#undef STRATA_MV_IQ3_XXS
