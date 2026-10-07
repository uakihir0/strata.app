// Standalone IQ2_XS row-dot twin for the grouped expert math in §5.10.
#include "types.h"
#include "reduce.h"
#include "dot.h"
#include "codebooks.h"

using namespace metal;

struct IQ2XSMVParams {
    uint n_in;
    uint n_out;
};

static inline char4 iq2_xs_signed_quartet(uint packed, uint signs,
                                           uint component) {
    const char4 magnitude = as_type<char4>(packed);
    return char4(
        (signs & uint(kmask_iq2xs[component + 0u])) != 0u
            ? -int(magnitude.x) : int(magnitude.x),
        (signs & uint(kmask_iq2xs[component + 1u])) != 0u
            ? -int(magnitude.y) : int(magnitude.y),
        (signs & uint(kmask_iq2xs[component + 2u])) != 0u
            ? -int(magnitude.z) : int(magnitude.z),
        (signs & uint(kmask_iq2xs[component + 3u])) != 0u
            ? -int(magnitude.w) : int(magnitude.w));
}

static inline int iq2_xs_integer_dot(device const block_iq2_xs* weight,
                                      uint part,
                                      device const Q81* activation) {
    const device ushort* codes = weight->qs + 4u * part;
    const device int* activation_words =
        reinterpret_cast<device const int*>(activation->qs);
    const int scale_pair = int(weight->scales[part]);
    const int ls0 = scale_pair & 0x0f;
    const int ls1 = scale_pair >> 4u;

    int sumi0 = 0;
    int sumi1 = 0;
    for (uint code_slot = 0u; code_slot < 4u; ++code_slot) {
        const uint codeword = uint(codes[code_slot]);
        const uint grid_index = codeword & 0x1ffu;
        const uint sign_index = codeword >> 9u;
        const uint signs = uint(ksigns_iq2xs[sign_index]);
        const ulong grid = iq2xs_grid[grid_index];
        const char4 low = iq2_xs_signed_quartet(
            uint(grid), signs, 0u);
        const char4 high = iq2_xs_signed_quartet(
            uint(grid >> 32u), signs, 4u);
        int partial = code_slot < 2u ? sumi0 : sumi1;
        partial = dot4_i8(as_type<int>(low),
                          activation_words[2u * code_slot]) + partial;
        partial = dot4_i8(as_type<int>(high),
                          activation_words[2u * code_slot + 1u]) + partial;
        if (code_slot < 2u) {
            sumi0 = partial;
        } else {
            sumi1 = partial;
        }
    }
    return (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
}

#define STRATA_MV_IQ2_XS(NC)                                                                   \
    [[max_total_threads_per_threadgroup(128)]]                                                  \
    kernel void strata_mv_iq2_xs_n##NC(                                                        \
        device const Q81* xq [[buffer(0)]],                                                     \
        device const block_iq2_xs* weights [[buffer(1)]],                                      \
        device float* y [[buffer(2)]],                                                          \
        device float* sink [[buffer(3)]],                                                       \
        device const uint* rows [[buffer(4)]],                                                   \
        constant IQ2XSMVParams& params [[buffer(5)]],                                           \
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
        const device block_iq2_xs* row_weights = weights + row * blocks_per_row;                \
        float accum[column_count];                                                              \
        for (uint j = 0u; j < column_count; ++j) accum[j] = 0.0f;                               \
        for (uint k = lane; k < calls_per_row; k += 32u) {                                      \
            const uint kbx = k / 8u;                                                            \
            const uint part = k % 8u;                                                           \
            const device block_iq2_xs* weight = row_weights + kbx;                              \
            for (uint j = 0u; j < column_count; ++j) {                                          \
                const uint input_column = j < live_rows ? j : 0u;                               \
                const device Q81* activation =                                                   \
                    xq + input_column * (params.n_in / 32u) + kbx * 8u + part;                  \
                const int sumi = iq2_xs_integer_dot(weight, part, activation);                   \
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

STRATA_MV_IQ2_XS(1)
STRATA_MV_IQ2_XS(2)
STRATA_MV_IQ2_XS(4)
STRATA_MV_IQ2_XS(8)

#undef STRATA_MV_IQ2_XS
