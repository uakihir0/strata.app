// BF16 weights with unrounded F32 activations (docs/mac/04-metal-kernels.md §5.9).
#include "types.h"
#include "math.h"
#include "reduce.h"

using namespace metal;

struct BF16MVParams {
    uint n_in;
    uint n_out;
};

#define STRATA_MV_BF16(BS, NC)                                                                    \
    [[max_total_threads_per_threadgroup(BS)]]                                                     \
    kernel void strata_mv_bf16_b##BS##_n##NC(                                                     \
        device const float* x [[buffer(0)]],                                                       \
        device const uint* weights2 [[buffer(1)]],                                                \
        device float* y [[buffer(2)]],                                                            \
        device float* sink [[buffer(3)]],                                                         \
        device const uint* rows [[buffer(4)]],                                                    \
        constant BF16MVParams& params [[buffer(5)]],                                              \
        uint3 tid3 [[thread_position_in_threadgroup]],                                            \
        uint3 group [[threadgroup_position_in_grid]]) {                                           \
        const uint tid = tid3.x;                                                                  \
        const uint row = group.x;                                                                 \
        const uint live_rows = *rows;                                                             \
        threadgroup float partials[NC][32];                                                       \
        float accum[NC];                                                                          \
        for (uint j = 0; j < NC; ++j) accum[j] = 0.0f;                                            \
        if (BS > 32u) {                                                                           \
            if (tid < 32u) {                                                                      \
                for (uint j = 0; j < NC; ++j) partials[j][tid] = 0.0f;                            \
            }                                                                                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
        }                                                                                         \
        for (uint pair = tid; pair < params.n_in / 2u; pair += BS) {                              \
            const uint word = weights2[row * (params.n_in / 2u) + pair];                          \
            const float w0 = bf16_to_f32(ushort(word & 0xffffu));                                 \
            const float w1 = bf16_to_f32(ushort(word >> 16u));                                    \
            for (uint j = 0; j < NC; ++j) {                                                       \
                const uint column = j < live_rows ? j : 0u;                                       \
                const device float2* x2 = (const device float2*)(x + column * params.n_in);        \
                const float2 input = x2[pair];                                                    \
                accum[j] = fma(w0, input.x, accum[j]);                                             \
                accum[j] = fma(w1, input.y, accum[j]);                                             \
            }                                                                                     \
        }                                                                                         \
        for (uint j = 0; j < NC; ++j) accum[j] = bfly_sum(accum[j]);                              \
        if (BS > 32u) {                                                                           \
            for (uint j = 0; j < NC; ++j) {                                                       \
                if ((tid & 31u) == 0u) partials[j][tid / 32u] = accum[j];                         \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
                if (tid < 32u) accum[j] = bfly_sum(partials[j][tid]);                             \
            }                                                                                     \
        }                                                                                         \
        if (tid == 0u) {                                                                          \
            for (uint j = 0; j < NC; ++j) {                                                       \
                device float* destination = j < live_rows ? y + j * params.n_out                  \
                                                           : sink + (j - live_rows) * params.n_out;\
                destination[row] = accum[j];                                                      \
            }                                                                                     \
        }                                                                                         \
    }

STRATA_MV_BF16(160, 1)
STRATA_MV_BF16(160, 2)
STRATA_MV_BF16(160, 4)
STRATA_MV_BF16(160, 8)
STRATA_MV_BF16(256, 1)
STRATA_MV_BF16(256, 2)
STRATA_MV_BF16(256, 4)
STRATA_MV_BF16(256, 8)

#undef STRATA_MV_BF16
