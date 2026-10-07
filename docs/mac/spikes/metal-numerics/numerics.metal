// Kernels for the numerics spike (README.md). build.sh compiles this file three times: with the safe flags of
// docs/mac/08-build-and-distribution.md section 5.3, with those flags minus -ffp-contract=off, and with the fast flags.
#include <metal_stdlib>
using namespace metal;

#define UNARY(name, expr)                                                                       \
    kernel void name(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],        \
                     uint i [[thread_position_in_grid]]) { float x = a[i]; o[i] = (expr); }
#define BINARY(name, expr)                                                                      \
    kernel void name(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],  \
                     device float *o [[buffer(2)]], uint i [[thread_position_in_grid]]) {        \
        float x = a[i], y = b[i]; o[i] = (expr); }

// a*b + c as written, and with fma(): which one the compiler emits for the first is the contraction question.
kernel void k_mad(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                  device const float *c [[buffer(2)]], device float *o [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) { o[i] = a[i] * b[i] + c[i]; }
kernel void k_fma(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                  device const float *c [[buffer(2)]], device float *o [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) { o[i] = fma(a[i], b[i], c[i]); }

BINARY(k_div, x / y)
BINARY(k_pdiv, precise::divide(x, y))
UNARY(k_recip, 1.0f / x)
UNARY(k_sqrt, sqrt(x))
UNARY(k_rsqrt, rsqrt(x))
UNARY(k_exp, exp(x))
UNARY(k_pexp, precise::exp(x))
// MSL has no log1p. Upstream's softplus is x > 20 ? x : log1pf(expf(x)) (gdn.cu, fused_gdn.cu and three more).
// k_log1p_naive is log(1 + x); k_log1p is the Goldberg/Kahan form, exact in u = 1 + x and correct to a few ulps.
static inline float log1p_k(float x) { float u = 1.0f + x; return u == 1.0f ? x : log(u) * (x / (u - 1.0f)); }
UNARY(k_log1p_naive, log(1.0f + x))
UNARY(k_log1p, log1p_k(x))
UNARY(k_softplus, x > 20.0f ? x : log1p_k(exp(x)))
UNARY(k_sigmoid, 1.0f / (1.0f + exp(-x)))
UNARY(k_rint, rint(x))
UNARY(k_round, round(x))
UNARY(k_trunc, trunc(x))

// fast:: functions are only built in the fast library; a safe file that calls them fails the check of doc 08
// section 5.4.
#ifdef SPIKE_FAST
UNARY(k_fexp, fast::exp(x))
UNARY(k_fdiv, fast::divide(x, 3.0f))
#endif

kernel void k_half(device const float *a [[buffer(0)]], device ushort *o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) { o[i] = as_type<ushort>(half(a[i])); }

// simd_sum against an explicit xor butterfly (o = 16, 8, 4, 2, 1), and simd_max against the same butterfly with max.
kernel void k_simd(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    float v = a[i];
    float s = simd_sum(v);
    float x = v;
    for (ushort m = 16; m > 0; m >>= 1) x += simd_shuffle_xor(x, m);
    float mx = simd_max(v);
    float y = v;
    for (ushort m = 16; m > 0; m >>= 1) y = max(y, simd_shuffle_xor(y, m));
    o[4 * i + 0] = s;
    o[4 * i + 1] = x;
    o[4 * i + 2] = mx;
    o[4 * i + 3] = y;
}

// Lane and simdgroup ids, and the ballot width.
kernel void k_ids(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint w [[threads_per_simdgroup]]) {
    simd_vote v = simd_ballot((lane & 1) != 0);
    ulong bits = ulong(simd_vote::vote_t(v));
    o[5 * i + 0] = lane;
    o[5 * i + 1] = sg;
    o[5 * i + 2] = w;
    o[5 * i + 3] = uint(bits);
    o[5 * i + 4] = uint(bits >> 32);
}

// Threadgroup memory: exactly 32 KiB of static threadgroup memory. tg_over.metal asks for 16 B more.
kernel void k_tg32k(device float *o [[buffer(0)]], uint i [[thread_position_in_threadgroup]]) {
    threadgroup float t[8192];
    for (uint j = i; j < 8192; j += 1024) t[j] = float(j);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = t[8191 - i];
}

// Codebook lookups with divergent indices, from a program-scope constant array, a constant buffer, a device buffer
// and a threadgroup copy. Three tables: the 16 signed bytes of kvalues_iq4nl (upstream measured 2.12x for
// __constant__ on CUDA with this one, s_gemv.cu), 512 x u64 (4 KiB, iq2xs_grid) and 1024 x u64 (8 KiB, iq2s_grid; the
// largest grid, as large as iq1s_grid_gpu). Each thread does N_LOOKUPS lookups: "cb" kernels with independent
// xorshift indices, "ch" kernels with four dependent chains, where each index depends on the previous value.
constant constexpr uint N_LOOKUPS = 2048;

static inline uint next(thread uint &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

#include "grid.inc"   // kKv[16] (char), kGrid[512] and kGrid1k[1024] (ulong), program-scope constant arrays

#define cb_LOOP(TABLE, MASK)                                                                       \
    uint s = i * 2654435761u + 1u; ulong acc = 0;                                                  \
    for (uint n = 0; n < N_LOOKUPS; ++n) acc += TABLE[next(s) & MASK];                             \
    o[i] = acc;
#define ch_LOOP(TABLE, MASK)                                                                       \
    uint c0 = i * 2654435761u + 1u, c1 = c0 + 1u, c2 = c0 + 2u, c3 = c0 + 3u;                      \
    for (uint n = 0; n < N_LOOKUPS / 4; ++n) {                                                     \
        c0 = (c0 ^ uint(TABLE[c0 & MASK] >> 7)) + n; c1 = (c1 ^ uint(TABLE[c1 & MASK] >> 7)) + n;  \
        c2 = (c2 ^ uint(TABLE[c2 & MASK] >> 7)) + n; c3 = (c3 ^ uint(TABLE[c3 & MASK] >> 7)) + n;  \
    }                                                                                              \
    o[i] = ulong(c0) + c1 + c2 + c3;
#define TG_COPY(T, MASK)                                                                           \
    threadgroup T tg[MASK + 1];                                                                    \
    for (uint j = t; j <= MASK; j += tn) tg[j] = g[j];                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define CB_SET(K, SUF, T, PROG, MASK)                                                              \
    kernel void k_##K##_program##SUF(device ulong *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { \
        K##_LOOP(PROG, MASK) }                                                                     \
    kernel void k_##K##_constant##SUF(constant T *g [[buffer(1)]], device ulong *o [[buffer(0)]],  \
                                      uint i [[thread_position_in_grid]]) { K##_LOOP(g, MASK) }    \
    kernel void k_##K##_device##SUF(device const T *g [[buffer(1)]], device ulong *o [[buffer(0)]], \
                                    uint i [[thread_position_in_grid]]) { K##_LOOP(g, MASK) }      \
    kernel void k_##K##_threadgroup##SUF(device const T *g [[buffer(1)]], device ulong *o [[buffer(0)]], \
                                         uint i [[thread_position_in_grid]],                       \
                                         uint t [[thread_position_in_threadgroup]],                \
                                         uint tn [[threads_per_threadgroup]]) {                    \
        TG_COPY(T, MASK) K##_LOOP(tg, MASK) }

CB_SET(cb, _16, char, kKv, 15)
CB_SET(ch, _16, char, kKv, 15)
CB_SET(cb, _512, ulong, kGrid, 511)
CB_SET(ch, _512, ulong, kGrid, 511)
CB_SET(cb, _1k, ulong, kGrid1k, 1023)
CB_SET(ch, _1k, ulong, kGrid1k, 1023)

// No lookup: the cost of the index arithmetic alone.
kernel void k_cb_none(device ulong *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    uint s = i * 2654435761u + 1u; ulong acc = 0;
    for (uint n = 0; n < N_LOOKUPS; ++n) acc += next(s);
    o[i] = acc;
}

// Last in the file, so the pragma cannot leak into another kernel: does it override the file's contraction mode?
#pragma METAL fp contract(off)
kernel void k_mad_pragma_off(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                             device const float *c [[buffer(2)]], device float *o [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) { o[i] = a[i] * b[i] + c[i]; }
