// Isolated Prefill BF16 MPP projection kernel (docs/mac/04 §6.2).
#include <metal_stdlib>
#include <metal_tensor>
#ifndef __HAVE_BFLOAT__
#define __HAVE_BFLOAT__ 1
#endif
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gemm_f16_mpp(
    device const half* x [[buffer(0)]],
    device const half* w_k_major [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& ldy [[buffer(6)]],
    constant uint& col0 [[buffer(7)]],
    constant uint& accumulate [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        const_cast<device half*>(x), dextents<int32_t, 2>(k, rows));
    auto tW = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        const_cast<device half*>(w_k_major), dextents<int32_t, 2>(n, k));
    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        y + col0, dextents<int32_t, 2>(n, rows),
        array<int32_t, 2>{1, static_cast<int32_t>(ldy)});

    auto mX = tX.slice(0, 64 * group.y);
    auto mW = tW.slice(64 * group.x, 0);
    auto mY = tY.slice(64 * group.x, 64 * group.y);
    if (accumulate != 0u) {
        constexpr auto desc = matmul2d_descriptor(
            64, 64, static_cast<int>(dynamic_extent), false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(mX, mW, mY);
    } else {
        constexpr auto desc = matmul2d_descriptor(
            64, 64, static_cast<int>(dynamic_extent), false, false, false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(mX, mW, mY);
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gemm_bf16_mpp(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& ldy [[buffer(6)]],
    constant uint& col0 [[buffer(7)]],
    constant uint& accumulate [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    // MPP's cooperative-operand traits currently require unqualified input
    // element types. The dispatch ABI remains const and these views are read
    // only throughout this kernel.
    auto tX = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        const_cast<device bfloat*>(x), dextents<int32_t, 2>(k, rows));
    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        const_cast<device bfloat*>(w), dextents<int32_t, 2>(k, n));
    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        y + col0, dextents<int32_t, 2>(n, rows),
        array<int32_t, 2>{1, static_cast<int32_t>(ldy)});

    auto mX = tX.slice(0, 64 * group.y);
    auto mW = tW.slice(0, 64 * group.x);
    auto mY = tY.slice(64 * group.x, 64 * group.y);
    if (accumulate != 0u) {
        constexpr auto desc = matmul2d_descriptor(
            64, 64, static_cast<int>(dynamic_extent), false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(mX, mW, mY);
    } else {
        constexpr auto desc = matmul2d_descriptor(
            64, 64, static_cast<int>(dynamic_extent), false, true, false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(mX, mW, mY);
    }
}

// SIMD-group matrix fallback for every Apple GPU family. K is a multiple of
// 32, so every threadgroup processes complete K tiles and masks only the T/n
// edges. Each of four SIMD-groups owns a disjoint 16x32 region of the 32x64
// output tile.
[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gemm_f16_simd(
    device const half* x [[buffer(0)]],
    device const half* w_k_major [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& ldy [[buffer(6)]],
    constant uint& col0 [[buffer(7)]],
    constant uint& accumulate [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sx[32][32];
    threadgroup half sw[32][64];
    threadgroup float so[32][64];

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 4u; ++c) {
            acc[i][c] = simdgroup_float8x8(0.0f);
        }
    }

    const uint t0 = 32u * group.y;
    const uint n0 = 64u * group.x;
    const uint row_base = 16u * (static_cast<uint>(sg) / 2u);
    const uint column_base = 32u * (static_cast<uint>(sg) % 2u);

    for (uint k0 = 0u; k0 < k; k0 += 32u) {
        for (uint i = tid; i < 32u * 32u; i += 128u) {
            const uint tile_row = i / 32u;
            const uint tile_k = i % 32u;
            const uint row = t0 + tile_row;
            sx[tile_row][tile_k] =
                row < rows
                    ? x[ulong(row) * ulong(k) + k0 + tile_k]
                    : half(0.0f);
        }
        for (uint i = tid; i < 32u * 64u; i += 128u) {
            const uint tile_k = i / 64u;
            const uint tile_column = i % 64u;
            const uint column = n0 + tile_column;
            sw[tile_k][tile_column] =
                column < n
                    ? w_k_major[ulong(k0 + tile_k) * ulong(n) + column]
                    : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a[2];
            simdgroup_half8x8 b[4];
            for (uint i = 0u; i < 2u; ++i) {
                simdgroup_load(a[i], &sx[row_base + 8u * i][kk], 32);
            }
            for (uint c = 0u; c < 4u; ++c) {
                simdgroup_load(b[c], &sw[kk][column_base + 8u * c], 64);
            }
            for (uint i = 0u; i < 2u; ++i) {
                for (uint c = 0u; c < 4u; ++c) {
                    simdgroup_multiply_accumulate(
                        acc[i][c], a[i], b[c], acc[i][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 4u; ++c) {
            simdgroup_store(acc[i][c],
                            &so[row_base + 8u * i][column_base + 8u * c],
                            64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 32u * 64u; i += 128u) {
        const uint tile_row = i / 64u;
        const uint tile_column = i % 64u;
        const uint row = t0 + tile_row;
        const uint column = n0 + tile_column;
        if (row < rows && column < n) {
            const ulong output_index =
                ulong(row) * ulong(ldy) + col0 + column;
            const float product = so[tile_row][tile_column];
            y[output_index] =
                accumulate != 0u ? product + y[output_index] : product;
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void strata_pf_gemm_bf16_simd(
    device const ushort* x_bf16 [[buffer(0)]],
    device const ushort* w_bf16 [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& ldy [[buffer(6)]],
    constant uint& col0 [[buffer(7)]],
    constant uint& accumulate [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sx[32][32];
    threadgroup float sw[32][64];
    threadgroup float so[32][64];

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 4u; ++c) {
            acc[i][c] = simdgroup_float8x8(0.0f);
        }
    }

    const uint t0 = 32u * group.y;
    const uint n0 = 64u * group.x;
    const uint row_base = 16u * (static_cast<uint>(sg) / 2u);
    const uint column_base = 32u * (static_cast<uint>(sg) % 2u);

    for (uint k0 = 0u; k0 < k; k0 += 32u) {
        for (uint i = tid; i < 32u * 32u; i += 128u) {
            const uint tile_row = i / 32u;
            const uint tile_k = i % 32u;
            const uint row = t0 + tile_row;
            const uint bits =
                row < rows
                    ? static_cast<uint>(
                          x_bf16[ulong(row) * ulong(k) + k0 + tile_k])
                           : 0u;
            sx[tile_row][tile_k] = as_type<float>(bits << 16u);
        }
        for (uint i = tid; i < 32u * 64u; i += 128u) {
            const uint tile_k = i / 64u;
            const uint tile_column = i % 64u;
            const uint column = n0 + tile_column;
            const uint bits =
                column < n
                    ? static_cast<uint>(
                          w_bf16[ulong(column) * ulong(k) + k0 + tile_k])
                    : 0u;
            // BF16 file layout is W[n][K], while the staged right operand is
            // transposed to the K-major tile expected by multiply-accumulate.
            sw[tile_k][tile_column] = as_type<float>(bits << 16u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_float8x8 a[2];
            simdgroup_float8x8 b[4];
            for (uint i = 0u; i < 2u; ++i) {
                simdgroup_load(a[i], &sx[row_base + 8u * i][kk], 32);
            }
            for (uint c = 0u; c < 4u; ++c) {
                simdgroup_load(b[c], &sw[kk][column_base + 8u * c], 64);
            }
            for (uint i = 0u; i < 2u; ++i) {
                for (uint c = 0u; c < 4u; ++c) {
                    simdgroup_multiply_accumulate(
                        acc[i][c], a[i], b[c], acc[i][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 2u; ++i) {
        for (uint c = 0u; c < 4u; ++c) {
            simdgroup_store(acc[i][c],
                            &so[row_base + 8u * i][column_base + 8u * c],
                            64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 32u * 64u; i += 128u) {
        const uint tile_row = i / 64u;
        const uint tile_column = i % 64u;
        const uint row = t0 + tile_row;
        const uint column = n0 + tile_column;
        if (row < rows && column < n) {
            const ulong output_index =
                ulong(row) * ulong(ldy) + col0 + column;
            const float product = so[tile_row][tile_column];
            y[output_index] =
                accumulate != 0u ? product + y[output_index] : product;
        }
    }
}
