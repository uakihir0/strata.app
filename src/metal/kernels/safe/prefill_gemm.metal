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
