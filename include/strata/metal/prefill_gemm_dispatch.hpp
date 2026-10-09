// Validated encoders for the isolated Prefill F16/BF16 MPP GEMM components.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGemmMPPParams {
    std::uint32_t rows = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    std::uint32_t ldy = 0;
    std::uint32_t col0 = 0;
    bool accumulate = false;
};

struct PrefillGemmMPPBuffers {
    ReadOnlyBufferRef x;
    ReadOnlyBufferRef w;
    BufferRef y;
};

using PrefillGemmBF16MPPParams = PrefillGemmMPPParams;
using PrefillGemmBF16MPPBuffers = PrefillGemmMPPBuffers;
using PrefillGemmF16MPPParams = PrefillGemmMPPParams;
using PrefillGemmF16MPPBuffers = PrefillGemmMPPBuffers;

/// Encode an Apple10 BF16 matrix product. X is row-major [rows, k], W is
/// row-major [n, k], and the output starts at column col0 in row-major Y with
/// leading dimension ldy. This isolated component is not yet wired into
/// production Prefill.
bool encode_prefill_gemm_bf16_mpp(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);

/// Standalone transactional wrapper for the Prefill BF16 MPP matrix product.
bool encode_prefill_gemm_bf16_mpp(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);

/// Encode an Apple10 F16 matrix product. X is row-major [rows, k], W is the
/// K-major scratch [k, n], and output layout, validation, and isolation match
/// the BF16 API.
bool encode_prefill_gemm_f16_mpp(
    Context& context, Encoder& encoder, const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);

/// Standalone transactional wrapper for the Prefill F16 MPP matrix product.
bool encode_prefill_gemm_f16_mpp(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);

/// Encode the F16/F32-accumulate SIMD-group matrix path on Apple Silicon.
/// F16 uses K-major scratch weights; the call is independent of the MPP path.
bool encode_prefill_gemm_f16_simd(
    Context& context, Encoder& encoder, const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);
bool encode_prefill_gemm_f16_simd(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);

/// Encode the BF16/F32-accumulate SIMD-group matrix path on Apple Silicon.
/// BF16 X and W use file layout; W is widened exactly to F32 in threadgroup
/// memory before the matrix operations.
bool encode_prefill_gemm_bf16_simd(
    Context& context, Encoder& encoder, const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);
bool encode_prefill_gemm_bf16_simd(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error);

}  // namespace strata::metal
