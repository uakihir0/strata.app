// Validated encoders for isolated and selected Prefill F16/BF16 GEMM paths.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>
#include <string_view>

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

enum class PrefillGemmBackend : std::uint8_t { MPP, SIMD };
enum class PrefillGemmFormat : std::uint8_t { F16, BF16 };

/// Resolve STRATA_METAL_GEMM for the projection shape. An empty setting uses
/// the device default. Projections with n < 64 always select SIMD. MPP
/// requires Apple10 until cross-family support is validated.
bool resolve_prefill_gemm_backend(
    bool apple10, std::uint32_t n, std::string_view setting,
    PrefillGemmBackend& backend, std::string& error);

/// Select a validated F16/BF16 encoder from STRATA_METAL_GEMM and the active
/// device family. The standalone and caller-owned Encoder forms share policy.
bool encode_prefill_gemm(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, PrefillGemmFormat format,
    std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);
bool encode_prefill_gemm(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, PrefillGemmFormat format,
    std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

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
