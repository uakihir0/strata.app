// Validated encoder for the isolated Prefill BF16 MPP GEMM component.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGemmBF16MPPParams {
    std::uint32_t rows = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    std::uint32_t ldy = 0;
    std::uint32_t col0 = 0;
    bool accumulate = false;
};

struct PrefillGemmBF16MPPBuffers {
    ReadOnlyBufferRef x;
    ReadOnlyBufferRef w;
    BufferRef y;
};

/// Encode an Apple10 BF16 matrix product. X is row-major [rows, k], W is
/// row-major [n, k], and the output starts at column col0 in row-major Y with
/// leading dimension ldy. The component is isolated and is not yet wired into
/// the production Prefill path.
bool encode_prefill_gemm_bf16_mpp(
    Context& context, Encoder& encoder,
    const PrefillGemmBF16MPPParams& params,
    const PrefillGemmBF16MPPBuffers& buffers, std::string& error);

/// Standalone transactional wrapper for the Prefill BF16 MPP matrix product.
bool encode_prefill_gemm_bf16_mpp(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmBF16MPPParams& params,
    const PrefillGemmBF16MPPBuffers& buffers, std::string& error);

}  // namespace strata::metal
