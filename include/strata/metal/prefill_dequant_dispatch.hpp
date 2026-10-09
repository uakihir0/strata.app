// GPU encoding for native-weight dequantization in the Metal Prefill path.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillDequantParams {
    std::uint32_t n_out = 0;
    std::uint32_t k = 0;
};

struct PrefillDequantBuffers {
    ReadOnlyBufferRef weights;
    BufferRef k_major_f16;
};

/// Expand row-major Q8_0 blocks to F16 K-major scratch `[k][n_out]`.
/// Input blocks are read-only; output must not overlap the weight view.
bool encode_prefill_dequant_q8_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

/// Standalone transactional wrapper for a fresh command buffer.
bool encode_prefill_dequant_q8_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

/// Expand row-major Q2_0 blocks to F16 K-major scratch `[k][n_out]`.
bool encode_prefill_dequant_q2_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);
bool encode_prefill_dequant_q2_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

/// Expand row-major IQ4_NL blocks to F16 K-major scratch `[k][n_out]`.
bool encode_prefill_dequant_iq4_nl(Context& context, Encoder& encoder,
                                   const PrefillDequantParams& params,
                                   const PrefillDequantBuffers& buffers,
                                   std::string& error);
bool encode_prefill_dequant_iq4_nl(
    Context& context, CommandBuffer& command_buffer,
    const PrefillDequantParams& params, const PrefillDequantBuffers& buffers,
    std::string& error);

/// Expand row-major Q4_0 blocks to F16 K-major scratch `[k][n_out]`.
bool encode_prefill_dequant_q4_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);
bool encode_prefill_dequant_q4_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

/// Expand row-major Q5_0 blocks to F16 K-major scratch `[k][n_out]`.
bool encode_prefill_dequant_q5_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);
bool encode_prefill_dequant_q5_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

/// Expand row-major Q4_K blocks to F16 K-major scratch `[k][n_out]`.
bool encode_prefill_dequant_q4_k(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);
bool encode_prefill_dequant_q4_k(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

/// Expand row-major Q5_K blocks to F16 K-major scratch `[k][n_out]`.
bool encode_prefill_dequant_q5_k(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);
bool encode_prefill_dequant_q5_k(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

/// Expand row-major Q6_K blocks to F16 K-major scratch `[k][n_out]`.
bool encode_prefill_dequant_q6_k(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);
bool encode_prefill_dequant_q6_k(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error);

}  // namespace strata::metal
