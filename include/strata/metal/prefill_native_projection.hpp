// Ordered native-weight dequantization plus F16 Prefill projection.
#pragma once

#include "strata/metal/prefill_gemm_dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

enum class PrefillNativeWeightFormat : std::uint8_t {
    Q2_0,
    IQ4_NL,
    Q4_0,
    Q5_0,
    Q8_0,
    Q3_K,
    IQ4_XS,
    IQ3_S,
    Q4_K,
    Q5_K,
    Q6_K,
    Invalid = 0xff,
};

struct PrefillNativeProjectionBuffers {
    ReadOnlyBufferRef x_f16;
    ReadOnlyBufferRef weights;
    BufferRef dequantized_f16;
    BufferRef y_f32;
};

struct PrefillNativeProjectionBufferSizes {
    std::size_t x_f16_bytes = 0u;
    std::size_t weights_bytes = 0u;
    std::size_t dequantized_f16_bytes = 0u;
    std::size_t y_f32_bytes = 0u;
    std::size_t weights_alignment = 1u;
};

/// Return the active byte extents required by one native-weight projection.
/// The output size spans through the final written column in the last row.
bool prefill_native_projection_buffer_sizes(
    const PrefillGemmMPPParams& params, PrefillNativeWeightFormat format,
    PrefillNativeProjectionBufferSizes& sizes, std::string& error);

/// Dequantize native weights to K-major F16 scratch, then encode the F16
/// projection into row-major F32 output in the same ordered Encoder segment.
/// The shared-Encoder form may return false after the dequant dispatch has
/// been encoded; callers must propagate failure from their enclosing
/// Encoder::encode body. Writable views must not overlap any active input or
/// each other.
bool encode_prefill_native_projection(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params, PrefillNativeWeightFormat format,
    const PrefillNativeProjectionBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

/// Transactional standalone wrapper. Any late GEMM validation or dispatch
/// failure discards the earlier dequant dispatch as well.
bool encode_prefill_native_projection(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params, PrefillNativeWeightFormat format,
    const PrefillNativeProjectionBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

}  // namespace strata::metal
