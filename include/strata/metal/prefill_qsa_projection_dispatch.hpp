// Ordered Prefill QSA input projections (docs/mac/04 §6.5, steps 1–2).
#pragma once

#include "strata/metal/prefill_gemm_dispatch.hpp"
#include "strata/metal/prefill_gr_norm_dispatch.hpp"
#include "strata/metal/prefill_native_projection.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillQsaProjectionBuffers {
    ReadOnlyBufferRef mixed_h;
    ReadOnlyBufferRef mixed_bf;
    /// Required in BF16X2 modes 1 and 2; empty in mode 0.
    ReadOnlyBufferRef mixed_bf_lo;

    ReadOnlyBufferRef attn_k_weights;
    PrefillNativeWeightFormat attn_k_format =
        PrefillNativeWeightFormat::Invalid;
    ReadOnlyBufferRef attn_v_weights;
    PrefillNativeWeightFormat attn_v_format =
        PrefillNativeWeightFormat::Invalid;
    ReadOnlyBufferRef attn_q_weights;
    PrefillNativeWeightFormat attn_q_format =
        PrefillNativeWeightFormat::Invalid;
    ReadOnlyBufferRef indexer_k_weights;
    ReadOnlyBufferRef indexer_q_weights;

    /// Reused in order by the attn_k, attn_v and attn_q projections.
    BufferRef dequantized_f16;
    BufferRef key;
    BufferRef value;
    BufferRef query_full;
    BufferRef indexer_key;
    BufferRef indexer_query;
};

struct PrefillQsaProjectionBufferSizes {
    PrefillGemmMPPParams attn_k_params;
    PrefillGemmMPPParams attn_v_params;
    PrefillGemmMPPParams attn_q_params;
    PrefillGemmMPPParams indexer_k_params;
    PrefillGemmMPPParams indexer_q_params;
    PrefillNativeProjectionBufferSizes attn_k;
    PrefillNativeProjectionBufferSizes attn_v;
    PrefillNativeProjectionBufferSizes attn_q;
    std::size_t mixed_bf_bytes = 0u;
    /// Maximum low-component extent needed by BF16X2 modes 1 and 2.
    std::size_t mixed_bf_lo_bytes = 0u;
    std::size_t indexer_k_weights_bytes = 0u;
    std::size_t indexer_q_weights_bytes = 0u;
    std::size_t dequantized_f16_bytes = 0u;
    std::size_t indexer_k_output_bytes = 0u;
    std::size_t indexer_q_output_bytes = 0u;
};

/// Return active input, weight, scratch and output extents for the five
/// projections in QSA steps 1–2. Native K/V/Q formats are independent.
bool prefill_qsa_projection_buffer_sizes(
    std::uint32_t rows, PrefillQsaProjectionBufferSizes& sizes,
    const PrefillQsaProjectionBuffers& buffers, std::string& error);

/// Encode QSA K/V/Q native projections followed by indexer K/Q BF16
/// projections in one ordered Encoder segment. The native dequant scratch is
/// reused serially. Modes 1 and 2 project both BF16 activation components.
/// Propagate false from the enclosing Encoder::encode body on failure.
bool encode_prefill_qsa_projection(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillQsaProjectionBuffers& buffers,
    std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

/// Transactional standalone form; a failure discards every earlier
/// projection encoded into the command buffer.
bool encode_prefill_qsa_projection(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillQsaProjectionBuffers& buffers,
    std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

}  // namespace strata::metal
