// Shared-expert Prefill MoE projection and gate composition.
#pragma once

#include "strata/metal/prefill_native_projection.hpp"
#include "strata/metal/prefill_gr_norm_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

/// Native formats for the shared expert's gate, up, and down projections.
struct PrefillMoeSharedFFNFormats {
    PrefillNativeWeightFormat gate = PrefillNativeWeightFormat::Invalid;
    PrefillNativeWeightFormat up = PrefillNativeWeightFormat::Invalid;
    PrefillNativeWeightFormat down = PrefillNativeWeightFormat::Invalid;
};

/// Caller-owned active views for Prefill MoE steps 7–10.
///
/// The three native projections each require dedicated dequantized-F16
/// scratch. `gate_f32` and `up_f32` feed the F16 `hidden` SwiGLU output;
/// `hidden` feeds `shared_f32` through the down projection. `router_gate`
/// supplies the BF16X2 shared gate projection and writes one F32 value per
/// row to its `y` view.
struct PrefillMoeSharedFFNBuffers {
    ReadOnlyBufferRef mixed_h;
    ReadOnlyBufferRef gate_weights;
    BufferRef gate_dequantized_f16;
    BufferRef gate_f32;
    ReadOnlyBufferRef up_weights;
    BufferRef up_dequantized_f16;
    BufferRef up_f32;
    BufferRef hidden_f16;
    ReadOnlyBufferRef down_weights;
    BufferRef down_dequantized_f16;
    BufferRef shared_f32;
    PrefillGemmBF16X2Buffers router_gate;
};

/// Encode Prefill MoE's shared expert: native gate/up projections, F16
/// SwiGLU, native down projection, then the BF16X2 shared gate logits.
/// Supports T=1..8. Writable outputs and dequantization scratch must be
/// mutually disjoint and disjoint from all inputs; read-only inputs may alias.
/// The BF16X2 shared gate requires projection mode 1 or 2. For caller-owned
/// use, propagate false from the enclosing `Encoder::encode` body so an
/// already encoded prefix is discarded.
bool encode_prefill_moe_shared_ffn(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeSharedFFNFormats& formats,
    const PrefillMoeSharedFFNBuffers& buffers, std::string& error);

/// Transactional standalone form. It preflights all active stage ranges
/// before encoding any projection.
bool encode_prefill_moe_shared_ffn(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeSharedFFNFormats& formats,
    const PrefillMoeSharedFFNBuffers& buffers, std::string& error);

}  // namespace strata::metal
