// Ordered Prefill GDN projection, transform, and output block (docs/mac/04 §6.4).
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/prefill_gr_norm_dispatch.hpp"
#include "strata/metal/prefill_native_projection.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

/// Inputs and scratch for one GDN layer chunk, from GR mixer output through
/// the GDN output projection. Native formats are supplied per tensor because
/// the model files use different quantization formats across layers.
struct PrefillGDNBlockBuffers {
    ReadOnlyBufferRef mixed_h;
    ReadOnlyBufferRef mixed_bf;
    /// Required in BF16X2 modes 1 and 2; must be empty in mode 0.
    ReadOnlyBufferRef mixed_bf_lo;

    ReadOnlyBufferRef attn_qkv_weights;
    PrefillNativeWeightFormat attn_qkv_format =
        PrefillNativeWeightFormat::Invalid;
    ReadOnlyBufferRef attn_gate_weights;
    PrefillNativeWeightFormat attn_gate_format =
        PrefillNativeWeightFormat::Invalid;
    ReadOnlyBufferRef ssm_alpha_weights;
    ReadOnlyBufferRef ssm_beta_weights;
    ReadOnlyBufferRef ssm_out_weights;
    PrefillNativeWeightFormat ssm_out_format =
        PrefillNativeWeightFormat::Invalid;

    ReadOnlyBufferRef dt_bias;
    ReadOnlyBufferRef a;
    ReadOnlyBufferRef conv_weights;
    ReadOnlyBufferRef gamma;

    BufferRef dequantized_f16;
    BufferRef qkv;
    BufferRef z;
    BufferRef alpha_beta;
    BufferRef history;
    BufferRef gate;
    BufferRef beta;
    BufferRef h;
    BufferRef state;
    BufferRef oc;
    BufferRef y_h;
    BufferRef bo;
};

/// Encode one GDN layer chunk in docs/mac/04 §6.4 order, consuming GR mixer
/// outputs and producing `bo`. Modes 1 and 2 apply BF16X2 to the GDN
/// alpha/beta projections; mode 0 uses their BF16 high components only.
///
/// All active views must belong to `context`. Simultaneously live writable
/// ranges must be disjoint from each other and from external read-only inputs.
/// `dequantized_f16` is reused serially for qkv, z, and ssm_out. `history` and
/// `state` are persistent in-place inputs/outputs.
/// This caller-owned form may return false after earlier stages have already
/// been encoded. The enclosing Encoder::encode body must propagate false so
/// the unsubmitted block is discarded.
bool encode_prefill_gdn_block(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGDNBlockBuffers& buffers,
    std::string& error);

/// Transactional standalone form. Any validation or late projection failure
/// discards all earlier encodes in the current command buffer.
bool encode_prefill_gdn_block(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGDNBlockBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
