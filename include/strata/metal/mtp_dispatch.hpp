// include/strata/metal/mtp_dispatch.hpp - MTP drafter kernel encoders.
#pragma once

#include "strata/metal/context.hpp"

#include <string>

namespace strata::metal {

class Encoder;

/// Inputs for stage 6 of one MTP pass. `params` must be a CPU-visible,
/// immutable MtpParams block whose T is in 1..8. All views must remain alive
/// and unchanged until GPU completion; `output` must not overlap any input.
struct MtpAddStreamsBuffers {
    ReadOnlyBufferRef params;
    ReadOnlyBufferRef h2;
    ReadOnlyBufferRef e2;
    BufferRef output;
};

/// MTP stage 7: run the attention hyperconnection read with apply=false.
/// `dead_h2_scratch` supplies distinct placeholders for the GR ABI's unused
/// block-output and previous-injection bindings. Stage 6 has already consumed
/// the active h2 rows, so this region is dead for the remainder of the MTP
/// pass. The wrapper slices only the required extents and rejects overlap
/// through the underlying GR encoder. Every view must belong to the Context
/// and remain alive through completion; params must not overlap a writable
/// output or any other active GR view.
struct MtpAttentionReadBuffers {
    ReadOnlyBufferRef params;
    BufferRef residual;
    BufferRef dead_h2_scratch;
    BufferRef norm;
    BufferRef normalized;
    BufferRef rs;
    BufferRef hc_down;
    BufferRef hc_inject;
    BufferRef lo;
    BufferRef inject_out;
    BufferRef hc_up;
    BufferRef mixed;
};

/// Inputs and outputs for MTP stage 8. The Q8_0 projections use the same
/// quantized activation rows and active T metadata, but have independent
/// weights and F32 outputs. `sink` receives padded compiled columns and is
/// reused by the ordered key and value matvec dispatches.
struct MtpAttentionKvProjectionBuffers {
    ReadOnlyBufferRef params;
    BufferRef mixed;
    BufferRef xq;
    ReadOnlyBufferRef key_weights;
    ReadOnlyBufferRef value_weights;
    BufferRef kcur;
    BufferRef vcur;
    BufferRef sink;
};

/// Add MTP hidden and embedding streams to the caller-owned Encoder segment.
/// On failure, the enclosing Encoder::encode body must return false so prior
/// work in that segment is discarded.
bool encode_mtp_add_streams(Context& context, Encoder& encoder,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error);

/// Encode MTP stage 6 transactionally into an unsubmitted command buffer.
bool encode_mtp_add_streams(Context& context, CommandBuffer& command_buffer,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error);

/// Encode the ordered MTP stage-7 attention GR read for T rows. This always
/// uses apply=false and leaves `residual` unchanged. The standalone
/// command-buffer form is transactional; callers of the Encoder form must
/// propagate failure from their enclosing `Encoder::encode`.
bool encode_mtp_attention_read(Context& context, Encoder& encoder,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error);
bool encode_mtp_attention_read(Context& context,
                               CommandBuffer& command_buffer,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error);

/// Encode MTP stage 8: quantize `mixed` to Q8_1 and run the key and value
/// Q8_0 projections for T rows. T is read from the CPU-visible MtpParams
/// block; the smallest supported compiled matvec width (1, 2, 4, or 8) is
/// selected automatically. Views must remain alive and unchanged until GPU
/// completion. The standalone form is transactional; the caller-owned form
/// must propagate failure from its enclosing Encoder::encode body.
bool encode_mtp_attention_kv_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error);
bool encode_mtp_attention_kv_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error);

}  // namespace strata::metal
