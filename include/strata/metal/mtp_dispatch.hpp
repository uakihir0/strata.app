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

}  // namespace strata::metal
