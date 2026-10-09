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

}  // namespace strata::metal
