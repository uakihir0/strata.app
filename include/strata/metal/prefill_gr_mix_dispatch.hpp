// GPU encoding for Prefill's GR stream mixer (docs/mac/04 §6.3).
#pragma once

#include "strata/metal/prefill_gr_norm_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGRMixBuffers {
    ReadOnlyBufferRef residual;
    ReadOnlyBufferRef rs;
    ReadOnlyBufferRef norm;
    ReadOnlyBufferRef gated;
    BufferRef mixed_bf;
    /// Required in BF16X2 modes 1 and 2; must be empty in mode 0.
    BufferRef mixed_bf_lo;
    BufferRef mixed_h;
};

/// Encode the Prefill-specific four-stream GR mix for T rows.
/// Modes 1 and 2 write the BF16 remainder; mode 0 does not bind that output.
bool encode_prefill_gr_mix(Context& context, Encoder& encoder,
                           std::uint32_t rows, PrefillBF16X2Mode mode,
                           const PrefillGRMixBuffers& buffers,
                           std::string& error);

/// Standalone wrapper. A validation failure discards the supplied fresh
/// command buffer, including any earlier dispatches in the same segment.
bool encode_prefill_gr_mix(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, PrefillBF16X2Mode mode,
                           const PrefillGRMixBuffers& buffers,
                           std::string& error);

}  // namespace strata::metal
