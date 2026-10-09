// GPU encoding for Prefill's fused GR write and next-read normalization
// (docs/mac/04 §6.3).
#pragma once

#include "strata/metal/prefill_gr_norm_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGRWriteNormBuffers {
    BufferRef residual;
    ReadOnlyBufferRef block_out;
    ReadOnlyBufferRef inject;
    ReadOnlyBufferRef next_norm;
    BufferRef rs;
    BufferRef xn16;
    /// Required only in all_projections mode; must be empty otherwise.
    BufferRef xn16_lo;
};

/// Fuse the Prefill GR residual write with normalization for the next read.
/// Mode 1 writes the BF16 remainder; modes 0 and 2 do not bind that output.
bool encode_prefill_gr_write_norm(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRWriteNormBuffers& buffers,
    std::string& error);

/// Standalone wrapper. A validation failure discards the supplied fresh
/// command buffer, including any earlier dispatches in the same segment.
bool encode_prefill_gr_write_norm(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRWriteNormBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
