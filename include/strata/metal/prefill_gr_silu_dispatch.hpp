// GPU encoding for Prefill's GR hyper-connection SiLU stage (docs/mac/04 §6.3).
#pragma once

#include "strata/metal/prefill_gr_norm_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGRSiluBuffers {
    ReadOnlyBufferRef low_rank;
    BufferRef high_bf16;
    /// Required only in all_projections mode; must be empty otherwise.
    BufferRef low_bf16;
};

/// Encode the Prefill GR activation and BF16 conversion for T rows.
/// Mode 1 writes the BF16 remainder; the other modes do not bind it.
bool encode_prefill_gr_silu(Context& context, Encoder& encoder,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRSiluBuffers& buffers,
                            std::string& error);

/// Standalone wrapper. A validation failure discards the supplied fresh
/// command buffer, including any earlier dispatches in the same segment.
bool encode_prefill_gr_silu(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRSiluBuffers& buffers,
                            std::string& error);

}  // namespace strata::metal
