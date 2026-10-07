// GPU encoding for Prefill's GR read-normalization stage (docs/mac/04 §6.3).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

/// The prompt-path interpretation of STRATA_PREFILL_BF16X2.
enum class PrefillBF16X2Mode : std::uint32_t {
    off = 0,
    all_projections = 1,
    except_gr = 2,
};

struct PrefillGRNormBuffers {
    BufferRef residual;
    BufferRef norm;
    BufferRef rs;
    BufferRef xn16;
    /// Required only in all_projections mode; must be empty otherwise.
    BufferRef xn16_lo;
};

/// Encode the Prefill-specific 256-thread GR normalization for T rows.
/// Mode 1 writes the BF16 remainder. The other modes leave it unallocated.
/// All views belong to `context`; read-only inputs may overlap, but outputs
/// must be disjoint from each other and from both inputs.
bool encode_prefill_gr_norm(Context& context, Encoder& encoder,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRNormBuffers& buffers,
                            std::string& error);

/// Standalone wrapper. A validation failure discards the supplied fresh
/// command buffer, including any earlier dispatches in the same segment.
bool encode_prefill_gr_norm(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRNormBuffers& buffers,
                            std::string& error);

}  // namespace strata::metal
