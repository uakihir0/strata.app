// Ordered Prefill GR read half through the four-stream mixer (docs/mac/04 §6.3).
#pragma once

#include "strata/metal/prefill_gr_mix_dispatch.hpp"
#include "strata/metal/prefill_gr_read_dispatch.hpp"

namespace strata::metal {

class Encoder;

struct PrefillGRReadMixBuffers {
    PrefillGRReadBuffers read;
    BufferRef mixed_bf;
    /// Required in BF16X2 modes 1 and 2; must be empty in mode 0.
    BufferRef mixed_bf_lo;
    BufferRef mixed_h;
};

/// Encode the complete Prefill GR read half, including its BF16/F16 mixer.
/// Intermediate read outputs and mixer outputs must occupy disjoint ranges.
bool encode_prefill_gr_read_mix(Context& context, Encoder& encoder,
                                std::uint32_t rows,
                                PrefillBF16X2Mode mode,
                                const PrefillGRReadMixBuffers& buffers,
                                std::string& error);

/// Transactional wrapper for a caller-supplied fresh command buffer.
bool encode_prefill_gr_read_mix(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
