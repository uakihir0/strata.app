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

/// Encode the complete GR read/mix half from rs/xn16 (and mode-1 xn16_lo)
/// prepared for the same active rows and BF16X2 mode by an earlier fused GR
/// write/norm transition. Order the producer before this consumer in the same
/// Encoder, or establish a supported cross-command-buffer dependency before
/// submission. The residual and norm weights must match that prepared state
/// because the mixer reads them.
bool encode_prefill_gr_read_mix_pre_normalized(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    std::string& error);

/// Transactional wrapper for a caller-supplied fresh command buffer.
bool encode_prefill_gr_read_mix(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    std::string& error);

/// Transactional wrapper for the pre-normalized read/mix path.
bool encode_prefill_gr_read_mix_pre_normalized(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
