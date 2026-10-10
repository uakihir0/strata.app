// Ordered Prefill attention-half GR-read/GDN transition (docs/mac/04 §6.3-6.4).
#pragma once

#include "strata/metal/prefill_gdn_block_dispatch.hpp"
#include "strata/metal/prefill_gr_read_mix_dispatch.hpp"

namespace strata::metal {

/// Buffers for one attention GR-read → GDN → residual-write sequence that
/// prepares the next GR read's normalization.
/// `gdn.mixed_h`, `gdn.mixed_bf`, and `gdn.mixed_bf_lo` must be empty; the
/// coordinator binds them from `gr_read_mix` outputs. The final write reuses
/// `gr_read_mix.read.rs`, `xn16`, and (in mode 1) `xn16_lo` after both the GR
/// read and mixer have finished. Those buffers must remain intact through GDN.
struct PrefillAttentionGDNTransitionBuffers {
    PrefillGRReadMixBuffers gr_read_mix;
    PrefillGDNBlockBuffers gdn;
    ReadOnlyBufferRef next_norm;
};

/// Encode one attention-half transition: GR read/mix, the complete GDN block,
/// then the residual write fused with normalization for the next GR read.
/// This does not encode the FFN GR half, MoE, or a chunk scheduler. A later GR
/// read/mix may consume the prepared normalization through
/// `encode_prefill_gr_read_mix_pre_normalized`; the caller must order that
/// consumer after this transition in the same Encoder segment or establish a
/// supported cross-command-buffer dependency. The caller-owned form may fail
/// after an earlier dispatch was encoded; propagate false from the enclosing
/// Encoder::encode body.
bool encode_prefill_attention_gdn_transition(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode,
    const PrefillAttentionGDNTransitionBuffers& buffers,
    std::string& error);

/// Transactional standalone form. Any failure discards every earlier
/// dispatch in this transition.
bool encode_prefill_attention_gdn_transition(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode,
    const PrefillAttentionGDNTransitionBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
