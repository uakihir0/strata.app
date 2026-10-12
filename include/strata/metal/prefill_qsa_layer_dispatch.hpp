// Ordered Prefill QSA projected-row layer stages.
#pragma once

#include "strata/metal/prefill_qsa_prepare_dispatch.hpp"
#include "strata/metal/prefill_qsa_tail_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillQsaLayerBuffers {
    /// Projected K/V/Q and indexer rows consumed by preparation steps 3–9.
    PrefillQsaPrepareBuffers preparation;
    /// Selection, attention and output buffers consumed by steps 10–14.
    PrefillQsaTailBuffers tail;
};

/// Compose Prefill QSA steps 3–14 for one layer from caller-produced
/// projections. This connects cache/indexer preparation to selection,
/// attention, the optional attention transform, gating and `attn_output`.
/// Projection steps 1–2 and production layer scheduling remain caller-owned.
///
/// The coordinator verifies every preparation-to-tail producer handoff
/// before encoding. Each child validates its active extents and aliases.
/// This ordered Encoder form rejects open concurrent groups; propagate false
/// from the enclosing `Encoder::encode` body to discard the whole layer.
bool encode_prefill_qsa_layer(
    Context& context, Encoder& encoder, const PrefillParams& params,
    QsaAttentionMode attention_mode, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv, PrefillNativeWeightFormat output_format,
    const PrefillQsaLayerBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

/// Transactional standalone form. Any preparation, handoff, selection,
/// attention or output failure discards the whole unsubmitted layer.
bool encode_prefill_qsa_layer(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, QsaAttentionMode attention_mode,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv, PrefillNativeWeightFormat output_format,
    const PrefillQsaLayerBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

}  // namespace strata::metal
