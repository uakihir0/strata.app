// Ordered Prefill QSA selection, attention and output tail.
#pragma once

#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_qsa_output_dispatch.hpp"
#include "strata/metal/qsa_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillQsaTailBuffers {
    QsaPrefillSelectionBuffers selection;
    QsaPrefillAttentionBuffers attention;
    PrefillQsaOutputBuffers output;
};

/// Compose Prefill QSA selection→sparse attention→optional attention
/// FWHT256→gate→native `attn_output` dequantization and F16 GEMM. Selection
/// and attention must share StepRows and the selection IDs must feed
/// attention. The attention merge output must feed the output gate. Writable
/// stage ranges must be disjoint except for these producer→consumer handoffs.
/// Input buffers remain caller-owned; the coordinator does not append KV or
/// indexer rows or schedule a production Prefill layer.
///
/// This ordered caller-owned form rejects an open concurrent dispatch group.
/// Propagate false from the enclosing `Encoder::encode` body so any failure
/// discards earlier selection or attention dispatches.
bool encode_prefill_qsa_tail(
    Context& context, Encoder& encoder, const PrefillParams& params,
    QsaAttentionMode attention_mode, std::uint32_t cell_capacity,
    PrefillNativeWeightFormat output_format,
    const PrefillQsaTailBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr,
    PrefillQsaAttentionTransform attention_transform =
        PrefillQsaAttentionTransform::none);

/// Transactional standalone wrapper. Any failure after selection, attention
/// or FWHT dispatch discards the complete unsubmitted QSA tail.
bool encode_prefill_qsa_tail(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, QsaAttentionMode attention_mode,
    std::uint32_t cell_capacity,
    PrefillNativeWeightFormat output_format,
    const PrefillQsaTailBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr,
    PrefillQsaAttentionTransform attention_transform =
        PrefillQsaAttentionTransform::none);

}  // namespace strata::metal
