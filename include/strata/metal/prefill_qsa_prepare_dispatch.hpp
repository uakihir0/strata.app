// Ordered Prefill QSA preparation from projected rows through cache append.
#pragma once

#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_qsa_output_dispatch.hpp"
#include "strata/metal/qsa_dispatch.hpp"

#include <cstdint>
#include <string>
#include <variant>

namespace strata::metal {

class Encoder;

using PrefillQsaKvAppendBuffers =
    std::variant<QsaKvAppendF16Buffers, QsaKvAppendI8Buffers,
                 QsaKvAppendQ4Buffers, QsaKvAppendK8V4Buffers>;

struct PrefillQsaPrepareBuffers {
    /// Projected attn_k output. The norm/RoPE stage reads and writes this
    /// buffer in place; the KV append consumes the same active rows.
    QsaNormRopeBuffers key_norm_rope;
    /// Projected attn_q output. The norm/RoPE stage reads its strided 512-wide
    /// rows and writes compact 256-wide query rows to `output`.
    QsaNormRopeBuffers query_norm_rope;
    /// Projected indexer.q_proj output. Normalization and RoPE are in place.
    QsaNormRopeBuffers indexer_query_norm_rope;
    PrefillQsaKvAppendBuffers kv_append;
    QsaPrefillIndexerAppendBuffers indexer_append;
};

/// Compose Prefill QSA steps 3–9 in order: key norm/RoPE, cache-format FWHT,
/// KV append, query norm/RoPE and optional FWHT, indexer-query norm/RoPE, and
/// batched indexer append. Inputs are the projected rows from steps 1–2.
///
/// q4 always transforms K, V and qcur; k8v4 always transforms V; and
/// `rotate_uncompressed_kv` selects the optional QSA rotation for f16/i8.
/// The resulting attention transform is returned so the caller can pass it
/// to `encode_prefill_qsa_tail`. Indexer outputs and KV cache outputs remain
/// caller-owned views for that later selection/attention stage.
///
/// The coordinator requires all norm, cache-append and indexer-append stages
/// to use the same StepRows, and preflights active cross-stage aliases before
/// encoding. The Encoder form requires an ordered segment; propagate false
/// from the enclosing `Encoder::encode` callback to discard the whole segment.
bool encode_prefill_qsa_prepare(
    Context& context, Encoder& encoder, const PrefillParams& params,
    QsaAttentionMode attention_mode, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv,
    const PrefillQsaPrepareBuffers& buffers, std::string& error,
    PrefillQsaAttentionTransform* attention_transform = nullptr);

/// Transactional standalone form. A validation or dispatch failure discards
/// every preparation stage encoded by this call.
bool encode_prefill_qsa_prepare(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, QsaAttentionMode attention_mode,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv,
    const PrefillQsaPrepareBuffers& buffers, std::string& error,
    PrefillQsaAttentionTransform* attention_transform = nullptr);

}  // namespace strata::metal
