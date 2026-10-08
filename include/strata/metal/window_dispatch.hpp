// T=1 Window input-stage composition for the Metal engine.
#pragma once

#include "strata/metal/embed_dispatch.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/window_gdn_dispatch.hpp"
#include "strata/metal/window_params.hpp"

namespace strata::metal {

class Encoder;

struct WindowEmbeddingBuffers {
    /// The complete ABI-compatible block built by WindowParamsRing.
    BufferRef params;
    /// Read-only token embedding tensor from full-mode model storage.
    ReadOnlyBufferRef table;
    /// T=1 embedding and four-stream residual scratch regions.
    BufferRef embedding;
    BufferRef streams;
};

/// Inputs for layer 0's T=1 attention read followed by its speculative GDN
/// block. `attention.mixed` and `gdn.mixed` must be the exact same live view;
/// every other cross-stage view must be disjoint. `params` must describe a
/// T=1 ordinary speculative step (`n_keep == 0`).
struct WindowT1AttentionGDNBuffers {
    BufferRef params;
    GRReadBuffers attention;
    WindowGDNBlockBuffers gdn;
};

/// Layer 0's complete attention-read, speculative GDN, then FFN-read chain.
/// All three `mixed` roles share one exact view. `gdn.block_out` and
/// `ffn.block_out` share one exact view; the attention `block_out` is a
/// separate, unused placeholder because `pending(0) == 0`. The two GR reads
/// share residual and `normalized`/`rs`/`lo`/`mixed` scratch. Their injection
/// views swap (`attention.inject_out == ffn.inject_prev` and
/// `attention.inject_prev == ffn.inject_out`). Attention and FFN model weights
/// are distinct layer-0 tensors and must not alias. All other cross-stage
/// overlaps are rejected. The FFN roles must use the layer-0 `hc_ffn_*`
/// weights; the attention roles use `hc_attn_*`.
struct WindowT1AttentionGDNFFNBuffers {
    BufferRef params;
    GRReadBuffers attention;
    WindowGDNBlockBuffers gdn;
    GRReadBuffers ffn;
};

/// Layer 0 followed by the T=1 router projection, top-10 routing and MoE
/// input quantization. The router reads the FFN `mixed` output; its logits
/// feed `route_ids`/`route_weights`; the final Q8_1 row reuses `gdn.xq` after
/// the GDN dispatches have finished. `router_matvec_sink` is separate scratch.
/// The router tensor is a read-only BF16 [experts, 2560] view; `experts` is
/// 256 for Coder or 512 otherwise. This prefix stops before MoePlan or expert
/// execution.
struct WindowT1MoeRouterPrefixBuffers {
    WindowT1AttentionGDNFFNBuffers layer0;
    ReadOnlyBufferRef router_tensor;
    BufferRef router_logits;
    BufferRef route_ids;
    BufferRef route_weights;
    BufferRef router_matvec_sink;
};

/// Encode the T=1 input stem: gather the token from WindowParams, then
/// broadcast the embedding into the four initial GR streams. The views must
/// remain alive and the parameter block must remain unchanged until GPU
/// completion. This does not encode the Window's model layers or imply that a
/// complete decode is available. When using the caller-owned Encoder overload,
/// propagate a false return from the enclosing Encoder::encode body so any
/// earlier gather in that segment is discarded.
bool encode_window_embedding(Context& context, Encoder& encoder,
                             std::uint32_t vocabulary_size,
                             EmbedRowsFormat format,
                             const WindowEmbeddingBuffers& buffers,
                             std::string& error);

/// Transactional standalone wrapper for a fresh command buffer.
bool encode_window_embedding(Context& context, CommandBuffer& command_buffer,
                             std::uint32_t vocabulary_size,
                             EmbedRowsFormat format,
                             const WindowEmbeddingBuffers& buffers,
                             std::string& error);

/// Encode the L9 T=1 layer-0 attention read (`pending(0) == 0`) followed by
/// its speculative GDN block in the same ordered segment. This assumes the
/// attention residual has been initialized by the Window input stem. It does
/// not commit GDN history/state or implement the remaining Window layers.
/// Propagate a false return from the enclosing Encoder::encode body.
bool encode_window_t1_attention_gdn(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNBuffers& buffers, std::string& error);
/// Standalone transactional form. Late failure discards the GR read too.
bool encode_window_t1_attention_gdn(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNBuffers& buffers, std::string& error);

/// Encode the layer-0 attention read (`apply=false`), speculative GDN block,
/// and FFN read (`apply=true`) in one ordered segment. The FFN read consumes
/// the GDN block output and attention injection, then writes the swapped
/// injection buffer. This does not route experts or perform the FFN write.
/// Propagate false from a caller-owned Encoder body to discard all prior work.
bool encode_window_t1_attention_gdn_ffn(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNFFNBuffers& buffers, std::string& error);
/// Standalone transactional form. A late FFN validation failure discards the
/// attention and GDN dispatches too.
bool encode_window_t1_attention_gdn_ffn(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNFFNBuffers& buffers, std::string& error);

/// Encode the T=1 layer-0 attention/GDN/FFN-read chain, BF16 router, top-10
/// Window route (`divide_weights=false`), then Q8_1 quantization of `mixed`
/// into the GDN `xq` scratch after GDN has consumed it. All stages are
/// ordered in one segment and reject an open concurrent group. The caller
/// must propagate false from its enclosing Encoder::encode body so the full
/// segment is discarded on failure.
bool encode_window_t1_moe_router_prefix(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    std::uint32_t experts,
    const WindowT1MoeRouterPrefixBuffers& buffers, std::string& error);
/// Standalone transactional form. A late validation failure discards every
/// earlier stage in the prefix.
bool encode_window_t1_moe_router_prefix(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoeRouterPrefixBuffers& buffers, std::string& error);

}  // namespace strata::metal
