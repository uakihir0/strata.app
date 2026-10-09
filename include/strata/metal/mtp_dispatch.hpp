// include/strata/metal/mtp_dispatch.hpp - MTP drafter kernel encoders.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/mtp_kv.hpp"

#include <string>

namespace strata::metal {

class Encoder;

/// Inputs for stage 6 of one MTP pass. `params` must be a CPU-visible,
/// immutable MtpParams block whose T is in 1..8. All views must remain alive
/// and unchanged until GPU completion; `output` must not overlap any input.
struct MtpAddStreamsBuffers {
    ReadOnlyBufferRef params;
    ReadOnlyBufferRef h2;
    ReadOnlyBufferRef e2;
    BufferRef output;
};

/// MTP stage 7: run the attention hyperconnection read with apply=false.
/// `dead_h2_scratch` supplies distinct placeholders for the GR ABI's unused
/// block-output and previous-injection bindings. Stage 6 has already consumed
/// the active h2 rows, so this region is dead for the remainder of the MTP
/// pass. The wrapper slices only the required extents and rejects overlap
/// through the underlying GR encoder. Every view must belong to the Context
/// and remain alive through completion; params must not overlap a writable
/// output or any other active GR view.
struct MtpAttentionReadBuffers {
    ReadOnlyBufferRef params;
    BufferRef residual;
    BufferRef dead_h2_scratch;
    BufferRef norm;
    BufferRef normalized;
    BufferRef rs;
    BufferRef hc_down;
    BufferRef hc_inject;
    BufferRef lo;
    BufferRef inject_out;
    BufferRef hc_up;
    BufferRef mixed;
};

/// Inputs and outputs for MTP stage 8. The Q8_0 projections use the same
/// quantized activation rows and active T metadata, but have independent
/// weights and F32 outputs. `sink` receives padded compiled columns and is
/// reused by the ordered key and value matvec dispatches.
struct MtpAttentionKvProjectionBuffers {
    ReadOnlyBufferRef params;
    BufferRef mixed;
    BufferRef xq;
    ReadOnlyBufferRef key_weights;
    ReadOnlyBufferRef value_weights;
    BufferRef kcur;
    BufferRef vcur;
    BufferRef sink;
};

/// Inputs and caller-owned cache views for MTP stage 9. K is normalized with
/// the MTP K norm and the shared RoPE table in place; when `kv_rot` is set,
/// both current K and V are also FWHT-rotated in place before append. The KV
/// layout must be the canonical layout returned by make_mtp_kv_layout, and
/// its views must come from that layout. K8v4 requests use the drafter's
/// documented int8 storage mapping.
///
/// Keep every view alive until command-buffer completion. The caller must
/// keep `params` and its active StepRows, `kcur`, `vcur`, `k_norm`, and `rope`
/// unchanged until completion. `params` and its StepRows are CPU-validated
/// before the GPU reads them. The encoder overwrites K in place, overwrites V
/// when `kv_rot` is set, and appends rows to the cache views.
struct MtpAttentionKvAppendBuffers {
    BufferRef params;
    BufferRef kcur;
    BufferRef vcur;
    BufferRef k_norm;
    BufferRef rope;
    MtpKvLayout kv_layout;
    MtpKvViews kv_views;
    bool kv_rot = false;
};

/// Inputs and outputs for MTP stage 10. This full-layer projection consumes
/// the single Q8_1 activation row already produced by stage 8 and writes the
/// 24×512 Q values into the full `q_full` tensor; `sink` must provide the
/// Q8_0 encoder's required one-float minimum. Keep all views alive through
/// command-buffer completion, and keep `params`, `xq` and `q_weights`
/// unchanged while the GPU reads them.
struct MtpAttentionQueryProjectionBuffers {
    ReadOnlyBufferRef params;
    BufferRef xq;
    ReadOnlyBufferRef q_weights;
    BufferRef q_full;
    BufferRef sink;
};

/// Inputs and outputs for MTP stage 11. Normalizes/RoPEs the 24 Q heads from
/// `q_full` into compact `qcur`; `kv_rot` applies an in-place FWHT to qcur
/// after normalization. Keep all views alive through completion and keep
/// params, q_full, q_norm and rope unchanged while the GPU reads them.
struct MtpAttentionQueryNormBuffers {
    BufferRef params;
    BufferRef q_full;
    BufferRef q_norm;
    BufferRef rope;
    BufferRef qcur;
    bool kv_rot = false;
};

/// Inputs and output for MTP stage 12's finite-window ID generation.
/// `params` is an immutable CPU-visible T=1 MtpParams block whose first
/// StepRow contains the host-published `n_kv` and matching `width`. `mtp_ids`
/// is the named int32 ID region with capacity at least `window`; only its
/// active prefix is written. Keep both views alive through completion and
/// keep params unchanged while the GPU reads it.
struct MtpWindowIdsBuffers {
    ReadOnlyBufferRef params;
    BufferRef mtp_ids;
    std::uint32_t window = 0;
};

/// Inputs and outputs for MTP stage 13's dense finite-window attention.
/// `id_row_stride` is the bind-layout capacity (a positive multiple of 64),
/// while the active StepRow width limits the IDs consumed by attention.
/// `att_scratch` is the complete named bind region, including its reserved
/// 64-float tail; only the chunk accumulator/max/denominator prefix is used.
/// The canonical MTP KV storage format selects the QSA attention mode;
/// K8v4 requests use the MTP store's documented int8 representation.
/// `kv_rot` must match the stage-9/11 rotation choice. Keep all views alive
/// through GPU completion, and keep params, qcur, mtp_ids, and KV views
/// unchanged until the encoder has consumed them.
struct MtpAttentionBuffers {
    BufferRef params;
    BufferRef qcur;
    BufferRef mtp_ids;
    MtpKvLayout kv_layout;
    MtpKvViews kv_views;
    BufferRef att_scratch;
    BufferRef attn;
    std::uint32_t id_row_stride = 0;
    bool kv_rot = false;
};

/// Inputs and outputs for MTP stage 14's attention gate and O projection.
/// The stage consumes stage 13's 24×256 attention values and stage 10's
/// q_full, gates the attention values, quantizes the 6144-wide row to Q8_1,
/// then projects through the read-only Q8_0 o_proj matrix into `bo`. Keep all
/// views alive through GPU completion, and keep params, inputs and weights
/// unchanged until the encoder has consumed them.
struct MtpAttentionOutputProjectionBuffers {
    ReadOnlyBufferRef params;
    BufferRef attn;
    BufferRef q_full;
    BufferRef attn32;
    BufferRef xq;
    ReadOnlyBufferRef o_weights;
    BufferRef bo;
    BufferRef sink;
};

/// Inputs and outputs for MTP stage 15's feed-forward hyperconnection read.
/// It applies the GR block output `bo` and prior injection `inj` to `residual`,
/// producing the updated residual, `inject_out`, and `mixed` values. Keep all
/// views alive through GPU completion and keep params and read-only inputs
/// unchanged until the encoder has consumed them.
struct MtpFeedForwardReadBuffers {
    ReadOnlyBufferRef params;
    BufferRef residual;
    BufferRef bo;
    BufferRef inj;
    BufferRef norm;
    BufferRef normalized;
    BufferRef rs;
    BufferRef hc_down;
    BufferRef hc_inject;
    BufferRef lo;
    BufferRef inject_out;
    BufferRef hc_up;
    BufferRef mixed;
};

/// Inputs and outputs for MTP stage 16's dense router and top-10 route.
struct MtpRouterBuffers {
    ReadOnlyBufferRef params;
    ReadOnlyBufferRef mixed;
    ReadOnlyBufferRef router_weights;
    BufferRef logits_r;
    BufferRef ids;
    BufferRef weights;
    BufferRef matvec_sink;
};

/// Add MTP hidden and embedding streams to the caller-owned Encoder segment.
/// On failure, the enclosing Encoder::encode body must return false so prior
/// work in that segment is discarded.
bool encode_mtp_add_streams(Context& context, Encoder& encoder,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error);

/// Encode MTP stage 6 transactionally into an unsubmitted command buffer.
bool encode_mtp_add_streams(Context& context, CommandBuffer& command_buffer,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error);

/// Encode the ordered MTP stage-7 attention GR read for T rows. This always
/// uses apply=false and leaves `residual` unchanged. The standalone
/// command-buffer form is transactional; callers of the Encoder form must
/// propagate failure from their enclosing `Encoder::encode`.
bool encode_mtp_attention_read(Context& context, Encoder& encoder,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error);
bool encode_mtp_attention_read(Context& context,
                               CommandBuffer& command_buffer,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error);

/// Encode MTP stage 8: quantize `mixed` to Q8_1 and run the key and value
/// Q8_0 projections for T rows. T is read from the CPU-visible MtpParams
/// block; the smallest supported compiled matvec width (1, 2, 4, or 8) is
/// selected automatically. Views must remain alive and unchanged until GPU
/// completion. The standalone form is transactional; the caller-owned form
/// must propagate failure from its enclosing Encoder::encode body.
bool encode_mtp_attention_kv_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error);
bool encode_mtp_attention_kv_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error);

/// Encode MTP stage 9: normalize/RoPE the two K heads, optionally rotate K/V
/// according to the selected KV mode, and append both heads at MtpParams
/// StepRow positions. This requires an ordered concurrent-group boundary.
/// Current K (and V when rotated) are modified in place. The standalone
/// command-buffer form is transactional; the caller-owned form must propagate
/// failure from its enclosing Encoder::encode body.
bool encode_mtp_attention_kv_append(
    Context& context, Encoder& encoder,
    const MtpAttentionKvAppendBuffers& buffers, std::string& error);
bool encode_mtp_attention_kv_append(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionKvAppendBuffers& buffers, std::string& error);

/// Encode MTP stage 10 for T=1: project stage 8's Q8_1 `xq` row through the
/// Q8_0 query matrix into the full 24×512 `q_full` tensor. The standalone form
/// is transactional; the caller-owned form must propagate failure from its
/// enclosing Encoder::encode body.
bool encode_mtp_attention_query_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionQueryProjectionBuffers& buffers, std::string& error);
bool encode_mtp_attention_query_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionQueryProjectionBuffers& buffers, std::string& error);

/// Encode MTP stage 11 for T=1: normalize/RoPE q_full into qcur, then apply
/// the optional in-place FWHT selected by kv_rot. Both forms require an
/// ordered concurrent-group boundary. The standalone form is transactional;
/// the caller-owned form must propagate failure from its Encoder::encode body.
bool encode_mtp_attention_query_norm(
    Context& context, Encoder& encoder,
    const MtpAttentionQueryNormBuffers& buffers, std::string& error);
bool encode_mtp_attention_query_norm(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionQueryNormBuffers& buffers, std::string& error);

/// Encode MTP stage 12 for a finite window. The active ID prefix contains
/// absolute KV cell positions from max(0, n_kv - window) to n_kv. Both forms
/// require an ordered concurrent-group boundary; the standalone form is
/// transactional and caller-owned failures must be propagated.
bool encode_mtp_window_ids(Context& context, Encoder& encoder,
                           const MtpWindowIdsBuffers& buffers,
                           std::string& error);
bool encode_mtp_window_ids(Context& context, CommandBuffer& command_buffer,
                           const MtpWindowIdsBuffers& buffers,
                           std::string& error);

/// Encode MTP stage 13: chunk attention, ordered merge, and optional O-class
/// output FWHT. Requires T=1, the canonical MTP KV layout/views and an ordered
/// concurrent-group boundary. The standalone form is transactional; callers
/// of the Encoder form must propagate failure from their enclosing
/// `Encoder::encode` body.
bool encode_mtp_attention(Context& context, Encoder& encoder,
                          const MtpAttentionBuffers& buffers,
                          std::string& error);
bool encode_mtp_attention(Context& context,
                          CommandBuffer& command_buffer,
                          const MtpAttentionBuffers& buffers,
                          std::string& error);

/// Encode MTP stage 14: apply the QSA gate, quantize its 6144-wide result to
/// Q8_1, then run the T=1 Q8_0 O projection. The standalone form is
/// transactional; callers of the Encoder form must propagate failure from
/// their enclosing `Encoder::encode` body.
bool encode_mtp_attention_output_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionOutputProjectionBuffers& buffers,
    std::string& error);
bool encode_mtp_attention_output_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionOutputProjectionBuffers& buffers,
    std::string& error);

/// Encode MTP stage 15 by composing the GR normalize/down/up chain with
/// apply=true. Requires T=1; the standalone form is transactional, and the
/// caller-owned form must propagate failure from its enclosing
/// `Encoder::encode` body.
bool encode_mtp_feed_forward_read(Context& context, Encoder& encoder,
                                  const MtpFeedForwardReadBuffers& buffers,
                                  std::string& error);
bool encode_mtp_feed_forward_read(
    Context& context, CommandBuffer& command_buffer,
    const MtpFeedForwardReadBuffers& buffers, std::string& error);

/// Encode MTP stage 16: project the T=1 mixed row to 512 expert logits and
/// select the top ten routes. The standalone form is transactional, and the
/// caller-owned form must propagate failure from its enclosing
/// `Encoder::encode` body.
bool encode_mtp_router(Context& context, Encoder& encoder,
                       const MtpRouterBuffers& buffers,
                       std::string& error);
bool encode_mtp_router(Context& context, CommandBuffer& command_buffer,
                       const MtpRouterBuffers& buffers,
                       std::string& error);

}  // namespace strata::metal
