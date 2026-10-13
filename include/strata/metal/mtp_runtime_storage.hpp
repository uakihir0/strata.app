// include/strata/metal/mtp_runtime_storage.hpp - transactionally owned MTP runtime allocations.
#pragma once

#include "strata/metal/mtp_kv.hpp"
#include "strata/metal/mtp_bind_layout.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "strata/metal/mtp_params_layout.hpp"
#include "strata/metal/mtp_storage.hpp"
#include "strata/metal/plan.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::metal {

class KvStore;

/// One immutable rotation choice shared by every attention-stage binder in
/// the same MTP pass. Construct once from the caller's runtime configuration.
class MtpAttentionRuntimeMode final {
public:
    explicit MtpAttentionRuntimeMode(bool kv_rot) noexcept : kv_rot_(kv_rot) {}
    MtpAttentionRuntimeMode(const MtpAttentionRuntimeMode&) = delete;
    MtpAttentionRuntimeMode& operator=(const MtpAttentionRuntimeMode&) = delete;

    bool kv_rot() const noexcept { return kv_rot_; }

private:
    bool kv_rot_ = false;
};

/// Runtime-bound MTP stages 9–11, sharing one parameter slot and rotation
/// choice. The contained encoder views are immutable to callers so they
/// cannot be independently reconfigured after binding.
class MtpAttentionQueryPrefixBuffers final {
public:
    MtpAttentionQueryPrefixBuffers() = default;

    const MtpAttentionKvAppendBuffers& kv_append() const noexcept {
        return kv_append_;
    }
    const MtpAttentionQueryProjectionBuffers& query_projection() const noexcept {
        return query_projection_;
    }
    const MtpAttentionQueryNormBuffers& query_norm() const noexcept {
        return query_norm_;
    }

private:
    MtpAttentionKvAppendBuffers kv_append_;
    MtpAttentionQueryProjectionBuffers query_projection_;
    MtpAttentionQueryNormBuffers query_norm_;

    friend class MtpRuntimeStorage;
    friend bool encode_mtp_attention_query_prefix(
        Context&, Encoder&, const MtpAttentionQueryPrefixBuffers&,
        std::string&);
    friend bool encode_mtp_attention_query_prefix(
        Context&, CommandBuffer&, const MtpAttentionQueryPrefixBuffers&,
        std::string&);
};

/// Encode the runtime-bound stage-9 append, stage-10 query projection and
/// stage-11 query norm in order. The views and rotation mode are validated as
/// one pass before any dispatch is recorded. Caller-owned encoding failures
/// must be propagated from the enclosing Encoder::encode body.
bool encode_mtp_attention_query_prefix(
    Context& context, Encoder& encoder,
    const MtpAttentionQueryPrefixBuffers& buffers, std::string& error);
bool encode_mtp_attention_query_prefix(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionQueryPrefixBuffers& buffers, std::string& error);

/// Runtime-bound full MTP attention pass, stages 9–14. Its private views
/// preserve one parameter slot, cache rotation choice and ordered scratch
/// handoffs from K/V append through O projection.
class MtpAttentionPassBuffers final {
public:
    MtpAttentionPassBuffers() = default;

    const MtpAttentionQueryPrefixBuffers& query_prefix() const noexcept {
        return query_prefix_;
    }
    const MtpWindowIdsBuffers& window_ids() const noexcept {
        return window_ids_;
    }
    const MtpAttentionBuffers& attention() const noexcept {
        return attention_;
    }
    const MtpAttentionOutputProjectionBuffers& output_projection() const noexcept {
        return output_projection_;
    }

private:
    MtpAttentionQueryPrefixBuffers query_prefix_;
    MtpWindowIdsBuffers window_ids_;
    MtpAttentionBuffers attention_;
    MtpAttentionOutputProjectionBuffers output_projection_;

    friend class MtpRuntimeStorage;
    friend bool encode_mtp_attention_pass(
        Context&, Encoder&, const MtpAttentionPassBuffers&, std::string&);
    friend bool encode_mtp_attention_pass(
        Context&, CommandBuffer&, const MtpAttentionPassBuffers&,
        std::string&);
};

/// Encode MTP stages 9–14 in order. Stage 12 refreshes the active IDs on every
/// pass: a positive finite window selects a suffix, while zero writes the full
/// identity prefix. Cross-stage parameters, views and rotation are validated
/// before dispatch. Caller-owned failures must be propagated from the
/// enclosing Encoder::encode body.
bool encode_mtp_attention_pass(Context& context, Encoder& encoder,
                               const MtpAttentionPassBuffers& buffers,
                               std::string& error);
bool encode_mtp_attention_pass(Context& context,
                               CommandBuffer& command_buffer,
                               const MtpAttentionPassBuffers& buffers,
                               std::string& error);

/// Caller-owned inputs needed to resolve stage-1–5 views from MTP storage.
/// The token table and residual belong to the same Context as the runtime.
struct MtpInputProjectionSources {
    std::uint32_t params_slot = 0;
    MtpInputTokenSource token_source = MtpInputTokenSource::params_tokens;
    ReadOnlyBufferRef selected_token;
    EmbedRowsFormat embedding_format = EmbedRowsFormat::bf16;
    std::uint32_t vocabulary_size = 0;
    ReadOnlyBufferRef token_embedding_table;
    BufferRef residual;
};

/// Caller-owned inputs that select one T=1 runtime-bound MTP full-layer pass.
/// The input projection must use the `selected_token` source, and the caller
/// must encode the GPU producer of that view earlier in the same ordered
/// command buffer. The coordinator checks the source mode and view, but cannot
/// prove who wrote the shared buffer. Its parameter slot is shared by stages
/// 1–20. Stages 21–22 (draft-head projection and sampling) are encoded
/// separately.
struct MtpFullLayerSources {
    MtpInputProjectionSources input_projection;
    std::uint32_t attention_window = 0;
    bool kv_rot = false;
};

/// Own MTP weights, bind scratch, parameter slots, and drafter K/V from one
/// fitting memory plan.
///
/// The bind scratch is one planned arena allocation with device-free,
/// 256-byte-aligned named views. Parameter blocks occupy their own array of
/// bounded 256-byte slots. Context must outlive this owner. If an enclosing
/// Buffers::Transaction rolls back, all buffer accessors return empty views
/// and is_valid() becomes false.
class MtpRuntimeStorage {
public:
    static std::unique_ptr<MtpRuntimeStorage> open(
        Context& context, const Geometry& geometry,
        const MtpRuntimeSources& sources,
        const PlanResult& memory_plan, std::string& error);
    /// Open speculative storage with a borrowed view of the matching main
    /// KvStore's shared RoPE table; no second table is allocated. The Context
    /// and main KV allocation must remain live while this owner uses that view.
    static std::unique_ptr<MtpRuntimeStorage> open_with_main_kv(
        Context& context, const KvStore& main_kv,
        const Geometry& geometry, const MtpRuntimeSources& sources,
        const PlanResult& memory_plan, std::string& error);
#if defined(STRATA_METAL_TEST_LOCK_ACCESS)
    static std::unique_ptr<MtpRuntimeStorage> open_for_test(
        Context& context, const Geometry& geometry,
        const MtpRuntimeSources& sources,
        const PlanResult& memory_plan, bool fail_after_weights,
        std::string& error);
#endif
    ~MtpRuntimeStorage();
    MtpRuntimeStorage(const MtpRuntimeStorage&) = delete;
    MtpRuntimeStorage& operator=(const MtpRuntimeStorage&) = delete;

    bool is_valid() const noexcept;
    const MtpRuntimeLayout& layout() const noexcept;
    const MtpKvLayout& kv_layout() const noexcept;
    const MtpBindLayout& bind_layout() const noexcept;
    const MtpParamsLayout& params_layout() const noexcept;
    const std::vector<std::uint32_t>& draft_vocab_ids() const noexcept;
    BufferRef expert(std::uint32_t expert_index) const noexcept;
    BufferRef dense_weight(const std::string& tensor_name) const noexcept;
    /// Return a mutable named scratch view. The cached device vocabulary map
    /// is intentionally withheld; bind_draft_head exposes it read-only.
    BufferRef bind_region(MtpBindRegionId id) const noexcept;
    BufferRef mtp_params_slot(std::uint32_t slot) const noexcept;
    /// Resolve the MTP-owned weights and scratch for stages 1–5. The main
    /// token table and current residual remain caller-owned inputs. Success
    /// resolves Context-owned views; the encoder remains responsible for
    /// format, extent, alignment, hid-count, token-value, and alias validation.
    /// Returned views are borrowed and valid only while this runtime is valid.
    bool bind_input_projection(Context& context,
                               const MtpInputProjectionSources& sources,
                               MtpInputProjectionBuffers& buffers,
                               std::string& error) const;
    /// Resolve the MTP parameter and scratch views for stage 6. The encoder
    /// validates active extents and aliases before dispatch.
    bool bind_add_streams(Context& context, std::uint32_t params_slot,
                          MtpAddStreamsBuffers& buffers,
                          std::string& error) const;
    /// Resolve the MTP weights and scratch views for stages 7–8. The encoder
    /// validates active extents and aliases before dispatch.
    bool bind_attention_read(Context& context, std::uint32_t params_slot,
                             MtpAttentionReadBuffers& buffers,
                             std::string& error) const;
    bool bind_attention_kv_projection(
        Context& context, std::uint32_t params_slot,
        MtpAttentionKvProjectionBuffers& buffers,
        std::string& error) const;
    /// Resolve runtime views for MTP stages 9–11 using one shared rotation
    /// mode for the whole MTP attention pass.
    bool bind_attention_kv_append(Context& context,
                                  std::uint32_t params_slot,
                                  const MtpAttentionRuntimeMode& mode,
                                  MtpAttentionKvAppendBuffers& buffers,
                                  std::string& error) const;
    bool bind_attention_query_projection(
        Context& context, std::uint32_t params_slot,
        MtpAttentionQueryProjectionBuffers& buffers,
        std::string& error) const;
    /// Resolve stages 9–11 into one immutable pass binding with shared params
    /// and `kv_rot` mode. Use this coordinator for runtime-bound MTP attention.
    bool bind_attention_query_prefix(
        Context& context, std::uint32_t params_slot,
        const MtpAttentionRuntimeMode& mode,
        MtpAttentionQueryPrefixBuffers& buffers,
        std::string& error) const;
    /// Resolve runtime views for stages 9–14. Stage 12 always refreshes the
    /// IDs: a positive `window` selects a suffix and zero selects full history.
    bool bind_attention_pass(
        Context& context, std::uint32_t params_slot,
        const MtpAttentionRuntimeMode& mode, std::uint32_t window,
        MtpAttentionPassBuffers& buffers, std::string& error) const;
    bool bind_feed_forward_read(
        Context& context, std::uint32_t params_slot,
        MtpFeedForwardReadBuffers& buffers, std::string& error) const;
    bool bind_router(Context& context, std::uint32_t params_slot,
                     MtpRouterBuffers& buffers, std::string& error) const;
    /// Resolve the stage-17 scratch and every bounded expert window. The
    /// encoder dispatches only route IDs owned by each window.
    bool bind_routed_experts(Context& context,
                             MtpRoutedExpertsBuffers& buffers,
                             std::string& error) const;
    /// Resolve stages 18–19 and their dead-region mat-vec sinks from the MTP
    /// arena. The residual and stage-15 GR injection are shared in place.
    bool bind_shared_ffn_write(Context& context,
                               std::uint32_t params_slot,
                               MtpSharedFFNWriteBuffers& buffers,
                               std::string& error) const;
    /// Resolve stage 20's mixer weights and its stage-21 sample output.
    bool bind_final_mixer(Context& context, std::uint32_t params_slot,
                          MtpFinalMixerBuffers& buffers,
                          std::string& error) const;
    /// Resolve stage 21 against the caller's read-only main output head.
    /// The inspected draft vocabulary is validated once when this runtime
    /// opens and reused by every returned draft-head binding.
    bool bind_draft_head(Context& context, std::uint32_t params_slot,
                         ReadOnlyBufferRef output_weights,
                         std::uint32_t full_head_vocab_size,
                         MtpDraftHeadBuffers& buffers,
                         std::string& error) const;
    /// Resolve stage 22 and require the exact vocabulary token returned by
    /// bind_draft_head, so logits and final token IDs use one mapping.
    bool bind_draft_sampling(
        Context& context, std::uint32_t params_slot,
        const MtpDraftVocabularyMap& draft_vocabulary,
        MtpDraftSamplingBuffers& buffers, std::string& error) const;
    /// Borrowed main RoPE view, or empty when unbound or this owner is invalid.
    BufferRef shared_rope() const noexcept;
    bool kv_views(MtpKvViews& views, std::string& error) const;

private:
    struct Impl;
    static std::unique_ptr<MtpRuntimeStorage> open_impl(
        Context& context, const Geometry& geometry,
        const MtpRuntimeSources& sources,
        const PlanResult& memory_plan, const KvStore* main_kv,
        bool fail_after_weights,
        std::string& error);
    explicit MtpRuntimeStorage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// Bind and encode the T=1 MTP full layer in stage order, from stages 1–5
/// through the stage-20 final mixer. All runtime views are resolved before
/// this pass records its first stage. Use the Encoder form inside
/// Encoder::encode and propagate false so a later-stage failure discards the
/// entire unsubmitted command buffer, including any earlier caller-encoded
/// work. The caller owns the selected-token view and must have encoded its GPU
/// producer earlier in the same ordered command buffer; this function cannot
/// verify the view's write provenance. The caller also owns the token
/// embedding table and main-model residual supplied in `sources`. Keep the
/// Context, runtime, main KV owner (when its RoPE is borrowed), parameters and
/// every view alive through GPU completion, and keep the selected MtpParams
/// slot and its StepRows unchanged until then. One runtime owns one set of
/// mutable scratch and drafter K/V, so callers must serialize passes using
/// the same runtime until each command buffer completes. The stage-20 sample
/// remains in the runtime's `sample` bind region for subsequent draft-head
/// and sampler encoders.
bool encode_mtp_full_layer(Context& context, Encoder& encoder,
                           const MtpRuntimeStorage& runtime,
                           const MtpFullLayerSources& sources,
                           std::string& error);
/// Transactional command-buffer form of the ordered MTP stages 1–20 pass.
bool encode_mtp_full_layer(Context& context, CommandBuffer& command_buffer,
                           const MtpRuntimeStorage& runtime,
                           const MtpFullLayerSources& sources,
                           std::string& error);

}  // namespace strata::metal
