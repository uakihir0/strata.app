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

}  // namespace strata::metal
