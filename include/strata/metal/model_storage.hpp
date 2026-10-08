// include/strata/metal/model_storage.hpp - owns the selected model weight storage mode.
#pragma once

#include "strata/metal/geometry.hpp"
#include "strata/metal/plan.hpp"
#include "strata/metal/tiered.hpp"
#include "strata/metal/window.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::kernels {
class PleTableReader;
}

namespace strata::metal {

class MtpRuntimeStorage;
class KvStore;
class WindowScratchStorage;
struct MtpRuntimeSources;

/// Own the opened weight storage selected by a fitting memory plan.
///
/// Every plan opens and owns its main-model QSA KV, indexer and shared RoPE
/// allocation. Speculative plans also open and own their complete MTP runtime
/// allocation. Qwen4exp plans also open the fixed-capacity Window scratch;
/// full mode opens the CPU-only PLE table reader over its shared GGUF mapping;
/// per-session recurrent state and PLE history remain owned by the session
/// lifecycle. Tiered mode does not publish a PLE reader while its startup
/// mapping is closed.
/// This does not own Context or initialized KV/RoPE contents.
/// Context must outlive this object. Full mode is returned only after its
/// no-copy windows have been warmed; tiered mode is returned after its startup
/// copies and residency request complete.
class ModelStorage {
public:
    static std::unique_ptr<ModelStorage> open(Context& context,
                                               const std::string& native_shard,
                                               const Geometry& geometry,
                                               const WindowSources& sources,
                                               const PlanResult& memory_plan,
                                               std::string& error,
                                               const MtpRuntimeSources* mtp_sources = nullptr);
    ~ModelStorage();
    ModelStorage(const ModelStorage&) = delete;
    ModelStorage& operator=(const ModelStorage&) = delete;

    const char* mode() const noexcept;
    std::uint64_t resident_experts() const noexcept;
    std::size_t window_count() const noexcept;
    bool full_mode_warmed() const noexcept;
    const KvStore* kv_store() const noexcept;
    const MtpRuntimeStorage* mtp_runtime() const noexcept;
    const WindowScratchStorage* window_scratch() const noexcept;
    /// Mutable MoE usage counters; valid until this model storage is destroyed.
    BufferRef moe_usage_table() const noexcept;
    /// Borrow the full-mode tensor/table resolver; null in tiered mode. The
    /// pointer does not outlive this ModelStorage, and its buffer views expose
    /// CPU mappings only through const pointers.
    const FullModeStorage* full_mode_storage() const noexcept;
    /// Available in full mode; null in tiered mode.
    strata::kernels::PleTableReader* ple_reader() noexcept;

private:
    ModelStorage() = default;

    // Declared before MTP so the main shared RoPE allocation outlives its
    // drafter storage if a later session owner binds that view.
    std::unique_ptr<KvStore> kv_store_;
    std::unique_ptr<MtpRuntimeStorage> mtp_runtime_;
    std::unique_ptr<WindowScratchStorage> window_scratch_;
    std::unique_ptr<FullModeStorage> full_;
    std::unique_ptr<TieredStorage> tiered_;
    std::uint64_t resident_experts_ = 0;
    std::size_t window_count_ = 0;
    bool full_mode_warmed_ = false;
};

}  // namespace strata::metal
