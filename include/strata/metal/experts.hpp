// include/strata/metal/experts.hpp - tiered expert miss reads and cache accounting.
#pragma once

#include "strata/metal/tiered.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata::metal {

inline constexpr std::size_t kExpertFetchWorkers = 8;

class Encoder;

/// Holds slot leases for one command buffer until its registered completion callback runs.
class ExpertLeaseSet {
public:
    ~ExpertLeaseSet();
    ExpertLeaseSet(const ExpertLeaseSet&) = delete;
    ExpertLeaseSet& operator=(const ExpertLeaseSet&) = delete;

    /// Release a layer's slot pins after its GPU expert kernels have finished
    /// reading them, so a later layer of the same parity can reuse those slots.
    void retire_layer(std::uint32_t layer) noexcept;
    /// Confirm the set was prepared for the command buffer being encoded.
    bool prepared_for(const Encoder& encoder) const noexcept;
    /// Confirm this layer still pins the complete GPU-address range.
    bool pins(std::uint32_t layer, std::uint64_t gpu_address,
              std::uint64_t bytes) const noexcept;

private:
    struct Impl;
    explicit ExpertLeaseSet(std::unique_ptr<Impl> impl);
    bool retain(std::uint32_t layer, std::vector<SlotPool::Lease>& leases,
                std::string& error);
    void complete() noexcept;
    std::unique_ptr<Impl> impl_;
    friend class ExpertStore;
};

struct ExpertLayerStats {
    std::uint64_t hits = 0;
    std::uint64_t lookups = 0;
    std::uint64_t ram_blobs = 0;
    std::uint64_t file_blobs = 0;
    std::uint64_t file_bytes = 0;
};

/// Resolves GPU-produced grouped plans into resident or slot addresses.
/// The owning TieredStorage and Context must outlive this object and its command buffers.
class ExpertStore {
public:
    static std::unique_ptr<ExpertStore> create(TieredStorage& storage,
                                               const Geometry& geometry,
                                               std::string& error);
    ~ExpertStore();
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;

    /// Register lease release before enqueue/commit. The returned set can then
    /// accept leases after a submitted command buffer produces its plan.
    std::shared_ptr<ExpertLeaseSet> prepare_command_buffer(
        CommandBuffer& command_buffer, std::string& error);

    /// Resolve a decoded window plan. On failure, callers must not signal the
    /// GPU's layer-ready event. Slot leases remain pinned until the prepared
    /// command buffer completes.
    bool resolve_layer(const std::shared_ptr<ExpertLeaseSet>& lease_set,
                       std::uint32_t layer,
                       const TieredMoePlan& plan, ExpertLayerStats& stats,
                       std::string& error);

    /// Wait for a GPU-produced shared plan, retire earlier layers whose expert
    /// reads precede this plan event, resolve its misses, then signal the matching
    /// layer-ready value on the same monotonic event. A skipped plan reads and
    /// validates only its four-word header. On failure, this leaves the layer-ready
    /// event unsignaled. Before releasing any queued GPU wait, the caller must ensure
    /// the waiting command buffer has no unsafe consumers after that wait, or that
    /// those consumers are guarded against request failure; releasing a wait does not
    /// cancel later GPU commands.
    bool resolve_layer_after_plan(
        const std::shared_ptr<ExpertLeaseSet>& lease_set, std::uint32_t layer,
        Events& events, std::uint64_t plan_ready_value,
        std::uint64_t layer_ready_value, std::uint64_t timeout_ms,
        ExpertLayerStats& stats, std::string& error);

    /// Prefetch every cold expert of a layer, used by large prefill chunks.
    bool prefetch_layer(std::uint32_t layer, std::string& error);
    /// Prefetch predicted cold ids, used by decode lookahead.
    bool prefetch_experts(std::uint32_t layer, std::span<const std::uint32_t> experts,
                          std::string& error);

private:
    struct Impl;
    bool resolve_layer_locked(const std::shared_ptr<ExpertLeaseSet>& lease_set,
                              std::uint32_t layer, const TieredMoePlan& plan,
                              ExpertLayerStats& stats, std::string& error);
    explicit ExpertStore(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
