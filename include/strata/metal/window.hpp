// include/strata/metal/window.hpp - full-mode tensor and expert address layout.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/geometry.hpp"
#include "strata/metal/window_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::kernels {
class PleTableReader;
}

namespace strata::metal {

inline constexpr std::uint32_t kFullModeMaxLayers = 48;
inline constexpr std::uint32_t kFullModeMaxExperts = 512;
inline constexpr std::uint32_t kFullModeExpertRoles = 3;
inline constexpr std::size_t kFullModeWarmWorkers = 8;

struct TensorGpuBinding {
    std::string tensor_name;
    std::uint64_t expert_index = kNoExpertIndex;
    std::uint64_t gpu_address = 0;
    std::uint64_t byte_count = 0;
};

struct FullModeLayout {
    // Whole dense tensors have one binding; expert tensors have one binding per slab.
    std::vector<TensorGpuBinding> tensors;
    // Converted dense-pack rows use a separate 16 KiB-rounded anonymous arena.
    std::uint64_t dense_pack_bytes = 0;
    std::uint64_t dense_pack_arena_bytes = 0;
    // Maximum-capacity buffers; active entries are packed using the model's n_expert row stride.
    // Addresses are [layer * n_expert + expert][gate, up, down]; unused tail entries are zero.
    std::vector<std::uint64_t> expert_addresses;
    // Active entries are one in full mode; unused tail entries are zero.
    std::vector<std::int32_t> resident;
};

struct FullModeTensorView {
    // GGML type in the live full-mode representation. Converted router rows
    // are BF16 even when their source GGUF tensor has another type.
    std::uint32_t ggml_type = 0;
    std::vector<std::uint64_t> shape;
    ReadOnlyBufferRef buffer;
};

/// Preserve profile order, then append all unranked experts in layer/expert order.
/// Consecutive groups of kFullModeWarmWorkers form the CPU warm-up waves.
bool order_full_mode_warm_experts(
    std::uint32_t n_layers, std::uint32_t n_expert,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& profile_order,
    std::vector<std::pair<std::uint32_t, std::uint32_t>>& ordered_experts,
    std::string& error);

/// Run fixed-size batches of warm-up work with a barrier before the next batch.
/// The observer runs before workers enter each batch and sees prior completions.
bool run_full_mode_warmup_waves(
    std::size_t work_count,
    const std::function<void(std::size_t worker, std::size_t item)>& work,
    const std::function<bool(std::size_t wave, std::size_t completed_items)>& observer,
    std::string& error);

/// Resolve every validated source slice to a GPU address and build the full-mode tables.
/// `window_gpu_addresses[file][window]` is the base address returned by the no-copy MTLBuffer.
bool build_full_mode_layout(const WindowSources& sources, const WindowPlan& plan,
                            const std::vector<std::vector<std::uint64_t>>& window_gpu_addresses,
                            std::uint64_t max_buffer_length, std::uint32_t n_layers,
                            std::uint32_t n_expert, FullModeLayout& layout, std::string& error,
                            std::uint64_t dense_pack_gpu_base = 0);

/// Owns the GGUF mmap and its CPU-only PLE table reader while Context retains
/// every no-copy MTLBuffer over it. The Context passed to open must outlive
/// this storage.
class FullModeStorage {
public:
    static std::unique_ptr<FullModeStorage> open(Context& context, const std::string& native_shard,
                                                  const Geometry& geometry,
                                                  const WindowSources& sources,
                                                  const WindowPlan& plan, std::string& error);
    ~FullModeStorage();
    FullModeStorage(const FullModeStorage&) = delete;
    FullModeStorage& operator=(const FullModeStorage&) = delete;

    const FullModeLayout& layout() const noexcept;
    const std::vector<std::vector<ReadOnlyBufferRef>>& windows() const noexcept;
    ReadOnlyBufferRef dense_pack_arena() const noexcept;
    ReadOnlyBufferRef expert_address_table() const noexcept;
    ReadOnlyBufferRef resident_table() const noexcept;
    /// Persistent per-layer expert-use counters, initialized to zero.
    BufferRef usage_table() const noexcept;
    /// Row access to the storage-owned PLE reader; lifecycle and I/O mode stay
    /// controlled by this owner.
    strata::kernels::PleTableReader* ple_reader() noexcept;
    /// Resolve a whole dense GGUF tensor to its bounded live buffer view.
    /// Per-expert slabs and the separate per-layer token embedding PLE table
    /// are not returned. Dense PLE projection, normalization, and convolution
    /// tensors remain resolvable through this accessor.
    bool dense_tensor_view(const std::string& tensor_name,
                           FullModeTensorView& view,
                           std::string& error) const;
    bool warm_up(const std::vector<std::pair<std::uint32_t, std::uint32_t>>& expert_order,
                 std::string& error);

private:
    struct Impl;
    explicit FullModeStorage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
