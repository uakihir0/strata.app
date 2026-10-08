// include/strata/metal/tiered.hpp - tiered dense/resident arena layout and storage.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/geometry.hpp"
#include "strata/metal/plan.hpp"
#include "strata/metal/slot_pool.hpp"
#include "strata/metal/window_plan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::metal {

inline constexpr std::uint32_t kTieredMaxLayers = 48;
inline constexpr std::uint32_t kTieredMaxExperts = 512;
inline constexpr std::uint32_t kTieredMoePlanGroups = 80;
inline constexpr std::size_t kTieredCopyWorkers = 8;

struct TieredMoePlan {
    std::uint32_t n_groups = 0;
    std::uint32_t n_entries = 0;
    std::uint32_t n_miss = 0;
    std::uint32_t skip = 0;
    std::int32_t grp_e[kTieredMoePlanGroups]{};
    std::int32_t grp_res[kTieredMoePlanGroups]{};
    std::int32_t grp_start[kTieredMoePlanGroups + 1]{};
    std::int32_t ent_tok[kTieredMoePlanGroups]{};
    std::int32_t ent_dst[kTieredMoePlanGroups]{};
};

static_assert(sizeof(TieredMoePlan) == 1'620);

struct TieredSourceRange {
    std::size_t shard_index = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t byte_count = 0;
};

struct TieredDenseCopy {
    std::string tensor_name;
    bool from_dense_pack = false;
    std::size_t shard_index = 0;
    std::uint64_t source_offset = 0;
    std::uint64_t source_byte_count = 0;
    std::uint64_t byte_count = 0;
    std::uint64_t arena_offset = 0;
};

struct TieredTensorBinding {
    std::string tensor_name;
    std::uint64_t gpu_address = 0;
    std::uint64_t byte_count = 0;
    std::uint64_t arena_offset = 0;
};

struct TieredExpert {
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;
    std::int32_t resident_slot = -1;
    std::uint64_t blob_bytes = 0;
    std::uint64_t resident_arena_offset = 0;
    std::array<TieredSourceRange, 3> roles{};
    std::array<std::uint64_t, 3> resident_addresses{};
};

struct TieredLayout {
    std::uint64_t dense_arena_bytes = 0;
    std::uint64_t resident_arena_bytes = 0;
    std::vector<TieredDenseCopy> dense_copies;
    std::vector<TieredTensorBinding> tensors;
    std::vector<TieredExpert> experts;
    // Maximum-capacity tables, indexed with active row stride n_expert.
    std::vector<std::uint64_t> expert_addresses;
    std::vector<std::int32_t> resident;
    std::array<TieredMoePlan, kTieredMaxLayers> plans{};
};

/// Build the exact dense/resident copy ranges and kernel tables without allocating Metal buffers.
bool build_tiered_layout(const Geometry& geometry, const WindowSources& sources,
                         const PlanResult& memory_plan, std::uint64_t dense_gpu_base,
                         std::uint64_t resident_gpu_base, TieredLayout& layout,
                         std::string& error);

/// Owns the tiered anonymous arenas and both descriptors per GGUF shard.
/// The Context passed to open must outlive this storage.
class TieredStorage {
public:
    static std::unique_ptr<TieredStorage> open(Context& context, const std::string& native_shard,
                                                const Geometry& geometry,
                                                const WindowSources& sources,
                                                const PlanResult& memory_plan,
                                                std::string& error);
    ~TieredStorage();
    TieredStorage(const TieredStorage&) = delete;
    TieredStorage& operator=(const TieredStorage&) = delete;

    const TieredLayout& layout() const noexcept;
    BufferRef dense_arena() const noexcept;
    BufferRef resident_arena() const noexcept;
    BufferRef slot_arena() const noexcept;
    SlotPool* slot_pool() noexcept;
    /// Read one exact cold-expert slab into a slot without changing the file offset.
    bool read_expert_role(std::uint32_t layer, std::uint32_t expert, std::uint32_t role,
                          void* destination, std::size_t destination_bytes,
                          std::string& error) const;
    /// Query all three slab extents with mincore. Returns false on any query failure so
    /// callers conservatively classify the blob as a file read.
    bool expert_pages_resident(std::uint32_t layer, std::uint32_t expert) const noexcept;
    BufferRef expert_address_table() const noexcept;
    BufferRef resident_table() const noexcept;
    BufferRef plan_buffer() const noexcept;
    /// Persistent per-layer expert-use counters, initialized to zero.
    BufferRef usage_table() const noexcept;

private:
    struct Impl;
    explicit TieredStorage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
