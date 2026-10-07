// include/strata/metal/kv.hpp - QSA KV and indexer state layout (docs/mac/04 sections 4.5-4.6).
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/plan.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::metal {

inline constexpr std::uint32_t kQsaKvLayerCount = 12;
inline constexpr std::uint32_t kQsaKvHeadCount = 2;
inline constexpr std::uint32_t kQsaKvHeadWidth = 256;
inline constexpr std::uint32_t kQsaKvCellsPerRowGroup = 4;
inline constexpr std::uint32_t kQsaIndexerKeyWidth = 128;
inline constexpr std::uint32_t kKvMaximumCells = 262'144;

/// Byte interval within one future Context-owned KV allocation.
struct KvRegion {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

/// One QSA layer's interleaved K/V pools and rollback-capable indexer state.
struct QsaKvLayerLayout {
    KvRegion k;
    KvRegion k_scale;
    KvRegion v;
    KvRegion v_scale;
    KvRegion indexer_tail;
    KvRegion indexer_dead;
    KvRegion indexer_pooled;
    KvRegion indexer_block_pos;
};

/// Metal views for one layer, each a subrange of the same KvStore allocation.
struct QsaKvLayerViews {
    BufferRef k;
    BufferRef k_scale;
    BufferRef v;
    BufferRef v_scale;
    BufferRef indexer_tail;
    BufferRef indexer_dead;
    BufferRef indexer_pooled;
    BufferRef indexer_block_pos;
};

/// Exact layout for the main model's QSA cache, indexer state and shared RoPE table.
/// `cell_capacity` is rounded up to a complete four-cell interleave group.
struct QsaKvLayout {
    KvFormat format = KvFormat::Fp16;
    std::uint32_t requested_cells = 0;
    std::uint32_t cell_capacity = 0;
    std::uint32_t kv_rows = 0;
    std::uint32_t pooled_rows_per_layer = 0;
    std::size_t allocation_bytes = 0;
    std::vector<QsaKvLayerLayout> layers;
    KvRegion rope;
};

/// Build the byte layout without a Metal device or allocation. Failure clears `layout`.
bool make_qsa_kv_layout(std::uint32_t requested_cells, KvFormat format,
                        QsaKvLayout& layout, std::string& error);

/// Rebuild and validate a planned main QSA allocation against its exact byte
/// count and the live device's single-buffer limit. Does not allocate.
bool validate_main_qsa_kv_plan(const PlanResult& plan, std::uint64_t max_buffer_length,
                               QsaKvLayout& layout, std::string& error);

/// Owns one contiguous allocation for the main-model QSA KV pools, indexer state and RoPE table.
/// The Context passed to open must outlive this store. KV rows and RoPE values are written by the
/// caller before use; indexer tail, dead and block-position state starts at zero.
class KvStore {
public:
    static std::unique_ptr<KvStore> open(Context& context, std::uint32_t requested_cells,
                                         KvFormat format, std::string& error);
    ~KvStore();
    KvStore(const KvStore&) = delete;
    KvStore& operator=(const KvStore&) = delete;

    const QsaKvLayout& layout() const noexcept;
    BufferRef allocation() const noexcept;
    BufferRef rope() const noexcept;
    bool layer(std::uint32_t index, QsaKvLayerViews& views, std::string& error) const;

private:
    struct Impl;
    explicit KvStore(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
