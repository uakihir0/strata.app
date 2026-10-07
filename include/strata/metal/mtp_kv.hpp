// include/strata/metal/mtp_kv.hpp - MTP drafter KV layout and Context-owned store.
#pragma once

#include "strata/metal/kv.hpp"

namespace strata::metal {

/// Exact byte layout for the drafter's single K/V layer.
/// K8v4 is represented by plain int8 for the drafter. Unlike the main QSA state,
/// this layout has no indexer state or pooled rows; the drafter borrows the main
/// KvStore's shared RoPE table.
struct MtpKvLayout {
    KvFormat requested_format = KvFormat::Fp16;
    KvFormat storage_format = KvFormat::Fp16;
    std::uint32_t requested_cells = 0;
    std::uint32_t cell_capacity = 0;
    std::uint32_t kv_rows = 0;
    std::size_t allocation_bytes = 0;
    KvRegion k;
    KvRegion k_scale;
    KvRegion v;
    KvRegion v_scale;
};

/// Build the drafter layout without a Metal device or allocation. Failure clears `layout`.
bool make_mtp_kv_layout(std::uint32_t requested_cells, KvFormat format,
                        MtpKvLayout& layout, std::string& error);

/// Views into the one Context-owned MTP K/V allocation.
struct MtpKvViews {
    BufferRef k;
    BufferRef k_scale;
    BufferRef v;
    BufferRef v_scale;
};

/// Own one contiguous allocation for the MTP drafter's dense-attention K/V.
/// The Context passed to open must outlive this store. K/V data is not initialized.
/// If opened inside a wider Buffers::Transaction, keep the store provisional until
/// that transaction commits and destroy it if the wider transaction rolls back.
class MtpKvStore {
public:
    static std::unique_ptr<MtpKvStore> open(Context& context, std::uint32_t requested_cells,
                                            KvFormat format, std::string& error);
    ~MtpKvStore();
    MtpKvStore(const MtpKvStore&) = delete;
    MtpKvStore& operator=(const MtpKvStore&) = delete;

    const MtpKvLayout& layout() const noexcept;
    /// Empty after an enclosing Buffers::Transaction rolls the allocation back.
    BufferRef allocation() const;
    bool views(MtpKvViews& result, std::string& error) const;

private:
    struct Impl;
    explicit MtpKvStore(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
