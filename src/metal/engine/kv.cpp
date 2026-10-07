// src/metal/engine/kv.cpp - contiguous main-model QSA storage.
#include "strata/metal/kv.hpp"

#include <cstring>
#include <limits>
#include <memory>

namespace strata::metal {
namespace {

bool region_view(BufferRef backing, KvRegion region, BufferRef& view, std::string& error) {
    view = {};
    if (region.bytes == 0) return true;
    if (!backing || backing.contents == nullptr || region.offset > backing.length ||
        region.bytes > backing.length - region.offset ||
        region.offset > std::numeric_limits<std::size_t>::max() - backing.offset ||
        region.offset > std::numeric_limits<std::uint64_t>::max() - backing.gpu_address) {
        error = "QSA state region is outside its Metal allocation";
        return false;
    }
    if (region.offset > std::numeric_limits<std::size_t>::max() ||
        region.bytes > std::numeric_limits<std::size_t>::max()) {
        error = "QSA state region is outside its Metal allocation";
        return false;
    }
    view = backing.slice(static_cast<std::size_t>(region.offset),
                         static_cast<std::size_t>(region.bytes));
    if (!view) {
        error = "QSA state region could not be represented as a bounded Metal view";
        return false;
    }
    return true;
}

void initialize_indexer_state(BufferRef allocation, const QsaKvLayout& layout) {
    for (const QsaKvLayerLayout& layer : layout.layers) {
        std::memset(static_cast<std::byte*>(allocation.contents) + layer.indexer_tail.offset,
                    0, layer.indexer_tail.bytes);
        std::memset(static_cast<std::byte*>(allocation.contents) + layer.indexer_dead.offset,
                    0, layer.indexer_dead.bytes);
        std::memset(static_cast<std::byte*>(allocation.contents) + layer.indexer_block_pos.offset,
                    0, layer.indexer_block_pos.bytes);
    }
}

}  // namespace

struct KvStore::Impl {
    QsaKvLayout layout;
    BufferRef allocation;
};

KvStore::KvStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
KvStore::~KvStore() = default;

std::unique_ptr<KvStore> KvStore::open(Context& context, std::uint32_t requested_cells,
                                       KvFormat format, std::string& error) {
    error.clear();
    auto candidate = std::make_unique<Impl>();
    if (!make_qsa_kv_layout(requested_cells, format, candidate->layout, error)) return nullptr;

    const DeviceFacts facts = context.device_facts();
    if (facts.max_buffer_length == 0 ||
        candidate->layout.allocation_bytes > facts.max_buffer_length) {
        error = "QSA KV state exceeds the Metal device's maxBufferLength";
        return nullptr;
    }
    candidate->allocation = context.buffers().allocate(
        "main QSA KV, indexer and RoPE state", candidate->layout.allocation_bytes, error);
    if (!candidate->allocation) return nullptr;
    if (!context.buffers().owns(candidate->allocation) ||
        candidate->allocation.length < candidate->layout.allocation_bytes ||
        candidate->allocation.contents == nullptr) {
        error = "Metal did not return a valid owned QSA state allocation";
        return nullptr;
    }

    initialize_indexer_state(candidate->allocation, candidate->layout);
    if (!context.buffers().commit_residency(error)) return nullptr;
    return std::unique_ptr<KvStore>(new KvStore(std::move(candidate)));
}

const QsaKvLayout& KvStore::layout() const noexcept { return impl_->layout; }
BufferRef KvStore::allocation() const noexcept { return impl_->allocation; }

BufferRef KvStore::rope() const noexcept {
    BufferRef result;
    std::string ignored;
    (void)region_view(impl_->allocation, impl_->layout.rope, result, ignored);
    return result;
}

bool KvStore::layer(std::uint32_t index, QsaKvLayerViews& views, std::string& error) const {
    error.clear();
    views = {};
    if (impl_ == nullptr || index >= impl_->layout.layers.size()) {
        error = "QSA KV layer index is out of range";
        return false;
    }

    const QsaKvLayerLayout& regions = impl_->layout.layers[index];
    if (!region_view(impl_->allocation, regions.k, views.k, error) ||
        !region_view(impl_->allocation, regions.k_scale, views.k_scale, error) ||
        !region_view(impl_->allocation, regions.v, views.v, error) ||
        !region_view(impl_->allocation, regions.v_scale, views.v_scale, error) ||
        !region_view(impl_->allocation, regions.indexer_tail, views.indexer_tail, error) ||
        !region_view(impl_->allocation, regions.indexer_dead, views.indexer_dead, error) ||
        !region_view(impl_->allocation, regions.indexer_pooled, views.indexer_pooled, error) ||
        !region_view(impl_->allocation, regions.indexer_block_pos, views.indexer_block_pos, error)) {
        views = {};
        return false;
    }
    return true;
}

}  // namespace strata::metal
