// src/metal/engine/mtp_kv.cpp - Context-owned MTP drafter KV allocation.
#include "strata/metal/mtp_kv.hpp"

#include <limits>
#include <memory>
#include <utility>

namespace strata::metal {
namespace {

bool region_view(BufferRef backing, KvRegion region, BufferRef& view, std::string& error) {
    view = {};
    if (region.bytes == 0) return true;
    if (!backing || backing.contents == nullptr ||
        region.offset > std::numeric_limits<std::size_t>::max() ||
        region.bytes > std::numeric_limits<std::size_t>::max() ||
        region.offset > backing.length ||
        region.bytes > backing.length - region.offset ||
        region.offset > std::numeric_limits<std::size_t>::max() - backing.offset ||
        region.offset > std::numeric_limits<std::uint64_t>::max() - backing.gpu_address) {
        error = "MTP KV region is outside its Metal allocation";
        return false;
    }
    const auto view_offset = backing.offset + region.offset;
    const auto gpu_address = backing.gpu_address + region.offset;
    if (region.bytes > std::numeric_limits<std::uint64_t>::max() - gpu_address ||
        region.bytes > std::numeric_limits<std::size_t>::max() - view_offset) {
        error = "MTP KV region address extent overflows";
        return false;
    }
    view = backing.slice(static_cast<std::size_t>(region.offset),
                         static_cast<std::size_t>(region.bytes));
    if (!view) {
        error = "MTP KV region could not be represented as a bounded Metal view";
        return false;
    }
    return true;
}

}  // namespace

struct MtpKvStore::Impl {
    Buffers* buffers = nullptr;
    std::uint64_t allocation_identity = 0;
    MtpKvLayout layout;
    BufferRef allocation;
};

MtpKvStore::MtpKvStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MtpKvStore::~MtpKvStore() = default;

std::unique_ptr<MtpKvStore> MtpKvStore::open(Context& context, std::uint32_t requested_cells,
                                             KvFormat format, std::string& error) {
    error.clear();
    auto candidate = std::make_unique<Impl>();
    if (!make_mtp_kv_layout(requested_cells, format, candidate->layout, error)) return nullptr;

    const DeviceFacts facts = context.device_facts();
    if (facts.max_buffer_length == 0 ||
        candidate->layout.allocation_bytes > facts.max_buffer_length) {
        error = "MTP drafter KV exceeds the Metal device's maxBufferLength";
        return nullptr;
    }

    candidate->buffers = &context.buffers();
    auto transaction = candidate->buffers->transaction();
    candidate->allocation = candidate->buffers->allocate(
        "MTP drafter K/V", candidate->layout.allocation_bytes, error);
    if (!candidate->allocation) return nullptr;
    candidate->allocation_identity =
        candidate->buffers->allocation_identity(candidate->allocation);
    if (candidate->allocation_identity == 0 ||
        !candidate->buffers->owns(candidate->allocation, candidate->allocation_identity) ||
        candidate->allocation.length < candidate->layout.allocation_bytes ||
        candidate->allocation.contents == nullptr) {
        error = "Metal did not return a valid owned MTP KV allocation";
        return nullptr;
    }
    if (!candidate->buffers->commit_residency(error)) return nullptr;

    auto result = std::unique_ptr<MtpKvStore>(new MtpKvStore(std::move(candidate)));
    transaction.commit();
    error.clear();
    return result;
}

const MtpKvLayout& MtpKvStore::layout() const noexcept { return impl_->layout; }
BufferRef MtpKvStore::allocation() const {
    if (impl_ == nullptr || impl_->buffers == nullptr ||
        !impl_->buffers->owns(impl_->allocation, impl_->allocation_identity)) {
        return {};
    }
    return impl_->allocation;
}

bool MtpKvStore::views(MtpKvViews& result, std::string& error) const {
    error.clear();
    result = {};
    if (impl_ == nullptr || impl_->buffers == nullptr ||
        !impl_->buffers->owns(impl_->allocation, impl_->allocation_identity)) {
        error = "MTP KV allocation is no longer owned by its Context";
        return false;
    }

    if (!region_view(impl_->allocation, impl_->layout.k, result.k, error) ||
        !region_view(impl_->allocation, impl_->layout.k_scale, result.k_scale, error) ||
        !region_view(impl_->allocation, impl_->layout.v, result.v, error) ||
        !region_view(impl_->allocation, impl_->layout.v_scale, result.v_scale, error)) {
        result = {};
        return false;
    }
    return true;
}

}  // namespace strata::metal
