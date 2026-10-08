// src/metal/host/buffers.mm - shared buffers, no-copy windows and fixed-address arenas.
#include "context_impl.h"

#import <Foundation/Foundation.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unistd.h>
#include <utility>

namespace strata::metal {

struct Buffers::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::recursive_mutex mutex;
    // Declared before residency/owned so those Metal handles release before their backing mmap owners.
    std::vector<std::shared_ptr<void>> backing_lifetimes;
    id<MTLResidencySet> residency = nil;
    NSMutableArray<id<MTLBuffer>>* owned = [NSMutableArray array];
    std::unordered_map<void*, std::shared_ptr<BufferValidity>> validity_tokens;
    std::uint64_t next_allocation_identity = 1;
    std::size_t max_buffer_length = 0;
};

static std::string ns_error(NSError* error) {
    if (error == nil) return "unknown error";
    const char* text = error.localizedDescription.UTF8String;
    return text != nullptr ? text : "unknown error";
}

static MTLResourceOptions shared_untracked_options() {
    return MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked |
           MTLResourceCPUCacheModeDefaultCache;
}

static bool poisoned(bool requested) {
    const char* value = std::getenv("STRATA_METAL_POISON");
    return requested || (value != nullptr && std::strcmp(value, "1") == 0);
}

Buffers::Buffers(void* device, void* queue, std::uint64_t max_buffer_length,
                 std::string& err) : impl_(std::make_unique<Impl>()) {
    impl_->device = (__bridge id<MTLDevice>)device;
    impl_->queue = (__bridge id<MTLCommandQueue>)queue;
    if (impl_->device == nil || impl_->queue == nil) {
        err = "cannot create Metal residency set without a device and command queue";
        return;
    }
    impl_->max_buffer_length = static_cast<std::size_t>(max_buffer_length);
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.label = @"strata";
    descriptor.initialCapacity = 64;
    NSError* error = nil;
    impl_->residency = [impl_->device newResidencySetWithDescriptor:descriptor error:&error];
    if (impl_->residency == nil) {
        err = "cannot create Metal residency set: " + ns_error(error);
        return;
    }
    [impl_->queue addResidencySet:impl_->residency];
    err.clear();
}

Buffers::~Buffers() {
    if (impl_) {
        for (const auto& [handle, token] : impl_->validity_tokens) {
            (void)handle;
            if (token != nullptr) token->invalidate();
        }
        if (impl_->queue != nil && impl_->residency != nil) {
            [impl_->queue removeResidencySet:impl_->residency];
        }
    }
}

Buffers::Transaction::Transaction(Buffers& buffers) : buffers_(&buffers) {
    if (!buffers.impl_) {
        buffers_ = nullptr;
        return;
    }
    lock_ = std::unique_lock<std::recursive_mutex>(buffers.impl_->mutex);
    allocation_count_ = static_cast<std::size_t>(buffers.impl_->owned.count);
    backing_lifetime_count_ = buffers.impl_->backing_lifetimes.size();
}

Buffers::Transaction::~Transaction() {
    if (buffers_ == nullptr || committed_ || !lock_.owns_lock() || !buffers_->impl_) return;
    Impl& impl = *buffers_->impl_;
    const bool added_allocations =
        static_cast<std::size_t>(impl.owned.count) > allocation_count_;
    const bool added_backing_lifetimes =
        impl.backing_lifetimes.size() > backing_lifetime_count_;
    if (!added_allocations && !added_backing_lifetimes) return;
    while (static_cast<std::size_t>(impl.owned.count) > allocation_count_) {
        id<MTLBuffer> buffer = impl.owned.lastObject;
        [impl.residency removeAllocation:buffer];
        void* handle = (__bridge void*)buffer;
        const auto token = impl.validity_tokens.find(handle);
        if (token != impl.validity_tokens.end()) {
            token->second->invalidate();
            impl.validity_tokens.erase(token);
        }
        [impl.owned removeLastObject];
    }
    if (impl.backing_lifetimes.size() > backing_lifetime_count_) {
        impl.backing_lifetimes.resize(backing_lifetime_count_);
    }
    if (impl.residency != nil) {
        [impl.residency commit];
        [impl.residency requestResidency];
    }
}

void Buffers::Transaction::commit() noexcept {
    committed_ = true;
    if (lock_.owns_lock()) lock_.unlock();
}

Buffers::Transaction Buffers::transaction() {
    return Transaction(*this);
}

BufferRef Buffers::allocate(const std::string& label, std::size_t bytes, std::string& err, bool poison) {
    if (!impl_ || impl_->device == nil || impl_->residency == nil) {
        err = "Metal buffers are unavailable because the residency set was not created";
        return {};
    }
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if (bytes == 0) {
        err = "cannot allocate an empty Metal buffer";
        return {};
    }
    if (bytes > impl_->max_buffer_length) {
        err = "Metal buffer " + label + " is " + std::to_string(bytes) + " bytes; maxBufferLength is " +
              std::to_string(impl_->max_buffer_length);
        return {};
    }
    id<MTLBuffer> buffer = [impl_->device newBufferWithLength:bytes options:shared_untracked_options()];
    if (buffer == nil) {
        err = "cannot allocate Metal buffer " + label + " (" + std::to_string(bytes) + " bytes)";
        return {};
    }
    buffer.label = @(label.c_str());
    if (buffer.contents == nullptr) {
        err = "Metal buffer " + label + " has no CPU-visible contents";
        return {};
    }
    if (impl_->next_allocation_identity == 0) {
        err = "Metal buffer allocation identity space is exhausted";
        return {};
    }
    const std::uint64_t identity = impl_->next_allocation_identity++;
    void* handle = (__bridge void*)buffer;
    auto validity = std::shared_ptr<BufferValidity>(new BufferValidity(
        handle, buffer.contents, buffer.gpuAddress, bytes, identity));
    if (poisoned(poison)) std::memset(buffer.contents, 0xff, bytes);
    [impl_->owned addObject:buffer];
    impl_->validity_tokens.emplace(handle, validity);
    [impl_->residency addAllocation:buffer];
    err.clear();
    return {handle, buffer.contents, buffer.gpuAddress, bytes, 0, validity};
}

bool Buffers::owns(ReadOnlyBufferRef buffer) const {
    return buffer && owns(buffer.buffer_);
}

bool Buffers::owns(BufferRef buffer) const {
    if (!impl_ || !buffer || buffer.validity_ == nullptr) return false;
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const auto token = impl_->validity_tokens.find(buffer.handle);
    return token != impl_->validity_tokens.end() &&
           token->second == buffer.validity_ &&
           token->second->valid_for(buffer.handle, buffer.contents, buffer.gpu_address,
                                    buffer.length, buffer.offset);
}

std::uint64_t Buffers::allocation_identity(BufferRef buffer) const {
    if (!impl_ || !buffer || buffer.validity_ == nullptr) return 0;
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const auto token = impl_->validity_tokens.find(buffer.handle);
    if (token == impl_->validity_tokens.end() ||
        token->second != buffer.validity_ ||
        !token->second->valid_for(buffer.handle, buffer.contents, buffer.gpu_address,
                                  buffer.length, buffer.offset)) {
        return 0;
    }
    return token->second->allocation_identity_;
}

bool Buffers::owns(BufferRef buffer, std::uint64_t identity) const {
    if (!impl_ || !buffer || buffer.validity_ == nullptr || identity == 0) return false;
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const auto token = impl_->validity_tokens.find(buffer.handle);
    return token != impl_->validity_tokens.end() &&
           token->second == buffer.validity_ &&
           token->second->allocation_identity_ == identity &&
           token->second->valid_for(buffer.handle, buffer.contents, buffer.gpu_address,
                                    buffer.length, buffer.offset);
}

BufferRef Buffers::wrap_no_copy(const std::string& label, void* address, std::size_t bytes, std::string& err) {
    return wrap_no_copy(label, address, bytes, {}, err);
}

BufferRef Buffers::wrap_no_copy(const std::string& label, void* address, std::size_t bytes,
                                std::shared_ptr<void> backing_lifetime, std::string& err) {
    if (!impl_ || impl_->device == nil || impl_->residency == nil) {
        err = "Metal buffers are unavailable because the residency set was not created";
        return {};
    }
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const long page = getpagesize();
    if (address == nullptr || bytes == 0 || page <= 0 ||
        reinterpret_cast<std::uintptr_t>(address) % static_cast<std::size_t>(page) != 0 ||
        bytes % static_cast<std::size_t>(page) != 0) {
        err = "no-copy Metal buffer " + label + " needs a non-null " + std::to_string(page) +
              "-byte-aligned address and length";
        return {};
    }
    if (bytes > impl_->max_buffer_length) {
        err = "no-copy Metal buffer " + label + " is " + std::to_string(bytes) +
              " bytes; maxBufferLength is " + std::to_string(impl_->max_buffer_length);
        return {};
    }
    id<MTLBuffer> buffer = [impl_->device newBufferWithBytesNoCopy:address length:bytes
                                                            options:shared_untracked_options() deallocator:nil];
    if (buffer == nil) {
        err = "cannot wrap no-copy Metal buffer " + label;
        return {};
    }
    buffer.label = @(label.c_str());
    if (buffer.contents == nullptr) {
        err = "no-copy Metal buffer " + label + " has no CPU-visible contents";
        return {};
    }
    if (impl_->next_allocation_identity == 0) {
        err = "Metal buffer allocation identity space is exhausted";
        return {};
    }
    const std::uint64_t identity = impl_->next_allocation_identity++;
    void* handle = (__bridge void*)buffer;
    auto validity = std::shared_ptr<BufferValidity>(new BufferValidity(
        handle, buffer.contents, buffer.gpuAddress, bytes, identity));
    if (backing_lifetime) impl_->backing_lifetimes.push_back(std::move(backing_lifetime));
    [impl_->owned addObject:buffer];
    impl_->validity_tokens.emplace(handle, validity);
    [impl_->residency addAllocation:buffer];
    err.clear();
    return {handle, buffer.contents, buffer.gpuAddress, bytes, 0, validity};
}

bool Buffers::commit_residency(std::string& err) {
    if (!impl_ || impl_->residency == nil) {
        err = "Metal residency set is unavailable";
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    [impl_->residency commit];
    [impl_->residency requestResidency];
    err.clear();
    return true;
}

std::size_t Buffers::allocation_count() const {
    if (!impl_) return 0;
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    return impl_ && impl_->residency != nil ? impl_->residency.allocationCount : 0;
}

std::uint64_t Buffers::allocated_size() const {
    if (!impl_) return 0;
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    return impl_ && impl_->residency != nil ? impl_->residency.allocatedSize : 0;
}

BufferArena::BufferArena(Buffers& buffers, const std::string& label, std::size_t capacity,
                         std::string& err, bool poison) {
    if (capacity == 0) {
        err = "cannot create an empty Metal arena";
        return;
    }
    backing_ = buffers.allocate(label, capacity, err, poison);
    if (backing_) capacity_ = backing_.length;
}

BufferRef BufferArena::allocate(std::size_t bytes, std::size_t alignment, std::string& err) {
    if (!backing_) {
        err = "Metal arena has no backing buffer";
        return {};
    }
    if (bytes == 0) {
        err = "cannot allocate an empty Metal arena slice";
        return {};
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        err = "Metal arena alignment must be a non-zero power of two";
        return {};
    }

    const auto base = reinterpret_cast<std::uintptr_t>(backing_.contents);
    if (base > std::numeric_limits<std::uintptr_t>::max() - used_ ||
        backing_.gpu_address > std::numeric_limits<std::uint64_t>::max() - used_) {
        err = "Metal arena address overflow";
        return {};
    }
    const auto cursor = base + used_;
    if (cursor > std::numeric_limits<std::uintptr_t>::max() - (alignment - 1)) {
        err = "Metal arena alignment overflow";
        return {};
    }
    const auto aligned = (cursor + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    const auto offset = static_cast<std::size_t>(aligned - base);
    if (offset > capacity_ || bytes > capacity_ - offset) {
        err = "Metal arena is out of space";
        return {};
    }
    if (backing_.gpu_address != 0 &&
        (backing_.gpu_address > std::numeric_limits<std::uint64_t>::max() - offset ||
         (backing_.gpu_address + offset) % alignment != 0)) {
        err = "Metal arena CPU and GPU addresses do not share the requested alignment";
        return {};
    }
    BufferRef slice = backing_.slice(offset, bytes);
    if (!slice) {
        err = "Metal arena backing view became invalid";
        return {};
    }
    used_ = offset + bytes;
    err.clear();
    return slice;
}

std::size_t BufferArena::capacity() const noexcept { return capacity_; }
std::size_t BufferArena::used() const noexcept { return used_; }

}  // namespace strata::metal
