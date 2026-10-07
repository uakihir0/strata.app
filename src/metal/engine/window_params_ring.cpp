// Fixed-address Shared storage for two in-flight-safe window parameter slots.
#include "strata/metal/window_params.hpp"

#include <cstring>
#include <memory>
#include <utility>

namespace strata::metal {

struct WindowParamsRing::Impl {
    Buffers* buffers = nullptr;
    BufferRef allocation;
    std::uint64_t allocation_identity = 0u;
    bool initialized[kWindowParamsRingSlotCount]{};
};

WindowParamsRing::WindowParamsRing(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

WindowParamsRing::~WindowParamsRing() = default;

std::unique_ptr<WindowParamsRing> WindowParamsRing::open(
    Context& context, std::string& error) {
    error.clear();
    Buffers& buffers = context.buffers();
    auto transaction = buffers.transaction();
    BufferRef allocation =
        buffers.allocate("window parameters", kWindowParamsRingBytes, error);
    if (!allocation) return nullptr;

    const std::uint64_t allocation_identity =
        buffers.allocation_identity(allocation);
    if (allocation_identity == 0u ||
        !buffers.owns(allocation, allocation_identity) ||
        allocation.length != kWindowParamsRingBytes ||
        allocation.contents == nullptr) {
        error = "Metal did not return the fixed WindowParams ring allocation";
        return nullptr;
    }
    if (!buffers.commit_residency(error)) return nullptr;

    auto candidate = std::make_unique<Impl>();
    candidate->buffers = &buffers;
    candidate->allocation = allocation;
    candidate->allocation_identity = allocation_identity;
    auto result = std::unique_ptr<WindowParamsRing>(
        new WindowParamsRing(std::move(candidate)));
    if (!result->is_valid()) {
        error = "WindowParams ring became invalid before publication";
        return nullptr;
    }
    transaction.commit();
    error.clear();
    return result;
}

bool WindowParamsRing::is_valid() const noexcept {
    return impl_ != nullptr && impl_->buffers != nullptr &&
           impl_->allocation_identity != 0u &&
           impl_->allocation.length == kWindowParamsRingBytes &&
           impl_->allocation.contents != nullptr &&
           impl_->buffers->owns(impl_->allocation,
                                impl_->allocation_identity);
}

BufferRef WindowParamsRing::view(std::uint32_t slot_index) const noexcept {
    if (!is_valid() || slot_index >= kWindowParamsRingSlotCount ||
        !impl_->initialized[slot_index]) {
        return {};
    }
    const std::size_t offset =
        static_cast<std::size_t>(slot_index) * kWindowParamsRingSlotStride;
    BufferRef result = impl_->allocation.slice(offset, sizeof(WindowParams));
    if (!result ||
        !impl_->buffers->owns(result, impl_->allocation_identity)) {
        return {};
    }
    return result;
}

BufferRef WindowParamsRing::build_slot(
    std::uint32_t slot_index, std::uint32_t pos0,
    std::span<const std::uint32_t> tokens,
    std::span<const WindowSamplerSettings> samplers,
    std::uint32_t n_keep, std::uint32_t seed_lo, std::uint32_t seed_hi,
    std::string& error) {
    error.clear();
    if (slot_index >= kWindowParamsRingSlotCount) {
        error = "WindowParams ring slot index is out of range";
        return {};
    }
    if (!is_valid()) {
        error = "WindowParams ring allocation is unavailable";
        return {};
    }

    WindowParams params{};
    if (!build_window_params(pos0, tokens, samplers, n_keep, seed_lo,
                             seed_hi, params, error)) {
        return {};
    }

    const std::size_t offset =
        static_cast<std::size_t>(slot_index) * kWindowParamsRingSlotStride;
    BufferRef destination =
        impl_->allocation.slice(offset, sizeof(WindowParams));
    if (!destination ||
        !impl_->buffers->owns(destination, impl_->allocation_identity)) {
        error = "WindowParams ring slot is not a live owned view";
        return {};
    }
    std::memcpy(destination.contents, &params, sizeof(params));
    impl_->initialized[slot_index] = true;
    error.clear();
    return destination;
}

}  // namespace strata::metal
