// include/strata/metal/slot_pool.hpp - fixed tiered expert slots and pinned cache tags.
#pragma once

#include "strata/metal/context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::metal {

struct ExpertSlotView {
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;
    std::uint32_t half = 0;
    std::uint64_t local_slot = 0;
    std::uint64_t arena_slot = 0;
    std::uint64_t arena_offset = 0;
    std::uint64_t gpu_address = 0;
    std::uint64_t byte_count = 0;
    void* contents = nullptr;
    bool cache_hit = false;
};

/// Thread-safe two-half slot cache. A lease pins a slot until its GPU reader is done.
class SlotPool {
    struct State;

public:
    class Lease {
    public:
        Lease() = default;
        ~Lease();
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        explicit operator bool() const noexcept;
        const ExpertSlotView& view() const noexcept;
        /// Publish a completed cache miss so later acquisitions may read it.
        bool publish(std::string& error);
        void reset() noexcept;

    private:
        Lease(std::shared_ptr<State> state, std::size_t slot_index,
              ExpertSlotView view);
        std::shared_ptr<State> state_;
        std::size_t slot_index_ = 0;
        ExpertSlotView view_{};
        bool owns_load_ = false;
        friend class SlotPool;
    };

    /// Slots are divided as evenly as possible. Layer parity selects a half;
    /// an odd extra slot belongs to half zero.
    static std::unique_ptr<SlotPool> create(BufferRef arena, std::uint32_t n_layers,
                                             std::uint32_t n_expert,
                                             std::uint64_t slot_count,
                                             std::uint64_t slot_stride_bytes,
                                             std::string& error);
    ~SlotPool();
    SlotPool(const SlotPool&) = delete;
    SlotPool& operator=(const SlotPool&) = delete;

    /// Return a ready cache-hit lease or claim an empty/LRU unpinned slot of
    /// layer % 2. A miss stays Loading until its lease publishes the completed
    /// read; duplicate requests while Loading fail instead of exposing bytes.
    Lease acquire(std::uint32_t layer, std::uint32_t expert,
                  std::uint64_t byte_count, std::string& error);
    /// Pin and touch a Ready entry without claiming or evicting on a miss.
    /// An empty lease with an empty error means the tag is not cached.
    Lease lookup_ready(std::uint32_t layer, std::uint32_t expert,
                       std::uint64_t byte_count, std::string& error);
    std::uint64_t half_capacity(std::uint32_t half) const noexcept;
    std::uint64_t slot_stride_bytes() const noexcept;

private:
    explicit SlotPool(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
};

}  // namespace strata::metal
