// src/metal/engine/slot_pool.cpp - ping-pong slot cache with explicit reader leases.
#include "strata/metal/slot_pool.hpp"

#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace strata::metal {

struct SlotPool::State {
    enum class SlotStatus { empty, loading, ready };

    struct Slot {
        SlotStatus status = SlotStatus::empty;
        std::uint32_t layer = 0;
        std::uint32_t expert = 0;
        std::uint64_t byte_count = 0;
        std::uint64_t last_use = 0;
        std::uint32_t pins = 0;
    };

    BufferRef arena;
    std::uint32_t n_layers = 0;
    std::uint32_t n_expert = 0;
    std::uint64_t slot_count = 0;
    std::uint64_t slot_stride_bytes = 0;
    std::uint64_t half_zero_slots = 0;
    std::uint64_t use_clock = 0;
    std::vector<Slot> slots;
    mutable std::mutex mutex;
};

namespace {

bool checked_multiply(std::uint64_t left, std::uint64_t right,
                      std::uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

}  // namespace

SlotPool::Lease::Lease(std::shared_ptr<State> state, std::size_t slot_index,
                       ExpertSlotView view)
    : state_(std::move(state)), slot_index_(slot_index), view_(view),
      owns_load_(!view.cache_hit) {}

SlotPool::Lease::~Lease() { reset(); }

SlotPool::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)),
      slot_index_(std::exchange(other.slot_index_, 0)),
      view_(std::exchange(other.view_, ExpertSlotView{})),
      owns_load_(std::exchange(other.owns_load_, false)) {}

SlotPool::Lease& SlotPool::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        slot_index_ = std::exchange(other.slot_index_, 0);
        view_ = std::exchange(other.view_, ExpertSlotView{});
        owns_load_ = std::exchange(other.owns_load_, false);
    }
    return *this;
}

SlotPool::Lease::operator bool() const noexcept {
    return state_ != nullptr && view_.contents != nullptr;
}

const ExpertSlotView& SlotPool::Lease::view() const noexcept { return view_; }

bool SlotPool::Lease::publish(std::string& error) {
    error.clear();
    if (!state_ || !owns_load_ || slot_index_ >= state_->slots.size()) {
        error = "tiered slot lease does not own an unpublished cache load";
        return false;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    State::Slot& slot = state_->slots[slot_index_];
    if (slot.status != State::SlotStatus::loading || slot.layer != view_.layer ||
        slot.expert != view_.expert || slot.pins == 0) {
        error = "tiered slot load is no longer publishable";
        return false;
    }
    slot.status = State::SlotStatus::ready;
    owns_load_ = false;
    return true;
}

void SlotPool::Lease::reset() noexcept {
    if (!state_) return;
    if (slot_index_ < state_->slots.size()) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        State::Slot& slot = state_->slots[slot_index_];
        if (slot.pins != 0) --slot.pins;
        if (slot.pins == 0 && slot.status == State::SlotStatus::loading) {
            slot = State::Slot{};
        }
    }
    state_.reset();
    slot_index_ = 0;
    view_ = {};
    owns_load_ = false;
}

std::unique_ptr<SlotPool> SlotPool::create(BufferRef arena, std::uint32_t n_layers,
                                            std::uint32_t n_expert,
                                            std::uint64_t slot_count,
                                            std::uint64_t slot_stride_bytes,
                                            std::string& error) {
    error.clear();
    if (n_layers == 0 || n_expert == 0) {
        error = "tiered slot pool needs nonzero model dimensions";
        return nullptr;
    }

    std::uint64_t arena_bytes = 0;
    if (slot_count == 0) {
        if (slot_stride_bytes != 0 || arena) {
            error = "empty tiered slot pool must not have an arena or slot stride";
            return nullptr;
        }
    } else if (slot_stride_bytes == 0 ||
               !checked_multiply(slot_count, slot_stride_bytes, arena_bytes) ||
               arena_bytes > std::numeric_limits<std::size_t>::max() ||
               !arena || arena.contents == nullptr ||
               arena_bytes > arena.length ||
               arena.gpu_address == 0 ||
               arena_bytes > std::numeric_limits<std::uint64_t>::max() - arena.gpu_address ||
               arena_bytes > std::numeric_limits<std::uintptr_t>::max() -
                                 reinterpret_cast<std::uintptr_t>(arena.contents) ||
               slot_count > std::numeric_limits<std::size_t>::max()) {
        error = "tiered slot arena does not cover its declared slots";
        return nullptr;
    }

    auto state = std::make_shared<State>();
    state->arena = arena;
    state->n_layers = n_layers;
    state->n_expert = n_expert;
    state->slot_count = slot_count;
    state->slot_stride_bytes = slot_stride_bytes;
    state->half_zero_slots = slot_count / 2 + slot_count % 2;
    state->slots.resize(static_cast<std::size_t>(slot_count));
    return std::unique_ptr<SlotPool>(new SlotPool(std::move(state)));
}

SlotPool::SlotPool(std::shared_ptr<State> state) : state_(std::move(state)) {}
SlotPool::~SlotPool() = default;

SlotPool::Lease SlotPool::acquire(std::uint32_t layer, std::uint32_t expert,
                                  std::uint64_t byte_count, std::string& error) {
    error.clear();
    if (!state_ || layer >= state_->n_layers || expert >= state_->n_expert ||
        byte_count == 0 || byte_count > state_->slot_stride_bytes) {
        error = "tiered slot request is outside the model or slot dimensions";
        return {};
    }
    if (state_->slot_count == 0) {
        error = "tiered slot pool has no capacity";
        return {};
    }

    const std::uint32_t half = layer % 2;
    const std::uint64_t begin = half == 0 ? 0 : state_->half_zero_slots;
    const std::uint64_t count = half == 0
        ? state_->half_zero_slots
        : state_->slot_count - state_->half_zero_slots;
    if (count == 0) {
        error = "tiered slot half has no capacity";
        return {};
    }

    std::lock_guard<std::mutex> lock(state_->mutex);
    std::uint64_t tick = 0;
    if (state_->use_clock == std::numeric_limits<std::uint64_t>::max()) {
        error = "tiered slot-cache LRU counter is exhausted";
        return {};
    }
    tick = ++state_->use_clock;

    std::size_t selected = std::numeric_limits<std::size_t>::max();
    for (std::uint64_t local = 0; local < count; ++local) {
        const std::size_t index = static_cast<std::size_t>(begin + local);
        State::Slot& slot = state_->slots[index];
        if (slot.status != State::SlotStatus::empty && slot.layer == layer &&
            slot.expert == expert) {
            if (slot.byte_count != byte_count) {
                error = "tiered slot-cache tag changed blob size";
                return {};
            }
            if (slot.status == State::SlotStatus::loading) {
                error = "tiered slot-cache load is already in progress";
                return {};
            }
            if (slot.pins == std::numeric_limits<std::uint32_t>::max()) {
                error = "tiered slot-cache pin count is exhausted";
                return {};
            }
            ++slot.pins;
            slot.last_use = tick;
            selected = index;
            break;
        }
    }

    bool cache_hit = selected != std::numeric_limits<std::size_t>::max();
    if (!cache_hit) {
        std::uint64_t oldest_use = std::numeric_limits<std::uint64_t>::max();
        for (std::uint64_t local = 0; local < count; ++local) {
            const std::size_t index = static_cast<std::size_t>(begin + local);
            const State::Slot& slot = state_->slots[index];
            if (slot.pins != 0) continue;
            if (slot.status == State::SlotStatus::empty) {
                selected = index;
                break;
            }
            if (selected == std::numeric_limits<std::size_t>::max() ||
                slot.last_use < oldest_use) {
                selected = index;
                oldest_use = slot.last_use;
            }
        }
        if (selected == std::numeric_limits<std::size_t>::max()) {
            error = "tiered slot half has no unpinned slot";
            return {};
        }

        State::Slot& slot = state_->slots[selected];
        slot.status = State::SlotStatus::loading;
        slot.layer = layer;
        slot.expert = expert;
        slot.byte_count = byte_count;
        slot.last_use = tick;
        slot.pins = 1;
    }

    const std::uint64_t local_slot = selected - static_cast<std::size_t>(begin);
    const std::uint64_t arena_slot = selected;
    const std::uint64_t arena_offset = arena_slot * state_->slot_stride_bytes;
    auto* const contents = static_cast<std::uint8_t*>(state_->arena.contents) +
                           static_cast<std::size_t>(arena_offset);
    ExpertSlotView view;
    view.layer = layer;
    view.expert = expert;
    view.half = half;
    view.local_slot = local_slot;
    view.arena_slot = arena_slot;
    view.arena_offset = arena_offset;
    view.gpu_address = state_->arena.gpu_address + arena_offset;
    view.byte_count = byte_count;
    view.contents = contents;
    view.cache_hit = cache_hit;
    return Lease(state_, selected, view);
}

SlotPool::Lease SlotPool::lookup_ready(std::uint32_t layer, std::uint32_t expert,
                                       std::uint64_t byte_count,
                                       std::string& error) {
    error.clear();
    if (!state_ || layer >= state_->n_layers || expert >= state_->n_expert ||
        byte_count == 0 || byte_count > state_->slot_stride_bytes) {
        error = "tiered slot lookup is outside the model or slot dimensions";
        return {};
    }
    if (state_->slot_count == 0) {
        error = "tiered slot pool has no capacity";
        return {};
    }

    const std::uint32_t half = layer % 2;
    const std::uint64_t begin = half == 0 ? 0 : state_->half_zero_slots;
    const std::uint64_t count = half == 0
        ? state_->half_zero_slots
        : state_->slot_count - state_->half_zero_slots;
    if (count == 0) {
        error = "tiered slot half has no capacity";
        return {};
    }

    std::lock_guard<std::mutex> lock(state_->mutex);
    for (std::uint64_t local = 0; local < count; ++local) {
        const std::size_t index = static_cast<std::size_t>(begin + local);
        State::Slot& slot = state_->slots[index];
        if (slot.status == State::SlotStatus::empty || slot.layer != layer ||
            slot.expert != expert) {
            continue;
        }
        if (slot.byte_count != byte_count) {
            error = "tiered slot-cache tag changed blob size";
            return {};
        }
        if (slot.status == State::SlotStatus::loading) {
            error = "tiered slot-cache load is already in progress";
            return {};
        }
        if (slot.pins == std::numeric_limits<std::uint32_t>::max()) {
            error = "tiered slot-cache pin count is exhausted";
            return {};
        }
        if (state_->use_clock == std::numeric_limits<std::uint64_t>::max()) {
            error = "tiered slot-cache LRU counter is exhausted";
            return {};
        }
        slot.last_use = ++state_->use_clock;
        ++slot.pins;
        const std::uint64_t local_slot = index - static_cast<std::size_t>(begin);
        const std::uint64_t arena_offset =
            static_cast<std::uint64_t>(index) * state_->slot_stride_bytes;
        ExpertSlotView view;
        view.layer = layer;
        view.expert = expert;
        view.half = half;
        view.local_slot = local_slot;
        view.arena_slot = index;
        view.arena_offset = arena_offset;
        view.gpu_address = state_->arena.gpu_address + arena_offset;
        view.byte_count = byte_count;
        view.contents = static_cast<std::uint8_t*>(state_->arena.contents) +
                        static_cast<std::size_t>(arena_offset);
        view.cache_hit = true;
        return Lease(state_, index, view);
    }
    return {};
}

std::uint64_t SlotPool::half_capacity(std::uint32_t half) const noexcept {
    if (!state_ || half > 1) return 0;
    return half == 0
        ? state_->half_zero_slots
        : state_->slot_count - state_->half_zero_slots;
}

std::uint64_t SlotPool::slot_stride_bytes() const noexcept {
    return state_ ? state_->slot_stride_bytes : 0;
}

}  // namespace strata::metal
