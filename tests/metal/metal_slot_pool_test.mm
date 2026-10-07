// tests/metal/metal_slot_pool_test.mm - tiered ping-pong slot-cache leases.
#include "strata/metal/context.hpp"
#include "../../src/metal/engine/metal_fence.hpp"
#include "strata/metal/slot_pool.hpp"

#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace strata::metal;

constexpr std::uint64_t kSlotStride = 16 * 1024;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void publish_loaded(SlotPool::Lease& lease, std::string& error) {
    require(lease.publish(error), "publish completed slot load: " + error);
}

class CommandGateCleanup {
public:
    CommandGateCleanup(Events* events,
                       id<MTLCommandBuffer> command_buffer,
                       std::uint64_t release_value)
        : events_(events), command_buffer_(command_buffer),
          release_value_(release_value) {}

    ~CommandGateCleanup() {
        if (pending_) {
            std::string ignored;
            (void)release_and_drain(ignored);
        }
    }

    bool release_and_drain(std::string& error) {
        if (!pending_) return true;
        if (!events_->signal_host(release_value_, error)) return false;
        pending_ = false;
        [command_buffer_ waitUntilCompleted];
        return true;
    }

private:
    Events* events_ = nullptr;
    id<MTLCommandBuffer> command_buffer_;
    std::uint64_t release_value_ = 0;
    bool pending_ = true;
};

struct CompletionSignal {
    std::mutex mutex;
    std::condition_variable condition;
    bool released = false;
    bool succeeded = false;
};

void test_slot_cache_and_halves(Context& context) {
    std::string error;
    BufferRef arena = context.buffers().allocate(
        "slot-pool test arena", static_cast<std::size_t>(4 * kSlotStride), error);
    require(static_cast<bool>(arena), "allocate slot-pool arena: " + error);
    auto pool = SlotPool::create(arena, 4, 8, 4, kSlotStride, error);
    require(pool != nullptr, "create slot pool: " + error);
    require(pool->half_capacity(0) == 2 && pool->half_capacity(1) == 2 &&
                pool->half_capacity(2) == 0,
            "four slots must split into two equal ping-pong halves");

    auto layer_zero = pool->acquire(0, 3, 512, error);
    require(layer_zero && !layer_zero.view().cache_hit &&
                layer_zero.view().half == 0 && layer_zero.view().local_slot == 0 &&
                layer_zero.view().arena_offset == 0 &&
                layer_zero.view().gpu_address == arena.gpu_address &&
                layer_zero.view().contents == arena.contents,
            "layer zero must claim the first slot of half zero");
    const ExpertSlotView original = layer_zero.view();
    publish_loaded(layer_zero, error);
    layer_zero.reset();

    auto layer_one = pool->acquire(1, 4, 768, error);
    require(layer_one && !layer_one.view().cache_hit &&
                layer_one.view().half == 1 && layer_one.view().local_slot == 0 &&
                layer_one.view().arena_slot == 2 &&
                layer_one.view().arena_offset == 2 * kSlotStride,
            "layer one must claim the first slot of the disjoint second half");
    publish_loaded(layer_one, error);
    layer_one.reset();

    auto hot = pool->acquire(0, 3, 512, error);
    require(hot && hot.view().cache_hit &&
                hot.view().arena_offset == original.arena_offset &&
                hot.view().gpu_address == original.gpu_address,
            "a repeated expert tag must hit its existing half-local cache entry");
    hot.reset();

    auto recent = pool->acquire(0, 5, 512, error);
    require(recent && !recent.view().cache_hit,
            "a second tag must claim the remaining half-zero slot");
    const std::uint64_t recent_offset = recent.view().arena_offset;
    publish_loaded(recent, error);
    recent.reset();

    auto touch_hot = pool->acquire(0, 3, 512, error);
    require(touch_hot && touch_hot.view().cache_hit,
            "touch the hot tag after filling the second slot");
    touch_hot.reset();

    auto least_recent = pool->acquire(0, 6, 512, error);
    require(least_recent && !least_recent.view().cache_hit &&
                least_recent.view().arena_offset == recent_offset,
            "the least-recently-used unpinned slot must be evicted first");
    publish_loaded(least_recent, error);
    least_recent.reset();

    auto still_hot = pool->acquire(0, 3, 512, error);
    require(still_hot && still_hot.view().cache_hit &&
                still_hot.view().arena_offset == original.arena_offset,
            "LRU eviction must preserve the more recently used expert");
}

void test_loading_slots_are_not_cache_hits(Context& context) {
    std::string error;
    BufferRef arena = context.buffers().allocate(
        "slot-pool loading-state arena", static_cast<std::size_t>(2 * kSlotStride), error);
    require(static_cast<bool>(arena), "allocate loading-state arena: " + error);
    auto pool = SlotPool::create(arena, 2, 8, 2, kSlotStride, error);
    require(pool != nullptr, "create loading-state slot pool: " + error);

    auto loading = pool->acquire(0, 3, 512, error);
    require(loading && !loading.view().cache_hit,
            "the first request must reserve a miss in Loading state");
    std::memset(loading.view().contents, 0x5a, 512);
    auto duplicate = pool->acquire(0, 3, 512, error);
    require(!duplicate && error.find("already in progress") != std::string::npos,
            "a concurrent request must not see a partially written loading slot");

    publish_loaded(loading, error);
    auto hit = pool->acquire(0, 3, 512, error);
    require(hit && hit.view().cache_hit,
            "the slot becomes a cache hit only after explicit publication");
    hit.reset();
    loading.reset();

    auto abandoned = pool->acquire(0, 4, 512, error);
    require(abandoned && !abandoned.view().cache_hit,
            "a new expert claims an unpinned slot as Loading");
    const std::uint64_t abandoned_offset = abandoned.view().arena_offset;
    abandoned.reset();
    auto retried = pool->acquire(0, 4, 512, error);
    require(retried && !retried.view().cache_hit &&
                retried.view().arena_offset == abandoned_offset,
            "dropping an unpublished load invalidates the tag for a clean retry");
}

void test_pinned_slot_waits_for_gpu_completion(Context& context) {
    std::string error;
    BufferRef arena = context.buffers().allocate(
        "slot-pool delayed-completion arena", static_cast<std::size_t>(4 * kSlotStride), error);
    require(static_cast<bool>(arena), "allocate delayed-completion arena: " + error);
    auto pool = SlotPool::create(arena, 4, 8, 4, kSlotStride, error);
    require(pool != nullptr, "create delayed-completion slot pool: " + error);

    auto first = pool->acquire(0, 1, 1024, error);
    auto second = pool->acquire(0, 2, 1024, error);
    require(first && second && !first.view().cache_hit && !second.view().cache_hit,
            "fill both half-zero slots with distinct in-flight experts");
    publish_loaded(second, error);
    const std::uint64_t first_offset = first.view().arena_offset;
    std::array<std::uint8_t, 1024> expected{};
    for (std::size_t byte = 0; byte < expected.size(); ++byte) {
        expected[byte] = static_cast<std::uint8_t>((byte * 29 + 7) & 0xff);
    }
    std::memcpy(first.view().contents, expected.data(), expected.size());
    publish_loaded(first, error);
    std::atomic_thread_fence(std::memory_order_release);
    BufferRef readback = context.buffers().allocate("slot-pool readback", expected.size(), error);
    require(static_cast<bool>(readback), "allocate slot-pool readback: " + error);
    require(context.buffers().commit_residency(error),
            "commit slot-pool test buffers to residency: " + error);
    auto completion_lease = std::make_shared<SlotPool::Lease>(std::move(first));
    auto completion_signal = std::make_shared<CompletionSignal>();

    auto arrival_events = context.make_events("slot-pool GPU arrival", error);
    require(arrival_events != nullptr, "create GPU-arrival event: " + error);
    auto gate_events = context.make_events("slot-pool host release gate", error);
    require(gate_events != nullptr, "create host-release event: " + error);
    const std::uint64_t arrival_value = arrival_events->reserve_value();
    const std::uint64_t release_value = gate_events->reserve_value();
    require(arrival_value != 0 && release_value != 0,
            "reserve separate GPU-arrival and host-release event values");
    auto command_buffer = context.make_command_buffer("slot-pool delayed reader", error);
    require(command_buffer != nullptr, "create delayed-reader command buffer: " + error);
    require(arrival_events->encode_signal(*command_buffer, arrival_value, error),
            "encode GPU arrival before the delayed slot-reader wait: " + error);
    require(gate_events->encode_wait(*command_buffer, release_value, error),
            "encode delayed slot-reader wait: " + error);

    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command_buffer->unsafe_native_handle();
    require(command_buffer->add_completion_handler(
                [completion_lease, completion_signal](bool succeeded) {
                    completion_lease->reset();
                    {
                        std::lock_guard<std::mutex> lock(completion_signal->mutex);
                        completion_signal->released = true;
                        completion_signal->succeeded = succeeded;
                    }
                    completion_signal->condition.notify_one();
                }, error),
            "register wrapper completion handler for slot release: " + error);
    id<MTLBlitCommandEncoder> blit = begin_blit_encoder(context, native);
    require(blit != nil, "create slot-pool reader blit encoder");
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)arena.handle
            sourceOffset:static_cast<NSUInteger>(first_offset)
                toBuffer:(__bridge id<MTLBuffer>)readback.handle
       destinationOffset:0
                    size:expected.size()];
    end_blit_encoder(context, blit);
    completion_lease.reset();
    require(command_buffer->enqueue(error), "enqueue delayed slot reader: " + error);
    require(command_buffer->commit(error), "commit delayed slot reader: " + error);
    CommandGateCleanup gate_cleanup(gate_events.get(), native, release_value);
    std::string arrival_error;
    const bool arrived = arrival_events->wait(arrival_value, 5000, arrival_error);
    std::string blocked_error;
    bool slot_was_blocked = false;
    if (arrived) {
        auto blocked = pool->acquire(0, 3, 1024, blocked_error);
        slot_was_blocked =
            !blocked && blocked_error.find("no unpinned slot") != std::string::npos;
    }
    std::string release_error;
    const bool released = gate_cleanup.release_and_drain(release_error);
    require(arrived, "GPU did not reach the delayed slot-reader gate: " + arrival_error);
    require(slot_was_blocked,
            "no slot may be reused while a delayed GPU command still owns its lease: " +
                blocked_error);
    require(released, "release the delayed GPU reader: " + release_error);
    require(native.status == MTLCommandBufferStatusCompleted,
            "the delayed slot-reader command buffer must complete successfully");
    {
        std::unique_lock<std::mutex> lock(completion_signal->mutex);
        require(completion_signal->condition.wait_for(
                    lock, std::chrono::seconds(5), [&] { return completion_signal->released; }),
                "the GPU completion handler must release its slot lease");
        require(completion_signal->succeeded,
                "the slot release callback must observe successful GPU completion");
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    require(std::memcmp(readback.contents, expected.data(), expected.size()) == 0,
            "the delayed GPU reader must copy the leased slot's exact bytes");

    auto after_completion = pool->acquire(0, 3, 1024, error);
    require(after_completion && !after_completion.view().cache_hit &&
                after_completion.view().arena_offset == first_offset,
            "a slot may be reused after its GPU completion handler releases the lease");
    publish_loaded(after_completion, error);
}

void test_odd_slot_split_and_invalid_arena(Context& context) {
    std::string error;
    BufferRef arena;
    auto pool = SlotPool::create(arena, 2, 2, 5, kSlotStride, error);
    require(pool == nullptr && error.find("does not cover") != std::string::npos,
            "a nonempty pool must reject a missing arena");

    arena = context.buffers().allocate(
        "odd slot split arena", static_cast<std::size_t>(5 * kSlotStride), error);
    require(static_cast<bool>(arena),
            "an odd-slot fixture needs an owned Metal arena: " + error);
    pool = SlotPool::create(arena, 4, 8, 5, kSlotStride, error);
    require(pool != nullptr && pool->half_capacity(0) == 3 &&
                pool->half_capacity(1) == 2,
            "an odd extra slot must belong to half zero");
}

}  // namespace

int main() {
    try {
        ContextOptions options;
        options.pipelines = false;
        std::string error;
        auto context = Context::open(options, error);
        require(context != nullptr, "open Metal context: " + error);
        test_slot_cache_and_halves(*context);
        test_loading_slots_are_not_cache_hits(*context);
        test_pinned_slot_waits_for_gpu_completion(*context);
        test_odd_slot_split_and_invalid_arena(*context);
        std::cout << "metal_slot_pool_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_slot_pool_test: " << exception.what() << '\n';
        return 1;
    }
}
