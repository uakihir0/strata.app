// src/metal/host/events.mm - monotonic MTLSharedEvent values for host/GPU ordering.
#include "context_impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace strata::metal {

struct Events::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    void* queue_identity = nullptr;
    id<MTLSharedEvent> event = nil;
    std::uint64_t reserved = 0;
    std::uint64_t last_encoded_signal = 0;
    struct PendingSignal {
        enum class Submission { encoded, enqueued, committed };
        void* command_buffer = nullptr;
        Submission submission = Submission::encoded;
    };
    struct PendingWait {
        enum class Submission { encoded, enqueued, committed };
        void* command_buffer = nullptr;
        std::uint64_t value = 0;
        std::uint64_t operation_order = 0;
        Submission submission = Submission::encoded;
    };
    std::map<std::uint64_t, PendingSignal> pending_gpu_signals;
    std::vector<PendingWait> pending_gpu_waits;
    std::mutex value_mutex;
    std::condition_variable waits_changed;
    bool released = false;
};

Events::Events(void* device, void* queue, const std::string& label, std::string& err)
    : impl_(std::make_shared<Impl>()) {
    impl_->device = (__bridge id<MTLDevice>)device;
    if (impl_->device == nil) {
        err = "cannot create a Metal shared event without a device";
        return;
    }
    impl_->queue = (__bridge id<MTLCommandQueue>)queue;
    impl_->queue_identity = queue;
    if (impl_->queue == nil) {
        err = "cannot create a Metal shared event without a command queue";
        return;
    }
    impl_->event = [impl_->device newSharedEvent];
    if (impl_->event == nil) {
        err = "cannot create a Metal shared event";
        return;
    }
    impl_->event.label = @(label.c_str());
    err.clear();
}

Events::~Events() = default;

std::uint64_t Events::reserve_value() {
    if (!impl_ || impl_->event == nil) return 0;
    std::lock_guard<std::mutex> lock(impl_->value_mutex);
    const std::uint64_t floor =
        std::max(impl_->reserved, std::max(impl_->event.signaledValue, impl_->last_encoded_signal));
    if (floor >= std::numeric_limits<std::uint64_t>::max() - 1) return 0;
    impl_->reserved = floor + 1;
    return impl_->reserved;
}

std::optional<std::uint64_t> Events::signaled_value() const {
    if (!impl_ || impl_->event == nil) return std::nullopt;
    std::lock_guard<std::mutex> lock(impl_->value_mutex);
    if (impl_->released) return std::nullopt;
    return impl_->event.signaledValue;
}

bool Events::release_gpu_waits(std::uint64_t timeout_ms, std::string& err) {
    if (!impl_ || impl_->event == nil) {
        err = "Metal shared event is unavailable";
        return false;
    }

    const auto started = std::chrono::steady_clock::now();
    const auto available_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::time_point::max() - started).count();
    if (available_ms < 0 || timeout_ms > static_cast<std::uint64_t>(available_ms)) {
        err = "Metal GPU-wait release timeout is too large";
        return false;
    }
    const auto deadline = started + std::chrono::milliseconds(
        static_cast<std::chrono::milliseconds::rep>(timeout_ms));
    {
        std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
        std::lock_guard<std::mutex> value_lock(impl_->value_mutex);
        if (!impl_->released) {
            std::atomic_thread_fence(std::memory_order_release);
            impl_->event.signaledValue = std::numeric_limits<std::uint64_t>::max();
            impl_->reserved = std::numeric_limits<std::uint64_t>::max();
            impl_->released = true;
        }
    }

    std::unique_lock<std::mutex> value_lock(impl_->value_mutex);
    const auto committed_waits_drained = [this] {
        return std::none_of(impl_->pending_gpu_waits.begin(),
                            impl_->pending_gpu_waits.end(), [](const Impl::PendingWait& wait) {
            return wait.submission != Impl::PendingWait::Submission::encoded;
        });
    };
    if (!impl_->waits_changed.wait_until(value_lock, deadline, committed_waits_drained)) {
        err = "timed out draining Metal command buffers after releasing GPU waits";
        return false;
    }
    err.clear();
    return true;
}

bool Events::signal_host(std::uint64_t value, std::string& err) {
    if (!impl_ || impl_->event == nil) {
        err = "Metal shared event is unavailable";
        return false;
    }
    if (value == 0 || value == std::numeric_limits<std::uint64_t>::max()) {
        err = "Metal shared event values must be positive and below the release sentinel";
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->value_mutex);
    if (impl_->released) {
        err = "Metal shared event GPU waits have been released";
        return false;
    }
    const std::uint64_t signaled = impl_->event.signaledValue;
    impl_->pending_gpu_signals.erase(impl_->pending_gpu_signals.begin(),
                                     impl_->pending_gpu_signals.upper_bound(signaled));
    const bool would_overtake_pending_gpu_signal =
        !impl_->pending_gpu_signals.empty() && value >= impl_->pending_gpu_signals.begin()->first;
    if (value <= signaled || would_overtake_pending_gpu_signal) {
        err = "Metal shared event values must increase monotonically";
        return false;
    }
    std::atomic_thread_fence(std::memory_order_release);
    impl_->event.signaledValue = value;
    impl_->reserved = std::max(impl_->reserved, value);
    err.clear();
    return true;
}

bool Events::wait(std::uint64_t value, std::uint64_t timeout_ms, std::string& err) const {
    if (!impl_ || impl_->event == nil) {
        err = "Metal shared event is unavailable";
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->value_mutex);
        if (impl_->released) {
            err = "Metal event waits were released; event values no longer indicate completion";
            return false;
        }
        if (impl_->event.signaledValue >= value) {
            err.clear();
            return true;
        }
    }

    const char* wait_mode = std::getenv("STRATA_METAL_WAIT");
    if (wait_mode != nullptr && std::strcmp(wait_mode, "spin") == 0) {
        const auto now = std::chrono::steady_clock::now();
        const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::time_point::max() - now).count();
        if (remaining_ms < 0 || timeout_ms > static_cast<std::uint64_t>(remaining_ms)) {
            err = "Metal event timeout is too large for a spin wait";
            return false;
        }
        const auto deadline = now + std::chrono::milliseconds(
            static_cast<std::chrono::milliseconds::rep>(timeout_ms));
        while (impl_->event.signaledValue < value && std::chrono::steady_clock::now() < deadline) {
#if defined(__aarch64__)
            __builtin_arm_yield();
#else
            std::this_thread::yield();
#endif
        }
        if (impl_->event.signaledValue >= value) {
            std::lock_guard<std::mutex> lock(impl_->value_mutex);
            if (impl_->released) {
                err = "Metal event waits were released; event values no longer indicate completion";
                return false;
            }
            err.clear();
            return true;
        }
    } else if ([impl_->event waitUntilSignaledValue:value timeoutMS:timeout_ms]) {
        std::lock_guard<std::mutex> lock(impl_->value_mutex);
        if (impl_->released) {
            err = "Metal event waits were released; event values no longer indicate completion";
            return false;
        }
        err.clear();
        return true;
    }
    err = "timed out waiting for Metal event value " + std::to_string(value);
    return false;
}

bool Events::encode_wait(CommandBuffer& command_buffer, std::uint64_t value, std::string& err) const {
    std::lock_guard<std::recursive_mutex> encoding_lock(command_buffer.encoding_mutex_);
    std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
    if (!impl_ || impl_->event == nil || !command_buffer.can_encode() ||
        command_buffer.queue_handle_ != impl_->queue_identity) {
        err = "cannot encode a Metal event wait without an event and command buffer";
        return false;
    }
    if (value == 0 || value == std::numeric_limits<std::uint64_t>::max()) {
        err = "Metal event wait values must be positive and below the release sentinel";
        return false;
    }
    const std::shared_ptr<Impl> state = impl_;
    void* const owner = command_buffer.handle_;
    const std::uint64_t operation_order = command_buffer.event_operation_count_;
    {
        std::lock_guard<std::mutex> lock(state->value_mutex);
        if (state->released) {
            err = "Metal shared event GPU waits have been released";
            return false;
        }
        state->pending_gpu_waits.push_back(Impl::PendingWait{owner, value, operation_order});
    }
    if (!command_buffer.add_discard_handler([state, owner, operation_order] {
            {
                std::lock_guard<std::mutex> discard_lock(state->value_mutex);
                std::erase_if(state->pending_gpu_waits, [owner, operation_order](const Impl::PendingWait& wait) {
                    return wait.command_buffer == owner && wait.operation_order == operation_order;
                });
            }
            state->waits_changed.notify_all();
        })) {
        std::lock_guard<std::mutex> lock(state->value_mutex);
        std::erase_if(state->pending_gpu_waits, [owner, operation_order](const Impl::PendingWait& wait) {
            return wait.command_buffer == owner && wait.operation_order == operation_order;
        });
        err = "cannot encode a Metal event wait into a committed or discarded command buffer";
        return false;
    }
    CommandBuffer::SubmissionHandler submission_handler;
    submission_handler.validate = [state, value, owner, operation_order](std::string& validation_error) {
        std::lock_guard<std::mutex> validation_lock(state->value_mutex);
        if (state->released) {
            const auto pending = std::find_if(
                state->pending_gpu_waits.begin(), state->pending_gpu_waits.end(),
                [owner, operation_order](const Impl::PendingWait& wait) {
                    return wait.command_buffer == owner && wait.operation_order == operation_order;
                });
            if (pending == state->pending_gpu_waits.end() ||
                pending->submission != Impl::PendingWait::Submission::enqueued) {
                validation_error = "cannot submit a Metal event wait after GPU waits were released";
                return false;
            }
            validation_error.clear();
            return true;
        }
        const std::uint64_t observed = state->event.signaledValue;
        state->pending_gpu_signals.erase(state->pending_gpu_signals.begin(),
                                         state->pending_gpu_signals.upper_bound(observed));
        for (auto it = state->pending_gpu_signals.begin(); it != state->pending_gpu_signals.end(); ++it) {
            if (it->first > observed && it->second.command_buffer != owner &&
                it->second.submission != Impl::PendingSignal::Submission::committed) {
                validation_error = "cannot submit Metal event wait " + std::to_string(value) +
                                   " before pending signal " + std::to_string(it->first) +
                                   " is committed";
                return false;
            }
        }
        validation_error.clear();
        return true;
    };
    submission_handler.notify = [state, owner, operation_order](bool committed) {
        {
            std::lock_guard<std::mutex> notify_lock(state->value_mutex);
            for (Impl::PendingWait& wait : state->pending_gpu_waits) {
                if (wait.command_buffer == owner && wait.operation_order == operation_order) {
                    wait.submission = committed ? Impl::PendingWait::Submission::committed
                                                : Impl::PendingWait::Submission::enqueued;
                    break;
                }
            }
        }
        state->waits_changed.notify_all();
    };
    if (!command_buffer.add_submission_handler(std::move(submission_handler), err)) {
        std::lock_guard<std::mutex> lock(state->value_mutex);
        std::erase_if(state->pending_gpu_waits, [owner, operation_order](const Impl::PendingWait& wait) {
            return wait.command_buffer == owner && wait.operation_order == operation_order;
        });
        return false;
    }
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer.unsafe_native_handle();
    [cb encodeWaitForEvent:impl_->event value:value];
    ++command_buffer.event_operation_count_;
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        {
            std::lock_guard<std::mutex> completion_lock(state->value_mutex);
            std::erase_if(state->pending_gpu_waits, [owner, operation_order](const Impl::PendingWait& wait) {
                return wait.command_buffer == owner && wait.operation_order == operation_order;
            });
        }
        state->waits_changed.notify_all();
    }];
    err.clear();
    return true;
}

bool Events::encode_signal(CommandBuffer& command_buffer, std::uint64_t value, std::string& err) const {
    std::lock_guard<std::recursive_mutex> encoding_lock(command_buffer.encoding_mutex_);
    std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
    if (!impl_ || impl_->event == nil || !command_buffer.can_encode() ||
        command_buffer.queue_handle_ != impl_->queue_identity) {
        err = "cannot encode a Metal event signal without an event and command buffer";
        return false;
    }
    if (value == 0 || value == std::numeric_limits<std::uint64_t>::max()) {
        err = "Metal shared event values must increase below the release sentinel";
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->value_mutex);
    if (impl_->released) {
        err = "Metal shared event GPU waits have been released";
        return false;
    }
    if (value <= impl_->event.signaledValue) {
        err = "Metal shared event values must increase monotonically";
        return false;
    }
    if (value <= impl_->last_encoded_signal) {
        err = "encoded Metal shared event values must increase monotonically";
        return false;
    }
    const void* owner = command_buffer.handle_;
    const std::uint64_t operation_order = command_buffer.event_operation_count_;
    const std::uint64_t observed = impl_->event.signaledValue;
    for (const Impl::PendingWait& wait : impl_->pending_gpu_waits) {
        if (wait.value <= observed) continue;
        if (wait.command_buffer == owner && wait.operation_order < operation_order &&
            value <= wait.value) {
            err = "GPU event signal would follow and block an unsignaled wait in the same command buffer";
            return false;
        }
        if (wait.command_buffer != owner && wait.submission != Impl::PendingWait::Submission::encoded) {
            err = "cannot encode a GPU event signal while an earlier command buffer waits on the event";
            return false;
        }
    }
    const bool already_owns_signal = std::any_of(
        impl_->pending_gpu_signals.begin(), impl_->pending_gpu_signals.end(),
        [owner](const auto& item) { return item.second.command_buffer == owner; });
    if (already_owns_signal && impl_->pending_gpu_signals.rbegin()->second.command_buffer != owner) {
        err = "GPU signals from one command buffer must be encoded as a contiguous value range";
        return false;
    }
    const std::shared_ptr<Impl> state = impl_;
    impl_->pending_gpu_signals.emplace(
        value, Impl::PendingSignal{command_buffer.handle_, Impl::PendingSignal::Submission::encoded});
    if (!command_buffer.add_discard_handler([state, value] {
            std::lock_guard<std::mutex> discard_lock(state->value_mutex);
            state->pending_gpu_signals.erase(value);
        })) {
        impl_->pending_gpu_signals.erase(value);
        err = "cannot encode a Metal event signal into a committed or discarded command buffer";
        return false;
    }
    const void* owner_for_checks = command_buffer.handle_;
    CommandBuffer::SubmissionHandler submission_handler;
    submission_handler.validate = [state, value, owner_for_checks](std::string& validation_error) {
        std::lock_guard<std::mutex> validation_lock(state->value_mutex);
        if (state->released) {
            const auto pending = state->pending_gpu_signals.find(value);
            if (pending == state->pending_gpu_signals.end() ||
                pending->second.command_buffer != owner_for_checks ||
                pending->second.submission != Impl::PendingSignal::Submission::enqueued) {
                validation_error = "cannot submit a Metal event signal after GPU waits were released";
                return false;
            }
            validation_error.clear();
            return true;
        }
        const std::uint64_t observed = state->event.signaledValue;
        state->pending_gpu_signals.erase(state->pending_gpu_signals.begin(),
                                         state->pending_gpu_signals.upper_bound(observed));
        for (auto it = state->pending_gpu_signals.begin();
             it != state->pending_gpu_signals.lower_bound(value); ++it) {
            if (it->second.command_buffer != owner_for_checks &&
                it->second.submission != Impl::PendingSignal::Submission::committed) {
                validation_error = "cannot submit Metal event signal " + std::to_string(value) +
                                   " before lower signal " + std::to_string(it->first) +
                                   " is committed";
                return false;
            }
        }
        for (const Impl::PendingWait& wait : state->pending_gpu_waits) {
            if (wait.command_buffer != owner_for_checks && wait.value > observed &&
                wait.submission != Impl::PendingWait::Submission::encoded) {
                validation_error = "cannot submit Metal event signal " + std::to_string(value) +
                                   " while another command buffer waits for " + std::to_string(wait.value);
                return false;
            }
        }
        validation_error.clear();
        return true;
    };
    submission_handler.notify = [state, value](bool committed) {
        std::lock_guard<std::mutex> notify_lock(state->value_mutex);
        auto it = state->pending_gpu_signals.find(value);
        if (it == state->pending_gpu_signals.end()) return;
        it->second.submission = committed ? Impl::PendingSignal::Submission::committed
                                          : Impl::PendingSignal::Submission::enqueued;
    };
    if (!command_buffer.add_submission_handler(std::move(submission_handler), err)) {
        impl_->pending_gpu_signals.erase(value);
        return false;
    }
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer.unsafe_native_handle();
    ++command_buffer.event_operation_count_;
    impl_->last_encoded_signal = value;
    impl_->reserved = std::max(impl_->reserved, value);
    [cb encodeSignalEvent:impl_->event value:value];
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        std::lock_guard<std::mutex> completion_lock(state->value_mutex);
        state->pending_gpu_signals.erase(value);
    }];
    err.clear();
    return true;
}

}  // namespace strata::metal
