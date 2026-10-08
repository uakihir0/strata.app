// tests/metal/metal_events_test.mm - monotonic host/GPU SharedEvent signaling and waits.
#include "strata/metal/context.hpp"

#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

using namespace strata::metal;

static int failures = 0;

static void check(bool ok, const char* name, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    failures += !ok;
}

int main() {
    ContextOptions options;
    options.pipelines = false;
    std::string err;
    auto context = Context::open(options, err);
    check(context != nullptr, "open device context", err);
    if (!context) return 1;

    auto invalid_timeout_event = context->make_events("strata invalid release timeout test", err);
    check(invalid_timeout_event != nullptr, "create event for release-timeout validation", err);
    if (invalid_timeout_event) {
        const bool invalid_timeout_rejected = !invalid_timeout_event->release_gpu_waits(
            std::numeric_limits<std::uint64_t>::max(), err);
        const auto still_live_value = invalid_timeout_event->signaled_value();
        check(invalid_timeout_rejected && still_live_value && *still_live_value == 0 &&
                  invalid_timeout_event->reserve_value() == 1,
              "reject an overflowing release timeout before changing event state", err);
    }

    auto event = context->make_events("strata test event", err);
    check(event != nullptr, "create shared event", err);
    if (!event) return 1;
    check(!event->signal_host(std::numeric_limits<std::uint64_t>::max(), err) &&
              err.find("release sentinel") != std::string::npos,
          "reserve the terminal event value for wait release", err);
    const auto first = event->reserve_value();
    const auto second = event->reserve_value();
    check(first == 1 && second == 2, "reserve monotonically increasing values");
    check(event->signal_host(first, err), "signal from host", err);
    check(event->wait(first, 100, err), "wait for an already signaled value", err);
    check(!event->wait(second, 1, err) && err.find("timed out") != std::string::npos,
          "timeout before a value is signaled", err);
    setenv("STRATA_METAL_WAIT", "spin", 1);
    check(!event->wait(second, std::numeric_limits<std::uint64_t>::max(), err) &&
              err.find("too large") != std::string::npos,
          "reject a spin timeout that would overflow its deadline", err);
    unsetenv("STRATA_METAL_WAIT");

    auto signal = context->make_command_buffer("strata event signal test", err);
    check(signal != nullptr, "create GPU signal command buffer", err);
    if (!signal) return 1;
    id<MTLCommandBuffer> native_signal =
        (__bridge id<MTLCommandBuffer>)signal->unsafe_native_handle();
    check(!event->encode_signal(*signal, std::numeric_limits<std::uint64_t>::max(), err) &&
              err.find("release sentinel") != std::string::npos,
          "reject the terminal value as an ordinary GPU signal", err);
    const auto gpu_signal = second + 3;   // Deliberately signal a value that was never reserved.
    check(event->encode_signal(*signal, gpu_signal, err), "encode GPU signal", err);
    check(signal->enqueue(err), "enqueue command buffer after encoding", err);
    check(signal->commit(err), "commit GPU signal", err);
    check(event->wait(gpu_signal, 5000, err), "wait for GPU signal", err);
    [native_signal waitUntilCompleted];
    check(native_signal.status == MTLCommandBufferStatusCompleted, "GPU signal command completed");
    const auto after_signal = event->reserve_value();
    check(after_signal == gpu_signal + 1, "reserve after an unreserved GPU signal");

    std::atomic<int> completion_calls{0};
    std::atomic<bool> completion_succeeded{false};
    std::mutex completion_mutex;
    std::condition_variable completion_condition;
    bool completion_notified = false;
    auto completion_test = context->make_command_buffer("strata completion callback test", err);
    check(completion_test != nullptr, "create command buffer for completion callback", err);
    id<MTLCommandBuffer> native_completion_test = completion_test
        ? (__bridge id<MTLCommandBuffer>)completion_test->unsafe_native_handle()
        : nil;
    if (completion_test) {
        check(completion_test->add_completion_handler(
                  [&](bool succeeded) {
                      completion_succeeded.store(succeeded, std::memory_order_release);
                      completion_calls.fetch_add(1, std::memory_order_relaxed);
                      {
                          std::lock_guard<std::mutex> lock(completion_mutex);
                          completion_notified = true;
                      }
                      completion_condition.notify_one();
                  }, err),
              "register command-buffer completion callback", err);
        check(completion_test->commit(err), "commit completion-callback command buffer", err);
        [native_completion_test waitUntilCompleted];
        {
            std::unique_lock<std::mutex> lock(completion_mutex);
            check(completion_condition.wait_for(
                      lock, std::chrono::seconds(5),
                      [&] { return completion_notified; }),
                  "completion callback returns after native command completion");
        }
        check(completion_calls.load(std::memory_order_relaxed) == 1 &&
                  completion_succeeded.load(std::memory_order_acquire),
              "completion callback runs exactly once after successful GPU completion");
        check(!completion_test->add_completion_handler([](bool) {}, err) &&
                  err.find("cannot register") != std::string::npos,
              "reject registering a completion callback after commit", err);
    }

    std::atomic<int> discard_completion_calls{0};
    std::atomic<bool> discard_completion_succeeded{true};
    auto discard_completion_test =
        context->make_command_buffer("strata discard completion callback test", err);
    check(discard_completion_test != nullptr,
          "create command buffer for discard completion callback", err);
    if (discard_completion_test) {
        check(discard_completion_test->add_completion_handler(
                  [&](bool succeeded) {
                      discard_completion_succeeded.store(succeeded, std::memory_order_release);
                      discard_completion_calls.fetch_add(1, std::memory_order_relaxed);
                  }, err),
              "register discard completion callback", err);
        discard_completion_test->discard();
        discard_completion_test->discard();
        check(discard_completion_calls.load(std::memory_order_relaxed) == 1 &&
                  !discard_completion_succeeded.load(std::memory_order_acquire),
              "discarded NotEnqueued command buffer reports one failed completion");
    }

    const auto lower_signal = event->reserve_value();
    const auto higher_signal = event->reserve_value();
    auto lower = context->make_command_buffer("strata lower ordered signal", err);
    auto higher = context->make_command_buffer("strata higher ordered signal", err);
    check(lower != nullptr && higher != nullptr, "create command buffers for signal submission ordering", err);
    if (lower && higher) {
        check(event->encode_signal(*lower, lower_signal, err), "encode lower signal first", err);
        check(event->encode_signal(*higher, higher_signal, err), "encode higher signal second", err);
        check(!higher->enqueue(err) && err.find("lower signal") != std::string::npos,
              "reject higher signal enqueued before its lower signal", err);
        check(!higher->commit(err) && err.find("lower signal") != std::string::npos,
              "reject higher signal committed before its lower signal", err);
        check(lower->commit(err), "commit lower signal first", err);
        check(higher->commit(err), "commit higher signal after lower signal", err);
        check(event->wait(higher_signal, 5000, err), "wait for ordered higher signal", err);
    }

    const auto produced_signal = event->reserve_value();
    auto producer = context->make_command_buffer("strata event producer", err);
    auto consumer = context->make_command_buffer("strata event consumer", err);
    check(producer != nullptr && consumer != nullptr, "create command buffers for GPU wait ordering", err);
    if (producer && consumer) {
        check(event->encode_signal(*producer, produced_signal, err), "encode GPU producer signal", err);
        check(event->encode_wait(*consumer, produced_signal, err), "encode dependent GPU wait", err);
        check(!consumer->enqueue(err) && err.find("pending signal") != std::string::npos,
              "reject consumer enqueued before its GPU producer", err);
        check(!consumer->commit(err) && err.find("pending signal") != std::string::npos,
              "reject consumer committed before its GPU producer", err);
        check(producer->commit(err), "commit GPU producer before consumer", err);
        check(consumer->commit(err), "commit GPU consumer after producer", err);
        check(event->wait(produced_signal, 5000, err), "wait for producer-consumer sequence", err);
    }

    const auto self_wait_value = event->reserve_value();
    auto self_wait = context->make_command_buffer("strata self-dependent event wait", err);
    check(self_wait != nullptr, "create self-dependent event command buffer", err);
    if (self_wait) {
        check(event->encode_wait(*self_wait, self_wait_value, err), "encode event wait before self-signal", err);
        check(!event->encode_signal(*self_wait, self_wait_value, err) && err.find("same command buffer") != std::string::npos,
              "reject a self-dependent GPU wait and signal", err);
    }

    const auto late_wait_value = event->reserve_value();
    const auto late_producer_value = event->reserve_value();
    auto queued_waiter = context->make_command_buffer("strata queued event waiter", err);
    auto late_producer = context->make_command_buffer("strata late event producer", err);
    check(queued_waiter != nullptr && late_producer != nullptr,
          "create command buffers for late GPU producer check", err);
    if (queued_waiter && late_producer) {
        check(event->encode_wait(*queued_waiter, late_wait_value, err), "encode wait before producer signal exists", err);
        check(queued_waiter->enqueue(err), "enqueue wait before producer signal exists", err);
        check(!event->encode_signal(*late_producer, late_producer_value, err) &&
                  err.find("earlier command buffer waits") != std::string::npos,
              "reject a producer encoded after its dependent wait was enqueued", err);
        check(event->signal_host(late_wait_value, err), "host release remains available for queued wait", err);
        check(queued_waiter->commit(err), "commit host-released GPU wait", err);
        check(event->wait(late_wait_value, 5000, err), "wait for host-released GPU wait", err);
    }

    const auto discarded_signal = event->reserve_value();
    auto abandoned = context->make_command_buffer("strata discarded signal test", err);
    check(abandoned != nullptr, "create command buffer that will be discarded", err);
    if (abandoned) {
        check(event->encode_signal(*abandoned, discarded_signal, err),
              "encode signal into command buffer that will be discarded", err);
        abandoned.reset();
        check(event->signal_host(discarded_signal, err),
              "discard removes its pending GPU signal reservation", err);
    }
    const auto enqueued_discard_signal = event->reserve_value();
    auto enqueued_discard = context->make_command_buffer("strata enqueued discard test", err);
    check(enqueued_discard != nullptr, "create enqueued command buffer that will be discarded", err);
    if (enqueued_discard) {
        check(event->encode_signal(*enqueued_discard, enqueued_discard_signal, err),
              "encode signal into enqueued command buffer", err);
        check(enqueued_discard->enqueue(err), "enqueue command buffer for discard test", err);
        enqueued_discard.reset();
        check(event->wait(enqueued_discard_signal, 5000, err),
              "destruction commits an enqueued command buffer", err);
    }

    const auto arrived = event->reserve_value();
    const auto wait_for = event->reserve_value();
    const auto release_signal = event->reserve_value();
    auto wait = context->make_command_buffer("strata event wait test", err);
    check(wait != nullptr, "create GPU wait command buffer", err);
    if (!wait) return 1;
    id<MTLCommandBuffer> native_wait =
        (__bridge id<MTLCommandBuffer>)wait->unsafe_native_handle();
    check(!event->encode_wait(*wait, std::numeric_limits<std::uint64_t>::max(), err) &&
              err.find("release sentinel") != std::string::npos,
          "reject the terminal value as an ordinary GPU wait", err);
    check(event->encode_signal(*wait, arrived, err),
          "encode GPU arrival before the wait", err);
    check(event->encode_wait(*wait, wait_for, err), "encode GPU wait", err);
    check(event->encode_signal(*wait, release_signal, err),
          "encode a signal after the GPU wait", err);
    const auto overtaking_signal = event->reserve_value();
    check(wait->commit(err), "commit GPU wait command", err);
    check(event->wait(arrived, 5000, err), "GPU reached the event wait", err);
    check(!event->wait(release_signal, 100, err) && err.find("timed out") != std::string::npos,
          "GPU wait blocks its following signal while the event is unsignaled", err);
    const auto before_release_value = event->signaled_value();
    check(before_release_value && *before_release_value < release_signal,
          "following GPU signal remains pending before host release");
    check(!event->signal_host(overtaking_signal, err) && err.find("monotonically") != std::string::npos,
          "host cannot overtake a pending GPU signal", err);
    check(event->signal_host(wait_for, err), "release GPU wait from host", err);
    check(event->wait(release_signal, 5000, err), "GPU wait releases its following signal", err);
    [native_wait waitUntilCompleted];
    check(native_wait.status == MTLCommandBufferStatusCompleted, "GPU wait command completed");
    const auto completed_wait_value = event->signaled_value();
    check(completed_wait_value && *completed_wait_value >= wait_for,
          "event value remains monotonic");
    check(!event->signal_host(first, err) && err.find("monotonically") != std::string::npos,
          "reject a backwards host signal", err);

    ContextOptions foreign_options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, err);
    check(foreign_context != nullptr, "open a second command queue for event ownership", err);
    if (foreign_context) {
        auto foreign_command = foreign_context->make_command_buffer("strata foreign event queue", err);
        check(foreign_command != nullptr, "create command buffer on the second queue", err);
        if (foreign_command) {
            const auto foreign_value = event->reserve_value();
            check(!event->encode_wait(*foreign_command, foreign_value, err) &&
                      err.find("event wait") != std::string::npos,
                  "reject waiting on an event from another queue", err);
            check(!event->encode_signal(*foreign_command, foreign_value, err) &&
                      err.find("event signal") != std::string::npos,
                  "reject signaling an event from another queue", err);
        }
    }
    foreign_context.reset();

    ContextOptions drain_options;
    drain_options.pipelines = false;
    std::string drain_error;
    auto drain_context = Context::open(drain_options, drain_error);
    check(drain_context != nullptr, "open context for teardown drain", drain_error);
    if (drain_context) {
        auto drain_event = drain_context->make_events("strata teardown wait", drain_error);
        check(drain_event != nullptr, "create teardown event", drain_error);
        if (drain_event) {
            const std::uint64_t arrived_value = drain_event->reserve_value();
            const std::uint64_t drain_value = drain_event->reserve_value();
            const std::uint64_t release_value = drain_event->reserve_value();
            auto pending = drain_context->make_command_buffer("strata teardown pending", drain_error);
            check(pending != nullptr, "create pending teardown command", drain_error);
            if (pending) {
                id<MTLCommandBuffer> pending_native =
                    (__bridge id<MTLCommandBuffer>)pending->unsafe_native_handle();
                check(drain_event->encode_signal(*pending, arrived_value, drain_error),
                      "encode teardown wait arrival", drain_error);
                check(drain_event->encode_wait(*pending, drain_value, drain_error),
                      "encode a pending teardown wait", drain_error);
                check(drain_event->encode_signal(*pending, release_value, drain_error),
                      "encode teardown signal after wait", drain_error);
                check(pending->commit(drain_error), "commit pending teardown command", drain_error);
                check(drain_event->wait(arrived_value, 5000, drain_error),
                      "GPU reached the teardown wait", drain_error);
                check(!drain_event->wait(release_value, 100, drain_error) &&
                          drain_error.find("timed out") != std::string::npos,
                      "teardown command remains blocked until the host releases it", drain_error);
                std::atomic<bool> host_signal_ok{false};
                std::string host_signal_error;
                std::thread signaler([&] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    host_signal_ok.store(drain_event->signal_host(drain_value, host_signal_error),
                                         std::memory_order_release);
                });
                drain_context.reset();
                check(pending_native.status == MTLCommandBufferStatusCompleted,
                      "Context teardown waits for committed GPU work");
                signaler.join();
                check(host_signal_ok.load(std::memory_order_acquire),
                      "release the pending teardown wait", host_signal_error);
            } else {
                drain_context.reset();
            }
        } else {
            drain_context.reset();
        }
    }

    auto release_event = context->make_events("strata released wait test", err);
    check(release_event != nullptr, "create one-shot released event", err);
    if (release_event) {
        const std::uint64_t arrived_value = release_event->reserve_value();
        const std::uint64_t blocked_value = release_event->reserve_value();
        const std::uint64_t trailing_signal = release_event->reserve_value();
        auto blocked = context->make_command_buffer("strata release blocked wait", err);
        check(blocked != nullptr, "create command buffer blocked on release event", err);
        if (blocked) {
            id<MTLCommandBuffer> native_blocked =
                (__bridge id<MTLCommandBuffer>)blocked->unsafe_native_handle();
            check(release_event->encode_signal(*blocked, arrived_value, err),
                  "encode arrival before the blocked event wait", err);
            check(release_event->encode_wait(*blocked, blocked_value, err),
                  "encode a GPU wait for the release test", err);
            check(release_event->encode_signal(*blocked, trailing_signal, err),
                  "encode a signal after the blocked event wait", err);

            const std::uint64_t staged_wait_value = release_event->reserve_value();
            auto staged = context->make_command_buffer("strata staged released wait", err);
            check(staged != nullptr, "create an unsubmitted wait before release", err);
            if (staged) {
                check(release_event->encode_wait(*staged, staged_wait_value, err),
                      "encode an unsubmitted event wait", err);
            }

            check(blocked->commit(err), "submit command buffer with a blocked event wait", err);
            check(release_event->wait(arrived_value, 5000, err),
                  "GPU reaches the blocked event wait", err);
            check(!release_event->wait(trailing_signal, 50, err) &&
                      err.find("timed out") != std::string::npos,
                  "trailing GPU signal stays blocked before release", err);
            check(release_event->release_gpu_waits(5000, err),
                  "release event waits and drain committed command buffers", err);
            [native_blocked waitUntilCompleted];
            check(native_blocked.status == MTLCommandBufferStatusCompleted,
                  "released GPU wait allows the submitted command buffer to complete");
            check(!release_event->signaled_value(),
                  "released event stops exposing a completion value");
            check(release_event->reserve_value() == 0,
                  "released event cannot reserve values for another request");
            check(!release_event->wait(trailing_signal, 100, err) &&
                      err.find("released") != std::string::npos,
                  "released event values cannot be used as completion evidence", err);
            check(!release_event->signal_host(trailing_signal + 1, err) &&
                      err.find("released") != std::string::npos,
                  "released event rejects later host signals", err);

            auto after_release = context->make_command_buffer(
                "strata post-release event operation", err);
            check(after_release != nullptr, "create a post-release command buffer", err);
            if (after_release) {
                check(!release_event->encode_wait(*after_release, 1, err) &&
                          err.find("released") != std::string::npos,
                      "released event rejects new GPU waits", err);
                check(!release_event->encode_signal(*after_release, 1, err) &&
                          err.find("released") != std::string::npos,
                      "released event rejects new GPU signals", err);
                after_release->discard();
            }
            if (staged) {
                check(!staged->commit(err) && err.find("released") != std::string::npos,
                      "released event rejects a previously encoded but unsubmitted wait", err);
                staged->discard();
            }
        }
    }

    auto enqueued_event = context->make_events("strata enqueued released wait test", err);
    auto completion_gate = context->make_events("strata release drain completion gate", err);
    check(enqueued_event != nullptr, "create event for an enqueued released wait", err);
    check(completion_gate != nullptr, "create a second gate for release-drain testing", err);
    if (enqueued_event && completion_gate) {
        const std::uint64_t completion_arrival = completion_gate->reserve_value();
        const std::uint64_t completion_blocked = completion_gate->reserve_value();
        const std::uint64_t completion_done = completion_gate->reserve_value();
        const std::uint64_t arrived_value = enqueued_event->reserve_value();
        const std::uint64_t blocked_value = enqueued_event->reserve_value();
        const std::uint64_t trailing_signal = enqueued_event->reserve_value();
        auto enqueued = context->make_command_buffer("strata enqueued released wait", err);
        check(enqueued != nullptr, "create an enqueued command buffer with an event wait", err);
        if (enqueued) {
            id<MTLCommandBuffer> native_enqueued =
                (__bridge id<MTLCommandBuffer>)enqueued->unsafe_native_handle();
            check(enqueued_event->encode_signal(*enqueued, arrived_value, err),
                  "encode arrival for an enqueued command buffer", err);
            check(enqueued_event->encode_wait(*enqueued, blocked_value, err),
                  "encode the enqueued command buffer wait", err);
            check(enqueued_event->encode_signal(*enqueued, trailing_signal, err),
                  "encode the enqueued command buffer trailing signal", err);
            check(completion_gate->encode_signal(*enqueued, completion_arrival, err),
                  "encode arrival at the independent completion gate", err);
            check(completion_gate->encode_wait(*enqueued, completion_blocked, err),
                  "hold the command buffer at the independent completion gate", err);
            check(completion_gate->encode_signal(*enqueued, completion_done, err),
                  "encode a completion signal after the independent gate", err);
            check(enqueued->enqueue(err), "enqueue without committing the blocked command buffer", err);

            std::mutex commit_gate_mutex;
            std::condition_variable commit_gate_condition;
            bool allow_late_commit = false;
            std::atomic<bool> late_commit_succeeded{false};
            std::string late_commit_error;
            std::thread late_committer([&] {
                {
                    std::unique_lock<std::mutex> gate_lock(commit_gate_mutex);
                    commit_gate_condition.wait(gate_lock, [&] { return allow_late_commit; });
                }
                late_commit_succeeded.store(enqueued->commit(late_commit_error),
                                            std::memory_order_release);
            });
            const auto release_started = std::chrono::steady_clock::now();
            const bool initial_release_succeeded =
                enqueued_event->release_gpu_waits(100, err);
            const auto release_elapsed = std::chrono::steady_clock::now() - release_started;
            const bool release_timed_out =
                !initial_release_succeeded &&
                err.find("timed out draining Metal command buffers") != std::string::npos;
            check(release_timed_out && release_elapsed >= std::chrono::milliseconds(80),
                  "release waits for its bounded drain timeout while commit is gated", err);
            check(!enqueued_event->signaled_value(),
                  "timed-out release still poisons the event and releases GPU waits");
            {
                std::lock_guard<std::mutex> gate_lock(commit_gate_mutex);
                allow_late_commit = true;
            }
            commit_gate_condition.notify_all();
            late_committer.join();
            check(late_commit_succeeded.load(std::memory_order_acquire),
                  "allow a previously enqueued command buffer to commit after release",
                  late_commit_error);
            check(completion_gate->wait(completion_arrival, 5000, err),
                  "late-committed command reaches its independent completion gate", err);
            const auto held_drain_started = std::chrono::steady_clock::now();
            const bool held_drain_succeeded =
                enqueued_event->release_gpu_waits(100, err);
            const auto held_drain_elapsed =
                std::chrono::steady_clock::now() - held_drain_started;
            check(!held_drain_succeeded &&
                      err.find("timed out draining Metal command buffers") != std::string::npos &&
                      held_drain_elapsed >= std::chrono::milliseconds(80),
                  "release drain waits while the command buffer is held after its event wait", err);
            check(completion_gate->signal_host(completion_blocked, err),
                  "release the independent command-buffer completion gate", err);
            std::string drain_retry_error;
            const bool drain_retry_succeeded = enqueued_event->release_gpu_waits(
                5000, drain_retry_error);
            check(drain_retry_succeeded,
                  "a later release call drains the completed command buffer",
                  drain_retry_error);
            check(completion_gate->wait(completion_done, 5000, err),
                  "command buffer signals after the independent gate is released", err);
            check(native_enqueued.status == MTLCommandBufferStatusCompleted,
                  "released enqueued wait is complete when the drain returns");
            check(!enqueued_event->signaled_value(),
                  "released enqueued event exposes no completion value");
        }
    }

    std::printf("metal_events_test: %s\n", failures ? "FAILED" : "OK");
    return failures != 0;
}
