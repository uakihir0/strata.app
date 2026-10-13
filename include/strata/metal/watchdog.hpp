// include/strata/metal/watchdog.hpp - progress beats, the serve watchdog and its stall report (docs/mac/06, section 14).
#pragma once

#include "strata/platform/memory_probe.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace strata::metal {

inline constexpr std::uint64_t kWatchdogDefaultSeconds = 60;
inline constexpr std::chrono::milliseconds kStallPassSpacing {2000};
inline constexpr std::uint64_t kGpuWaitReleaseTimeoutMs = 2000;
inline constexpr std::string_view kCrashReportHint =
    "strata serve: a crash report with every thread will be in ~/Library/Logs/DiagnosticReports/";

/// STRATA_WATCHDOG_S: nullptr or an empty string gives the default of 60 s, "0" turns the watchdog off, and any other
/// value must be a decimal integer that fits in 64 bits.
bool parse_watchdog_limit(const char* text, std::uint64_t& seconds, std::string& error);

struct ProgressState {
    bool busy = false;
    std::uint64_t requests = 0;  // Requests begun, so a new request is seen even when no beat happened in between.
    std::uint64_t beats = 0;     // Advances per token, prompt chunk or window.
    std::uint64_t ticks = 0;     // Advances per layer served.
    std::uint64_t ticks_at_beat = 0;
    std::string where;
    std::string detail;
};

/// The engine's progress record. The request thread updates it and the watchdog thread reads snapshots.
class Progress {
public:
    void begin_request();
    void end_request();
    void beat();
    void tick();
    void set_stage(std::string_view where, std::string_view detail = {});
    ProgressState snapshot() const;

private:
    mutable std::mutex mutex_;
    ProgressState state_;
};

/// Brackets one request: busy from construction to destruction.
class BusyScope {
public:
    explicit BusyScope(Progress& progress) : progress_(progress) { progress_.begin_request(); }
    ~BusyScope() { progress_.end_request(); }
    BusyScope(const BusyScope&) = delete;
    BusyScope& operator=(const BusyScope&) = delete;

private:
    Progress& progress_;
};

/// The engine's part of a stall report. The watchdog writes the memory line itself.
class StallReporter {
public:
    virtual ~StallReporter() = default;
    /// The engine's own diagnostics for one pass (Window::diag(), or the MissResolver state when tiered).
    virtual void write_engine_pass(std::FILE* out) = 0;
    /// The Mac `gpu:` line for one pass, without a newline.
    virtual std::string gpu_line() = 0;
    /// Releases the GPU waits that section 14.2 allows. Returns false when some wait must stay in place.
    virtual bool release_gpu_waits(std::uint64_t timeout_ms, std::FILE* out) = 0;
};

/// The reporter used when no engine is attached: it has no engine lines and no GPU waits to release.
class NoEngineStallReporter final : public StallReporter {
public:
    void write_engine_pass(std::FILE*) override {}
    std::string gpu_line() override;
    bool release_gpu_waits(std::uint64_t, std::FILE*) override { return true; }
};

std::string format_no_progress_line(std::uint64_t limit_seconds, std::string_view stage);
std::string format_stall_header(std::string_view engine, std::string_view stage, std::uint64_t stalled_seconds,
                                std::uint64_t layers_since_step);
std::string format_memory_line(const std::optional<platform::ProcessMemory>& process,
                               std::optional<std::uint64_t> available_bytes);
std::string format_gpu_line(std::string_view label, std::string_view status, std::uint64_t budget_mib,
                            std::uint64_t allocated_mib);

/// Stall detection on a steady clock. `poll` makes one check, and `start` runs `poll` once per second on a thread.
/// The default hooks use the system clock, sleep and std::abort; tests substitute their own.
class Watchdog {
public:
    struct Hooks {
        std::function<std::chrono::steady_clock::time_point()> now;
        std::function<void(std::chrono::milliseconds)> sleep;
        std::function<void()> abort;
    };
    static Hooks system_hooks();

    Watchdog(Progress& progress, std::uint64_t limit_seconds, std::string engine, StallReporter& reporter,
             std::FILE* diagnostics, Hooks hooks = system_hooks());
    ~Watchdog();
    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    void start();
    void stop();
    /// One check at `now`. Fires when the request is busy and neither the request count nor the beat count has changed
    /// for the limit. Returns true when it fired.
    bool poll(std::chrono::steady_clock::time_point now);
    bool fired() const { return fired_.load(); }

private:
    void run();
    void write_pass();
    void report_and_abort(const ProgressState& state, std::uint64_t stalled_seconds);

    Progress& progress_;
    std::uint64_t limit_seconds_;
    std::string engine_;
    StallReporter& reporter_;
    std::FILE* diagnostics_;
    Hooks hooks_;

    bool have_baseline_ = false;
    std::uint64_t baseline_requests_ = 0;
    std::uint64_t baseline_beats_ = 0;
    std::chrono::steady_clock::time_point baseline_time_ {};
    std::atomic<bool> fired_ {false};

    std::mutex thread_mutex_;
    std::condition_variable wake_;
    bool stop_requested_ = false;
    std::thread thread_;
};

}  // namespace strata::metal
