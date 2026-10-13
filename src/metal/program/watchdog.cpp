// src/metal/program/watchdog.cpp - serve progress, the watchdog thread and the stall report (docs/mac/06, section 14).
#include "strata/metal/watchdog.hpp"

#include <charconv>
#include <cstdlib>
#include <system_error>

namespace strata::metal {
namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

std::mutex& report_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string stage_text(const ProgressState& state) {
    return state.detail.empty() ? state.where : state.where + " " + state.detail;
}

}  // namespace

bool parse_watchdog_limit(const char* text, std::uint64_t& seconds, std::string& error) {
    if (text == nullptr || *text == '\0') {
        seconds = kWatchdogDefaultSeconds;
        return true;
    }
    const std::string_view view(text);
    std::uint64_t value = 0;
    const auto result = std::from_chars(view.data(), view.data() + view.size(), value);
    if (result.ec != std::errc {} || result.ptr != view.data() + view.size()) {
        error = "STRATA_WATCHDOG_S needs a decimal number of seconds (0 turns the watchdog off)";
        return false;
    }
    seconds = value;
    return true;
}

void Progress::begin_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.busy = true;
    ++state_.requests;
    state_.where = "request";
    state_.detail.clear();
}

void Progress::end_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.busy = false;
    state_.where.clear();
    state_.detail.clear();
}

void write_report_line(std::FILE* out, std::string_view text) {
    if (out == nullptr) return;
    std::string line;
    line.reserve(text.size() + 1);
    line.append(text);
    line.push_back('\n');
    const std::lock_guard<std::mutex> lock(report_mutex());
    std::fwrite(line.data(), 1, line.size(), out);
    std::fflush(out);
}

void Progress::beat() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++state_.beats;
    state_.ticks_at_beat = state_.ticks;
}

void Progress::beat(std::string_view where, std::string_view detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++state_.beats;
    state_.ticks_at_beat = state_.ticks;
    state_.where.assign(where);
    state_.detail.assign(detail);
}

void Progress::tick() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++state_.ticks;
}

void Progress::set_stage(std::string_view where, std::string_view detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.where.assign(where);
    state_.detail.assign(detail);
}

ProgressState Progress::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

std::string NoEngineStallReporter::gpu_line() {
    return "gpu: unavailable (no engine is attached to this serve loop)";
}

std::string format_no_progress_line(std::uint64_t limit_seconds, std::string_view stage) {
    return "strata serve: no progress for " + std::to_string(limit_seconds) + " s during a request (" +
           std::string(stage) + ") - stopping the engine so the server starts it again (issue #29)";
}

std::string format_stall_header(std::string_view engine, std::string_view stage, std::uint64_t stalled_seconds,
                                std::uint64_t layers_since_step) {
    return "strata serve: stall report (engine " + std::string(engine) + "): stage \"" + std::string(stage) +
           "\" for " + std::to_string(stalled_seconds) + " s; " + std::to_string(layers_since_step) +
           " layers served since the last finished step (0 = stopped, more = slow)";
}

std::string format_memory_line(const std::optional<platform::ProcessMemory>& process,
                               std::optional<std::uint64_t> available_bytes) {
    if (!process || !available_bytes) return "memory: unavailable (the memory probe failed)";
    return "memory: " + std::to_string(process->resident / kMiB) + " MiB resident, " +
           std::to_string(process->swap_used / kMiB) + " MiB in swap, " +
           std::to_string(*available_bytes / kMiB) + " MiB RAM available; " +
           std::to_string(process->major_faults) + " major page faults so far";
}

std::string format_gpu_line(std::string_view label, std::string_view status, std::uint64_t budget_mib,
                            std::uint64_t allocated_mib) {
    return "gpu: last command buffer \"" + std::string(label) + "\" status " + std::string(status) + ", " +
           std::to_string(budget_mib) + " MiB budget, " + std::to_string(allocated_mib) + " MiB allocated";
}

Watchdog::Hooks Watchdog::system_hooks() {
    return Hooks {
        [] { return std::chrono::steady_clock::now(); },
        [](std::chrono::milliseconds duration) { std::this_thread::sleep_for(duration); },
        [] { std::abort(); },
    };
}

Watchdog::Watchdog(Progress& progress, std::uint64_t limit_seconds, std::string engine, StallReporter& reporter,
                   std::FILE* diagnostics, Hooks hooks)
    : progress_(progress),
      limit_seconds_(limit_seconds),
      engine_(std::move(engine)),
      reporter_(reporter),
      diagnostics_(diagnostics),
      hooks_(std::move(hooks)) {}

Watchdog::~Watchdog() { stop(); }

bool Watchdog::start() {
    if (limit_seconds_ == 0 || thread_.joinable()) return true;
    try {
        thread_ = std::thread([this] { run(); });
    } catch (const std::system_error&) {
        return false;
    }
    return true;
}

void Watchdog::stop() {
    {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        stop_requested_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void Watchdog::run() {
    std::unique_lock<std::mutex> lock(thread_mutex_);
    while (!stop_requested_) {
        if (wake_.wait_for(lock, std::chrono::seconds(1), [this] { return stop_requested_; })) break;
        lock.unlock();
        poll(hooks_.now());
        lock.lock();
    }
}

bool Watchdog::poll(std::chrono::steady_clock::time_point now) {
    if (limit_seconds_ == 0 || fired_.load()) return false;
    const ProgressState state = progress_.snapshot();
    if (!state.busy) {
        have_baseline_ = false;
        return false;
    }
    if (!have_baseline_ || state.requests != baseline_requests_ || state.beats != baseline_beats_) {
        have_baseline_ = true;
        baseline_requests_ = state.requests;
        baseline_beats_ = state.beats;
        baseline_time_ = now;
        return false;
    }
    // Compare whole seconds, as unsigned. Converting the limit to a duration of nanoseconds overflows at about 9.2e9 s,
    // so a huge limit that should never fire would otherwise fire about one second into the request.
    const auto stalled = std::chrono::duration_cast<std::chrono::seconds>(now - baseline_time_).count();
    if (stalled < 0 || static_cast<std::uint64_t>(stalled) < limit_seconds_) return false;
    fired_.store(true);
    report_and_abort(state, static_cast<std::uint64_t>(stalled));
    return true;
}

void Watchdog::write_pass() {
    reporter_.write_engine_pass(diagnostics_);
    write_report_line(diagnostics_, format_memory_line(platform::process_memory(), platform::available_memory()));
    write_report_line(diagnostics_, reporter_.gpu_line());
}

void Watchdog::report_and_abort(const ProgressState& state, std::uint64_t stalled_seconds) {
    const std::string stage = stage_text(state);
    write_report_line(diagnostics_, format_no_progress_line(limit_seconds_, stage));
    write_report_line(diagnostics_,
                      format_stall_header(engine_, stage, stalled_seconds, state.ticks - state.ticks_at_beat));
    write_pass();
    hooks_.sleep(kStallPassSpacing);
    write_pass();
    write_report_line(diagnostics_, kCrashReportHint);
    reporter_.release_gpu_waits(kGpuWaitReleaseTimeoutMs, diagnostics_);
    std::fflush(diagnostics_);
    hooks_.abort();
}

}  // namespace strata::metal
