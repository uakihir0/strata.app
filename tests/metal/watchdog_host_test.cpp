// tests/metal/watchdog_host_test.cpp - the serve watchdog and its stall report (docs/mac/06, section 14), host side.
//
// The fake-clock cases call Watchdog::poll (through WatchdogTestAccess) with explicit times and check when it fires,
// what it prints and in what order it aborts. The process cases run the real serve loop in a child process with
// STRATA_WATCHDOG_S=1 and a stub backend that stalls, keeps beating, or sits idle, then check the exit status and the
// stderr lines. A stall in a child ends through ServeLoopOptions::watchdog_abort (exit 86), so the run writes no crash
// report; the real std::abort is metal_watchdog_test's case.
#include "strata/metal/serve.hpp"
#include "strata/metal/watchdog.hpp"
#include "strata/platform/memory_probe.hpp"

#include <sys/stat.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

using namespace std::chrono_literals;
using namespace strata::metal;
namespace platform = strata::platform;

namespace strata::metal {
// The test's hook into Watchdog::poll, which is private (watchdog.hpp, friend class).
class WatchdogTestAccess {
public:
    static bool poll(Watchdog& watchdog, std::chrono::steady_clock::time_point now) { return watchdog.poll(now); }
};
}  // namespace strata::metal

namespace {

int failures = 0;

void check(bool condition, const std::string& name, const std::string& detail = {}) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s%s%s\n", name.c_str(), detail.empty() ? "" : " - ", detail.c_str());
}

constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr const char* kGpuLine = "gpu: last command buffer \"tf-test\" status completed, 1024 MiB budget, 512 MiB allocated";
constexpr const char* kIssueLine =
    "strata serve: no progress for 5 s during a request (decode token 3) - stopping the engine so the server starts "
    "it again (issue #29)";
constexpr const char* kHeaderLine =
    "strata serve: stall report (engine strata-test): stage \"decode token 3\" for 5 s; 3 layers served since the last "
    "finished step (0 = stopped, more = slow)";
constexpr const char* kHintLine =
    "strata serve: a crash report with every thread will be in ~/Library/Logs/DiagnosticReports/";
// The child serve loops end a stall through the watchdog_abort seam: one line on stderr, then _exit(86). No SIGABRT, so
// no crash report is written (docs/mac/06, section 14.3; metal_watchdog_test keeps the real abort).
constexpr int kSeamExitCode = 86;
constexpr const char* kSeamLine = "watchdog_host_test: watchdog abort seam reached";
// A child that hangs ends by SIGALRM, so the test fails instead of waiting forever.
constexpr unsigned kChildAlarmSeconds = 20;

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Everything the fake hooks record, in call order.
struct FakeWorld {
    TimePoint now = TimePoint {} + std::chrono::seconds(1000);
    std::vector<std::string> events;
    std::vector<std::int64_t> sleeps_ms;
    int aborts = 0;
    std::string output_at_abort;
    std::FILE* sink = nullptr;
    char* buffer = nullptr;
    std::size_t buffer_size = 0;

    FakeWorld() { sink = open_memstream(&buffer, &buffer_size); }
    ~FakeWorld() {
        if (sink != nullptr) std::fclose(sink);
        std::free(buffer);
    }
    std::string output() {
        std::fflush(sink);
        return std::string(buffer, buffer_size);
    }
};

class FakeStall final : public StallReporter {
public:
    FakeStall(FakeWorld& world, bool release_ok) : world_(world), release_ok_(release_ok) {}
    void write_engine_pass(std::FILE* out) override { std::fprintf(out, "engine pass %d\n", ++passes_); }
    std::string gpu_line() override { return kGpuLine; }
    bool release_gpu_waits(std::uint64_t timeout_ms, std::FILE*) override {
        world_.events.push_back("release " + std::to_string(timeout_ms));
        return release_ok_;
    }

private:
    FakeWorld& world_;
    bool release_ok_;
    int passes_ = 0;
};

Watchdog::Hooks fake_hooks(FakeWorld& world) {
    return Watchdog::Hooks {
        [&world] { return world.now; },
        [&world](std::chrono::milliseconds duration) {
            world.sleeps_ms.push_back(duration.count());
            world.events.push_back("sleep " + std::to_string(duration.count()));
            world.now += duration;
        },
        [&world] {
            ++world.aborts;
            world.events.push_back("abort");
            world.output_at_abort = world.output();
        },
    };
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::string current;
    for (const char c : text) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) lines.push_back(current);
    return lines;
}

bool starts_with(const std::string& text, const std::string& prefix) { return text.rfind(prefix, 0) == 0; }

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// "memory: R MiB resident, S MiB in swap, A MiB RAM available; F major page faults so far", with R > 0 for this process.
bool is_real_memory_line(const std::string& line) {
    if (!starts_with(line, "memory: ")) return false;
    if (line.find(" MiB resident, ") == std::string::npos || line.find(" MiB in swap, ") == std::string::npos ||
        line.find(" MiB RAM available; ") == std::string::npos ||
        !ends_with(line, " major page faults so far")) {
        return false;
    }
    return std::stoull(line.substr(8)) > 0;
}

// Fires at the first poll that is `limit` seconds after the first poll that saw the request busy with no beats.
void test_fake_clock_fires_at_the_limit() {
    FakeWorld world;
    Progress progress;
    FakeStall stall(world, true);
    Watchdog watchdog(progress, 5, "strata-test", stall, world.sink, fake_hooks(world));

    // One layer before the last beat and three after it, so the header's "layers since the last step" is the ticks after
    // the beat (3), not the total (4) and not zero. A mutant that reports the total is then caught.
    progress.begin_request();
    progress.tick();
    progress.beat("decode", "token 3");
    for (int tick = 0; tick < 3; ++tick) progress.tick();

    const TimePoint start = world.now;
    int fired_at = -1;
    for (int second = 0; second <= 9; ++second) {
        if (WatchdogTestAccess::poll(watchdog, start + std::chrono::seconds(second))) {
            fired_at = second;
            break;
        }
    }
    check(fired_at == 5, "a stall fires when the limit has passed, not before", std::to_string(fired_at));
    check(watchdog.fired(), "fired() reports the firing");
    check(world.aborts == 1, "the watchdog aborts once", std::to_string(world.aborts));
    check(!WatchdogTestAccess::poll(watchdog, start + std::chrono::seconds(60)), "a fired watchdog does not fire again");
    check(world.aborts == 1, "no second abort after the first");

    const std::vector<std::string> lines = split_lines(world.output_at_abort);
    check(lines.size() == 9, "the stall report has nine lines before the abort", std::to_string(lines.size()));
    if (lines.size() >= 9) {
        check(lines[0] == kIssueLine, "issue #29 line is verbatim", lines[0]);
        check(lines[1] == kHeaderLine, "stall report header", lines[1]);
        check(lines[2] == "engine pass 1", "first pass starts with the engine diagnostics", lines[2]);
        check(is_real_memory_line(lines[3]), "first pass memory line uses real probe values", lines[3]);
        check(lines[4] == kGpuLine, "first pass gpu line", lines[4]);
        check(lines[5] == "engine pass 2", "second pass starts with the engine diagnostics", lines[5]);
        check(is_real_memory_line(lines[6]), "second pass memory line", lines[6]);
        check(lines[7] == kGpuLine, "second pass gpu line", lines[7]);
        check(lines[8] == kHintLine, "crash report hint comes last, before the abort", lines[8]);
    }
    const std::vector<std::string> expected_events {"sleep 2000", "release 2000", "abort"};
    check(world.events == expected_events, "two passes 2 s apart, then GPU-wait release, then abort");
    check(world.sleeps_ms == std::vector<std::int64_t> {2000}, "exactly one 2 s spacing between the passes");
}

// A stall with the watchdog's full limit of GPU-wait release still aborts when the release refuses.
void test_refused_release_still_aborts() {
    FakeWorld world;
    Progress progress;
    FakeStall stall(world, false);
    Watchdog watchdog(progress, 3, "strata-test", stall, world.sink, fake_hooks(world));
    progress.begin_request();
    progress.beat();
    const TimePoint start = world.now;
    check(!WatchdogTestAccess::poll(watchdog, start), "first poll only sets the baseline");
    check(!WatchdogTestAccess::poll(watchdog, start + 2s), "no stall two seconds in");
    check(WatchdogTestAccess::poll(watchdog, start + 3s), "stall fires at the limit");
    check(world.aborts == 1, "refused release still aborts");
}

// A beat in every poll interval keeps the request alive however long it runs.
void test_beats_keep_the_watchdog_quiet() {
    FakeWorld world;
    Progress progress;
    FakeStall stall(world, true);
    Watchdog watchdog(progress, 5, "strata-test", stall, world.sink, fake_hooks(world));
    progress.begin_request();
    const TimePoint start = world.now;
    for (int second = 0; second < 40; ++second) {
        progress.beat();
        if (WatchdogTestAccess::poll(watchdog, start + std::chrono::seconds(second))) break;
    }
    check(!watchdog.fired() && world.aborts == 0, "beats in every interval never fire");
}

void test_idle_never_fires() {
    FakeWorld world;
    Progress progress;
    FakeStall stall(world, true);
    Watchdog watchdog(progress, 5, "strata-test", stall, world.sink, fake_hooks(world));
    const TimePoint start = world.now;
    bool any = false;
    for (int second = 0; second < 100; ++second) any = any || WatchdogTestAccess::poll(watchdog, start + std::chrono::seconds(second));
    check(!any && world.aborts == 0 && world.sleeps_ms.empty(), "an idle engine never fires, however long it waits");
}

void test_zero_limit_is_off() {
    FakeWorld world;
    Progress progress;
    FakeStall stall(world, true);
    Watchdog watchdog(progress, 0, "strata-test", stall, world.sink, fake_hooks(world));
    progress.begin_request();
    const TimePoint start = world.now;
    bool any = false;
    any = any || WatchdogTestAccess::poll(watchdog, start);
    any = any || WatchdogTestAccess::poll(watchdog, start + std::chrono::seconds(10000));
    const bool started = watchdog.start();
    check(!any && !watchdog.fired() && world.aborts == 0 && started, "a zero limit turns the watchdog off");
}

// A new request with no beat in between restarts the clock.
void test_new_request_restarts_the_clock() {
    FakeWorld world;
    Progress progress;
    FakeStall stall(world, true);
    Watchdog watchdog(progress, 5, "strata-test", stall, world.sink, fake_hooks(world));
    const TimePoint start = world.now;
    progress.begin_request();
    check(!WatchdogTestAccess::poll(watchdog, start), "first request sets the baseline");
    progress.end_request();
    progress.begin_request();
    check(!WatchdogTestAccess::poll(watchdog, start + 1s), "a second request resets the clock");
    check(!WatchdogTestAccess::poll(watchdog, start + 5s), "no stall 4 s after the second request began");
    check(WatchdogTestAccess::poll(watchdog, start + 6s), "stall 5 s after the second request began");
}

// The limit is a 64-bit count of seconds. Converting it to nanoseconds overflows at about 9.2e9 s, and a limit that large
// must never fire, however long the stall: this one runs for 20 s of fake time.
void test_huge_limits_never_fire() {
    for (const std::uint64_t limit : {std::uint64_t {9'300'000'000ull}, UINT64_MAX}) {
        FakeWorld world;
        Progress progress;
        FakeStall stall(world, true);
        Watchdog watchdog(progress, limit, "strata-test", stall, world.sink, fake_hooks(world));
        progress.begin_request();
        progress.beat();
        const TimePoint start = world.now;
        bool any = false;
        for (int second = 0; second <= 20; ++second) {
            any = any || WatchdogTestAccess::poll(watchdog, start + std::chrono::seconds(second));
        }
        check(!any && !watchdog.fired() && world.aborts == 0, "a huge limit does not fire after a 20 s stall",
              std::to_string(limit));
    }
}

// The beat and its stage are one locked update. A snapshot taken while the request thread beats never pairs a new beat
// count with the stage of an earlier beat.
void test_beat_and_stage_update_together() {
    Progress progress;
    progress.begin_request();
    std::atomic<bool> finished {false};
    std::atomic<int> torn {0};
    std::thread reader([&progress, &finished, &torn] {
        while (!finished.load()) {
            const ProgressState state = progress.snapshot();
            if (state.beats > 0 && state.detail != std::to_string(state.beats)) torn.fetch_add(1);
        }
    });
    for (std::uint64_t i = 1; i <= 200000; ++i) progress.beat("decode", std::to_string(i));
    finished.store(true);
    reader.join();
    check(torn.load() == 0, "a snapshot never pairs a beat with an earlier stage", std::to_string(torn.load()));
    check(progress.snapshot().where == "decode", "the stage is set by the beat itself");
}

void test_report_line_is_written_whole() {
    FakeWorld world;
    write_report_line(world.sink, "first line");
    write_report_line(nullptr, "discarded");
    check(world.output() == "first line\n", "a report line is written whole, with one newline", world.output());
}

void test_busy_scope_and_counters() {
    Progress progress;
    check(!progress.snapshot().busy, "a new engine is idle");
    {
        const BusyScope scope(progress);
        check(progress.snapshot().busy, "BusyScope marks the request busy");
    }
    check(!progress.snapshot().busy, "BusyScope ends the busy state");

    progress.begin_request();
    progress.beat();
    for (int i = 0; i < 3; ++i) progress.tick();
    progress.beat();
    for (int i = 0; i < 2; ++i) progress.tick();
    const ProgressState state = progress.snapshot();
    // The BusyScope above was the first request, so this one is the second.
    check(state.requests == 2 && state.beats == 2 && state.ticks == 5, "requests, beats and ticks are counted",
          std::to_string(state.requests) + " " + std::to_string(state.beats) + " " + std::to_string(state.ticks));
    check(state.ticks - state.ticks_at_beat == 2, "layers since the last step are the ticks after the last beat",
          std::to_string(state.ticks - state.ticks_at_beat));
}

void test_limit_parsing() {
    std::uint64_t seconds = 99;
    std::string error;
    check(strata::metal::parse_watchdog_limit(nullptr, seconds, error) && seconds == kWatchdogDefaultSeconds,
          "unset STRATA_WATCHDOG_S defaults to 60 s");
    seconds = 99;
    check(parse_watchdog_limit("", seconds, error) && seconds == 60, "an empty STRATA_WATCHDOG_S defaults to 60 s");
    seconds = 99;
    check(parse_watchdog_limit("0", seconds, error) && seconds == 0, "0 turns the watchdog off");
    check(parse_watchdog_limit("5", seconds, error) && seconds == 5, "a decimal number is the limit");
    check(parse_watchdog_limit("18446744073709551615", seconds, error) && seconds == UINT64_MAX,
          "the largest 64-bit value is accepted");
    for (const char* bad : {"-1", "soon", "1.5", " 5", "+5", "5s", "18446744073709551616"}) {
        seconds = 99;
        check(!parse_watchdog_limit(bad, seconds, error) && !error.empty(), "a malformed limit is rejected", bad);
    }
}

void test_report_formats() {
    check(format_no_progress_line(60, "decode token 3") ==
              "strata serve: no progress for 60 s during a request (decode token 3) - stopping the engine so the server "
              "starts it again (issue #29)",
          "no-progress line format");
    check(format_stall_header("0.1.34", "decode", 5, 0) ==
              "strata serve: stall report (engine 0.1.34): stage \"decode\" for 5 s; 0 layers served since the last "
              "finished step (0 = stopped, more = slow)",
          "stall header format with no layers since the last step");
    platform::ProcessMemory process;
    process.phys_footprint = 5 * kMiB;
    process.resident = 300 * kMiB + 999;
    process.swap_used = 2 * kMiB + 5;
    process.major_faults = 7;
    check(format_memory_line(process, 4096 * kMiB + 1) ==
              "memory: 300 MiB resident, 2 MiB in swap, 4096 MiB RAM available; 7 major page faults so far",
          "memory line rounds down to whole MiB");
    const std::string missing_process = format_memory_line(std::nullopt, 4096 * kMiB);
    const std::string missing_available = format_memory_line(process, std::nullopt);
    check(missing_process == "memory: unavailable (the memory probe failed)" &&
              missing_available == missing_process,
          "a failed probe prints unavailable, not zeros");
    check(format_gpu_line("tf-prefill", "completed", 37888, 1024) ==
              "gpu: last command buffer \"tf-prefill\" status completed, 37888 MiB budget, 1024 MiB allocated",
          "gpu line format");
    check(std::string(kCrashReportHint) == kHintLine, "crash report hint text");
}

void test_real_probes_report_this_process() {
    const auto process = platform::process_memory();
    const auto available = platform::available_memory();
    check(process.has_value() && process->resident > 0, "process_memory reports a resident size for this process");
    check(available.has_value() && *available > 0, "available_memory reports free RAM");
    check(is_real_memory_line(format_memory_line(process, available)), "the real memory line has the documented form");
}

// ---- Process cases: a real serve loop in a child, with a stub backend -------------------------------------------

class StubBackend final : public ServeBackend {
public:
    enum class Behaviour { Stall, Beat, PromptStall };

    explicit StubBackend(Behaviour behaviour) : behaviour_(behaviour) {
        info_.context = 4096;
        info_.engine = "strata-test";
        info_.vram_free_mib = 4096;
    }

    const ServeInfo& serve_info() const noexcept override { return info_; }

    bool run_request(const ServeRequest& request, const ServeCallbacks& callbacks, ServeOutcome& outcome,
                     ServeFailure& failure) override {
        const auto prompt = static_cast<std::int64_t>(request.token_ids.size());
        if (!callbacks.resume(0)) {
            failure.message = "stub resume callback failed";
            return false;
        }
        if (behaviour_ == Behaviour::PromptStall) {
            // Prompt progress every 400 ms, so the beats span 2.8 s, longer than the limit. Then no callbacks: the
            // watchdog has to end the process in the prompt stage. The loop is a bound, not a pass.
            for (std::int64_t done = 1; done < prompt; ++done) {
                std::this_thread::sleep_for(400ms);
                if (!callbacks.prompt_progress(done, prompt, 1.0, 10.0)) return false;
            }
            for (int i = 0; i < 600; ++i) std::this_thread::sleep_for(100ms);
            failure.kind = ServeFailureKind::Fatal;
            failure.message = "the watchdog did not end the stalled prompt";
            return false;
        }
        if (!callbacks.prompt_progress(prompt, prompt, 1.0, 10.0) || !callbacks.prompt_reused(0)) {
            failure.message = "stub prompt callbacks failed";
            return false;
        }
        if (behaviour_ == Behaviour::Stall) {
            // One token, then no callbacks: the watchdog has to end the process. The loop is a bound, not a pass.
            if (!callbacks.token(100)) return false;
            for (int i = 0; i < 600; ++i) std::this_thread::sleep_for(100ms);
            failure.kind = ServeFailureKind::Fatal;
            failure.message = "the watchdog did not end the stalled request";
            return false;
        }
        // Gaps of 200 ms keep a beat inside every one-second poll window. Eight of them make the request run 1.6 s,
        // longer than the one-second limit, so the watchdog polls while it runs.
        for (std::int64_t i = 0; i < request.max_new; ++i) {
            std::this_thread::sleep_for(200ms);
            if (!callbacks.token(static_cast<std::int32_t>(100 + i))) return false;
        }
        outcome.prompt_ms = 1.0;
        outcome.decode_ms = 1.0;
        outcome.finish = "length";
        return true;
    }

private:
    ServeInfo info_;
    Behaviour behaviour_;
};

struct ChildResult {
    bool signaled = false;
    int signal = 0;
    bool exited = false;
    int exit_code = -1;
    std::string out;
    std::string err;
    std::chrono::milliseconds elapsed {0};
};

std::string read_to_eof(int fd) {
    std::string text;
    char chunk[1024];
    for (;;) {
        const ssize_t count = ::read(fd, chunk, sizeof(chunk));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        text.append(chunk, static_cast<std::size_t>(count));
    }
    return text;
}

// One child serve loop: its STRATA_WATCHDOG_S text, the stub behaviour, whether GENI is enabled, and `feed`, which writes
// the protocol input and may wait.
struct ChildSpec {
    const char* limit_text;
    StubBackend::Behaviour behaviour;
    bool vision;
    std::function<void(int)> feed;
};

// Runs every spec's serve loop in its own forked child, all at once, so the stalls and passes overlap. The feeds run in
// spec order, then the parent waits for each child and reads what it wrote. The watchdog's abort is the seam, and each
// child sets an alarm, so a child that hangs ends by SIGALRM instead of holding the test.
std::vector<ChildResult> run_serve_children(const std::vector<ChildSpec>& specs) {
    struct Run {
        pid_t pid = -1;
        int input = -1;
        int output = -1;
        int error = -1;
        Clock::time_point started {};
        Clock::time_point ended {};
        int status = 0;
        bool finished = false;
        bool waited = false;
    };
    std::vector<Run> runs(specs.size());
    std::fflush(nullptr);
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const ChildSpec& spec = specs[i];
        int input[2] = {-1, -1};
        int output[2] = {-1, -1};
        int error[2] = {-1, -1};
        if (::pipe(input) != 0 || ::pipe(output) != 0 || ::pipe(error) != 0) {
            check(false, "pipes for the child serve loop");
            continue;
        }
        runs[i].started = Clock::now();
        const pid_t pid = ::fork();
        if (pid == 0) {
            (void) ::dup2(input[0], STDIN_FILENO);
            (void) ::dup2(output[1], STDOUT_FILENO);
            (void) ::dup2(error[1], STDERR_FILENO);
            for (const int fd : {input[0], input[1], output[0], output[1], error[0], error[1]}) (void) ::close(fd);
            // Pipe ends of the children forked before this one are inherited; close them so no child holds another's stdin.
            for (const Run& earlier : runs) {
                for (const int fd : {earlier.input, earlier.output, earlier.error}) {
                    if (fd >= 0) (void) ::close(fd);
                }
            }
            (void) ::alarm(kChildAlarmSeconds);
            ::setenv("STRATA_WATCHDOG_S", spec.limit_text, 1);
            StubBackend backend(spec.behaviour);
            ServeLoopOptions options;
            options.vision_enabled = spec.vision;
            options.watchdog_abort = [] {
                const std::string line = std::string(kSeamLine) + "\n";
                (void) ::write(STDERR_FILENO, line.data(), line.size());
                ::_exit(kSeamExitCode);
            };
            ::_exit(run_serve_loop(backend, options));
        }
        (void) ::close(input[0]);
        (void) ::close(output[1]);
        (void) ::close(error[1]);
        if (pid < 0) {
            check(false, "fork the child serve loop");
            (void) ::close(input[1]);
            (void) ::close(output[0]);
            (void) ::close(error[0]);
            continue;
        }
        runs[i].pid = pid;
        runs[i].input = input[1];
        runs[i].output = output[0];
        runs[i].error = error[0];
    }

    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (runs[i].pid > 0) specs[i].feed(runs[i].input);
    }

    for (bool pending = true; pending;) {
        pending = false;
        for (Run& run : runs) {
            if (run.pid <= 0 || run.finished) continue;
            const pid_t reaped = ::waitpid(run.pid, &run.status, WNOHANG);
            if (reaped == run.pid) {
                run.finished = run.waited = true;
                run.ended = Clock::now();
            } else if (reaped < 0 && errno != EINTR) {
                run.finished = true;
                check(false, "wait for a child serve loop", std::strerror(errno));
            } else {
                pending = true;
            }
        }
        if (pending) std::this_thread::sleep_for(5ms);
    }

    std::vector<ChildResult> results(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        Run& run = runs[i];
        if (run.pid <= 0) continue;
        ChildResult& result = results[i];
        result.out = read_to_eof(run.output);
        result.err = read_to_eof(run.error);
        (void) ::close(run.input);
        (void) ::close(run.output);
        (void) ::close(run.error);
        if (!run.waited) continue;
        result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(run.ended - run.started);
        result.signaled = WIFSIGNALED(run.status);
        result.signal = result.signaled ? WTERMSIG(run.status) : 0;
        result.exited = WIFEXITED(run.status);
        result.exit_code = result.exited ? WEXITSTATUS(run.status) : -1;
    }
    return results;
}

void write_text(int fd, const std::string& text) {
    std::size_t offset = 0;
    while (offset < text.size()) {
        const ssize_t count = ::write(fd, text.data() + offset, text.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return;
        offset += static_cast<std::size_t>(count);
    }
}

std::size_t position_of(const std::vector<std::string>& lines, const std::string& prefix, std::size_t from) {
    for (std::size_t i = from; i < lines.size(); ++i) {
        if (starts_with(lines[i], prefix)) return i;
    }
    return lines.size();
}

struct ProcessResults {
    ChildResult bad_limit;
    ChildResult beats;
    ChildResult idle;
    ChildResult stall;
    ChildResult prompt;
    ChildResult image;
};

// Every process case, run once and shared by the tests below. The cases overlap, so the group takes about as long as its
// slowest child (the prompt stall) rather than the sum of them.
const ProcessResults& process_results() {
    static const ProcessResults results = [] {
        // The image case's file is a FIFO that nothing writes to, so the child's open blocks until the watchdog ends it.
        char pattern[] = "/tmp/strata-watchdog-XXXXXX";
        const char* made = ::mkdtemp(pattern);
        const std::string fifo = made == nullptr ? std::string() : std::string(made) + "/image.fifo";
        if (made == nullptr || ::mkfifo(fifo.c_str(), 0600) != 0) check(false, "a FIFO for the image load");
        const std::string image_request = "GENI 1 " + fifo + " 1,2\n";

        // The idle child's feed waits 2.5 s before it sends QUIT, so it is fed last and the others are already running.
        const std::vector<ChildSpec> specs = {
            {"soon", StubBackend::Behaviour::Beat, false, [](int) {}},
            {"1", StubBackend::Behaviour::Stall, false, [](int input) { write_text(input, "GEN 2 5,6\n"); }},
            {"1", StubBackend::Behaviour::PromptStall, false,
             [](int input) { write_text(input, "GEN 2 1,2,3,4,5,6,7,8\n"); }},
            {"1", StubBackend::Behaviour::Beat, true, [&image_request](int input) { write_text(input, image_request); }},
            {"1", StubBackend::Behaviour::Beat, false, [](int input) { write_text(input, "GEN 8 5,6\nQUIT\n"); }},
            {"1", StubBackend::Behaviour::Beat, false,
             [](int input) {
                 std::this_thread::sleep_for(2500ms);
                 write_text(input, "QUIT\n");
             }},
        };
        std::vector<ChildResult> children = run_serve_children(specs);

        ProcessResults out;
        out.bad_limit = children[0];
        out.stall = children[1];
        out.prompt = children[2];
        out.image = children[3];
        out.beats = children[4];
        out.idle = children[5];
        if (made != nullptr) {
            std::error_code ignored;
            std::filesystem::remove_all(made, ignored);
        }
        return out;
    }();
    return results;
}

// A real stall: the process ends through the abort seam after the issue #29 line, two passes and the crash-report hint.
// The seam's line comes last, and the exit code is 86. The real SIGABRT path is metal_watchdog_test's job.
void test_process_stall_aborts_with_report() {
    const ChildResult& result = process_results().stall;
    check(result.exited && result.exit_code == kSeamExitCode, "the stalled engine ends through the abort seam",
          result.signaled ? "signal " + std::to_string(result.signal) : "exit " + std::to_string(result.exit_code));
    check(result.out.find("READY ") != std::string::npos, "the child reached READY before the stall");
    check(result.out.find("RESUME 0\n") != std::string::npos && result.out.find("T 100\n") != std::string::npos,
          "the protocol emitted the first token before the stall");

    const std::vector<std::string> lines = split_lines(result.err);
    const std::size_t issue = position_of(lines, "strata serve: no progress for 1 s during a request (decode token 1)", 0);
    const std::size_t header = position_of(
        lines, "strata serve: stall report (engine strata-test): stage \"decode token 1\" for ", issue + 1);
    check(issue < lines.size(), "issue #29 line with the limit and the stage", result.err);
    check(header < lines.size() && ends_with(lines[header], "; 0 layers served since the last finished step (0 = "
                                                             "stopped, more = slow)"),
          "stall report header names the stage, and no layers were served", result.err);
    const std::size_t memory_1 = position_of(lines, "memory: ", header + 1);
    const std::size_t gpu_1 = position_of(lines, "gpu: unavailable", memory_1 + 1);
    const std::size_t memory_2 = position_of(lines, "memory: ", gpu_1 + 1);
    const std::size_t gpu_2 = position_of(lines, "gpu: unavailable", memory_2 + 1);
    const std::size_t hint = position_of(lines, kHintLine, gpu_2 + 1);
    check(hint < lines.size() && issue < header && header < memory_1 && memory_1 < gpu_1 && gpu_1 < memory_2 &&
              memory_2 < gpu_2 && gpu_2 < hint,
          "the stall lines come in the documented order: issue, header, two passes, hint", result.err);
    check(memory_1 < lines.size() && is_real_memory_line(lines[memory_1]), "a real memory line in the first pass");
    check(memory_2 < lines.size() && is_real_memory_line(lines[memory_2]), "a real memory line in the second pass");
    const std::size_t seam = position_of(lines, kSeamLine, hint + 1);
    check(seam == lines.size() - 1, "the abort seam's line comes last", result.err);
    check(result.elapsed < 15s, "the stall ends within the limit plus the passes", std::to_string(result.elapsed.count()));
}

// The prompt stage is named in the stall report. The stub beats through the prompt phase for 2.8 s, longer than the
// limit, and then stalls there. If the prompt callback did not beat, the watchdog would fire inside the phase and name
// an earlier count, so this check fails.
void test_process_prompt_stall_names_the_prompt_stage() {
    const ChildResult& result = process_results().prompt;
    check(result.exited && result.exit_code == kSeamExitCode, "a stall in the prompt stage ends through the seam",
          result.signaled ? "signal " + std::to_string(result.signal) : "exit " + std::to_string(result.exit_code));
    check(result.out.find("RESUME 0\n") != std::string::npos, "the protocol resumed before the prompt stall");
    check(result.out.find("PP 7 8 ") != std::string::npos, "prompt progress reached 7 of 8 before the stall", result.out);

    const std::vector<std::string> lines = split_lines(result.err);
    const std::size_t issue = position_of(lines, "strata serve: no progress for 1 s during a request (prompt 7/8)", 0);
    const std::size_t header = position_of(
        lines, "strata serve: stall report (engine strata-test): stage \"prompt 7/8\" for ", issue + 1);
    check(issue < lines.size(), "the issue #29 line names the prompt stage", result.err);
    check(header < lines.size(), "the stall report header names the prompt stage", result.err);
    check(position_of(lines, kSeamLine, header + 1) < lines.size(), "the seam ends the prompt stall", result.err);
}

// The image load runs inside the request's busy scope. Its file is a FIFO that nothing writes to, so the open blocks,
// and only the watchdog can end the child, with the stage named "image". Without the scope, the child would hang until
// SIGALRM.
void test_process_image_load_is_watched() {
    const ChildResult& result = process_results().image;
    check(result.exited && result.exit_code == kSeamExitCode, "a stalled image load ends through the seam",
          result.signaled ? "signal " + std::to_string(result.signal) : "exit " + std::to_string(result.exit_code));
    check(result.out.find("READY ") != std::string::npos && result.out.find("RESUME ") == std::string::npos,
          "the stall happens before the backend starts the request");
    const std::vector<std::string> lines = split_lines(result.err);
    check(position_of(lines, "strata serve: no progress for 1 s during a request (image) - ", 0) < lines.size(),
          "the issue #29 line names the image stage", result.err);
    check(position_of(lines, "strata serve: stall report (engine strata-test): stage \"image\" for ", 0) < lines.size(),
          "the stall report header names the image stage", result.err);
}

// A request that keeps beating runs past the limit without a report, and ends normally.
void test_process_beats_keep_request_alive() {
    const ChildResult& result = process_results().beats;
    check(result.exited && result.exit_code == 0, "a beating request ends with a clean exit",
          result.signaled ? "signal " + std::to_string(result.signal) : "exit " + std::to_string(result.exit_code));
    check(result.out.find("\nDONE 8 2 ") != std::string::npos, "the eight tokens were produced and finished");
    check(result.err.find("no progress") == std::string::npos && result.err.find("stall report") == std::string::npos,
          "no stall report while the engine beats");
    check(result.elapsed > 1s, "the request outlived the one-second limit", std::to_string(result.elapsed.count()));
}

// An engine waiting for input is not busy, so the watchdog stays quiet past the limit.
void test_process_idle_engine_is_quiet() {
    const ChildResult& result = process_results().idle;
    check(result.exited && result.exit_code == 0, "an idle engine quits cleanly after QUIT",
          result.signaled ? "signal " + std::to_string(result.signal) : "exit " + std::to_string(result.exit_code));
    check(result.err.find("no progress") == std::string::npos && result.err.find("stall report") == std::string::npos,
          "an idle engine is not reported as stalled");
    check(result.elapsed >= 2500ms, "the engine idled past the limit");
}

// A malformed STRATA_WATCHDOG_S fails the serve startup with exit 2 before any protocol line.
void test_process_bad_limit_exits_2() {
    const ChildResult& result = process_results().bad_limit;
    check(result.exited && result.exit_code == 2, "a malformed STRATA_WATCHDOG_S exits 2",
          result.signaled ? "signal " + std::to_string(result.signal) : "exit " + std::to_string(result.exit_code));
    check(result.err.find("strata: STRATA_WATCHDOG_S needs a decimal number of seconds") != std::string::npos,
          "the startup error names the variable", result.err);
    check(result.out.empty(), "no protocol output before the startup error");
}

}  // namespace

int main() {
    test_limit_parsing();
    test_report_formats();
    test_report_line_is_written_whole();
    test_busy_scope_and_counters();
    test_beat_and_stage_update_together();
    test_real_probes_report_this_process();
    test_fake_clock_fires_at_the_limit();
    test_huge_limits_never_fire();
    test_refused_release_still_aborts();
    test_beats_keep_the_watchdog_quiet();
    test_idle_never_fires();
    test_zero_limit_is_off();
    test_new_request_restarts_the_clock();
    test_process_bad_limit_exits_2();
    test_process_beats_keep_request_alive();
    test_process_idle_engine_is_quiet();
    test_process_stall_aborts_with_report();
    test_process_prompt_stall_names_the_prompt_stage();
    test_process_image_load_is_watched();
    std::printf("watchdog_host_test: %s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
