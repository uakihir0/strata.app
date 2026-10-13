// tests/metal/serve_transcript_test.cpp - protocol goldens and process lifecycle checks.
#include "fake_window.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <poll.h>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace strata::metal;
using strata::metal::test::FakeWindow;

#ifndef STRATA_SERVE_FIXTURE_DIR
#error STRATA_SERVE_FIXTURE_DIR must name the checked-in serve fixtures
#endif

namespace {

int failures = 0;

void check(bool condition, std::string_view name, std::string_view detail = {}) {
    std::cout << (condition ? "ok  " : "FAIL ") << name;
    if (!detail.empty()) std::cout << ": " << detail;
    std::cout << '\n';
    failures += !condition;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string make_sve_file(const std::array<std::int32_t, 5>& header,
                          std::size_t payload_floats = 5'120, std::size_t trailing_bytes = 0) {
    std::array<char, 64> pattern {};
    const std::string base = "/tmp/strata serve SVE1.XXXXXX";
    std::copy(base.begin(), base.end(), pattern.begin());
    const int descriptor = ::mkstemp(pattern.data());
    if (descriptor < 0) return {};
    std::FILE* file = ::fdopen(descriptor, "wb");
    if (file == nullptr) {
        (void) ::close(descriptor);
        (void) ::unlink(pattern.data());
        return {};
    }
    const bool wrote_header =
        std::fwrite(header.data(), sizeof(header.front()), header.size(), file) == header.size();
    std::vector<float> payload(payload_floats, 0.0f);
    const bool wrote_payload = payload.empty() ||
        std::fwrite(payload.data(), sizeof(float), payload.size(), file) == payload.size();
    const std::array<unsigned char, 3> trailing {0xa5, 0x5a, 0xff};
    const bool wrote_trailing = trailing_bytes <= trailing.size() &&
        (trailing_bytes == 0 || std::fwrite(trailing.data(), 1, trailing_bytes, file) == trailing_bytes);
    const bool closed = std::fclose(file) == 0;
    if (!wrote_header || !wrote_payload || !wrote_trailing || !closed) {
        (void) ::unlink(pattern.data());
        return {};
    }
    return pattern.data();
}

std::string make_sve_records_file(const std::vector<std::array<std::int32_t, 5>>& records) {
    std::array<char, 64> pattern {};
    const std::string base = "/tmp/strata serve SVE1.XXXXXX";
    std::copy(base.begin(), base.end(), pattern.begin());
    const int descriptor = ::mkstemp(pattern.data());
    if (descriptor < 0) return {};
    std::FILE* file = ::fdopen(descriptor, "wb");
    if (file == nullptr) {
        (void) ::close(descriptor);
        (void) ::unlink(pattern.data());
        return {};
    }
    bool wrote = true;
    for (const auto& header : records) {
        const std::size_t payload_floats =
            static_cast<std::size_t>(header[1]) * static_cast<std::size_t>(header[4]);
        std::vector<float> payload(payload_floats, 0.0f);
        wrote = wrote && std::fwrite(header.data(), sizeof(header.front()), header.size(), file) == header.size() &&
                std::fwrite(payload.data(), sizeof(float), payload.size(), file) == payload.size();
    }
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed) {
        (void) ::unlink(pattern.data());
        return {};
    }
    return pattern.data();
}

void replace_once(std::string& text, std::string_view needle, std::string_view replacement) {
    const std::size_t position = text.find(needle);
    if (position != std::string::npos) text.replace(position, needle.size(), replacement);
}

std::string read_stream(std::FILE* file) {
    std::fflush(file);
    if (std::fseek(file, 0, SEEK_END) != 0) return {};
    const long size = std::ftell(file);
    if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) return {};
    std::string result(static_cast<std::size_t>(size), '\0');
    if (!result.empty() && std::fread(result.data(), 1, result.size(), file) != result.size()) return {};
    return result;
}

bool write_all(int descriptor, std::string_view text);

void run_fixture(std::string_view name, bool vision, bool fail_gpu = false) {
    const std::filesystem::path root(STRATA_SERVE_FIXTURE_DIR);
    const auto input_path = root / (std::string(name) + ".in");
    const auto expected_out_path = root / (std::string(name) + ".out");
    const auto expected_err_path = root / (std::string(name) + ".err");
    std::string input_text = read_file(input_path);
    std::string image_path;
    if (name == "image") {
        image_path = make_sve_records_file({
            {0x31455653, 4, 2, 2, kServeImageEmbeddingWidth},
            {0x31455653, 2, 2, 1, kServeImageEmbeddingWidth},
        });
        replace_once(input_text, "IMAGE_PATH", image_path);
    }
    int input_pipe[2] = {-1, -1};
    const bool input_pipe_open = ::pipe(input_pipe) == 0;
    std::FILE* output = std::tmpfile();
    std::FILE* diagnostics = std::tmpfile();
    check(input_pipe_open && output != nullptr && diagnostics != nullptr,
          std::string(name) + " fixture files open");
    if (!input_pipe_open || output == nullptr || diagnostics == nullptr) {
        if (input_pipe[0] >= 0) (void) ::close(input_pipe[0]);
        if (input_pipe[1] >= 0) (void) ::close(input_pipe[1]);
        if (output != nullptr) std::fclose(output);
        if (diagnostics != nullptr) std::fclose(diagnostics);
        return;
    }
    const bool input_written = write_all(input_pipe[1], input_text);
    check(input_written, std::string(name) + " input transcript is ready");
    if (!input_written) {
        (void) ::close(input_pipe[0]);
        (void) ::close(input_pipe[1]);
        std::fclose(output);
        std::fclose(diagnostics);
        if (!image_path.empty()) (void) ::unlink(image_path.c_str());
        return;
    }

    FakeWindow backend(fail_gpu);
    ServeLoopOptions options;
    options.vision_enabled = vision;
    options.input_fd = input_pipe[0];
    options.output = output;
    options.diagnostics = diagnostics;
    const int result = run_serve_loop(backend, options);
    const std::string actual_out = read_stream(output);
    const std::string actual_err = read_stream(diagnostics);
    const std::string expected_out = read_file(expected_out_path);
    const std::string expected_err = read_file(expected_err_path);
    check(actual_out == expected_out, std::string(name) + " stdout transcript matches byte for byte",
          actual_out == expected_out ? "" : "see actual transcript printed below");
    if (actual_out != expected_out) std::cout << actual_out;
    check(actual_err == expected_err, std::string(name) + " stderr transcript matches byte for byte",
          actual_err == expected_err ? "" : "see actual diagnostics printed below");
    if (actual_err != expected_err) std::cout << actual_err;
    const bool should_fail = fail_gpu;
    check(result == (should_fail ? 1 : 0), std::string(name) + " returns the specified process status");
    (void) ::close(input_pipe[0]);
    (void) ::close(input_pipe[1]);
    std::fclose(output);
    std::fclose(diagnostics);
    if (!image_path.empty()) (void) ::unlink(image_path.c_str());
}

enum class BackendFault {
    TooManyTokens,
    ProgressExceedsTotal,
    ProgressWrongTotal,
    ProgressBelowResume,
    ProgressRegresses,
    ProgressIncomplete,
    CancelWithoutStop,
    StopWithoutEos,
    TokenAfterEos,
    EosStop
};

class AdversarialBackend final : public ServeBackend {
public:
    explicit AdversarialBackend(BackendFault fault) : fault_(fault) {
        info_.context = 32;
        info_.engine = "test";
        info_.metal_rev = 1;
        info_.stop_eos = fault == BackendFault::StopWithoutEos || fault == BackendFault::TokenAfterEos ||
                         fault == BackendFault::EosStop;
    }

    const ServeInfo& serve_info() const noexcept override { return info_; }
    void set_context(std::int64_t context) { info_.context = context; }
    bool saw_stop_after_eos() const noexcept { return saw_stop_after_eos_; }

    bool run_request(const ServeRequest& request, const ServeCallbacks& callbacks, ServeOutcome& outcome,
                     ServeFailure&) override {
        const std::int64_t prompt = static_cast<std::int64_t>(request.token_ids.size());
        const std::int64_t reused = fault_ == BackendFault::ProgressBelowResume ? prompt : 0;
        if (!callbacks.resume(reused)) return false;
        if (fault_ == BackendFault::ProgressExceedsTotal) {
            (void) callbacks.prompt_progress(prompt + 1, prompt, 1.0, 1.0);
        } else if (fault_ == BackendFault::ProgressWrongTotal) {
            (void) callbacks.prompt_progress(prompt, prompt + 1, 1.0, 1.0);
        } else if (fault_ == BackendFault::ProgressBelowResume) {
            (void) callbacks.prompt_progress(0, prompt, 1.0, 1.0);
        } else if (fault_ == BackendFault::ProgressRegresses) {
            if (!callbacks.prompt_progress(1, prompt, 1.0, 1.0)) return false;
            (void) callbacks.prompt_progress(0, prompt, 1.0, 1.0);
        } else if (fault_ == BackendFault::ProgressIncomplete) {
            if (!callbacks.prompt_progress(prompt - 1, prompt, 1.0, 1.0)) return false;
        } else if (!callbacks.prompt_progress(prompt, prompt, 1.0, 1.0)) {
            return false;
        }
        if (callbacks.prompt_reused(reused)) {
            if (fault_ == BackendFault::TooManyTokens) {
                (void) callbacks.token(11);
                (void) callbacks.token(12);
            } else if (fault_ == BackendFault::StopWithoutEos) {
                (void) callbacks.token(11);
            } else if (fault_ == BackendFault::TokenAfterEos) {
                (void) callbacks.token(248044);
                (void) callbacks.token(9);
            } else if (fault_ == BackendFault::EosStop) {
                if (!callbacks.token(248044)) return false;
                saw_stop_after_eos_ = callbacks.should_stop();
            }
        }
        outcome.prompt_ms = 1.0;
        outcome.decode_ms = 1.0;
        if (fault_ == BackendFault::CancelWithoutStop) outcome.finish = "cancel";
        else if (fault_ == BackendFault::StopWithoutEos) outcome.finish = "stop";
        else outcome.finish = "length";
        return true;
    }

private:
    BackendFault fault_;
    ServeInfo info_;
    bool saw_stop_after_eos_ = false;
};

void test_backend_protocol_limits() {
    struct TestCase {
        BackendFault fault;
        std::string_view name;
        std::size_t expected_tokens;
        int expected_status;
        bool expected_error;
    };
    const std::array cases {
        TestCase {BackendFault::TooManyTokens, "backend cannot emit more than max_new tokens", 1, 1, true},
        TestCase {BackendFault::ProgressExceedsTotal, "PP cannot report done greater than total", 0, 1, true},
        TestCase {BackendFault::ProgressWrongTotal, "PP total must equal the prompt length", 0, 1, true},
        TestCase {BackendFault::ProgressBelowResume, "PP cannot regress before the resumed prompt prefix", 0, 1,
                  true},
        TestCase {BackendFault::ProgressRegresses, "PP progress cannot regress", 0, 1, true},
        TestCase {BackendFault::ProgressIncomplete, "REUSED requires final prompt progress to be complete", 0, 1,
                  true},
        TestCase {BackendFault::CancelWithoutStop, "backend cannot report cancellation without STOP or EOF", 0, 1,
                  true},
        TestCase {BackendFault::StopWithoutEos, "finish=stop requires an emitted configured EOS token", 1, 1, true},
        TestCase {BackendFault::TokenAfterEos, "backend cannot emit tokens after EOS", 1, 1, true},
        TestCase {BackendFault::EosStop, "EOS requests backend stop and overrides length", 1, 0, false},
    };
    for (const auto& test_case : cases) {
        const auto& [fault, name, expected_tokens, expected_status, expected_error] = test_case;
        int input_pipe[2] = {-1, -1};
        const bool input_pipe_open = ::pipe(input_pipe) == 0;
        std::FILE* output = std::tmpfile();
        std::FILE* diagnostics = std::tmpfile();
        check(input_pipe_open && output != nullptr && diagnostics != nullptr,
              std::string(name) + " streams open");
        if (!input_pipe_open || output == nullptr || diagnostics == nullptr) {
            if (input_pipe[0] >= 0) (void) ::close(input_pipe[0]);
            if (input_pipe[1] >= 0) (void) ::close(input_pipe[1]);
            if (output != nullptr) std::fclose(output);
            if (diagnostics != nullptr) std::fclose(diagnostics);
            continue;
        }
        constexpr std::string_view transcript = "GEN 1 7\nQUIT\n";
        const bool ready = write_all(input_pipe[1], transcript);
        check(ready, std::string(name) + " transcript is ready");
        AdversarialBackend backend(fault);
        ServeLoopOptions options;
        options.input_fd = input_pipe[0];
        options.output = output;
        options.diagnostics = diagnostics;
        const int result = ready ? run_serve_loop(backend, options) : 2;
        const std::string actual = read_stream(output);
        std::size_t token_lines = 0;
        for (std::size_t position = actual.find("\nT "); position != std::string::npos;
             position = actual.find("\nT ", position + 1)) {
            ++token_lines;
        }
        const bool has_protocol_error = actual.find("ERR internal serve protocol error:") != std::string::npos;
        const bool valid_eos = actual.find("DONE 1 1 1.0 1.0 stop ") != std::string::npos;
        check(result == expected_status && token_lines == static_cast<std::size_t>(expected_tokens) &&
                  (expected_error ? has_protocol_error : (!has_protocol_error && valid_eos)) &&
                  (fault != BackendFault::EosStop || backend.saw_stop_after_eos()),
              name, actual);
        (void) ::close(input_pipe[0]);
        (void) ::close(input_pipe[1]);
        std::fclose(output);
        std::fclose(diagnostics);
    }

    std::FILE* input = std::tmpfile();
    std::FILE* output = std::tmpfile();
    std::FILE* diagnostics = std::tmpfile();
    check(input != nullptr && output != nullptr && diagnostics != nullptr,
          "oversized-context startup streams open");
    if (input != nullptr && output != nullptr && diagnostics != nullptr) {
        AdversarialBackend backend(BackendFault::TooManyTokens);
        backend.set_context(262'145);
        ServeLoopOptions options;
        options.input_fd = fileno(input);
        options.output = output;
        options.diagnostics = diagnostics;
        const int result = run_serve_loop(backend, options);
        check(result == 2 && read_stream(output).empty() &&
                  read_stream(diagnostics).find("invalid serve startup metadata") != std::string::npos,
              "startup rejects contexts above the host memory bound");
    }
    if (input != nullptr) std::fclose(input);
    if (output != nullptr) std::fclose(output);
    if (diagnostics != nullptr) std::fclose(diagnostics);
}

void test_preclosed_eof() {
    int input_pipe[2] = {-1, -1};
    std::FILE* output = std::tmpfile();
    std::FILE* diagnostics = std::tmpfile();
    const bool input_ready = ::pipe(input_pipe) == 0 && write_all(input_pipe[1], "GEN 5 17\n") &&
                             ::close(input_pipe[1]) == 0;
    check(input_ready && output != nullptr && diagnostics != nullptr,
          "preclosed EOF streams are ready");
    if (!input_ready || output == nullptr || diagnostics == nullptr) {
        if (input_pipe[0] >= 0) (void) ::close(input_pipe[0]);
        if (input_pipe[1] >= 0) (void) ::close(input_pipe[1]);
        if (output != nullptr) std::fclose(output);
        if (diagnostics != nullptr) std::fclose(diagnostics);
        return;
    }

    FakeWindow backend;
    ServeLoopOptions options;
    options.input_fd = input_pipe[0];
    options.output = output;
    options.diagnostics = diagnostics;
    const int result = run_serve_loop(backend, options);
    const std::string actual = read_stream(output);
    check(result == 0 && actual.find("DONE 0 1 4.0 6.0 cancel ") != std::string::npos,
          "a fast backend sees a request as cancelled when EOF was already pending before dequeue", actual);
    (void) ::close(input_pipe[0]);
    std::fclose(output);
    std::fclose(diagnostics);
}

bool write_all(int descriptor, std::string_view text) {
    std::size_t offset = 0;
    while (offset < text.size()) {
        const ssize_t count = ::write(descriptor, text.data() + offset, text.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

class ChildEngine {
public:
    bool start(const std::filesystem::path& executable, std::string_view mode) {
        int input_pipe[2] = {-1, -1};
        int output_pipe[2] = {-1, -1};
        if (::pipe(input_pipe) != 0 || ::pipe(output_pipe) != 0) return false;
        pid_ = ::fork();
        if (pid_ < 0) return false;
        if (pid_ == 0) {
            (void) ::dup2(input_pipe[0], STDIN_FILENO);
            (void) ::dup2(output_pipe[1], STDOUT_FILENO);
            const int null_fd = ::open("/dev/null", O_WRONLY);
            if (null_fd >= 0) (void) ::dup2(null_fd, STDERR_FILENO);
            (void) ::close(input_pipe[0]);
            (void) ::close(input_pipe[1]);
            (void) ::close(output_pipe[0]);
            (void) ::close(output_pipe[1]);
            const std::string exe = executable.string();
            const std::string child_mode(mode);
            char* const arguments[] = {const_cast<char*>(exe.c_str()),
                                       const_cast<char*>("--serve-child"),
                                       const_cast<char*>(child_mode.c_str()), nullptr};
            ::execv(exe.c_str(), arguments);
            _exit(127);
        }
        (void) ::close(input_pipe[0]);
        (void) ::close(output_pipe[1]);
        input_ = input_pipe[1];
        output_ = output_pipe[0];
        return true;
    }

    ~ChildEngine() {
        if (input_ >= 0) (void) ::close(input_);
        if (output_ >= 0) (void) ::close(output_);
        if (pid_ > 0 && !exited_) {
            (void) ::kill(pid_, SIGKILL);
            int status = 0;
            (void) ::waitpid(pid_, &status, 0);
        }
    }

    bool send(std::string_view text) { return input_ >= 0 && write_all(input_, text); }

    bool next_line(std::string& line, std::chrono::milliseconds timeout) {
        if (!lines_.empty()) {
            line = std::move(lines_.front());
            lines_.pop_front();
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::array<char, 1024> chunk {};
        while (std::chrono::steady_clock::now() < deadline) {
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            pollfd descriptor {output_, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, static_cast<int>(remaining.count()));
            if (ready < 0 && errno == EINTR) continue;
            if (ready <= 0) return false;
            const ssize_t count = ::read(output_, chunk.data(), chunk.size());
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) return false;
            pending_.append(chunk.data(), static_cast<std::size_t>(count));
            std::size_t newline = 0;
            while ((newline = pending_.find('\n')) != std::string::npos) {
                lines_.push_back(pending_.substr(0, newline));
                pending_.erase(0, newline + 1);
            }
            if (!lines_.empty()) {
                line = std::move(lines_.front());
                lines_.pop_front();
                return true;
            }
        }
        return false;
    }

    bool wait_for_exit(std::chrono::milliseconds timeout, int& exit_code) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        int status = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            const pid_t result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                exited_ = true;
                exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
                pid_ = -1;
                return true;
            }
            if (result < 0 && errno != EINTR) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }

    bool signal(int value) { return pid_ > 0 && ::kill(pid_, value) == 0; }
    void close_input() {
        if (input_ >= 0) {
            (void) ::close(input_);
            input_ = -1;
        }
    }

private:
    pid_t pid_ = -1;
    int input_ = -1;
    int output_ = -1;
    bool exited_ = false;
    std::string pending_;
    std::deque<std::string> lines_;
};

bool wait_until_done(ChildEngine& child, std::vector<std::string>& transcript, std::string& done) {
    std::string line;
    while (child.next_line(line, std::chrono::seconds(5))) {
        transcript.push_back(line);
        if (line.starts_with("DONE ")) {
            done = line;
            return true;
        }
        if (line.starts_with("ERR ")) return false;
    }
    return false;
}

void test_child_lifecycle(const std::filesystem::path& executable) {
    ChildEngine child;
    check(child.start(executable, "quick"), "start serve executable as a child");
    if (std::string line; child.next_line(line, std::chrono::seconds(5))) {
        check(line.starts_with("INFO "), "child emits INFO before READY");
        check(child.next_line(line, std::chrono::seconds(5)) && line == "READY 32 stop",
              "child advertises STOP support");
    } else {
        check(false, "child emits INFO before READY", "no startup line");
        return;
    }

    std::vector<std::string> first_transcript;
    std::string first_done;
    check(child.send("GEN 3 17\n"), "send first child request");
    check(wait_until_done(child, first_transcript, first_done) &&
              first_done == "DONE 3 1 4.0 6.0 length 0 0 0 3 4 1 2 0.5",
          "first child request completes before signals");
    check(child.signal(SIGINT), "SIGINT is delivered between requests");
    check(child.signal(SIGHUP), "SIGHUP is delivered between requests");

    std::vector<std::string> second_transcript;
    std::string second_done;
    check(child.send("GEN 2 29\n"), "send second child request");
    check(wait_until_done(child, second_transcript, second_done) &&
              second_done == "DONE 2 1 4.0 6.0 length 0 0 0 3 4 1 2 0.5",
          "second child request completes after both signals");
    child.close_input();
    int exit_code = -1;
    check(child.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "stdin EOF exits cleanly within five seconds");

    std::string line;
    ChildEngine after_done;
    check(after_done.start(executable, "slow"), "start child for STOP immediately after DONE");
    if (!after_done.next_line(line, std::chrono::seconds(5)) ||
        !after_done.next_line(line, std::chrono::seconds(5))) {
        check(false, "post-DONE child reaches READY", "missing startup lines");
        return;
    }
    check(after_done.send("GEN 1 3\n"), "send request A before post-DONE STOP");
    std::vector<std::string> after_done_first_transcript;
    std::string after_done_first;
    check(wait_until_done(after_done, after_done_first_transcript, after_done_first) &&
              after_done_first.starts_with("DONE 1 1 4.0 6.0 length "),
          "request A completes before the next request arrives");
    check(after_done.send("GEN 20 5\nSTOP\n"), "send request B and STOP immediately after A's DONE");
    std::vector<std::string> after_done_second_transcript;
    std::string after_done_second;
    check(wait_until_done(after_done, after_done_second_transcript, after_done_second) &&
              after_done_second.starts_with("DONE ") && after_done_second.ends_with(" cancel 0 0 0 3 4 1 2 0.5"),
          "STOP after A's DONE cancels queued or active request B");
    check(after_done.send("QUIT\n"), "send QUIT after post-DONE STOP");
    check(after_done.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "post-DONE STOP child exits cleanly");

    ChildEngine queued_stop;
    check(queued_stop.start(executable, "quick"), "start child for queued STOP ordering");
    if (!queued_stop.next_line(line, std::chrono::seconds(5)) ||
        !queued_stop.next_line(line, std::chrono::seconds(5))) {
        check(false, "queued STOP child reaches READY", "missing startup lines");
        return;
    }
    check(queued_stop.send("GEN 20 33\nSTOP\n"), "send GEN and STOP before reading the response");
    std::vector<std::string> queued_stop_transcript;
    std::string queued_stop_done;
    check(wait_until_done(queued_stop, queued_stop_transcript, queued_stop_done) &&
              queued_stop_done.starts_with("DONE 0 1 4.0 6.0 cancel "),
          "STOP queued immediately after GEN cancels that request");
    check(queued_stop.send("QUIT\n"), "send QUIT after queued STOP");
    check(queued_stop.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "queued STOP child exits cleanly");

    ChildEngine stopping;
    check(stopping.start(executable, "slow"), "start child for asynchronous STOP");
    if (!stopping.next_line(line, std::chrono::seconds(5)) ||
        !stopping.next_line(line, std::chrono::seconds(5))) {
        check(false, "slow child reaches READY", "missing startup lines");
        return;
    }
    check(stopping.send("GEN 20 7\n"), "send cancellable generation request");
    bool saw_token = false;
    while (stopping.next_line(line, std::chrono::seconds(5))) {
        if (line.starts_with("T ")) {
            saw_token = true;
            break;
        }
    }
    check(saw_token, "slow child streams a token before STOP");
    check(stopping.send("STOP\n"), "send STOP while generation is active");
    std::vector<std::string> stopped_transcript;
    std::string stopped_done;
    check(wait_until_done(stopping, stopped_transcript, stopped_done) &&
              stopped_done.starts_with("DONE 1 1 4.0 6.0 cancel "),
          "STOP ends at a window boundary and emits DONE cancel");
    check(stopping.send("QUIT\n"), "send QUIT after cancellation");
    check(stopping.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "QUIT exits cleanly");

    ChildEngine stop_length;
    check(stop_length.start(executable, "slow-length"), "start child whose backend reports length after STOP");
    if (!stop_length.next_line(line, std::chrono::seconds(5)) ||
        !stop_length.next_line(line, std::chrono::seconds(5))) {
        check(false, "length-after-STOP child reaches READY", "missing startup lines");
        return;
    }
    check(stop_length.send("GEN 20 9\n"), "send request for STOP finish normalization");
    saw_token = false;
    while (stop_length.next_line(line, std::chrono::seconds(5))) {
        if (line.starts_with("T ")) {
            saw_token = true;
            break;
        }
    }
    check(saw_token, "length-after-STOP child begins decoding");
    check(stop_length.send("STOP\n"), "send STOP while length-after-STOP backend is active");
    std::vector<std::string> stop_length_transcript;
    std::string stop_length_done;
    check(wait_until_done(stop_length, stop_length_transcript, stop_length_done) &&
              stop_length_done.starts_with("DONE 1 1 4.0 6.0 cancel "),
          "the host converts a backend's length result to cancel when STOP wins completion");
    check(stop_length.send("QUIT\n"), "send QUIT after STOP finish normalization");
    check(stop_length.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "length-after-STOP child exits cleanly");

    ChildEngine eof_stopping;
    check(eof_stopping.start(executable, "slow"), "start child for EOF cancellation");
    if (!eof_stopping.next_line(line, std::chrono::seconds(5)) ||
        !eof_stopping.next_line(line, std::chrono::seconds(5))) {
        check(false, "EOF cancellation child reaches READY", "missing startup lines");
        return;
    }
    check(eof_stopping.send("GEN 20 11\n"), "send request before closing stdin");
    saw_token = false;
    while (eof_stopping.next_line(line, std::chrono::seconds(5))) {
        if (line.starts_with("T ")) {
            saw_token = true;
            break;
        }
    }
    check(saw_token, "EOF cancellation child begins decoding");
    eof_stopping.close_input();
    std::vector<std::string> eof_transcript;
    std::string eof_done;
    check(wait_until_done(eof_stopping, eof_transcript, eof_done) &&
              eof_done.starts_with("DONE 1 1 4.0 6.0 cancel "),
          "stdin EOF cancels an active request at a window boundary");
    check(eof_stopping.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "active-request EOF exits cleanly");

    ChildEngine eof_with_quit;
    check(eof_with_quit.start(executable, "slow"), "start child for EOF after a queued QUIT");
    if (!eof_with_quit.next_line(line, std::chrono::seconds(5)) ||
        !eof_with_quit.next_line(line, std::chrono::seconds(5))) {
        check(false, "EOF-with-QUIT child reaches READY", "missing startup lines");
        return;
    }
    check(eof_with_quit.send("GEN 20 12\n"), "send request before EOF-with-QUIT case");
    saw_token = false;
    while (eof_with_quit.next_line(line, std::chrono::seconds(5))) {
        if (line.starts_with("T ")) {
            saw_token = true;
            break;
        }
    }
    check(saw_token, "EOF-with-QUIT child begins decoding");
    check(eof_with_quit.send("QUIT\n"), "queue QUIT while generation is active");
    eof_with_quit.close_input();
    std::vector<std::string> eof_with_quit_transcript;
    std::string eof_with_quit_done;
    check(wait_until_done(eof_with_quit, eof_with_quit_transcript, eof_with_quit_done) &&
              eof_with_quit_done.starts_with("DONE 1 1 4.0 6.0 cancel "),
          "EOF cancels active generation even when QUIT is queued");
    check(eof_with_quit.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "EOF-with-QUIT child exits cleanly");

    ChildEngine immediate_eof;
    check(immediate_eof.start(executable, "slow"), "start child for EOF immediately after a queued request");
    if (!immediate_eof.next_line(line, std::chrono::seconds(5)) ||
        !immediate_eof.next_line(line, std::chrono::seconds(5))) {
        check(false, "immediate-EOF child reaches READY", "missing startup lines");
        return;
    }
    check(immediate_eof.send("GEN 20 13\n"), "queue request immediately before EOF");
    immediate_eof.close_input();
    std::vector<std::string> immediate_eof_transcript;
    std::string immediate_eof_done;
    check(wait_until_done(immediate_eof, immediate_eof_transcript, immediate_eof_done) &&
              immediate_eof_done.starts_with("DONE ") && immediate_eof_done.ends_with(" cancel 0 0 0 3 4 1 2 0.5"),
          "EOF deterministically cancels a request sent immediately before close");
    check(immediate_eof.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "immediate queued-request EOF exits cleanly");

    ChildEngine eof_cr;
    check(eof_cr.start(executable, "slow"), "start child for CR-terminated EOF request");
    if (!eof_cr.next_line(line, std::chrono::seconds(5)) ||
        !eof_cr.next_line(line, std::chrono::seconds(5))) {
        check(false, "CR-terminated EOF child reaches READY", "missing startup lines");
        return;
    }
    check(eof_cr.send("GEN 2 14\r"), "send a request with a trailing CR and no newline");
    eof_cr.close_input();
    std::vector<std::string> eof_cr_transcript;
    std::string eof_cr_done;
    check(wait_until_done(eof_cr, eof_cr_transcript, eof_cr_done) &&
              eof_cr_done.starts_with("DONE 0 1 4.0 6.0 cancel "),
          "EOF strips CR before selecting the queued request");
    check(eof_cr.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "CR-terminated EOF request exits cleanly");
}

void test_input_limits(const std::filesystem::path& executable) {
    std::string line_too_long(16 * 1024 * 1024, 'x');
    line_too_long.append("\nQUIT\n");
    ChildEngine long_line_child;
    check(long_line_child.start(executable, "quick"), "start child for input line limit");
    std::string line;
    if (!long_line_child.next_line(line, std::chrono::seconds(5)) ||
        !long_line_child.next_line(line, std::chrono::seconds(5))) {
        check(false, "long-line child reaches READY", "missing startup lines");
        return;
    }
    check(long_line_child.send(line_too_long), "send a request line above the byte limit");
    bool saw_line_error = false;
    while (long_line_child.next_line(line, std::chrono::seconds(5))) {
        if (line == "ERR request line exceeds the 4194304-byte limit") {
            saw_line_error = true;
            break;
        }
    }
    check(saw_line_error, "oversized input lines receive a bounded error");
    int exit_code = -1;
    check(long_line_child.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 0,
          "line-limit child exits on the following QUIT");

    ChildEngine full_queue_child;
    check(full_queue_child.start(executable, "quick"), "start child for queued command limit");
    if (!full_queue_child.next_line(line, std::chrono::seconds(5)) ||
        !full_queue_child.next_line(line, std::chrono::seconds(5))) {
        check(false, "full-queue child reaches READY", "missing startup lines");
        return;
    }
    std::string many_lines;
    for (std::size_t index = 0; index < 1'025; ++index) many_lines.append("WHAT\n");

    ChildEngine active_queue_child;
    check(active_queue_child.start(executable, "slow"), "start child for queue overflow during generation");
    if (!active_queue_child.next_line(line, std::chrono::seconds(5)) ||
        !active_queue_child.next_line(line, std::chrono::seconds(5))) {
        check(false, "active queue-limit child reaches READY", "missing startup lines");
        return;
    }
    check(active_queue_child.send("GEN 20 7\n"), "start a request before overflowing its input queue");
    bool saw_active_token = false;
    while (active_queue_child.next_line(line, std::chrono::seconds(5))) {
        if (line.starts_with("T ")) {
            saw_active_token = true;
            break;
        }
    }
    check(saw_active_token, "active queue-limit request begins decoding");
    check(active_queue_child.send(many_lines), "send excess commands while generation is active");
    std::vector<std::string> active_queue_transcript;
    std::string active_queue_done;
    const bool active_cancelled =
        wait_until_done(active_queue_child, active_queue_transcript, active_queue_done) &&
        active_queue_done.starts_with("DONE ") && active_queue_done.ends_with(" cancel 0 0 0 3 4 1 2 0.5");
    check(active_cancelled, "queue overflow cancels active generation before reporting the limit error");
    const bool active_queue_error =
        active_queue_child.next_line(line, std::chrono::seconds(5)) &&
        line == "ERR serve input exceeded the queued-command limit";
    check(active_queue_error, "queue overflow discards queued commands and preserves one bounded error");
    int active_queue_exit = -1;
    check(active_queue_child.wait_for_exit(std::chrono::seconds(5), active_queue_exit) && active_queue_exit == 2,
          "active queue overflow exits with the input protocol status");

    check(full_queue_child.send(many_lines), "send more commands than the queue can hold");
    bool saw_queue_error = false;
    while (full_queue_child.next_line(line, std::chrono::seconds(5))) {
        if (line == "ERR serve input exceeded the queued-command limit") {
            saw_queue_error = true;
            break;
        }
    }
    check(saw_queue_error, "command queue limit fails closed with a bounded error");
    check(full_queue_child.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 2,
          "queue-limit child exits with the input protocol status");

    ChildEngine full_queue_eof_child;
    check(full_queue_eof_child.start(executable, "quick"), "start child for queued limit followed by EOF");
    if (!full_queue_eof_child.next_line(line, std::chrono::seconds(5)) ||
        !full_queue_eof_child.next_line(line, std::chrono::seconds(5))) {
        check(false, "queue-limit EOF child reaches READY", "missing startup lines");
        return;
    }
    check(full_queue_eof_child.send(many_lines), "send more queued commands than allowed before EOF");
    full_queue_eof_child.close_input();
    saw_queue_error = false;
    while (full_queue_eof_child.next_line(line, std::chrono::seconds(5))) {
        if (line == "ERR serve input exceeded the queued-command limit") {
            saw_queue_error = true;
            break;
        }
    }
    check(saw_queue_error, "EOF preserves an already-detected queue-limit error");
    check(full_queue_eof_child.wait_for_exit(std::chrono::seconds(5), exit_code) && exit_code == 2,
          "queue-limit EOF child exits with the input protocol status");
}

void test_request_parser() {
    const ServeParseResult image =
        parse_serve_request("GENI 4 temperature=0.7 unknown=x pcie_frac=1.4 spec_min_p=-2 "
                            "/tmp/with spaces/vision.sve 248056,7");
    check(image.kind == ServeParseKind::Request && image.request.image,
          "GENI with options parses as an image request");
    check(image.request.embeddings_path == "/tmp/with spaces/vision.sve",
          "GENI preserves spaces in the embeddings path");
    check(image.request.token_ids == std::vector<std::int32_t>({248056, 7}),
          "GENI parses the final id list");
    check(image.request.sampling.temperature == 0.7f &&
              image.request.sampling.pcie_frac == 1.0 && image.request.sampling.spec_min_p == 0.0,
          "known keys parse and bounded request settings clamp");
    check(image.request.pcie_frac_is_set && image.request.spec_min_p_is_set,
          "explicit tuning keys override their INFO defaults");
    const ServeParseResult malformed = parse_serve_request("GEN 2 1,not-an-id");
    check(malformed.kind == ServeParseKind::BadRequest &&
              malformed.error == "invalid token id: expected an integer in [0, 2147483647]",
          "malformed token lists return the upstream diagnostic");
    const ServeParseResult empty = parse_serve_request("GEN 2 temperature=0.4");
    check(empty.kind == ServeParseKind::BadRequest && empty.error == "token list was empty",
          "empty token lists are rejected");
    const ServeParseResult bad_float = parse_serve_request("GEN 2 temperature=nan 1");
    check(bad_float.kind == ServeParseKind::BadRequest && bad_float.error == "temperature",
          "non-finite sampling values are rejected");
    const ServeParseResult bad_top_k = parse_serve_request("GEN 2 temperature=0.4 top_k=65 1");
    check(bad_top_k.kind == ServeParseKind::BadRequest && bad_top_k.error == "top_k",
          "sampled top_k is bounded to the sampler's supported range");
    const ServeParseResult unknown_key = parse_serve_request("GEN 1 future_key=nan 1");
    check(unknown_key.kind == ServeParseKind::Request,
          "unknown key=value settings remain silently ignored");
    std::string over_context = "GEN 1 ";
    for (std::int64_t index = 0; index <= 262'144; ++index) {
        if (index != 0) over_context.push_back(',');
        over_context.push_back('1');
    }
    const ServeParseResult over_context_result = parse_serve_request(over_context);
    check(over_context_result.kind == ServeParseKind::BadRequest &&
              over_context_result.error == "prompt exceeds the maximum serve context",
          "request parsing caps prompt IDs before they can grow without bound");
    check(parse_serve_request("QUIT").kind == ServeParseKind::Quit, "QUIT is recognized");
    check(parse_serve_request("STOP").kind == ServeParseKind::InvalidCommand,
          "STOP is reserved for the asynchronous reader");
}

void test_sve1_loader() {
    const std::array<std::int32_t, 5> valid_header {0x31455653, 2, 2, 1, kServeImageEmbeddingWidth};
    const std::string valid_path = make_sve_file(valid_header);
    check(!valid_path.empty(), "create a temporary SVE1 record with a spaced path");
    if (valid_path.empty()) return;

    const std::string missing_path = "/tmp/strata no such SVE1 file " + std::to_string(getpid());
    ServeRequest missing;
    missing.embeddings_path = missing_path;
    missing.token_ids = {7};
    std::string error;
    check(!load_serve_image_embeddings(missing, 32, error) && error == "cannot open " + missing_path,
          "missing image file reports the path");

    const std::string bad_path = make_sve_file({0x12345678, 2, 2, 1, kServeImageEmbeddingWidth});
    ServeRequest bad;
    bad.embeddings_path = bad_path;
    bad.token_ids = {kServeImagePadToken, kServeImagePadToken, 7};
    check(!bad_path.empty() && !load_serve_image_embeddings(bad, 32, error) &&
              error == "bad embeddings file (expected strata-vision records of width 2560)",
          "invalid SVE1 header is rejected");

    const std::string short_path = make_sve_file(valid_header, 1);
    ServeRequest short_file;
    short_file.embeddings_path = short_path;
    short_file.token_ids = {kServeImagePadToken, kServeImagePadToken, 7};
    check(!short_path.empty() && !load_serve_image_embeddings(short_file, 32, error) &&
              error == "short embeddings file",
          "truncated SVE1 payload is rejected");

    for (std::size_t trailing_bytes = 1; trailing_bytes <= 3; ++trailing_bytes) {
        const std::string partial_path = make_sve_file(valid_header, 5'120, trailing_bytes);
        ServeRequest partial;
        partial.embeddings_path = partial_path;
        partial.token_ids = {kServeImagePadToken, kServeImagePadToken, 7};
        check(!partial_path.empty() && !load_serve_image_embeddings(partial, 32, error) &&
                  error == "bad embeddings file (expected strata-vision records of width 2560)",
              "partial SVE1 header of " + std::to_string(trailing_bytes) + " byte(s) is rejected");
        if (!partial_path.empty()) (void) ::unlink(partial_path.c_str());
    }

    const std::string huge_path =
        make_sve_file({0x31455653, std::numeric_limits<std::int32_t>::max(),
                       std::numeric_limits<std::int32_t>::max(), 1, kServeImageEmbeddingWidth}, 0);
    ServeRequest huge;
    huge.embeddings_path = huge_path;
    huge.token_ids = {kServeImagePadToken, kServeImagePadToken, 7};
    check(!huge_path.empty() && !load_serve_image_embeddings(huge, 32, error) &&
              error == "image 1 has 2147483647 rows but fewer <|image_pad|> tokens",
          "oversized SVE1 dimensions are rejected before payload allocation");

    constexpr std::int32_t over_embedding_budget_rows = 26'215;
    const std::string over_budget_path =
        make_sve_file({0x31455653, over_embedding_budget_rows, over_embedding_budget_rows, 1,
                       kServeImageEmbeddingWidth}, 0);
    ServeRequest over_budget;
    over_budget.embeddings_path = over_budget_path;
    over_budget.token_ids.assign(over_embedding_budget_rows, kServeImagePadToken);
    check(!over_budget_path.empty() && !load_serve_image_embeddings(over_budget, 262'144, error) &&
              error == "image embeddings exceed the 268435456-byte per-request limit",
          "aggregate vision payload limit rejects large aligned image grids before allocation");

    const std::string multi_path = make_sve_records_file({
        {0x31455653, 4, 2, 2, kServeImageEmbeddingWidth},
        {0x31455653, 2, 2, 1, kServeImageEmbeddingWidth},
    });
    ServeRequest multi;
    multi.embeddings_path = multi_path;
    multi.token_ids = {kServeImagePadToken, kServeImagePadToken, kServeImagePadToken, kServeImagePadToken,
                       7, kServeImagePadToken, kServeImagePadToken, 8};
    check(!multi_path.empty() && load_serve_image_embeddings(multi, 32, error) &&
              multi.images.size() == 2 &&
              multi.mrope_positions == std::vector<std::array<std::int32_t, 3>>{
                                           {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {2, 2, 2},
                                           {3, 3, 3}, {3, 3, 4}, {5, 5, 5}},
          "2x2 and 2x1 image records preserve every two-dimensional M-RoPE position");
    (void) ::unlink(multi_path.c_str());

    ServeRequest over_context;
    over_context.embeddings_path = valid_path;
    over_context.token_ids = {kServeImagePadToken, kServeImagePadToken};
    check(!load_serve_image_embeddings(over_context, 1, error) &&
              error == "prompt exceeds the maximum serve context",
          "image loading rejects prompts above the bounded context before allocating");

    ServeRequest extra_prompt;
    extra_prompt.embeddings_path = valid_path;
    extra_prompt.token_ids = {kServeImagePadToken, kServeImagePadToken, 7,
                              kServeImagePadToken, kServeImagePadToken, 8};
    check(!load_serve_image_embeddings(extra_prompt, 32, error) &&
              error == "the prompt has more images than the embeddings file",
          "each image-pad run requires a matching record");

    ServeRequest too_few_rows;
    too_few_rows.embeddings_path = valid_path;
    too_few_rows.token_ids = {kServeImagePadToken, 7};
    check(!load_serve_image_embeddings(too_few_rows, 32, error) &&
              error == "image 1 has 2 rows but fewer <|image_pad|> tokens",
          "record row count cannot exceed its image-pad run");

    ServeRequest extra_record;
    extra_record.embeddings_path = valid_path;
    extra_record.token_ids = {7};
    check(!load_serve_image_embeddings(extra_record, 32, error) &&
              error == "the embeddings file has more images than the prompt",
          "unmatched image records are rejected");

    ServeRequest trailing_image;
    trailing_image.embeddings_path = valid_path;
    trailing_image.token_ids = {kServeImagePadToken, kServeImagePadToken};
    check(!load_serve_image_embeddings(trailing_image, 32, error) &&
              error == "the prompt cannot end in an image",
          "prompt ending in image rows is rejected");

    (void) ::unlink(valid_path.c_str());
    if (!bad_path.empty()) (void) ::unlink(bad_path.c_str());
    if (!short_path.empty()) (void) ::unlink(short_path.c_str());
    if (!huge_path.empty()) (void) ::unlink(huge_path.c_str());
    if (!over_budget_path.empty()) (void) ::unlink(over_budget_path.c_str());
}

void test_strict_request_grammar() {
    struct Malformed {
        std::string_view line;
        std::string_view error;
    };
    constexpr std::string_view id_error = "invalid token id: expected an integer in [0, 2147483647]";
    const std::array malformed_requests {
        Malformed {"GEN 2 1,,2", id_error},
        Malformed {"GEN 2 1 2", id_error},
        Malformed {"GEN 2 1,2,", id_error},
        Malformed {"GEN 2 ,1", id_error},
        Malformed {"GEN 2 1,2 ", id_error},
        Malformed {"GEN 2 1 , 2", id_error},
        Malformed {"GEN 2 +1", id_error},
        Malformed {"GEN 2 temperature=-0.5 1", "temperature"},
        Malformed {"GEN 2 min_p=1.5 1", "min_p"},
        Malformed {"GEN 2 min_p=-0.1 1", "min_p"},
    };
    for (const auto& test_case : malformed_requests) {
        const ServeParseResult result = parse_serve_request(test_case.line);
        check(result.kind == ServeParseKind::BadRequest && result.error == test_case.error,
              "malformed request is rejected: " + std::string(test_case.line));
    }
    const ServeParseResult strict = parse_serve_request("GEN 2 min_p=0.5 temperature=0 1,2");
    check(strict.kind == ServeParseKind::Request && strict.request.token_ids == std::vector<std::int32_t>({1, 2}) &&
              strict.request.sampling.min_p == 0.5f && strict.request.sampling.temperature == 0.0f,
          "a comma-separated id list after in-range sampling keys still parses");
}

void test_image_runs_match_records() {
    const std::string two_records = make_sve_records_file({
        {0x31455653, 2, 2, 1, kServeImageEmbeddingWidth},
        {0x31455653, 2, 2, 1, kServeImageEmbeddingWidth},
    });
    const std::string one_record = make_sve_records_file({
        {0x31455653, 2, 2, 1, kServeImageEmbeddingWidth},
    });
    check(!two_records.empty() && !one_record.empty(), "create SVE1 records for run matching");
    std::string error;

    ServeRequest split;
    split.embeddings_path = two_records;
    split.token_ids = {kServeImagePadToken, kServeImagePadToken, kServeImagePadToken, kServeImagePadToken, 7};
    check(!load_serve_image_embeddings(split, 32, error) && error == "image 1 has 2 rows but more <|image_pad|> tokens",
          "one run of pad tokens cannot be shared by two records");

    ServeRequest longer;
    longer.embeddings_path = one_record;
    longer.token_ids = {kServeImagePadToken, kServeImagePadToken, kServeImagePadToken, kServeImagePadToken, 7};
    check(!load_serve_image_embeddings(longer, 32, error) &&
              error == "image 1 has 2 rows but more <|image_pad|> tokens",
          "a run longer than its record is rejected rather than handed to a later record");

    ServeRequest exact;
    exact.embeddings_path = one_record;
    exact.token_ids = {kServeImagePadToken, kServeImagePadToken, 7};
    check(load_serve_image_embeddings(exact, 32, error) && exact.images.size() == 1 &&
              exact.mrope_positions == std::vector<std::array<std::int32_t, 3>>{{0, 0, 0}, {0, 0, 1}, {2, 2, 2}},
          "a run equal to its record's rows loads with the two-dimensional M-RoPE positions");

    if (!two_records.empty()) (void) ::unlink(two_records.c_str());
    if (!one_record.empty()) (void) ::unlink(one_record.c_str());
}

int child_main(std::string_view mode) {
    const bool slow = mode == "slow" || mode == "slow-length";
    const bool report_length_after_stop = mode == "slow-length";
    FakeWindow backend(false, slow ? std::chrono::milliseconds(25) : std::chrono::milliseconds(0),
                       report_length_after_stop);
    ServeLoopOptions options;
    return run_serve_loop(backend, options);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--serve-child") return child_main(argv[2]);

    test_request_parser();
    test_strict_request_grammar();
    test_image_runs_match_records();
    run_fixture("basic", false);
    run_fixture("image", true);
    run_fixture("gpu_error", false, true);
    test_sve1_loader();
    test_backend_protocol_limits();
    test_preclosed_eof();
    test_child_lifecycle(std::filesystem::absolute(argv[0]));
    test_input_limits(std::filesystem::absolute(argv[0]));
    std::cout << "serve_transcript_test: " << (failures == 0 ? "OK" : "FAILED") << '\n';
    return failures != 0;
}
