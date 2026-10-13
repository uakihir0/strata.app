// src/metal/program/serve.cpp - the host-side line protocol (docs/mac/06, section 5).
#include "strata/metal/serve.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <mutex>
#include <new>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace strata::metal {
namespace {

constexpr std::string_view kExpectedRequest =
    "expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>";
constexpr std::int64_t kServeMaximumContext = 262'144;
constexpr std::uint64_t kServeMaximumImageEmbeddingBytes = 256ull * 1024 * 1024;
constexpr std::size_t kServeMaximumInputLineBytes = 4 * 1024 * 1024;
constexpr std::size_t kServeMaximumQueuedBytes = 8 * 1024 * 1024;
constexpr std::size_t kServeMaximumQueuedLines = 1'024;
constexpr std::size_t kServeMaximumDrainBytes = 8 * 1024 * 1024;
constexpr std::size_t kServeReadChunkBytes = 16 * 1024;
constexpr std::size_t kServeMaximumCompletionDrainBytes = 64 * 1024 * 1024;

enum class InputError : std::uint8_t { None, LineTooLong, QueueLimit, CompletionDrainLimit };

enum class DrainResult : std::uint8_t { Idle, LimitReached, Ended, Failed };

bool parse_i64(std::string_view text, std::int64_t& value) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

// The id list is `<id>(,<id>)*`: base-10 integers in [0, 2147483647] with single commas and no spaces, as the
// server writes it. Empty fields, spaces and a trailing comma are malformed, not skipped.
bool parse_token_list(std::string_view text, std::vector<std::int32_t>& tokens, std::string& error) {
    tokens.clear();
    if (text.empty()) {
        error = "token list was empty";
        return false;
    }
    std::size_t begin = 0;
    for (;;) {
        const std::size_t comma = text.find(',', begin);
        const std::string_view item =
            text.substr(begin, comma == std::string_view::npos ? std::string_view::npos : comma - begin);
        std::int32_t token = 0;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), token);
        if (item.empty() || parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() || token < 0) {
            error = "invalid token id: expected an integer in [0, 2147483647]";
            tokens.clear();
            return false;
        }
        if (tokens.size() >= static_cast<std::size_t>(kServeMaximumContext)) {
            error = "prompt exceeds the maximum serve context";
            tokens.clear();
            return false;
        }
        tokens.push_back(token);
        if (comma == std::string_view::npos) return true;
        begin = comma + 1;
    }
}

std::string_view trim_spaces(std::string_view text) {
    while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
    while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
    return text;
}

bool parse_float_value(std::string_view text, float& parsed_value) {
    if (text.empty()) return false;
    const std::string value(text);
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value.c_str(), &end);
    if (errno == ERANGE || end == value.c_str() || end == nullptr || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    parsed_value = parsed;
    return true;
}

bool parse_i32(std::string_view text, std::int32_t& value) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_u64(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

std::string protocol_atom(std::string_view value) {
    std::string result(value);
    for (char& ch : result) {
        const auto byte = static_cast<unsigned char>(ch);
        if (byte <= 0x20 || byte == 0x7f) ch = '_';
    }
    return result;
}

bool write_line(std::FILE* output, std::string_view line) {
    if (std::fwrite(line.data(), 1, line.size(), output) != line.size() ||
        std::fputc('\n', output) == EOF) {
        return false;
    }
    return std::fflush(output) == 0;
}

void write_error(std::FILE* output, std::string_view message) {
    std::string line = "ERR ";
    line.append(message);
    (void) write_line(output, line);
}

void write_request_error(std::FILE* output, std::string_view message) {
    std::string line = "bad request: ";
    line.append(message);
    write_error(output, line);
}

class SignalPolicy {
public:
    bool install() {
        struct sigaction ignored {};
        ignored.sa_handler = SIG_IGN;
        sigemptyset(&ignored.sa_mask);
        if (sigaction(SIGINT, &ignored, &old_int_) != 0) return false;
        int_installed_ = true;
        if (sigaction(SIGHUP, &ignored, &old_hup_) != 0) {
            restore();
            return false;
        }
        hup_installed_ = true;
        return true;
    }

    ~SignalPolicy() { restore(); }

private:
    void restore() noexcept {
        if (hup_installed_) {
            (void) sigaction(SIGHUP, &old_hup_, nullptr);
            hup_installed_ = false;
        }
        if (int_installed_) {
            (void) sigaction(SIGINT, &old_int_, nullptr);
            int_installed_ = false;
        }
    }

    struct sigaction old_int_ {};
    struct sigaction old_hup_ {};
    bool int_installed_ = false;
    bool hup_installed_ = false;
};

class InputReader {
public:
    InputReader(int descriptor, std::atomic<bool>& stop_requested, std::atomic<bool>& request_active)
        : descriptor_(descriptor), stop_requested_(stop_requested), request_active_(request_active) {}

    bool start() {
        try {
            thread_ = std::thread([this] { read_loop(); });
        } catch (const std::system_error&) {
            return false;
        }
        return true;
    }

    ~InputReader() { stop(); }

    bool next(std::string& line, InputError& error) {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return !lines_.empty() || eof_ || shutting_down_; });
        if (lines_.empty()) return false;
        PendingLine pending = std::move(lines_.front());
        lines_.pop_front();
        queued_bytes_ -= queue_cost(pending);
        const bool generation_request = is_request(pending);
        line = std::move(pending.text);
        error = pending.error;
        stop_requested_.store(pending.cancel_on_start);
        request_active_.store(generation_request);
        lock.unlock();
        return true;
    }

    bool request_finished() noexcept {
        std::unique_lock lock(mutex_);
        if (drain_available_locked(kServeMaximumCompletionDrainBytes, true) == DrainResult::LimitReached) {
            fail_input_locked(InputError::CompletionDrainLimit);
        }
        const bool was_stopped = stop_requested_.load();
        request_active_.store(false);
        lock.unlock();
        ready_.notify_all();
        return was_stopped;
    }

    void stop() noexcept {
        shutting_down_.store(true);
        ready_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    struct PendingLine {
        std::string text;
        bool cancel_on_start = false;
        InputError error = InputError::None;
    };

    static std::size_t queue_cost(const PendingLine& line) noexcept {
        return sizeof(PendingLine) + 64 + line.text.size();
    }

    static bool is_request(const PendingLine& line) {
        return line.error == InputError::None &&
               (line.text.starts_with("GEN ") || line.text.starts_with("GENI "));
    }

    void clear_queue_locked() {
        lines_.clear();
        queued_bytes_ = 0;
    }

    void fail_input_locked(InputError error) {
        if (request_active_.load()) stop_requested_.store(true);
        clear_queue_locked();
        PendingLine failure {{}, false, error};
        queued_bytes_ = queue_cost(failure);
        lines_.push_back(std::move(failure));
        eof_ = true;
    }

    void stop_queued_request_locked() {
        for (PendingLine& pending : lines_) {
            if (is_request(pending)) {
                pending.cancel_on_start = true;
                return;
            }
        }
    }

    bool enqueue_line_locked(PendingLine line) {
        if (line.error == InputError::None && !line.text.empty() && line.text.back() == '\r') {
            line.text.pop_back();
        }
        if (line.error == InputError::None && line.text == "STOP") {
            if (request_active_.load()) stop_requested_.store(true);
            else stop_queued_request_locked();
            return true;
        }
        const std::size_t cost = queue_cost(line);
        if (lines_.size() >= kServeMaximumQueuedLines || cost > kServeMaximumQueuedBytes - queued_bytes_) {
            fail_input_locked(InputError::QueueLimit);
            return false;
        }
        queued_bytes_ += cost;
        lines_.push_back(std::move(line));
        return true;
    }

    void finish_input_locked() {
        if (eof_) return;
        if (dropping_long_line_) {
            dropping_long_line_ = false;
            partial_.clear();
            if (!enqueue_line_locked({{}, false, InputError::LineTooLong})) return;
        } else if (!partial_.empty()) {
            PendingLine last {std::move(partial_), false, InputError::None};
            partial_.clear();
            if (!enqueue_line_locked(std::move(last))) return;
        }
        if (request_active_.load()) {
            stop_requested_.store(true);
            clear_queue_locked();
        } else {
            PendingLine* terminal = nullptr;
            for (PendingLine& queued : lines_) {
                if (is_request(queued) || (queued.error == InputError::None && queued.text == "QUIT")) {
                    terminal = &queued;
                    break;
                }
            }
            if (terminal != nullptr) {
                PendingLine pending = std::move(*terminal);
                clear_queue_locked();
                if (is_request(pending)) pending.cancel_on_start = true;
                queued_bytes_ = queue_cost(pending);
                lines_.push_back(std::move(pending));
            }
        }
        eof_ = true;
    }

    bool process_bytes_locked(const char* bytes, std::size_t count) {
        for (std::size_t index = 0; index < count; ++index) {
            const char byte = bytes[index];
            if (byte == '\n') {
                PendingLine line;
                if (dropping_long_line_) {
                    line.error = InputError::LineTooLong;
                    dropping_long_line_ = false;
                    partial_.clear();
                } else {
                    line.text = std::move(partial_);
                    partial_.clear();
                }
                if (!enqueue_line_locked(std::move(line))) return false;
            } else if (!dropping_long_line_) {
                if (partial_.size() == kServeMaximumInputLineBytes) {
                    partial_.clear();
                    dropping_long_line_ = true;
                } else {
                    partial_.push_back(byte);
                }
            }
        }
        return true;
    }

    DrainResult drain_available_locked(std::size_t maximum_bytes, bool probe_limit = false) {
        std::array<char, kServeReadChunkBytes> chunk {};
        std::size_t drained = 0;
        while (!eof_ && !shutting_down_.load() && drained < maximum_bytes) {
            pollfd descriptor {descriptor_, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, 0);
            if (ready < 0) {
                if (errno == EINTR) continue;
                finish_input_locked();
                return DrainResult::Ended;
            }
            if (ready == 0) return DrainResult::Idle;
            if ((descriptor.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) == 0) return DrainResult::Idle;
            const std::size_t read_size = std::min(chunk.size(), maximum_bytes - drained);
            const ssize_t count = ::read(descriptor_, chunk.data(), read_size);
            if (count == 0) {
                finish_input_locked();
                return DrainResult::Ended;
            }
            if (count < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) return DrainResult::Idle;
                finish_input_locked();
                return DrainResult::Ended;
            }
            const auto amount = static_cast<std::size_t>(count);
            drained += amount;
            if (!process_bytes_locked(chunk.data(), amount)) return DrainResult::Failed;
        }
        if (eof_) return DrainResult::Ended;
        if (shutting_down_.load() || drained < maximum_bytes) return DrainResult::Idle;
        if (!probe_limit) return DrainResult::LimitReached;

        for (;;) {
            pollfd descriptor {descriptor_, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, 0);
            if (ready < 0) {
                if (errno == EINTR) continue;
                finish_input_locked();
                return DrainResult::Ended;
            }
            if (ready == 0) return DrainResult::Idle;
            if ((descriptor.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) == 0) return DrainResult::Idle;
            char extra = '\0';
            const ssize_t count = ::read(descriptor_, &extra, 1);
            if (count == 0) {
                finish_input_locked();
                return DrainResult::Ended;
            }
            if (count < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) return DrainResult::Idle;
                finish_input_locked();
                return DrainResult::Ended;
            }
            return DrainResult::LimitReached;
        }
    }

    void read_loop() {
        while (!shutting_down_.load()) {
            pollfd descriptor {descriptor_, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, 100);
            if (ready < 0) {
                if (errno == EINTR) continue;
                {
                    std::lock_guard lock(mutex_);
                    finish_input_locked();
                }
                ready_.notify_all();
                return;
            }
            if (ready == 0) continue;
            bool input_ended = false;
            {
                std::unique_lock lock(mutex_);
                (void) drain_available_locked(kServeMaximumDrainBytes);
                input_ended = eof_;
            }
            ready_.notify_all();
            if (input_ended) return;
        }
    }

    int descriptor_;
    std::atomic<bool>& stop_requested_;
    std::atomic<bool>& request_active_;
    std::atomic<bool> shutting_down_ {false};
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<PendingLine> lines_;
    std::size_t queued_bytes_ = 0;
    std::string partial_;
    bool dropping_long_line_ = false;
    bool eof_ = false;
    std::thread thread_;
};

enum class CallbackStage : std::uint8_t { BeforeResume, Prompt, Decode, Complete, Failed };

struct CallbackState {
    std::FILE* output = nullptr;
    std::int64_t prompt_tokens = 0;
    std::int64_t max_new = 0;
    bool stop_eos = false;
    std::vector<std::int32_t> eos_ids;
    CallbackStage stage = CallbackStage::BeforeResume;
    std::int64_t resume_count = -1;
    std::int64_t emitted_tokens = 0;
    std::int64_t progress_done = 0;
    std::int64_t progress_total = -1;
    std::int32_t last_token = -1;
    bool progress_seen = false;
    bool eos_seen = false;
    std::string error;

    bool resume(std::int64_t reused) {
        if (stage != CallbackStage::BeforeResume || reused < 0 || reused > prompt_tokens) {
            return fail("backend emitted RESUME out of order or with an invalid count");
        }
        if (std::fprintf(output, "RESUME %lld\n", static_cast<long long>(reused)) < 0 ||
            std::fflush(output) != 0) {
            return fail("could not write RESUME to stdout");
        }
        resume_count = reused;
        stage = CallbackStage::Prompt;
        return true;
    }

    bool progress(std::int64_t done, std::int64_t total, double ms, double tokens_per_second) {
        if (stage != CallbackStage::Prompt || done < resume_count || total != prompt_tokens ||
            done > total ||
            (progress_total >= 0 && (total != progress_total || done < progress_done)) ||
            !std::isfinite(ms) ||
            !std::isfinite(tokens_per_second) || ms < 0.0 || tokens_per_second < 0.0) {
            return fail("backend emitted invalid PP progress");
        }
        if (std::fprintf(output, "PP %lld %lld %.0f %.1f\n", static_cast<long long>(done),
                         static_cast<long long>(total), ms, tokens_per_second) < 0 ||
            std::fflush(output) != 0) {
            return fail("could not write PP to stdout");
        }
        progress_done = done;
        progress_total = total;
        progress_seen = true;
        return true;
    }

    bool reused(std::int64_t count) {
        if (stage != CallbackStage::Prompt || count != resume_count ||
            (progress_seen && progress_done != prompt_tokens)) {
            return fail("backend emitted REUSED out of order or inconsistent with RESUME");
        }
        if (std::fprintf(output, "REUSED %lld\n", static_cast<long long>(count)) < 0 ||
            std::fflush(output) != 0) {
            return fail("could not write REUSED to stdout");
        }
        stage = CallbackStage::Decode;
        return true;
    }

    bool token(std::int32_t id) {
        if (stage != CallbackStage::Decode || id < 0 || id >= kServeVocabularySize || eos_seen) {
            return fail("backend emitted a token outside the vocabulary or out of order");
        }
        if (emitted_tokens >= max_new) {
            return fail("backend emitted more tokens than max_new");
        }
        if (std::fprintf(output, "T %d\n", id) < 0 || std::fflush(output) != 0) {
            return fail("could not write T to stdout");
        }
        ++emitted_tokens;
        last_token = id;
        eos_seen = stop_eos && std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end();
        return true;
    }

    bool fail(std::string message) {
        if (error.empty()) error = std::move(message);
        stage = CallbackStage::Failed;
        return false;
    }
};

bool valid_outcome(const ServeOutcome& outcome) {
    return (outcome.finish == "stop" || outcome.finish == "cancel" || outcome.finish == "length") &&
           std::isfinite(outcome.prompt_ms) && outcome.prompt_ms >= 0.0 &&
           std::isfinite(outcome.decode_ms) && outcome.decode_ms >= 0.0 &&
           outcome.drafts_accepted >= 0 && outcome.drafts_offered >= outcome.drafts_accepted &&
           outcome.checkpoints >= 0 && outcome.hits >= 0 && outcome.lookups >= outcome.hits &&
           outcome.ram_blobs >= 0 && outcome.file_blobs >= 0 && std::isfinite(outcome.file_mb) &&
           outcome.file_mb >= 0.0 && std::isfinite(outcome.resident_ram_gib) &&
           outcome.resident_ram_gib >= 0.0 && outcome.exchanges >= 0 &&
           outcome.cumulative_file_reads >= 0 && outcome.cumulative_ram_blobs >= 0 &&
           outcome.cumulative_expert_file_blobs >= 0 &&
           std::isfinite(outcome.cumulative_expert_file_mb) && outcome.cumulative_expert_file_mb >= 0.0;
}

std::string request_context_error(const ServeRequest& request, std::int64_t context) {
    if (request.token_ids.size() > static_cast<std::uint64_t>(kServeMaximumContext)) {
        return "prompt exceeds the maximum serve context";
    }
    const std::int64_t prompt_size = static_cast<std::int64_t>(request.token_ids.size());
    if (prompt_size > std::numeric_limits<std::int64_t>::max() - 8 ||
        request.max_new > std::numeric_limits<std::int64_t>::max() - prompt_size - 8 ||
        prompt_size + request.max_new + 8 > context) {
        return "prompt (" + std::to_string(prompt_size) + " tokens) + max_new (" +
               std::to_string(request.max_new) + ") exceeds the context (" + std::to_string(context) + ")";
    }
    return {};
}

void log_outcome(std::FILE* diagnostics, const ServeRequest& request, std::int64_t reused,
                 std::int64_t generated, const ServeOutcome& outcome) {
    const std::int64_t fresh = static_cast<std::int64_t>(request.token_ids.size()) - reused;
    std::fprintf(diagnostics,
                 "strata serve: prompt %zu tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), "
                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %lld checkpoints%s\n",
                 request.token_ids.size(), static_cast<long long>(reused), static_cast<long long>(fresh),
                 outcome.prompt_ms, outcome.prompt_ms > 0.0 ? 1000.0 * static_cast<double>(fresh) / outcome.prompt_ms : 0.0,
                 static_cast<long long>(generated), outcome.decode_ms,
                 outcome.decode_ms > 0.0 ? 1000.0 * static_cast<double>(generated) / outcome.decode_ms : 0.0,
                 static_cast<long long>(outcome.drafts_accepted), static_cast<long long>(outcome.drafts_offered),
                 static_cast<long long>(outcome.checkpoints), outcome.finish == "cancel" ? " (cancelled)" : "");
    if (outcome.lookups > 0) {
        std::fprintf(diagnostics, "strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)\n",
                     100.0 * static_cast<double>(outcome.hits) / static_cast<double>(outcome.lookups),
                     static_cast<long long>(outcome.hits), static_cast<long long>(outcome.lookups));
    }
    if (outcome.report_resident_ram) {
        std::fprintf(diagnostics, "strata serve: resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                                 "VRAM tier, %lld blob reads from the file\n",
                     outcome.resident_ram_gib, static_cast<long long>(outcome.exchanges),
                     static_cast<long long>(outcome.cumulative_file_reads));
    }
    if (outcome.report_expert_tiers) {
        std::fprintf(diagnostics,
                     "strata serve: expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, "
                     "files %lld blobs %.1f MB read\n",
                     static_cast<long long>(outcome.hits), static_cast<long long>(outcome.cumulative_ram_blobs),
                     static_cast<long long>(outcome.cumulative_expert_file_blobs),
                     outcome.cumulative_expert_file_mb);
    }
    std::fflush(diagnostics);
}

int run_one_request(ServeBackend& backend, const ServeRequest& request, std::atomic<bool>& stop_requested,
                    std::FILE* output, std::FILE* diagnostics,
                    const std::function<bool()>& request_finished) {
    const std::int64_t prompt_size = static_cast<std::int64_t>(request.token_ids.size());
    const std::int64_t max_context = std::min(backend.serve_info().context, kServeMaximumContext);
    const std::string context_error = request_context_error(request, max_context);
    if (!context_error.empty()) {
        request_finished();
        std::fprintf(output, "ERR %s\n", context_error.c_str());
        std::fflush(output);
        return 0;
    }
    for (const std::int32_t token : request.token_ids) {
        if (token >= kServeVocabularySize) {
            request_finished();
            write_error(output, "a token id is outside the vocabulary");
            return 0;
        }
    }

    CallbackState state;
    state.output = output;
    state.prompt_tokens = prompt_size;
    state.max_new = request.max_new;
    state.stop_eos = backend.serve_info().stop_eos;
    state.eos_ids = backend.serve_info().eos_ids;
    ServeCallbacks callbacks;
    callbacks.resume = [&](std::int64_t reused) { return state.resume(reused); };
    callbacks.prompt_progress = [&](std::int64_t done, std::int64_t total, double ms, double tps) {
        return state.progress(done, total, ms, tps);
    };
    callbacks.prompt_reused = [&](std::int64_t reused) { return state.reused(reused); };
    callbacks.token = [&](std::int32_t id) { return state.token(id); };
    callbacks.should_stop = [&] { return stop_requested.load() || state.eos_seen; };

    ServeOutcome outcome;
    ServeFailure failure;
    bool succeeded = false;
    try {
        succeeded = backend.run_request(request, callbacks, outcome, failure);
    } catch (const std::exception& exception) {
        failure.kind = ServeFailureKind::Fatal;
        failure.message = std::string("serve backend threw: ") + exception.what();
    } catch (...) {
        failure.kind = ServeFailureKind::Fatal;
        failure.message = "serve backend threw an unknown exception";
    }

    if (!state.error.empty()) {
        request_finished();
        write_error(output, "internal serve protocol error: " + state.error);
        std::fprintf(diagnostics, "strata: internal serve protocol error: %s\n", state.error.c_str());
        std::fflush(diagnostics);
        return 1;
    }
    if (!succeeded) {
        if (failure.kind == ServeFailureKind::Gpu) {
            request_finished();
            write_error(output, "gpu error: " + failure.message);
            std::fprintf(diagnostics, "strata: GPU error in \"%s\": %s\n",
                         failure.label.empty() ? "window" : failure.label.c_str(), failure.message.c_str());
            std::fflush(diagnostics);
            return 1;
        }
        request_finished();
        write_error(output, failure.message.empty() ? "request failed" : failure.message);
        if (failure.kind == ServeFailureKind::Fatal) {
            std::fprintf(diagnostics, "strata: %s\n",
                         failure.message.empty() ? "fatal serve error" : failure.message.c_str());
            std::fflush(diagnostics);
            return 1;
        }
        return 0;
    }
    const bool was_stopped = request_finished();
    if (was_stopped && outcome.finish == "length") outcome.finish = "cancel";
    else if (state.eos_seen && outcome.finish == "length") outcome.finish = "stop";
    const ServeInfo& info = backend.serve_info();
    const bool eos_finish_valid = outcome.finish != "stop" || (info.stop_eos && state.eos_seen);
    if (state.stage != CallbackStage::Decode || !valid_outcome(outcome) ||
        (outcome.finish == "length" && state.emitted_tokens != request.max_new) ||
        (outcome.finish == "cancel" && !was_stopped) || !eos_finish_valid) {
        write_error(output, "internal serve protocol error: incomplete or invalid backend response");
        std::fprintf(diagnostics, "strata: internal serve protocol error: incomplete or invalid backend response\n");
        std::fflush(diagnostics);
        return 1;
    }
    std::fprintf(output, "DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f\n",
                 static_cast<long long>(state.emitted_tokens), static_cast<long long>(prompt_size), outcome.prompt_ms,
                 outcome.decode_ms, outcome.finish.c_str(), static_cast<long long>(outcome.drafts_accepted),
                 static_cast<long long>(outcome.drafts_offered), static_cast<long long>(state.resume_count),
                 static_cast<long long>(outcome.hits), static_cast<long long>(outcome.lookups),
                 static_cast<long long>(outcome.ram_blobs), static_cast<long long>(outcome.file_blobs), outcome.file_mb);
    std::fflush(output);
    log_outcome(diagnostics, request, state.resume_count, state.emitted_tokens, outcome);
    return 0;
}

}  // namespace

ServeParseResult parse_serve_request(std::string_view line) {
    ServeParseResult result;
    if (line == "QUIT") {
        result.kind = ServeParseKind::Quit;
        return result;
    }

    const bool image = line.starts_with("GENI ");
    const bool text = line.starts_with("GEN ");
    if (!image && !text) {
        result.kind = ServeParseKind::InvalidCommand;
        result.error = std::string(kExpectedRequest);
        return result;
    }
    result.request.image = image;
    const std::size_t command_size = image ? 5 : 4;
    const std::size_t max_end = line.find(' ', command_size);
    const std::string_view max_text =
        max_end == std::string_view::npos ? line.substr(command_size) : line.substr(command_size, max_end - command_size);
    if (!parse_i64(max_text, result.request.max_new) || result.request.max_new < 1) {
        result.kind = ServeParseKind::BadRequest;
        result.error = "max_new";
        return result;
    }

    std::size_t payload_start = max_end == std::string_view::npos ? line.size() : max_end + 1;
    while (payload_start < line.size() && line[payload_start] == ' ') ++payload_start;
    for (;;) {
        if (payload_start >= line.size()) break;
        const std::size_t token_end = line.find(' ', payload_start);
        const std::size_t end = token_end == std::string_view::npos ? line.size() : token_end;
        const std::string_view key_value = line.substr(payload_start, end - payload_start);
        if (key_value.find('=') == std::string_view::npos) break;
        const std::size_t equals = key_value.find('=');
        const std::string_view key = key_value.substr(0, equals);
        const std::string_view value = key_value.substr(equals + 1);
        if (key == "cvec") {
            std::int32_t parsed = 0;
            if (!parse_i32(value, parsed) || (parsed != 0 && parsed != 1)) {
                result.kind = ServeParseKind::BadRequest;
                result.error = "cvec";
                return result;
            }
            result.request.sampling.cvec = parsed;
        } else if (key == "temperature" || key == "top_p" || key == "min_p" ||
                   key == "penalty_repeat" || key == "penalty_freq" || key == "penalty_present" ||
                   key == "pcie_frac" || key == "spec_min_p") {
            float parsed = 0.0f;
            if (!parse_float_value(value, parsed) || (key == "temperature" && parsed < 0.0f) ||
                (key == "min_p" && (parsed < 0.0f || parsed > 1.0f))) {
                result.kind = ServeParseKind::BadRequest;
                result.error = std::string(key);
                return result;
            }
            if (key == "temperature") result.request.sampling.temperature = parsed;
            else if (key == "top_p") result.request.sampling.top_p = parsed;
            else if (key == "min_p") result.request.sampling.min_p = parsed;
            else if (key == "penalty_repeat") result.request.sampling.penalty_repeat = parsed;
            else if (key == "penalty_freq") result.request.sampling.penalty_freq = parsed;
            else if (key == "penalty_present") result.request.sampling.penalty_present = parsed;
            else if (key == "pcie_frac") {
                result.request.sampling.pcie_frac =
                    std::clamp(static_cast<double>(parsed), 0.0, 1.0);
                result.request.pcie_frac_is_set = true;
            } else {
                result.request.sampling.spec_min_p = std::clamp(static_cast<double>(parsed), 0.0, 1.0);
                result.request.spec_min_p_is_set = true;
            }
        } else if (key == "top_k" || key == "penalty_last_n") {
            std::int32_t parsed = 0;
            if (!parse_i32(value, parsed)) {
                result.kind = ServeParseKind::BadRequest;
                result.error = std::string(key);
                return result;
            }
            if (key == "top_k") result.request.sampling.top_k = parsed;
            else result.request.sampling.penalty_last_n = std::clamp(parsed, 0, 4096);
        } else if (key == "seed") {
            if (!parse_u64(value, result.request.sampling.seed)) {
                result.kind = ServeParseKind::BadRequest;
                result.error = "seed";
                return result;
            }
        }
        if (token_end == std::string_view::npos) {
            payload_start = line.size();
            break;
        }
        payload_start = token_end + 1;
        while (payload_start < line.size() && line[payload_start] == ' ') ++payload_start;
    }
    if (result.request.sampling.temperature > 0.0f &&
        (result.request.sampling.top_k < 1 || result.request.sampling.top_k > 64)) {
        result.kind = ServeParseKind::BadRequest;
        result.error = "top_k";
        return result;
    }

    std::string_view token_text;
    if (image) {
        const std::string_view rest = line.substr(payload_start);
        const std::size_t last_space = rest.find_last_of(' ');
        if (last_space == std::string_view::npos) {
            result.kind = ServeParseKind::BadRequest;
            result.error = "max_new";
            return result;
        }
        result.request.embeddings_path = std::string(trim_spaces(rest.substr(0, last_space)));
        token_text = trim_spaces(rest.substr(last_space + 1));
        if (result.request.embeddings_path.empty()) {
            result.kind = ServeParseKind::BadRequest;
            result.error = "max_new";
            return result;
        }
    } else {
        token_text = line.substr(payload_start);
    }

    if (!parse_token_list(token_text, result.request.token_ids, result.error)) {
        result.kind = ServeParseKind::BadRequest;
        return result;
    }
    result.kind = ServeParseKind::Request;
    return result;
}

bool load_serve_image_embeddings(ServeRequest& request, std::int64_t max_context, std::string& error) {
    request.images.clear();
    request.mrope_positions.clear();
    error.clear();
    const std::string bad_header =
        "bad embeddings file (expected strata-vision records of width " +
        std::to_string(kServeImageEmbeddingWidth) + ")";
    const auto fail = [&](std::string message) {
        request.images.clear();
        request.mrope_positions.clear();
        error = std::move(message);
        return false;
    };

    const std::int64_t context_limit = std::min(max_context, kServeMaximumContext);
    if (max_context < 1 || request.token_ids.size() > static_cast<std::uint64_t>(context_limit)) {
        return fail("prompt exceeds the maximum serve context");
    }
    const std::size_t prompt_size = request.token_ids.size();
    std::size_t prompt_cursor = 0;

    std::FILE* file = std::fopen(request.embeddings_path.c_str(), "rb");
    if (file == nullptr) return fail("cannot open " + request.embeddings_path);
    std::uint64_t total_payload_bytes = 0;
    for (;;) {
        std::array<unsigned char, sizeof(std::int32_t) * 5> bytes {};
        const std::size_t got = std::fread(bytes.data(), 1, bytes.size(), file);
        if (got == 0) {
            if (std::ferror(file) != 0) {
                (void) std::fclose(file);
                return fail("short embeddings file");
            }
            break;
        }
        if (got != bytes.size()) {
            (void) std::fclose(file);
            return fail(bad_header);
        }
        std::int32_t header[5] {};
        std::memcpy(header, bytes.data(), bytes.size());
        if (header[0] != 0x31455653 || header[1] < 1 || header[2] < 1 ||
            header[3] < 1 ||
            static_cast<std::int64_t>(header[2]) * static_cast<std::int64_t>(header[3]) != header[1] ||
            header[4] != kServeImageEmbeddingWidth) {
            (void) std::fclose(file);
            return fail(bad_header);
        }

        while (prompt_cursor < prompt_size && request.token_ids[prompt_cursor] != kServeImagePadToken) {
            ++prompt_cursor;
        }
        if (prompt_cursor == prompt_size) {
            (void) std::fclose(file);
            return fail("the embeddings file has more images than the prompt");
        }
        std::size_t image_pad_run = 0;
        while (prompt_cursor + image_pad_run < prompt_size &&
               request.token_ids[prompt_cursor + image_pad_run] == kServeImagePadToken) {
            ++image_pad_run;
        }
        const std::size_t image_number = request.images.size() + 1;
        if (static_cast<std::uint64_t>(header[1]) > image_pad_run) {
            (void) std::fclose(file);
            return fail("image " + std::to_string(image_number) + " has " +
                        std::to_string(header[1]) + " rows but fewer <|image_pad|> tokens");
        }
        // One run of pad tokens is one image: a longer run would leave pad rows for a later record to claim.
        if (static_cast<std::uint64_t>(header[1]) < image_pad_run) {
            (void) std::fclose(file);
            return fail("image " + std::to_string(image_number) + " has " +
                        std::to_string(header[1]) + " rows but more <|image_pad|> tokens");
        }

        const std::uint64_t element_count =
            static_cast<std::uint64_t>(header[1]) * static_cast<std::uint64_t>(header[4]);
        if (element_count > std::vector<float>().max_size()) {
            (void) std::fclose(file);
            return fail(bad_header);
        }
        const off_t payload_start = ::ftello(file);
        struct stat file_status {};
        if (payload_start < 0 || ::fstat(::fileno(file), &file_status) != 0 || file_status.st_size < payload_start) {
            (void) std::fclose(file);
            return fail("short embeddings file");
        }
        const std::uint64_t payload_bytes = element_count * sizeof(float);
        if (payload_bytes > kServeMaximumImageEmbeddingBytes - total_payload_bytes) {
            (void) std::fclose(file);
            return fail("image embeddings exceed the 268435456-byte per-request limit");
        }
        if (payload_bytes > static_cast<std::uint64_t>(file_status.st_size - payload_start)) {
            (void) std::fclose(file);
            return fail("short embeddings file");
        }
        ServeImageRecord image;
        image.n_tokens = header[1];
        image.nx = header[2];
        image.ny = header[3];
        try {
            image.embeddings.resize(static_cast<std::size_t>(element_count));
        } catch (const std::bad_alloc&) {
            (void) std::fclose(file);
            return fail("short embeddings file");
        } catch (const std::length_error&) {
            (void) std::fclose(file);
            return fail(bad_header);
        }
        if (std::fread(image.embeddings.data(), 1, static_cast<std::size_t>(payload_bytes), file) !=
            static_cast<std::size_t>(payload_bytes)) {
            (void) std::fclose(file);
            return fail("short embeddings file");
        }
        try {
            request.images.push_back(std::move(image));
        } catch (const std::bad_alloc&) {
            (void) std::fclose(file);
            return fail("short embeddings file");
        } catch (const std::length_error&) {
            (void) std::fclose(file);
            return fail(bad_header);
        }
        total_payload_bytes += payload_bytes;
        prompt_cursor += static_cast<std::size_t>(header[1]);
    }
    if (std::fclose(file) != 0) return fail("short embeddings file");

    try {
        request.mrope_positions.resize(prompt_size);
    } catch (const std::bad_alloc&) {
        return fail("short embeddings file");
    } catch (const std::length_error&) {
        return fail(bad_header);
    }
    std::int64_t position = 0;
    std::size_t token = 0;
    std::size_t image_index = 0;
    while (token < prompt_size) {
        if (request.token_ids[token] != kServeImagePadToken) {
            request.mrope_positions[token] = {static_cast<std::int32_t>(position),
                                               static_cast<std::int32_t>(position),
                                               static_cast<std::int32_t>(position)};
            ++position;
            ++token;
            continue;
        }
        if (image_index >= request.images.size()) {
            return fail("the prompt has more images than the embeddings file");
        }
        const ServeImageRecord& image = request.images[image_index++];
        for (std::int32_t row = 0; row < image.n_tokens; ++row) {
            const std::size_t row_index = token + static_cast<std::size_t>(row);
            if (row_index >= prompt_size || request.token_ids[row_index] != kServeImagePadToken) {
                return fail("image " + std::to_string(image_index) + " has " +
                            std::to_string(image.n_tokens) +
                            " rows but fewer <|image_pad|> tokens");
            }
        }
        for (std::int32_t row = 0; row < image.n_tokens; ++row) {
            const std::int64_t y = row / image.nx;
            const std::int64_t x = row % image.nx;
            request.mrope_positions[token + static_cast<std::size_t>(row)] =
                {static_cast<std::int32_t>(position), static_cast<std::int32_t>(position + y),
                 static_cast<std::int32_t>(position + x)};
        }
        token += static_cast<std::size_t>(image.n_tokens);
        position += std::max(image.nx, image.ny);
    }
    if (image_index != request.images.size()) {
        return fail("the embeddings file has more images than the prompt");
    }
    if (!request.token_ids.empty() && request.token_ids.back() == kServeImagePadToken) {
        return fail("the prompt cannot end in an image");
    }
    return true;
}

std::string serve_info_line(const ServeInfo& info) {
    std::ostringstream line;
    line.imbue(std::locale::classic());
    const std::string gpu = protocol_atom(info.gpu);
    line << "INFO context=" << info.context << " kv=" << protocol_atom(info.kv) << " kv_resident=0"
         << " expert_slots=" << info.expert_slots << " expert_cache_mib=" << info.expert_cache_mib
         << " expert_slots_primary=" << info.expert_slots
         << " expert_cache_primary_mib=" << info.expert_cache_mib << " spec=" << info.spec
         << " mtp_max=" << info.mtp_max << " lookup=" << info.lookup
         << " vram_free_mib=" << info.vram_free_mib << " cvec=" << protocol_atom(info.cvec)
         << " arena_mib=" << info.arena_mib << " pool_workers=0 pcie_frac=0.00"
         << " spec_min_p=" << std::fixed << std::setprecision(2) << info.spec_min_p
         << " conversation_cache_mib=" << info.conversation_cache_mib
         << " conversation_cache_slots=" << info.conversation_cache_slots
         << " conversation_cache_min_free_mib=" << info.conversation_cache_min_free_mib
         << " engine=" << protocol_atom(info.engine) << " backend=metal metal_rev=" << info.metal_rev
         << " gpu=" << gpu << " gpu_budget_mib=" << info.gpu_budget_mib
         << " residency=" << protocol_atom(info.residency) << " ssd_experts=" << info.ssd_experts
         << " plan_mib=" << info.plan_mib;
    return line.str();
}

int run_serve_loop(ServeBackend& backend, const ServeLoopOptions& options) {
    std::FILE* output = options.output == nullptr ? stdout : options.output;
    std::FILE* diagnostics = options.diagnostics == nullptr ? stderr : options.diagnostics;
    (void) std::setvbuf(output, nullptr, _IONBF, 0);

    SignalPolicy signals;
    if (!signals.install()) {
        std::fprintf(diagnostics, "strata: failed to install serve signal handling\n");
        return 2;
    }
    std::atomic<bool> stop_requested {false};
    std::atomic<bool> request_active {false};
    InputReader reader(options.input_fd, stop_requested, request_active);
    if (!reader.start()) {
        std::fprintf(diagnostics, "strata: could not start the stdin reader\n");
        return 2;
    }

    const ServeInfo& info = backend.serve_info();
    const bool eos_config_valid = (!info.stop_eos || !info.eos_ids.empty()) &&
                                  std::all_of(info.eos_ids.begin(), info.eos_ids.end(), [](std::int32_t token) {
                                      return token >= 0 && token < kServeVocabularySize;
                                  });
    if (info.context < 1 || info.context > kServeMaximumContext || info.engine.empty() || info.metal_rev < 1 ||
        !std::isfinite(info.spec_min_p) || info.spec_min_p < 0.0 || info.spec_min_p > 1.0 || !eos_config_valid) {
        std::fprintf(diagnostics, "strata: invalid serve startup metadata\n");
        reader.stop();
        return 2;
    }
    if (info.vram_free_mib >= 256) {
        std::fprintf(diagnostics, "strata serve: %lld MiB of the GPU working set free with everything loaded\n",
                     static_cast<long long>(info.vram_free_mib));
    } else {
        const std::int64_t suggested =
            info.vram_reserve_mib > std::numeric_limits<std::int64_t>::max() - 512
                ? std::numeric_limits<std::int64_t>::max()
                : std::max<std::int64_t>(0, info.vram_reserve_mib + 512 - info.vram_free_mib);
        std::fprintf(diagnostics,
                     "strata serve: %lld MiB of the GPU working set free with everything loaded - LOW: requests may "
                     "stall; add --vram-reserve-mib %lld to the config's args (or lower --max-context)\n",
                     static_cast<long long>(info.vram_free_mib), static_cast<long long>(suggested));
    }
    const std::string info_line = serve_info_line(info);
    if (!write_line(output, info_line) ||
        !write_line(output, "READY " + std::to_string(info.context) + " stop")) {
        reader.stop();
        return 1;
    }

    std::string line;
    InputError input_error = InputError::None;
    while (reader.next(line, input_error)) {
        const auto finish_request = [&reader] { return reader.request_finished(); };
        if (input_error == InputError::LineTooLong) {
            finish_request();
            write_error(output, "request line exceeds the 4194304-byte limit");
            continue;
        }
        if (input_error == InputError::QueueLimit) {
            finish_request();
            write_error(output, "serve input exceeded the queued-command limit");
            reader.stop();
            return 2;
        }
        if (input_error == InputError::CompletionDrainLimit) {
            finish_request();
            write_error(output, "serve input exceeded the 67108864-byte request-finish drain limit");
            reader.stop();
            return 2;
        }
        const ServeParseResult parsed = parse_serve_request(line);
        if (parsed.kind == ServeParseKind::Quit) {
            finish_request();
            break;
        }
        if (parsed.kind == ServeParseKind::InvalidCommand) {
            finish_request();
            write_error(output, parsed.error.empty() ? kExpectedRequest : parsed.error);
            continue;
        }
        if (parsed.kind == ServeParseKind::BadRequest) {
            finish_request();
            write_request_error(output, parsed.error.empty() ? "max_new" : parsed.error);
            continue;
        }
        if (parsed.request.image && !options.vision_enabled) {
            finish_request();
            write_error(output, "this engine was started without --vision");
            continue;
        }
        ServeRequest request = parsed.request;
        const std::int64_t context_limit = std::min(info.context, kServeMaximumContext);
        const std::string context_error = request_context_error(request, context_limit);
        if (!context_error.empty()) {
            finish_request();
            write_error(output, context_error);
            continue;
        }
        if (request.image) {
            std::string image_error;
            if (!load_serve_image_embeddings(request, context_limit, image_error)) {
                finish_request();
                write_error(output, image_error);
                continue;
            }
        }
        if (!request.spec_min_p_is_set) request.sampling.spec_min_p = info.spec_min_p;
        const int result = run_one_request(backend, request, stop_requested, output, diagnostics, finish_request);
        if (result != 0) {
            reader.stop();
            return result;
        }
    }
    reader.stop();
    return 0;
}

}  // namespace strata::metal
