// Metal twin for the split sampler and its CPU FP64 tail (docs/mac/04-metal-kernels.md §5.12).
#import <Metal/Metal.h>

#include "metallib_sha256.inc"
#include "strata/metal/sampler_tail.hpp"
#include "strata/metal/sampler_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/context.hpp"
#include "twin.hpp"
#include "../../src/metal/kernels/common/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using namespace strata::metal;

namespace {

using strata::kernels::SamplerParams;

constexpr std::uint32_t kBlockSpan = 4096u;
constexpr std::uint32_t kCandidateStride = 64u;
constexpr int kMaxTopK = 64;

int g_failures = 0;

void check(bool condition, const std::string& label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

bool concurrent_dispatch_mode() {
    const char* const value = std::getenv("STRATA_METAL_DISPATCH");
    return value != nullptr && std::string_view(value) == "concurrent";
}

template <class Attempt>
bool expect_sampler_rejected_in_open_concurrent_group(
    Context& context, const std::string& label,
    std::string_view expected_error, Attempt&& attempt) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;

    bool saw_open_group = false;
    bool accepted = false;
    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& body_error) {
            if (!encoder.group_begin(body_error)) return false;
            saw_open_group = encoder.concurrent_group_open();
            if (!saw_open_group) {
                body_error = "test requires STRATA_METAL_DISPATCH=concurrent";
                return false;
            }
            std::string attempt_error;
            accepted = attempt(encoder, attempt_error);
            std::string group_error;
            const bool group_closed = encoder.group_end(group_error);
            body_error = !group_closed
                ? group_error
                : accepted
                    ? "sampler stage unexpectedly succeeded in an open concurrent group"
                    : attempt_error;
            return false;
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool rejected = saw_open_group && !encoded && !accepted &&
        error.find(expected_error) != std::string::npos && discarded;
    check(rejected, label + " rejects open concurrent-group dependency", error);
    command->discard();
    return rejected;
}

template <class T>
std::span<const std::uint8_t> bytes_of(const std::vector<T>& values) {
    return {reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(T)};
}

template <class T>
std::span<const std::uint8_t> bytes_of(const T& value) {
    return {reinterpret_cast<const std::uint8_t*>(&value), sizeof(T)};
}

std::uint32_t float_bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

float candidate_value(const SampCandidate& candidate) {
    return std::bit_cast<float>(candidate.logit_bits);
}

SampCandidate make_candidate(std::int32_t id, float value) {
    return {id, float_bits(value)};
}

int effective_k(int top_k, std::uint32_t n_vocab) {
    const int requested = top_k > 0 && top_k < kMaxTopK ? top_k : kMaxTopK;
    return std::min<int>(requested, static_cast<int>(n_vocab));
}

int history_count(std::span<const int> history, int token) {
    return static_cast<int>(std::count(history.begin(), history.end(), token));
}

float penalize(float value, int count, const SamplerParams& params) {
    if (count <= 0) return value;
    value = value <= 0.0f ? value * params.penalty_repeat
                          : value / params.penalty_repeat;
    return value - std::fma(static_cast<float>(count), params.penalty_freq,
                            params.penalty_present);
}

bool candidate_before(const SampCandidate& left, const SampCandidate& right) {
    const float a = candidate_value(left);
    const float b = candidate_value(right);
    if (a != b) return a > b;
    return left.id < right.id;
}

std::vector<SampCandidate> reference_block(const float* row_logits, std::uint32_t n_vocab,
                                           std::uint32_t block,
                                           std::span<const int> history,
                                           const SamplerParams& params, int kpart) {
    const std::uint32_t begin = block * kBlockSpan;
    const std::uint32_t end = std::min(n_vocab, begin + kBlockSpan);
    std::vector<SampCandidate> candidates;
    candidates.reserve(end - begin);
    for (std::uint32_t token = begin; token < end; ++token) {
        const int count = history_count(history, static_cast<int>(token));
        const float value = penalize(row_logits[token], count, params);
        if (std::isnan(value) || value == -std::numeric_limits<float>::infinity()) continue;
        candidates.push_back(make_candidate(static_cast<int>(token), value));
    }
    std::sort(candidates.begin(), candidates.end(), candidate_before);
    if (candidates.size() > static_cast<std::size_t>(kpart)) {
        candidates.resize(static_cast<std::size_t>(kpart));
    }
    while (candidates.size() < static_cast<std::size_t>(kpart)) {
        candidates.push_back(make_candidate(static_cast<int>(n_vocab),
                                            -std::numeric_limits<float>::infinity()));
    }
    return candidates;
}

std::vector<SampCandidate> reference_merge(
    const std::vector<std::vector<SampCandidate>>& block_lists, std::uint32_t n_vocab, int k) {
    std::vector<SampCandidate> all;
    for (const auto& list : block_lists) {
        for (const SampCandidate& item : list) {
            if (item.id >= 0 && static_cast<std::uint32_t>(item.id) < n_vocab) all.push_back(item);
        }
    }
    std::sort(all.begin(), all.end(), candidate_before);
    if (all.size() > static_cast<std::size_t>(k)) all.resize(static_cast<std::size_t>(k));
    while (all.size() < static_cast<std::size_t>(k)) {
        all.push_back(make_candidate(0, -std::numeric_limits<float>::infinity()));
    }
    return all;
}

float reference_philox_uniform(std::uint64_t seed, std::uint64_t counter) {
    std::array<std::uint32_t, 4> words{
        static_cast<std::uint32_t>(counter),
        static_cast<std::uint32_t>(counter >> 32),
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32)};
    for (std::uint32_t round = 0; round < 10; ++round) {
        const std::uint64_t product0 = std::uint64_t{0x9E3779B9u} * words[0];
        const std::uint64_t product1 = std::uint64_t{0xBB67AE85u} * words[2];
        const std::array<std::uint32_t, 4> next{
            static_cast<std::uint32_t>(product1 >> 32) ^ words[1] ^ round,
            static_cast<std::uint32_t>(product1),
            static_cast<std::uint32_t>(product0 >> 32) ^ words[3],
            static_cast<std::uint32_t>(product0)};
        words = next;
    }
    return static_cast<float>(words[0] >> 8) * (1.0f / 16777216.0f);
}

TailPick reference_tail(std::span<const SampCandidate> selected, int k,
                        const SamplerParams& params, std::uint64_t counter) {
    if (params.greedy || params.temperature <= 0.0f) {
        return {selected[0].id, 1.0f};
    }

    std::array<float, 64> logits{};
    std::array<double, 64> ex{};
    for (int i = 0; i < k; ++i) logits[static_cast<std::size_t>(i)] = candidate_value(selected[i]);

    int n_keep = k;
    float maximum = logits[0];
    for (int i = 1; i < k; ++i) maximum = std::fmax(maximum, logits[static_cast<std::size_t>(i)]);
    if (params.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) {
            ex[static_cast<std::size_t>(i)] =
                std::exp(static_cast<double>(logits[static_cast<std::size_t>(i)]) -
                         static_cast<double>(maximum));
            sum += ex[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < k; ++i) ex[static_cast<std::size_t>(i)] /= sum;
        double cumulative = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cumulative += ex[static_cast<std::size_t>(i)];
            if (cumulative >= static_cast<double>(params.top_p)) {
                cut = i + 1;
                break;
            }
        }
        if (cut < params.min_keep) cut = params.min_keep < k ? params.min_keep : k;
        n_keep = cut;
    }
    if (params.min_p > 0.0f) {
        const float threshold = logits[0] + std::log(params.min_p);
        for (int i = 0; i < n_keep; ++i) {
            if (logits[static_cast<std::size_t>(i)] < threshold) {
                n_keep = i;
                break;
            }
        }
    }

    const float inverse_temperature =
        params.temperature > 0.0f ? 1.0f / params.temperature : 0.0f;
    float scaled_maximum = logits[0] * inverse_temperature;
    for (int i = 1; i < n_keep; ++i) {
        scaled_maximum =
            std::fmax(scaled_maximum, logits[static_cast<std::size_t>(i)] * inverse_temperature);
    }
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        ex[static_cast<std::size_t>(i)] =
            std::exp(static_cast<double>(logits[static_cast<std::size_t>(i)] * inverse_temperature) -
                     static_cast<double>(scaled_maximum));
        sum += ex[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < n_keep; ++i) ex[static_cast<std::size_t>(i)] /= sum;

    const float uniform = reference_philox_uniform(params.seed, counter);
    int pick = n_keep > 0 ? n_keep - 1 : 0;
    double cumulative = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cumulative += ex[static_cast<std::size_t>(i)];
        if (static_cast<double>(uniform) < cumulative) {
            pick = i;
            break;
        }
    }
    return {selected[static_cast<std::size_t>(pick)].id,
            n_keep > 0 ? static_cast<float>(ex[static_cast<std::size_t>(pick)]) : 1.0f};
}

SampRow make_row(const SamplerParams& params, std::uint32_t history_offset,
                 std::uint32_t history_length) {
    SampRow row{};
    row.temperature = params.temperature;
    row.top_p = params.top_p;
    row.min_p = params.min_p;
    row.penalty_repeat = params.penalty_repeat;
    row.penalty_freq = params.penalty_freq;
    row.penalty_present = params.penalty_present;
    row.top_k = params.top_k;
    row.min_keep = params.min_keep;
    row.greedy = params.greedy ? 1 : 0;
    row.counter = static_cast<std::uint32_t>(params.counter);
    row.hist_off = history_offset;
    row.hist_len = history_length;
    return row;
}

std::optional<twin::Failure> compare_ranges(std::span<const std::uint8_t> actual,
                                            std::span<const std::uint8_t> expected,
                                            std::span<const twin::ByteRange> ranges) {
    for (const twin::ByteRange range : ranges) {
        const auto failure = twin::compare_bytes(actual.subspan(range.offset, range.length),
                                                 expected.subspan(range.offset, range.length));
        if (failure) {
            twin::Failure adjusted = *failure;
            adjusted.index += range.offset;
            return adjusted;
        }
    }
    return std::nullopt;
}

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     const std::string& label, std::string& error) {
    if (!command || !command->commit(error)) return false;
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        error = label + " command buffer became unavailable after commit";
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = label + " command buffer failed";
        if (native.error.localizedDescription.UTF8String != nullptr) {
            error += ": ";
            error += native.error.localizedDescription.UTF8String;
        }
        return false;
    }
    error.clear();
    return true;
}

template <class Attempt>
bool expect_shared_sampler_rejected(Context& context, const std::string& label,
                                    std::string_view expected_error,
                                    Attempt&& attempt) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;

    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& body_error) {
            return attempt(encoder, body_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool rejected = !encoded && discarded &&
        error.find(expected_error) != std::string::npos;
    check(rejected, label + " shared Encoder rejection discards its segment", error);
    command->discard();
    return rejected;
}

bool run_greedy_case(Context& context, std::uint32_t rows, std::uint32_t n_vocab,
                     const std::vector<float>& logits, const std::vector<int>& history,
                     std::uint32_t history_stride, const twin::Options& options,
                     const std::string& name, bool apply_penalties = true,
                     bool shared_encoder = false) {
    if (!sampler_shape_supported(rows, n_vocab) ||
        logits.size() != static_cast<std::size_t>(rows) * n_vocab ||
        history.size() != static_cast<std::size_t>(rows) * history_stride) {
        check(false, name + " input shape");
        return false;
    }
    std::vector<SampRow> row_params(rows);
    std::vector<int> expected(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        SamplerParams params;
        params.top_k = 0;
        params.top_p = 1.0f;
        params.temperature = 0.0f;
        params.greedy = true;
        params.penalty_last_n = apply_penalties ? static_cast<int>(history_stride) : 0;
        params.penalty_repeat = 1.25f;
        params.penalty_freq = 0.125f;
        params.penalty_present = 0.25f;
        const std::size_t offset = static_cast<std::size_t>(row) * history_stride;
        const std::span<const int> row_history = apply_penalties && history_stride > 0u
            ? std::span<const int>(history.data() + offset, history_stride)
            : std::span<const int>{};
        row_params[row] = make_row(
            params, static_cast<std::uint32_t>(offset),
            apply_penalties ? history_stride : 0u);
        float best_value = -std::numeric_limits<float>::infinity();
        int best_id = static_cast<int>(n_vocab);
        const float* row_logits =
            logits.data() + static_cast<std::size_t>(row) * n_vocab;
        for (std::uint32_t token = 0; token < n_vocab; ++token) {
            const float value = penalize(row_logits[token],
                                         history_count(row_history, static_cast<int>(token)),
                                         params);
            if (value > best_value) {
                best_value = value;
                best_id = static_cast<int>(token);
            }
        }
        expected[row] = best_id < static_cast<int>(n_vocab) ? best_id : 0;
    }

    std::string err;
    twin::GuardedBuffer input, hist, rows_buffer, output;
    check(input.allocate(context, name + " logits", logits.size() * sizeof(float), err),
          name + " logits buffer", err);
    check(hist.allocate(context, name + " history", history.size() * sizeof(int), err),
          name + " history buffer", err);
    check(rows_buffer.allocate(context, name + " rows", row_params.size() * sizeof(SampRow), err),
          name + " row buffer", err);
    check(output.allocate(context, name + " output", expected.size() * sizeof(int), err),
          name + " output buffer", err);
    if (!input.view() || !hist.view() || !rows_buffer.view() || !output.view()) return false;
    std::memcpy(input.payload().data(), logits.data(), logits.size() * sizeof(float));
    std::memcpy(hist.payload().data(), history.data(), history.size() * sizeof(int));
    std::memcpy(rows_buffer.payload().data(), row_params.data(),
                row_params.size() * sizeof(SampRow));
    check(output.poison_bytes_against(bytes_of(expected), err), name + " output poison", err);

    const SamplerBuffers buffers{
        input.view(), hist.view(), rows_buffer.view(), {}, {}, output.view()};
    input.capture_before_dispatch();
    hist.capture_before_dispatch();
    rows_buffer.capture_before_dispatch();
    output.capture_before_dispatch();
    auto command = context.make_command_buffer(name, err);
    if (!command) {
        check(false, name + " command buffer", err);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, name,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_sampler_greedy(
                      context, encoder, rows, n_vocab, buffers, encode_error);
              },
              err)
        : encode_sampler_greedy(context, *command, rows, n_vocab, buffers, err);
    if (!encoded) {
        command->discard();
        check(false, name + " production sampler encode", err);
        return false;
    }
    if (!submit_and_wait(command, name, err)) {
        check(false, name + " production sampler submit", err);
        return false;
    }

    const auto failure = compare_ranges(output.payload(), bytes_of(expected),
                                        std::array{twin::ByteRange{0, expected.size() * sizeof(int)}});
    if (failure) twin::report_failure(name.c_str(), options, rows, *failure);
    if (failure) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            int got = -1;
            std::memcpy(&got, output.payload().data() + row * sizeof(int), sizeof(got));
            if (got == expected[row]) continue;
            const std::size_t history_offset = static_cast<std::size_t>(row) * history_stride;
            const std::span<const int> row_history(history.data() + history_offset, history_stride);
            const auto adjusted = [&](int token) {
                if (token < 0 || static_cast<std::uint32_t>(token) >= n_vocab) {
                    return std::pair{-std::numeric_limits<float>::infinity(), 0};
                }
                const int count = history_count(row_history, token);
                SamplerParams p;
                p.penalty_repeat = 1.25f;
                p.penalty_freq = 0.125f;
                p.penalty_present = 0.25f;
                return std::pair{penalize(logits[static_cast<std::size_t>(row) * n_vocab +
                                                 static_cast<std::uint32_t>(token)],
                                          count, p),
                                 count};
            };
            const auto [want_value, want_count] = adjusted(expected[row]);
            const auto [got_value, got_count] = adjusted(got);
            std::printf("  row %u expected id=%d value=%g hits=%d, got id=%d value=%g hits=%d\n",
                        row, expected[row], static_cast<double>(want_value), want_count,
                        got, static_cast<double>(got_value), got_count);
        }
    }
    check(!failure, name + " greedy IDs");
    check(!input.first_unwritten_change({}), name + " logits unchanged");
    check(!hist.first_unwritten_change({}), name + " history unchanged");
    check(!rows_buffer.first_unwritten_change({}), name + " rows unchanged");
    check(!output.first_unwritten_change(
              std::array{twin::ByteRange{0, expected.size() * sizeof(int)}}),
          name + " output and guards");
    return !failure;
}

bool run_split_case(Context& context, std::uint32_t rows, std::uint32_t n_vocab,
                    const std::vector<float>& logits, const std::vector<int>& history,
                    std::uint32_t history_stride,
                    const std::vector<SamplerParams>& params,
                    const twin::Options& options, const std::string& name,
                    bool prepenalized = false, std::uint32_t kpart = 0u,
                    bool composite_encoder = false,
                    bool shared_segment = false,
                    bool shared_sampled = false) {
    if (!sampler_shape_supported(rows, n_vocab) ||
        params.size() != rows ||
        logits.size() != static_cast<std::size_t>(rows) * n_vocab ||
        history.size() != static_cast<std::size_t>(rows) * history_stride) {
        check(false, name + " input shape");
        return false;
    }
    const std::uint32_t n_blocks = (n_vocab + kBlockSpan - 1u) / kBlockSpan;
    std::vector<SampRow> row_params(rows);
    std::vector<int> row_k(rows);
    std::vector<std::vector<std::vector<SampCandidate>>> block_lists(rows);
    std::vector<std::vector<SampCandidate>> selected_lists(rows);
    std::vector<SampCandidate> expected_candidates(
        static_cast<std::size_t>(rows) * n_blocks * kCandidateStride);
    std::vector<SampCandidate> expected_selected(
        static_cast<std::size_t>(rows) * kCandidateStride);
    std::vector<twin::ByteRange> candidate_ranges;
    std::vector<twin::ByteRange> selected_ranges;
    candidate_ranges.reserve(static_cast<std::size_t>(rows) * n_blocks);

    for (std::uint32_t row = 0; row < rows; ++row) {
        const int k = effective_k(params[row].top_k, n_vocab);
        row_k[row] = k;
        const std::uint32_t history_length = params[row].penalty_last_n <= 0
            ? 0u
            : std::min<std::uint32_t>(history_stride,
                                      static_cast<std::uint32_t>(params[row].penalty_last_n));
        const std::uint32_t row_history_offset =
            row * history_stride + history_stride - history_length;
        row_params[row] = make_row(params[row], row_history_offset, history_length);
        const std::size_t history_base = static_cast<std::size_t>(row) * history_stride;
        const std::span<const int> row_history =
            prepenalized || history_length == 0u || history_stride == 0
                ? std::span<const int>{}
                : std::span<const int>(history.data() + history_base +
                                           history_stride - history_length,
                                       history_length);

        const int part_k = kpart != 0u ? static_cast<int>(kpart) : k;
        block_lists[row].reserve(n_blocks);
        for (std::uint32_t block = 0; block < n_blocks; ++block) {
            auto local = reference_block(
                logits.data() + static_cast<std::size_t>(row) * n_vocab, n_vocab,
                block, row_history, params[row], part_k);
            const std::size_t base =
                (static_cast<std::size_t>(row) * n_blocks + block) * kCandidateStride;
            std::copy(local.begin(), local.end(), expected_candidates.begin() + base);
            candidate_ranges.push_back({base * sizeof(SampCandidate),
                                        static_cast<std::size_t>(part_k) * sizeof(SampCandidate)});
            block_lists[row].push_back(std::move(local));
        }
        selected_lists[row] = reference_merge(block_lists[row], n_vocab, k);
        const std::size_t selected_base =
            static_cast<std::size_t>(row) * kCandidateStride;
        std::copy(selected_lists[row].begin(), selected_lists[row].end(),
                  expected_selected.begin() + selected_base);
        selected_ranges.push_back({selected_base * sizeof(SampCandidate),
                                   static_cast<std::size_t>(k) * sizeof(SampCandidate)});
    }

    std::string err;
    twin::GuardedBuffer input, hist, rows_buffer, candidates, selected;
    check(input.allocate(context, name + " logits", logits.size() * sizeof(float), err),
          name + " logits buffer", err);
    check(hist.allocate(context, name + " history",
                        std::max<std::size_t>(history.size(), 1u) * sizeof(int), err),
          name + " history buffer", err);
    check(rows_buffer.allocate(context, name + " rows", row_params.size() * sizeof(SampRow), err),
          name + " row buffer", err);
    check(candidates.allocate(context, name + " candidates",
                              expected_candidates.size() * sizeof(SampCandidate), err),
          name + " candidates buffer", err);
    check(selected.allocate(context, name + " selected",
                            expected_selected.size() * sizeof(SampCandidate), err),
          name + " selected buffer", err);
    if (!input.view() || !hist.view() || !rows_buffer.view() ||
        !candidates.view() || !selected.view()) return false;
    std::memcpy(input.payload().data(), logits.data(), logits.size() * sizeof(float));
    if (!history.empty()) {
        std::memcpy(hist.payload().data(), history.data(), history.size() * sizeof(int));
    } else {
        std::memset(hist.payload().data(), 0, hist.payload().size());
    }
    std::memcpy(rows_buffer.payload().data(), row_params.data(),
                row_params.size() * sizeof(SampRow));
    check(candidates.poison_bytes_against(bytes_of(expected_candidates), err),
          name + " candidates poison", err);
    check(selected.poison_bytes_against(bytes_of(expected_selected), err),
          name + " selected poison", err);

    const SamplerBuffers buffers{
        input.view(), hist.view(), rows_buffer.view(),
        candidates.view(), selected.view(), {}};
    if (concurrent_dispatch_mode() && rows == 1u && n_vocab == 512u &&
        !prepenalized && kpart == 0u) {
        if (shared_segment &&
            !expect_sampler_rejected_in_open_concurrent_group(
                context, name + " split-merge concurrent-group preflight",
                "ordered concurrent-group boundary",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_sampler_split_merge(
                        context, encoder, rows, n_vocab, rows_buffer.view(),
                        candidates.view(), selected.view(), kpart,
                        encode_error);
                })) {
            return false;
        }
        if (shared_sampled &&
            !expect_sampler_rejected_in_open_concurrent_group(
                context, name + " sampled concurrent-group preflight",
                "ordered concurrent-group boundary",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_sampler_sampled(
                        context, encoder, rows, n_vocab, buffers, false,
                        encode_error, kpart);
                })) {
            return false;
        }
    }
    if (shared_segment || shared_sampled) {
        input.capture_before_dispatch();
        hist.capture_before_dispatch();
        rows_buffer.capture_before_dispatch();
        candidates.capture_before_dispatch();
        selected.capture_before_dispatch();
        auto command = context.make_command_buffer(name, err);
        if (!command) {
            check(false, name + " shared Encoder command buffer", err);
            return false;
        }
        const bool encoded = Encoder::encode(
            context, *command, name,
            [&](Encoder& encoder, std::string& encode_error) {
                if (shared_sampled) {
                    return encode_sampler_sampled(
                        context, encoder, rows, n_vocab, buffers,
                        prepenalized, encode_error, kpart);
                }
                return encode_sampler_split_part(
                           context, encoder, rows, n_vocab, buffers,
                           prepenalized, kpart, encode_error) &&
                       encode_sampler_split_merge(
                           context, encoder, rows, n_vocab,
                           rows_buffer.view(), candidates.view(),
                           selected.view(), kpart, encode_error);
            },
            err);
        if (!encoded) {
            command->discard();
            check(false, name + " shared sampler encode", err);
            return false;
        }
        if (!submit_and_wait(command, name + " shared sampler", err)) {
            check(false, name + " shared sampler submit", err);
            return false;
        }
    } else if (composite_encoder) {
        input.capture_before_dispatch();
        hist.capture_before_dispatch();
        rows_buffer.capture_before_dispatch();
        candidates.capture_before_dispatch();
        selected.capture_before_dispatch();
        auto command = context.make_command_buffer(name, err);
        if (!command) {
            check(false, name + " composite command buffer", err);
            return false;
        }
        if (!encode_sampler_sampled(context, *command, rows, n_vocab, buffers,
                                    prepenalized, err, kpart)) {
            command->discard();
            check(false, name + " production sampler composite encode", err);
            return false;
        }
        if (!submit_and_wait(command, name, err)) {
            check(false, name + " production sampler composite submit", err);
            return false;
        }
    } else {
        input.capture_before_dispatch();
        hist.capture_before_dispatch();
        rows_buffer.capture_before_dispatch();
        candidates.capture_before_dispatch();
        auto part_command = context.make_command_buffer(name + " split part", err);
        if (!part_command) {
            check(false, name + " split-part command buffer", err);
            return false;
        }
        if (!encode_sampler_split_part(context, *part_command, rows, n_vocab, buffers,
                                       prepenalized, kpart, err)) {
            part_command->discard();
            check(false, name + " production sampler split-part encode", err);
            return false;
        }
        if (!submit_and_wait(part_command, name + " split part", err)) {
            check(false, name + " production sampler split-part submit", err);
            return false;
        }
    }

    const auto part_failure =
        compare_ranges(candidates.payload(), bytes_of(expected_candidates), candidate_ranges);
    if (part_failure) twin::report_failure((name + " split-part candidates").c_str(),
                                           options, rows, *part_failure);
    check(!part_failure, name + " per-block top-k IDs and logit bits");
    check(!input.first_unwritten_change({}), name + " split-part logits unchanged");
    check(!hist.first_unwritten_change({}), name + " split-part history unchanged");
    check(!rows_buffer.first_unwritten_change({}), name + " split-part rows unchanged");
    check(!candidates.first_unwritten_change(candidate_ranges),
          name + " split-part candidate range and guards");
    if (part_failure) return false;

    if (!composite_encoder && !shared_segment && !shared_sampled) {
        candidates.capture_before_dispatch();
        rows_buffer.capture_before_dispatch();
        selected.capture_before_dispatch();
        auto merge_command = context.make_command_buffer(name + " split merge", err);
        if (!merge_command) {
            check(false, name + " split-merge command buffer", err);
            return false;
        }
        if (!encode_sampler_split_merge(context, *merge_command, rows, n_vocab,
                                        rows_buffer.view(), candidates.view(),
                                        selected.view(), kpart, err)) {
            merge_command->discard();
            check(false, name + " production sampler split-merge encode", err);
            return false;
        }
        if (!submit_and_wait(merge_command, name + " split merge", err)) {
            check(false, name + " production sampler split-merge submit", err);
            return false;
        }
    }

    const auto merge_failure =
        compare_ranges(selected.payload(), bytes_of(expected_selected), selected_ranges);
    if (merge_failure) twin::report_failure((name + " merged top-k").c_str(),
                                            options, rows, *merge_failure);
    check(!merge_failure, name + " ordered merged top-k IDs and logit bits");
    if (!composite_encoder && !shared_segment && !shared_sampled) {
        check(!candidates.first_unwritten_change({}), name + " candidates unchanged by merge");
    }
    check(!rows_buffer.first_unwritten_change({}), name + " merge rows unchanged");
    check(!selected.first_unwritten_change(selected_ranges),
          name + " merged selection range and guards");
    if (merge_failure) return false;

    const auto* actual_selected =
        reinterpret_cast<const SampCandidate*>(selected.payload().data());
    int tail_mismatches = 0;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const int k = row_k[row];
        const std::span<const SampCandidate> actual_row(
            actual_selected + static_cast<std::size_t>(row) * kCandidateStride,
            static_cast<std::size_t>(k));
        const std::uint64_t counter = params[row].counter + row;
        const auto actual_pick = sampler_tail(actual_row, k, params[row], counter);
        const auto expected_pick =
            reference_tail(selected_lists[row], k, params[row], counter);
        if (!actual_pick ||
            actual_pick->id != expected_pick.id ||
            float_bits(actual_pick->probability) != float_bits(expected_pick.probability)) {
            ++tail_mismatches;
            if (tail_mismatches <= 4) {
                std::printf("  tail row %u: expected (%d, %08x), got (%d, %08x)\n",
                            row, expected_pick.id, float_bits(expected_pick.probability),
                            actual_pick ? actual_pick->id : -1,
                            actual_pick ? float_bits(actual_pick->probability) : 0u);
            }
        }
    }
    check(tail_mismatches == 0, name + " CPU tail token and probability");
    return tail_mismatches == 0;
}

std::vector<float> make_small_logits(std::uint32_t rows, std::uint32_t n_vocab,
                                     std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::normal_distribution<float> normal(0.0f, 1.5f);
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_vocab);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t token = 0; token < n_vocab; ++token) {
            logits[static_cast<std::size_t>(row) * n_vocab + token] =
                normal(rng);
        }
    }
    return logits;
}

std::vector<SamplerParams> make_params(std::uint32_t rows, std::uint64_t seed,
                                       bool penalties) {
    constexpr std::array top_ks{1, 2, 20, 64, 0, 100, -3, 63};
    constexpr std::array top_ps{1.0f, 0.9f, 0.75f, 0.5f, 1.0f, 0.85f, 0.99f, 0.7f};
    constexpr std::array min_ps{0.0f, 0.0f, 0.05f, 0.0f, 0.2f, 0.01f, 0.0f, 0.05f};
    constexpr std::array penalty_windows{64, 0, 16, 32, 64, 8, 0, 48};
    std::vector<SamplerParams> params(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        SamplerParams& p = params[row];
        p.top_k = top_ks[row % top_ks.size()];
        p.top_p = top_ps[row % top_ps.size()];
        p.min_p = min_ps[row % min_ps.size()];
        p.temperature = 2.5f;
        p.min_keep = row % 3u == 0u ? 2 : 1;
        p.penalty_last_n = penalties ? penalty_windows[row % penalty_windows.size()] : 0;
        p.penalty_repeat = 1.25f;
        p.penalty_freq = 0.25f;
        p.penalty_present = 0.5f;
        p.seed = seed ^ (0x9e3779b97f4a7c15ULL * (row + 1u));
        p.counter = (std::uint64_t{1} << 32) + seed + row * 17u;
    }
    return params;
}

std::vector<SamplerParams> make_greedy_boundary_params(
    std::uint32_t rows, std::uint32_t history_stride, bool apply_penalties) {
    std::vector<SamplerParams> params(rows);
    for (SamplerParams& p : params) {
        p.top_k = 1;
        p.penalty_last_n = apply_penalties ? static_cast<int>(history_stride) : 0;
        p.penalty_repeat = 1.25f;
        p.penalty_freq = 0.125f;
        p.penalty_present = 0.25f;
    }
    return params;
}

std::vector<int> make_history(std::uint32_t rows, std::uint32_t history_stride,
                              std::uint32_t n_vocab, std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::vector<int> history(static_cast<std::size_t>(rows) * history_stride, -1);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t i = 0; i < history_stride; ++i) {
            int token = static_cast<int>(rng() % n_vocab);
            if (i % 3u == 1u && i != 0u) {
                token = history[static_cast<std::size_t>(row) * history_stride + i - 1u];
            }
            history[static_cast<std::size_t>(row) * history_stride + i] = token;
        }
    }
    if (history_stride >= 12u) {
        const std::size_t base = static_cast<std::size_t>(rows - 1u) * history_stride;
        const std::array edge_ids{0, 1023, 1024, 4095, 4096, 8191,
                                  static_cast<int>(n_vocab)};
        for (std::size_t i = 0; i < edge_ids.size(); ++i) {
            history[base + i] = edge_ids[i];
        }
        history[base + 7u] = -5;
        history[base + 8u] = static_cast<int>(n_vocab) + 77;
        history[base + 9u] = 0x7fffffff;
        if (history_stride >= 4u) {
            history[base + history_stride - 4u] = static_cast<int>(n_vocab);
            history[base + history_stride - 3u] = -5;
            history[base + history_stride - 2u] = static_cast<int>(n_vocab) + 77;
            history[base + history_stride - 1u] = 0x7fffffff;
        }
    }
    return history;
}

bool random_top_k_boundaries_are_separated(
    const std::vector<float>& logits, std::uint32_t rows, std::uint32_t n_vocab,
    const std::vector<int>& history, std::uint32_t history_stride,
    const std::vector<SamplerParams>& params) {
    constexpr float kMinimumRelativeGap = 1.0f / 65536.0f;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const int k = effective_k(params[row].top_k, n_vocab);
        if (k >= static_cast<int>(n_vocab)) continue;
        const std::uint32_t history_length = params[row].penalty_last_n <= 0
            ? 0u
            : std::min<std::uint32_t>(
                  history_stride, static_cast<std::uint32_t>(params[row].penalty_last_n));
        const std::size_t history_offset =
            static_cast<std::size_t>(row) * history_stride +
            history_stride - history_length;
        const std::span<const int> row_history =
            history_length == 0u
                ? std::span<const int>{}
                : std::span<const int>(history.data() + history_offset, history_length);
        const float* row_logits = logits.data() + static_cast<std::size_t>(row) * n_vocab;
        std::vector<SampCandidate> ranked;
        ranked.reserve(n_vocab);
        for (std::uint32_t token = 0; token < n_vocab; ++token) {
            const float value = penalize(
                row_logits[token], history_count(row_history, static_cast<int>(token)), params[row]);
            if (!std::isnan(value) && value != -std::numeric_limits<float>::infinity()) {
                ranked.push_back(make_candidate(static_cast<int>(token), value));
            }
        }
        if (ranked.size() <= static_cast<std::size_t>(k)) continue;
        const auto boundary = ranked.begin() + (k - 1);
        std::nth_element(ranked.begin(), boundary, ranked.end(), candidate_before);
        const auto next = std::min_element(boundary + 1, ranked.end(), candidate_before);
        const float selected = candidate_value(*boundary);
        const float rejected = candidate_value(*next);
        const float scale = std::max({1.0f, std::fabs(selected), std::fabs(rejected)});
        if (!(selected - rejected > scale * kMinimumRelativeGap)) return false;
    }
    return true;
}

int floor_div(int numerator, int denominator) {
    const int quotient = numerator / denominator;
    const int remainder = numerator % denominator;
    return quotient - (remainder != 0 && numerator < 0 ? 1 : 0);
}

int portable_fixture_normal_bin(std::mt19937& rng, int quantization_scale) {
    // The standardized mt19937 output feeds a fixed-point Irwin-Hall approximation.
    // Integer-only quantization avoids implementation-defined normal_distribution output.
    std::int64_t centered_sum = 0;
    for (int sample = 0; sample < 12; ++sample) {
        const std::uint32_t uniform = static_cast<std::uint32_t>(rng()) >> 16u;
        centered_sum += static_cast<std::int64_t>(uniform) - 32768;
    }
    const std::int64_t scaled = centered_sum * quantization_scale;
    constexpr std::int64_t denominator = 65536;
    const std::int64_t quotient = scaled / denominator;
    const std::int64_t remainder = scaled % denominator;
    return static_cast<int>(
        quotient - (remainder != 0 && scaled < 0 ? 1 : 0));
}

bool test_portable_fixture_rng_known_answers() {
    constexpr std::array fixture16_expected{-5, 4, 0, -5, 4, 4, -11, -1};
    constexpr std::array fixture17_expected{-3, 1, -4, 0, -4, -4, -7, -2};
    std::mt19937 fixture16_rng(2600u);
    std::mt19937 fixture17_rng(2212u);
    std::array<int, fixture16_expected.size()> fixture16_actual{};
    std::array<int, fixture17_expected.size()> fixture17_actual{};
    for (int& value : fixture16_actual) {
        value = portable_fixture_normal_bin(fixture16_rng, 4);
    }
    for (int& value : fixture17_actual) {
        value = portable_fixture_normal_bin(fixture17_rng, 3);
    }
    const bool passed =
        fixture16_actual == fixture16_expected &&
        fixture17_actual == fixture17_expected;
    check(passed, "portable sampler fixture RNG known answers");
    return passed;
}

std::vector<float> make_large_logits(std::uint32_t rows, std::uint32_t n_vocab,
                                     std::uint64_t seed);

std::vector<float> make_random_logits_with_separated_choices(
    std::uint32_t rows, std::uint32_t n_vocab, std::uint64_t seed,
    const std::vector<int>& history, std::uint32_t history_stride,
    const std::vector<SamplerParams>& params,
    const std::vector<SamplerParams>* additional_params = nullptr) {
    constexpr std::uint64_t kRetryStride = 0x9e3779b97f4a7c15ULL;
    std::vector<float> logits;
    for (std::uint64_t attempt = 0; attempt < 128u; ++attempt) {
        const std::uint64_t attempt_seed = seed + attempt * kRetryStride;
        logits = n_vocab > kBlockSpan
            ? make_large_logits(rows, n_vocab, attempt_seed)
            : make_small_logits(rows, n_vocab, attempt_seed);
        if (random_top_k_boundaries_are_separated(
                logits, rows, n_vocab, history, history_stride, params) &&
            (additional_params == nullptr ||
             random_top_k_boundaries_are_separated(
                 logits, rows, n_vocab, history, history_stride, *additional_params))) {
            return logits;
        }
    }
    check(false, "random sampler inputs find separated top-k boundaries");
    return logits;
}

std::vector<float> make_large_logits(std::uint32_t rows, std::uint32_t n_vocab,
                                    std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::normal_distribution<float> normal(0.0f, 1.5f);
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_vocab);
    for (float& value : logits) value = normal(rng);
    return logits;
}

std::vector<float> make_tie_fixture(std::uint32_t rows, std::uint32_t n_vocab,
                                    std::uint64_t seed) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_vocab);
    for (float& value : logits) value = std::floor(normal(rng) * 4.0f) / 4.0f;

    auto spread = [n_vocab](int j) {
        return static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(j) * 7919u + 13u) % n_vocab);
    };
    float* row0 = logits.data();
    for (int j = 0; j < 320; ++j) row0[spread(j)] = j < 20 ? 6.25f : 6.0f;
    row0[n_vocab / 2u] = inf;
    row0[n_vocab - 1u] = inf;

    if (rows > 1u) {
        float* row1 = logits.data() + n_vocab;
        for (std::uint32_t token = 0; token < n_vocab; ++token) {
            row1[token] = -std::fabs(row1[token]);
            if (row1[token] == 0.0f) row1[token] = (token & 1u) ? 0.0f : -0.0f;
        }
        row1[3] = inf;
    }

    if (rows > 2u) {
        float* row2 = logits.data() + 2u * n_vocab;
        std::fill_n(row2, n_vocab, -inf);
        for (int j = 0; j < 10; ++j) row2[spread(j + 400)] = j < 5 ? 1.0f : 0.5f;
        row2[spread(500)] = nan;
        row2[spread(501)] = nan;
        row2[spread(502)] = inf;
    }

    if (rows > 3u) {
        float* row3 = logits.data() + 3u * n_vocab;
        std::fill_n(row3, n_vocab, -8.0f);
        for (int j = 0; j < 30; ++j) row3[spread(j + 600)] = j < 15 ? 3.75f : 3.5f;
        row3[n_vocab / 2u] = inf;
    }
    return logits;
}

void make_upstream_fixture16(std::uint32_t n_vocab, std::vector<float>& logits,
                             std::vector<int>& history) {
    constexpr std::uint32_t rows = 4u;
    constexpr std::uint32_t history_size = 64u;
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rng(1600u + n_vocab);
    logits.resize(static_cast<std::size_t>(rows) * n_vocab);
    history.assign(static_cast<std::size_t>(rows) * history_size, -1);
    for (float& value : logits) {
        value = static_cast<float>(portable_fixture_normal_bin(rng, 4)) * 0.25f;
    }

    const auto spread = [n_vocab](int index) {
        return static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(index) * 7919u + 13u) % n_vocab);
    };

    float* row0 = logits.data();
    for (int j = 0; j < 320; ++j) row0[spread(j)] = j < 20 ? 6.25f : 6.0f;
    row0[n_vocab / 2u] = infinity;
    row0[n_vocab - 1u] = infinity;

    float* row1 = logits.data() + n_vocab;
    for (std::uint32_t token = 0; token < n_vocab; ++token) {
        row1[token] = -std::fabs(row1[token]);
        if (row1[token] == 0.0f) row1[token] = (token & 1u) ? 0.0f : -0.0f;
    }
    row1[3] = infinity;

    float* row2 = logits.data() + 2u * n_vocab;
    std::fill_n(row2, n_vocab, -infinity);
    for (int j = 0; j < 10; ++j) row2[spread(j + 400)] = j < 5 ? 1.0f : 0.5f;
    row2[spread(500)] = nan;
    row2[spread(501)] = nan;
    row2[spread(502)] = infinity;

    float* row3 = logits.data() + 3u * n_vocab;
    int history_count_written = 0;
    const std::array edges{0, 1023, 1024, 4095, 4096, 8191, 8192, 12345,
                           static_cast<int>(n_vocab / 2u + 1u),
                           static_cast<int>(n_vocab - 2u)};
    int* row_history = history.data() + 3u * history_size;
    for (const int token : edges) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= n_vocab ||
            static_cast<std::uint32_t>(token) == n_vocab / 2u) {
            continue;
        }
        row_history[history_count_written++] = token;
        row3[token] = 9.0f;
        if (history_count_written % 3 == 0) row_history[history_count_written++] = token;
    }
    row_history[history_count_written++] = static_cast<int>(n_vocab);
    row_history[history_count_written++] = static_cast<int>(n_vocab + 77u);
    row_history[history_count_written++] = -5;
    for (int j = 0; j < 30; ++j) row3[spread(j + 600)] = j < 15 ? 3.75f : 3.5f;
    row3[n_vocab / 2u] = infinity;
}

void make_upstream_fixture17(std::uint32_t n_vocab, std::vector<float>& logits,
                             std::vector<int>& history) {
    constexpr std::uint32_t rows = 17u;
    constexpr std::uint32_t history_size = 64u;
    std::mt19937 rng(1700u + n_vocab);
    logits.resize(static_cast<std::size_t>(rows) * n_vocab);
    history.resize(static_cast<std::size_t>(rows) * history_size);
    for (float& value : logits) {
        value = static_cast<float>(portable_fixture_normal_bin(rng, 3)) * 0.5f;
    }
    if (n_vocab == 512u) {
        for (float& value : logits) {
            const int half_step_bin = static_cast<int>(value * 2.0f);
            value = static_cast<float>(floor_div(half_step_bin, 4));
        }
    }

    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* row_logits = logits.data() + static_cast<std::size_t>(row) * n_vocab;
        const float maximum = *std::max_element(row_logits, row_logits + n_vocab);
        std::vector<int> head;
        for (std::uint32_t token = 0; token < n_vocab; ++token) {
            if (row_logits[token] >= maximum - 1.0f) {
                head.push_back(static_cast<int>(token));
            }
        }
        int* row_history = history.data() + static_cast<std::size_t>(row) * history_size;
        for (std::uint32_t i = 0; i < history_size; ++i) {
            row_history[i] = (i & 1u)
                ? static_cast<int>(rng() % n_vocab)
                : head[static_cast<std::size_t>(rng() % head.size())];
        }
    }
}

std::vector<float> make_fixture_logits(std::uint32_t rows, std::uint32_t n_vocab,
                                       std::uint64_t seed) {
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_vocab);
    const int phase = static_cast<int>(seed % 23u);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t token = 0; token < n_vocab; ++token) {
            const std::uint32_t bucket = (token + row * 7u + static_cast<std::uint32_t>(phase)) % 29u;
            float value = static_cast<float>(static_cast<int>(bucket) - 14) * 0.25f;
            if ((token + row + static_cast<std::uint32_t>(phase)) % 127u == 0u) value = 4.0f;
            logits[static_cast<std::size_t>(row) * n_vocab + token] = value;
        }
    }
    return logits;
}

std::vector<float> make_test_logits(std::uint32_t rows, std::uint32_t n_vocab,
                                    std::uint64_t seed, twin::InputSet input_set) {
    if (input_set == twin::InputSet::edge) {
        return make_tie_fixture(rows, n_vocab, seed);
    }
    if (input_set == twin::InputSet::fixture) {
        return make_fixture_logits(rows, n_vocab, seed);
    }
    return n_vocab > kBlockSpan
        ? make_large_logits(rows, n_vocab, seed)
        : make_small_logits(rows, n_vocab, seed);
}

bool test_coupled_penalty(Context& context, std::vector<float>& expected_logits,
                          std::vector<int>& history_out,
                          std::vector<SamplerParams>& params_out,
                          bool shared_encoder = false) {
    constexpr std::uint32_t nv = 64u;
    constexpr std::uint32_t n_vocab = 256u;
    constexpr std::uint32_t history_length = 12u;
    std::vector<float> original(nv);
    for (std::uint32_t i = 0; i < nv; ++i) original[i] = 8.0f - 0.125f * i;

    std::vector<int> id_to_sub(n_vocab, -1);
    id_to_sub[5] = 0;
    id_to_sub[7] = 1;
    id_to_sub[9] = 2;
    id_to_sub[11] = 3;
    std::vector<int> ring{5, 5, 7, 9, 9, 9, 11, 300, -3, 7, 255, 5};
    history_out = ring;

    SamplerParams params;
    params.top_k = 20;
    params.top_p = 0.9f;
    params.min_p = 0.05f;
    params.temperature = 1.5f;
    params.min_keep = 2;
    params.penalty_last_n = history_length;
    params.penalty_repeat = 1.5f;
    params.penalty_freq = 0.25f;
    params.penalty_present = 0.5f;
    params.seed = 0x123456789abcdef0ULL;
    params.counter = (std::uint64_t{1} << 32) + 75u;
    params_out.assign(1u, params);

    expected_logits = original;
    const std::array tokens{5, 7, 9, 11};
    for (const int token : tokens) {
        const int subset = id_to_sub[static_cast<std::size_t>(token)];
        expected_logits[static_cast<std::size_t>(subset)] =
            penalize(expected_logits[static_cast<std::size_t>(subset)],
                     history_count(ring, token), params);
    }
    SampRow row = make_row(params, 0u, history_length);
    SamplerParams sampled_params = params;
    sampled_params.penalty_last_n = 0;
    const SampRow sampled_row = make_row(sampled_params, 0u, 0u);
    const int sampled_k = effective_k(sampled_params.top_k, nv);
    const auto expected_sampled_block = reference_block(
        expected_logits.data(), nv, 0u, {}, sampled_params, sampled_k);
    const auto expected_sampled = reference_merge(
        std::vector<std::vector<SampCandidate>>{expected_sampled_block},
        nv, sampled_k);
    std::vector<SampCandidate> expected_candidate_bytes(kCandidateStride);
    std::vector<SampCandidate> expected_selected_bytes(kCandidateStride);
    std::copy(expected_sampled_block.begin(), expected_sampled_block.end(),
              expected_candidate_bytes.begin());
    std::copy(expected_sampled.begin(), expected_sampled.end(),
              expected_selected_bytes.begin());

    std::string err;
    twin::GuardedBuffer logits, map, history, row_buffer;
    twin::GuardedBuffer sampled_rows, candidates, selected;
    check(logits.allocate(context, "coupled penalty logits", original.size() * sizeof(float), err),
          "coupled penalty logits buffer", err);
    check(map.allocate(context, "coupled penalty inverse map", id_to_sub.size() * sizeof(int), err),
          "coupled penalty map buffer", err);
    check(history.allocate(context, "coupled penalty history", ring.size() * sizeof(int), err),
          "coupled penalty history buffer", err);
    check(row_buffer.allocate(context, "coupled penalty row", sizeof(row), err),
          "coupled penalty row buffer", err);
    if (!logits.view() || !map.view() || !history.view() || !row_buffer.view()) return false;
    std::memcpy(logits.payload().data(), original.data(), original.size() * sizeof(float));
    std::memcpy(map.payload().data(), id_to_sub.data(), id_to_sub.size() * sizeof(int));
    std::memcpy(history.payload().data(), ring.data(), ring.size() * sizeof(int));
    std::memcpy(row_buffer.payload().data(), &row, sizeof(row));
    SamplerBuffers sampled_buffers{};
    if (shared_encoder) {
        check(sampled_rows.allocate(context, "coupled sampled rows",
                                    sizeof(sampled_row), err),
              "coupled sampled row buffer", err);
        check(candidates.allocate(context, "coupled sampled candidates",
                                  expected_candidate_bytes.size() * sizeof(SampCandidate), err),
              "coupled sampled candidates buffer", err);
        check(selected.allocate(context, "coupled sampled selected",
                                expected_selected_bytes.size() * sizeof(SampCandidate), err),
              "coupled sampled selected buffer", err);
        if (!sampled_rows.view() || !candidates.view() || !selected.view()) return false;
        std::memcpy(sampled_rows.payload().data(), &sampled_row, sizeof(sampled_row));
        check(candidates.poison_bytes_against(bytes_of(expected_candidate_bytes), err),
              "coupled sampled candidates poison", err);
        check(selected.poison_bytes_against(bytes_of(expected_selected_bytes), err),
              "coupled sampled selected poison", err);
        sampled_buffers = {logits.view(), history.view(), sampled_rows.view(),
                           candidates.view(), selected.view(), {}};
        sampled_rows.capture_before_dispatch();
        candidates.capture_before_dispatch();
        selected.capture_before_dispatch();
    }

    logits.capture_before_dispatch();
    map.capture_before_dispatch();
    history.capture_before_dispatch();
    row_buffer.capture_before_dispatch();
    auto command = context.make_command_buffer("coupled draft penalty", err);
    if (!command) {
        check(false, "coupled penalty command buffer", err);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "coupled draft penalty shared Encoder",
              [&](Encoder& encoder, std::string& encode_error) {
                  if (!encode_sampler_coupled_penalty(
                          context, encoder, n_vocab, nv, true, logits.view(),
                          map.view(), history.view(), row_buffer.view(),
                          encode_error)) {
                      return false;
                  }
                  return encode_sampler_sampled(
                      context, encoder, 1u, nv, sampled_buffers, true,
                      encode_error, 0u);
              },
              err)
        : encode_sampler_coupled_penalty(
              context, *command, n_vocab, nv, true, logits.view(), map.view(),
              history.view(), row_buffer.view(), err);
    if (!encoded) {
        command->discard();
        check(false, "production coupled sampler penalty encode", err);
        return false;
    }
    if (!submit_and_wait(command, "coupled draft penalty", err)) {
        check(false, "production coupled sampler penalty submit", err);
        return false;
    }

    int wrong = 0;
    const auto actual = logits.payload();
    for (std::uint32_t i = 0; i < nv; ++i) {
        float got = 0.0f;
        std::memcpy(&got, actual.data() + i * sizeof(float), sizeof(got));
        if (float_bits(got) != float_bits(expected_logits[i])) ++wrong;
    }
    check(wrong == 0, "coupled penalty applies duplicate frequency and boolean presence exactly");
    std::vector<twin::ByteRange> written;
    for (const int token : tokens) {
        const std::size_t offset =
            static_cast<std::size_t>(id_to_sub[static_cast<std::size_t>(token)]) * sizeof(float);
        written.push_back({offset, sizeof(float)});
    }
    if (wrong == 0) {
        for (std::uint32_t i = 0; i < nv; ++i) {
            std::memcpy(&expected_logits[i],
                        actual.data() + i * sizeof(float), sizeof(float));
        }
    }
    check(!logits.first_unwritten_change(written),
          "coupled penalty leaves unrelated logits and guards unchanged");
    check(!map.first_unwritten_change({}), "coupled penalty map unchanged");
    check(!history.first_unwritten_change({}), "coupled penalty history unchanged");
    check(!row_buffer.first_unwritten_change({}), "coupled penalty row unchanged");
    bool shared_sampled_passed = true;
    if (shared_encoder) {
        const std::array sample_range{
            twin::ByteRange{0u, static_cast<std::size_t>(sampled_k) *
                                    sizeof(SampCandidate)}};
        const auto candidate_failure = compare_ranges(
            candidates.payload(), bytes_of(expected_candidate_bytes), sample_range);
        const auto selected_failure = compare_ranges(
            selected.payload(), bytes_of(expected_selected_bytes), sample_range);
        check(!candidate_failure,
              "coupled shared sampled stage sees the penalty result");
        check(!selected_failure,
              "coupled shared merge returns the penalized top-k list");
        check(!sampled_rows.first_unwritten_change({}),
              "coupled shared sampled row unchanged");
        check(!candidates.first_unwritten_change(sample_range),
              "coupled shared sampled candidates range and guards");
        check(!selected.first_unwritten_change(sample_range),
              "coupled shared selected range and guards");
        shared_sampled_passed = !candidate_failure && !selected_failure;
    }
    return wrong == 0 && shared_sampled_passed;
}

bool test_greedy_matrix(Context& context, const twin::Options& options) {
    for (std::uint32_t rows = 1; rows <= kSamplerMaxRows; ++rows) {
        constexpr std::uint32_t n_vocab = 512u;
        auto history = make_history(rows, 16u, n_vocab,
                                    options.seed ^ (0x5200u + rows));
        auto penalized_params = make_greedy_boundary_params(rows, 16u, true);
        auto unpenalized_params = make_greedy_boundary_params(rows, 16u, false);
        const auto* additional_params = rows == 1u ? &unpenalized_params : nullptr;
        auto logits = options.input_set == twin::InputSet::random
            ? make_random_logits_with_separated_choices(
                  rows, n_vocab, options.seed ^ (0x5100u + rows),
                  history, 16u, penalized_params, additional_params)
            : make_test_logits(rows, n_vocab,
                               options.seed ^ (0x5100u + rows), options.input_set);
        if (!run_greedy_case(context, rows, n_vocab, logits, history, 16u,
                             options, "greedy T=" + std::to_string(rows))) {
            return false;
        }
        if (rows == 1u &&
            !run_greedy_case(context, rows, n_vocab, logits, history, 16u,
                             options, "greedy ignores stale history", false)) {
            return false;
        }
    }

    constexpr std::uint32_t rows = 8u;
    constexpr std::uint32_t n_vocab = 248320u;
    auto history = make_history(rows, 64u, n_vocab, options.seed ^ 0x5400u);
    auto boundary_params = make_greedy_boundary_params(rows, 64u, true);
    auto logits = options.input_set == twin::InputSet::random
        ? make_random_logits_with_separated_choices(
              rows, n_vocab, options.seed ^ 0x5300u,
              history, 64u, boundary_params)
        : make_test_logits(rows, n_vocab, options.seed ^ 0x5300u, options.input_set);
    if (!run_greedy_case(context, rows, n_vocab, logits, history, 64u,
                         options, "greedy 248320 T=8")) {
        return false;
    }

    constexpr std::uint32_t max_vocab_rows = 8u;
    constexpr std::uint32_t max_vocab = kSamplerMaxVocab;
    auto max_history = make_history(max_vocab_rows, 64u, max_vocab,
                                    options.seed ^ 0x5600u);
    auto max_boundary_params =
        make_greedy_boundary_params(max_vocab_rows, 64u, true);
    auto max_logits = options.input_set == twin::InputSet::random
        ? make_random_logits_with_separated_choices(
              max_vocab_rows, max_vocab, options.seed ^ 0x5500u,
              max_history, 64u, max_boundary_params)
        : make_test_logits(max_vocab_rows, max_vocab,
                           options.seed ^ 0x5500u, options.input_set);
    return run_greedy_case(context, max_vocab_rows, max_vocab,
                           max_logits, max_history, 64u,
                           options, "greedy 262144 T=8");
}

bool test_upstream_fixture16(Context& context, const twin::Options& options);
bool test_upstream_fixture17(Context& context, const twin::Options& options);

bool test_sampler_matrix(Context& context, const twin::Options& options) {
    constexpr std::uint32_t small_vocab = 512u;
    for (std::uint32_t rows = 1; rows <= kSamplerMaxRows; ++rows) {
        auto small_history = make_history(rows, 64u, small_vocab,
                                          options.seed ^ (0x6200u + rows));
        auto small_params = make_params(rows, options.seed ^ (0x6300u + rows), true);
        auto small_logits = options.input_set == twin::InputSet::random
            ? make_random_logits_with_separated_choices(
                  rows, small_vocab, options.seed ^ (0x6100u + rows),
                  small_history, 64u, small_params)
            : make_test_logits(rows, small_vocab,
                               options.seed ^ (0x6100u + rows), options.input_set);
        const std::string split_name =
            "split 512 T=" + std::to_string(rows) + " ties/filters/penalties";
        if (!run_split_case(context, rows, small_vocab, small_logits, small_history,
                            64u, small_params, options, split_name)) {
            return false;
        }
        if (!run_split_case(context, rows, small_vocab, small_logits, small_history,
                            64u, small_params, options,
                            split_name + " shared split stages", false, 0u,
                            false, true)) {
            return false;
        }
        if (rows == 1u || rows == kSamplerMaxRows) {
            if (!run_split_case(context, rows, small_vocab, small_logits,
                                small_history, 64u, small_params, options,
                                split_name + " shared sampled composite", false,
                                0u, false, false, true)) {
                return false;
            }
        }
    }

    constexpr std::uint32_t model_rows = 8u;
    constexpr std::uint32_t model_vocab = 248320u;
    auto model_history = make_history(model_rows, 64u, model_vocab,
                                      options.seed ^ 0x7200u);
    auto model_params = make_params(model_rows, options.seed ^ 0x7300u, true);
    auto model_logits = options.input_set == twin::InputSet::random
        ? make_random_logits_with_separated_choices(
              model_rows, model_vocab, options.seed ^ 0x7100u,
              model_history, 64u, model_params)
        : make_test_logits(model_rows, model_vocab,
                           options.seed ^ 0x7100u, options.input_set);
    if (!run_split_case(context, model_rows, model_vocab, model_logits, model_history,
                        64u, model_params, options, "split 248320 / 61 blocks")) {
        return false;
    }

    constexpr std::uint32_t edge_rows = 4u;
    constexpr std::uint32_t edge_vocab = 262144u;
    auto edge_logits = make_tie_fixture(edge_rows, edge_vocab, options.seed ^ 0x8100u);
    auto edge_history = make_history(edge_rows, 64u, edge_vocab, options.seed ^ 0x8200u);
    auto edge_params = make_params(edge_rows, options.seed ^ 0x8300u, true);
    for (std::uint32_t row = 0; row < edge_rows; ++row) {
        edge_params[row].top_k = 64;
        edge_params[row].top_p = row & 1u ? 0.5f : 1.0f;
        edge_params[row].min_p = 0.0f;
    }
    if (!run_split_case(context, edge_rows, edge_vocab, edge_logits, edge_history,
                        64u, edge_params, options, "split 262144 / ties and sentinels")) {
        return false;
    }
    if (options.input_set == twin::InputSet::fixture) {
        return test_upstream_fixture16(context, options) &&
               test_upstream_fixture17(context, options);
    }
    return true;
}

bool test_sampler_shared_large_sizes(Context& context,
                                     const twin::Options& options) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t history_stride = 64u;
    constexpr std::array vocab_sizes{248320u, kSamplerMaxVocab};
    for (const std::uint32_t n_vocab : vocab_sizes) {
        auto history = make_history(rows, history_stride, n_vocab,
                                    options.seed ^ n_vocab);
        auto params = make_params(rows, options.seed ^ (n_vocab + 17u), true);
        auto logits = options.input_set == twin::InputSet::random
            ? make_random_logits_with_separated_choices(
                  rows, n_vocab, options.seed ^ (n_vocab + 31u),
                  history, history_stride, params)
            : make_test_logits(rows, n_vocab, options.seed ^ (n_vocab + 31u),
                               options.input_set);
        const std::string name = "shared sampled " + std::to_string(n_vocab) +
            " vocabulary merge scratch";
        if (!run_split_case(context, rows, n_vocab, logits, history,
                            history_stride, params, options, name,
                            false, 0u, false, false, true)) {
            return false;
        }
    }
    return true;
}

struct SamplerFixtureTask {
    std::uint32_t source_row;
    SamplerParams params;
};

bool run_sampler_fixture_tasks(Context& context, std::uint32_t n_vocab,
                               std::uint32_t history_stride,
                               const std::vector<float>& source_logits,
                               const std::vector<int>& source_history,
                               const std::vector<SamplerFixtureTask>& tasks,
                               const twin::Options& options,
                               const std::string& fixture_name) {
    const std::size_t source_row_logits = n_vocab;
    for (std::size_t first = 0; first < tasks.size(); first += kSamplerMaxRows) {
        const std::uint32_t rows = static_cast<std::uint32_t>(
            std::min<std::size_t>(kSamplerMaxRows, tasks.size() - first));
        std::vector<float> logits(static_cast<std::size_t>(rows) * n_vocab);
        std::vector<int> history(static_cast<std::size_t>(rows) * history_stride);
        std::vector<SamplerParams> params(rows);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const SamplerFixtureTask& task = tasks[first + row];
            const std::size_t source = static_cast<std::size_t>(task.source_row);
            std::copy_n(source_logits.begin() + source * source_row_logits, n_vocab,
                        logits.begin() + static_cast<std::size_t>(row) * n_vocab);
            std::copy_n(source_history.begin() + source * history_stride, history_stride,
                        history.begin() + static_cast<std::size_t>(row) * history_stride);
            params[row] = task.params;
            params[row].counter += task.source_row;
            params[row].counter -= row;
        }
        const std::string name = fixture_name + " batch=" + std::to_string(first / kSamplerMaxRows);
        if (!run_split_case(context, rows, n_vocab, logits, history, history_stride,
                            params, options, name)) {
            return false;
        }
    }
    return true;
}

bool test_upstream_fixture16(Context& context, const twin::Options& options) {
    if (!test_portable_fixture_rng_known_answers()) return false;

    const bool accepted_limit = sampler_shape_supported(4u, kSamplerMaxVocab);
    const bool rejected_above_limit = !sampler_shape_supported(4u, kSamplerMaxVocab + 1u);
    check(accepted_limit && rejected_above_limit,
          "upstream fixture 16 vocabulary limit boundary");
    if (!accepted_limit || !rejected_above_limit) return false;

    constexpr std::uint32_t rows = 4u;
    constexpr std::uint32_t history_stride = 64u;
    constexpr std::array vocabularies{248320u, 100003u, 262144u, 1000u};
    for (const std::uint32_t n_vocab : vocabularies) {
        std::vector<float> logits;
        std::vector<int> history;
        make_upstream_fixture16(n_vocab, logits, history);
        std::vector<SamplerFixtureTask> tasks;
        tasks.reserve(67u * rows);
        for (int top_k_case = 0; top_k_case <= 66; ++top_k_case) {
            SamplerParams params;
            params.top_k = top_k_case == 65 ? 100 : top_k_case == 66 ? -3 : top_k_case;
            params.top_p = (top_k_case & 1) ? 0.5f : 1.0f;
            params.min_p = 0.0f;
            params.temperature = 0.8f;
            params.seed = 16u;
            params.counter = static_cast<std::uint64_t>(top_k_case);
            params.penalty_last_n = static_cast<int>(history_stride);
            params.penalty_repeat = 2.0f;
            params.penalty_freq = 0.25f;
            params.penalty_present = 0.5f;
            for (std::uint32_t row = 0; row < rows; ++row) {
                tasks.push_back({row, params});
            }
        }
        if (!run_sampler_fixture_tasks(
                context, n_vocab, history_stride, logits, history, tasks, options,
                "upstream fixture 16 n_vocab=" + std::to_string(n_vocab))) {
            return false;
        }
    }
    return true;
}

bool test_upstream_fixture17(Context& context, const twin::Options& options) {
    constexpr std::uint32_t source_rows = 17u;
    constexpr std::uint32_t history_stride = 64u;
    constexpr std::array vocabularies{248320u, 512u};
    constexpr std::array top_ks{1, 20, 64};
    constexpr std::array top_ps{0.9f, 1.0f};
    constexpr std::array min_ps{0.0f, 0.05f};

    for (const std::uint32_t n_vocab : vocabularies) {
        std::vector<float> logits;
        std::vector<int> history;
        make_upstream_fixture17(n_vocab, logits, history);

        std::array<std::vector<std::vector<SampCandidate>>, 2> ordered_top64;
        for (int penalty_mode = 0; penalty_mode < 2; ++penalty_mode) {
            auto& row_lists = ordered_top64[static_cast<std::size_t>(penalty_mode)];
            row_lists.resize(source_rows);
            SamplerParams rank_params;
            rank_params.penalty_last_n = penalty_mode ? static_cast<int>(history_stride) : 0;
            rank_params.penalty_repeat = 1.25f;
            rank_params.penalty_freq = 0.25f;
            rank_params.penalty_present = 0.5f;
            const std::uint32_t n_blocks =
                (n_vocab + kBlockSpan - 1u) / kBlockSpan;
            for (std::uint32_t source_row = 0; source_row < source_rows; ++source_row) {
                const std::size_t row_offset =
                    static_cast<std::size_t>(source_row) * n_vocab;
                const std::span<const int> row_history = penalty_mode
                    ? std::span<const int>(
                          history.data() + static_cast<std::size_t>(source_row) * history_stride,
                          history_stride)
                    : std::span<const int>{};
                std::vector<std::vector<SampCandidate>> block_lists;
                block_lists.reserve(n_blocks);
                for (std::uint32_t block = 0; block < n_blocks; ++block) {
                    block_lists.push_back(reference_block(
                        logits.data() + row_offset, n_vocab, block, row_history, rank_params, 64));
                }
                row_lists[source_row] = reference_merge(block_lists, n_vocab, 64);
            }
        }

        std::vector<SamplerFixtureTask> tasks;
        tasks.reserve(816u);
        std::uint64_t config = 0;
        int tied_picks = 0;
        for (int penalty_mode = 0; penalty_mode < 2; ++penalty_mode) {
            for (const int top_k : top_ks) {
                for (const float top_p : top_ps) {
                    for (const float min_p : min_ps) {
                        // Preserve both upstream draw counters; Metal submits both through this test's queue.
                        for (int counter_variant = 0; counter_variant < 2; ++counter_variant) {
                            SamplerParams params;
                            params.top_k = top_k;
                            params.top_p = top_p;
                            params.min_p = min_p;
                            params.temperature = 2.5f;
                            params.seed = 17u;
                            params.counter = 1000u * ++config;
                            params.penalty_last_n =
                                penalty_mode ? static_cast<int>(history_stride) : 0;
                            params.penalty_repeat = 1.25f;
                            params.penalty_freq = 0.25f;
                            params.penalty_present = 0.5f;
                            const int k = effective_k(top_k, n_vocab);
                            for (std::uint32_t source_row = 0;
                                 source_row < source_rows; ++source_row) {
                                const auto& selected =
                                    ordered_top64[static_cast<std::size_t>(penalty_mode)][source_row];
                                const TailPick pick = reference_tail(
                                    std::span<const SampCandidate>(selected.data(),
                                                                  static_cast<std::size_t>(k)),
                                    k, params, params.counter + source_row);
                                const auto picked = std::find_if(
                                    selected.begin(), selected.begin() + k,
                                    [&pick](const SampCandidate& candidate) {
                                        return candidate.id == pick.id;
                                    });
                                if (picked != selected.begin() + k) {
                                    const float picked_logit = candidate_value(*picked);
                                    tied_picks += std::count_if(
                                        selected.begin(), selected.begin() + k,
                                        [picked_logit](const SampCandidate& candidate) {
                                            return candidate_value(candidate) == picked_logit;
                                        }) > 1;
                                }
                                tasks.push_back({source_row, params});
                            }
                        }
                    }
                }
            }
        }
        check(tasks.size() == 816u && tied_picks > 0,
              "upstream fixture 17 exposes sampled ties n_vocab=" +
                  std::to_string(n_vocab));
        if (tasks.size() != 816u || tied_picks == 0) return false;
        if (!run_sampler_fixture_tasks(
                context, n_vocab, history_stride, logits, history, tasks, options,
                "upstream fixture 17 n_vocab=" + std::to_string(n_vocab))) {
            return false;
        }
    }
    return true;
}

bool test_sampler_semantic_fixtures(Context& context, const twin::Options& options) {
    {
        constexpr std::uint32_t n_vocab = 512u;
        constexpr std::uint32_t history_stride = 8u;
        std::vector<float> logits(n_vocab, -8.0f);
        logits[0] = 6.0f;
        logits[3] = 6.5f;
        const std::vector<int> history{0, 0, 0, 0, 3, 3, 3, 3};
        SamplerParams tail_params;
        tail_params.top_k = 2;
        tail_params.top_p = 1.0f;
        tail_params.temperature = 0.0f;
        tail_params.penalty_last_n = 4;
        tail_params.penalty_repeat = 3.0f;
        tail_params.penalty_freq = 0.3f;
        tail_params.penalty_present = 0.5f;
        const std::span<const int> tail_history(history.data() + 4, 4);
        const auto tail_candidates =
            reference_block(logits.data(), n_vocab, 0u, tail_history, tail_params, 2);
        const auto full_candidates =
            reference_block(logits.data(), n_vocab, 0u, history, tail_params, 2);
        const bool tail_window_is_observable =
            tail_candidates[0].id != full_candidates[0].id;
        check(tail_window_is_observable,
              "penalty_last_n selects only the newest history suffix");
        if (!tail_window_is_observable) return false;
        if (!run_split_case(context, 1u, n_vocab, logits, history, history_stride,
                            std::vector<SamplerParams>{tail_params}, options,
                            "penalty history tail")) {
            return false;
        }
    }

    constexpr std::uint32_t rows = 8u;
    constexpr std::uint32_t n_vocab = 512u;
    constexpr std::uint32_t history_stride = 8u;
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_vocab, -12.0f);
    std::vector<int> history(static_cast<std::size_t>(rows) * history_stride, -1);
    std::vector<SamplerParams> params(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t favorite = 100u + row;
        const std::uint32_t runner_up = 200u + row;
        logits[static_cast<std::size_t>(row) * n_vocab + favorite] = 8.0f;
        logits[static_cast<std::size_t>(row) * n_vocab + runner_up] = 6.5f;
        history[static_cast<std::size_t>(row) * history_stride] =
            static_cast<int>(favorite);

        params[row].top_k = 2;
        params[row].top_p = 1.0f;
        params[row].min_p = 0.0f;
        params[row].temperature = 1.0f;
        params[row].penalty_last_n = static_cast<int>(history_stride);
        params[row].penalty_repeat = 1.25f;
        params[row].penalty_freq = 0.125f;
        params[row].penalty_present = 0.25f;
        params[row].seed = 0x1200u + row;
        params[row].counter = 0x3400u + row;
    }

    const std::span<const int> own_history(
        history.data() + history_stride, history_stride);
    const std::span<const int> row_zero_history(history.data(), history_stride);
    const auto own_candidates =
        reference_block(logits.data() + n_vocab, n_vocab, 0u, own_history, params[1], 2);
    const auto broadcast_candidates =
        reference_block(logits.data() + n_vocab, n_vocab, 0u, row_zero_history, params[1], 2);
    const bool row_history_is_observable =
        own_candidates[0].id != broadcast_candidates[0].id;
    check(row_history_is_observable,
          "per-row sampler history changes the selected top candidate");
    if (!row_history_is_observable) return false;

    if (!run_greedy_case(context, rows, n_vocab, logits, history, history_stride,
                         options, "per-row history greedy")) {
        return false;
    }
    if (!run_greedy_case(context, rows, n_vocab, logits, history, history_stride,
                         options, "per-row history greedy shared Encoder", true,
                         true)) {
        return false;
    }
    return run_split_case(context, rows, n_vocab, logits, history, history_stride,
                          params, options, "per-row history sampled");
}

bool test_coupled_chain(Context& context, const twin::Options& options,
                        bool shared_encoder = false) {
    std::vector<float> penalized_logits;
    std::vector<int> history;
    std::vector<SamplerParams> params;
    if (!test_coupled_penalty(context, penalized_logits, history, params,
                              shared_encoder)) {
        return false;
    }
    constexpr std::uint32_t nv = 64u;
    return run_split_case(
        context, 1u, nv, penalized_logits, history,
        static_cast<std::uint32_t>(history.size()), params, options,
        shared_encoder ? "coupled selected list and CPU tail shared sampled"
                       : "coupled selected list and CPU tail",
        true, nv, false, false, shared_encoder);
}

bool test_sampler_encoder_validation(Context& context) {
    std::string error;

    auto shape_command = context.make_command_buffer("sampler invalid shape", error);
    if (!shape_command) {
        check(false, "sampler invalid-shape command buffer", error);
        return false;
    }
    const bool rejected_shape =
        !encode_sampler_greedy(context, *shape_command, 9u, 4u, {}, error) &&
        error.find("T=1..8") != std::string::npos;
    check(rejected_shape, "sampler encoder rejects unsupported row count", error);
    shape_command->discard();

    auto coupled_shape_command =
        context.make_command_buffer("coupled sampler invalid vocabulary", error);
    if (!coupled_shape_command) {
        check(false, "coupled sampler invalid-vocabulary command buffer", error);
        return false;
    }
    const bool rejected_invalid_subset =
        !encode_sampler_coupled_penalty(
            context, *coupled_shape_command, 4u, 5u, true,
            {}, {}, {}, {}, error) &&
        error.find("valid subset") != std::string::npos;
    check(rejected_invalid_subset,
          "coupled sampler rejects a sampled vocabulary larger than its base", error);
    coupled_shape_command->discard();

    BufferRef backing = context.buffers().allocate(
        "sampler encoder validation backing", 4096u, error, true);
    if (!backing) {
        check(false, "sampler encoder validation backing", error);
        return false;
    }

    SamplerBuffers invalid_buffers{};
    invalid_buffers.logits = backing;
    invalid_buffers.logits.length = sizeof(float) * 3u;
    auto short_command = context.make_command_buffer("sampler short buffer", error);
    if (!short_command) {
        check(false, "sampler short-buffer command buffer", error);
        return false;
    }
    const bool rejected_short_buffer =
        !encode_sampler_greedy(context, *short_command, 1u, 4u,
                               invalid_buffers, error) &&
        error.find("undersized") != std::string::npos;
    check(rejected_short_buffer, "sampler encoder rejects undersized buffer", error);
    short_command->discard();
    const bool shared_short_buffer = expect_shared_sampler_rejected(
        context, "sampler shared undersized buffer", "undersized",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_sampler_greedy(context, encoder, 1u, 4u,
                                         invalid_buffers, encode_error);
        });

    invalid_buffers.logits = backing;
    invalid_buffers.logits = backing.slice(1u, backing.length - 1u);
    auto misaligned_command = context.make_command_buffer("sampler misaligned buffer", error);
    if (!misaligned_command) {
        check(false, "sampler misaligned-buffer command buffer", error);
        return false;
    }
    const bool rejected_misaligned_buffer =
        !encode_sampler_greedy(context, *misaligned_command, 1u, 4u,
                               invalid_buffers, error) &&
        error.find("misaligned") != std::string::npos;
    check(rejected_misaligned_buffer, "sampler encoder rejects misaligned buffer", error);
    misaligned_command->discard();
    const bool shared_misaligned_buffer = expect_shared_sampler_rejected(
        context, "sampler shared misaligned buffer", "misaligned",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_sampler_greedy(context, encoder, 1u, 4u,
                                         invalid_buffers, encode_error);
        });

    SampRow row{};
    row.temperature = 1.0f;
    row.top_p = 1.0f;
    row.min_keep = 1;
    row.top_k = 2;
    std::memcpy(backing.contents, &row, sizeof(row));

    SamplerBuffers overlapping_buffers{
        backing, backing, backing, backing, backing, backing};
    auto overlap_command = context.make_command_buffer("sampler overlapping buffers", error);
    if (!overlap_command) {
        check(false, "sampler overlapping-buffer command buffer", error);
        return false;
    }
    const bool rejected_overlap =
        !encode_sampler_greedy(context, *overlap_command, 1u, 4u,
                               overlapping_buffers, error) &&
        error.find("views overlap") != std::string::npos;
    check(rejected_overlap, "sampler encoder rejects overlapping buffer views", error);
    overlap_command->discard();
    const bool shared_overlap = expect_shared_sampler_rejected(
        context, "sampler shared overlapping buffer views", "views overlap",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_sampler_greedy(context, encoder, 1u, 4u,
                                         overlapping_buffers, encode_error);
        });

    auto subview = [&](std::size_t relative_offset, std::size_t length) {
        return backing.slice(relative_offset, length);
    };
    SamplerBuffers valid_buffers{
        subview(0u, sizeof(float) * 4u),
        subview(16u, sizeof(std::int32_t)),
        subview(32u, sizeof(SampRow)),
        subview(128u, 64u * sizeof(SampCandidate)),
        subview(640u, 64u * sizeof(SampCandidate)),
        subview(1152u, sizeof(std::int32_t)),
    };
    std::memcpy(valid_buffers.rows.contents, &row, sizeof(row));

    SamplerBuffers forged_cpu_metadata = valid_buffers;
    forged_cpu_metadata.rows.contents =
        static_cast<std::uint8_t*>(valid_buffers.rows.contents) + sizeof(std::uint32_t);
    auto forged_cpu_command =
        context.make_command_buffer("sampler forged CPU metadata", error);
    if (!forged_cpu_command) {
        check(false, "sampler forged-CPU-metadata command buffer", error);
        return false;
    }
    const bool rejects_forged_cpu_metadata =
        !encode_sampler_greedy(context, *forged_cpu_command, 1u, 4u,
                               forged_cpu_metadata, error) &&
        error.find("not owned by this Context") != std::string::npos;
    check(rejects_forged_cpu_metadata,
          "sampler rejects forged CPU backing metadata", error);
    forged_cpu_command->discard();
    const bool shared_forged_cpu_metadata = expect_shared_sampler_rejected(
        context, "sampler shared forged CPU metadata", "not owned by this Context",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_sampler_greedy(context, encoder, 1u, 4u,
                                         forged_cpu_metadata, encode_error);
        });

    SamplerBuffers forged_gpu_metadata = valid_buffers;
    forged_gpu_metadata.rows.gpu_address += sizeof(std::uint32_t);
    auto forged_gpu_command =
        context.make_command_buffer("sampler forged GPU metadata", error);
    if (!forged_gpu_command) {
        check(false, "sampler forged-GPU-metadata command buffer", error);
        return false;
    }
    const bool rejects_forged_gpu_metadata =
        !encode_sampler_greedy(context, *forged_gpu_command, 1u, 4u,
                               forged_gpu_metadata, error) &&
        error.find("not owned by this Context") != std::string::npos;
    check(rejects_forged_gpu_metadata,
          "sampler rejects forged GPU backing metadata", error);
    forged_gpu_command->discard();
    const bool shared_forged_gpu_metadata = expect_shared_sampler_rejected(
        context, "sampler shared forged GPU metadata", "not owned by this Context",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_sampler_greedy(context, encoder, 1u, 4u,
                                         forged_gpu_metadata, encode_error);
        });

    ContextOptions foreign_options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    if (!foreign_context) {
        check(false, "open foreign sampler Context", error);
        return false;
    }
    auto foreign_command =
        foreign_context->make_command_buffer("foreign sampler command", error);
    const bool rejects_foreign_command =
        foreign_command &&
        !encode_sampler_sampled(context, *foreign_command, 1u, 4u,
                                valid_buffers, false, error) &&
        error.find("different Context") != std::string::npos;
    check(rejects_foreign_command,
          "sampler rejects a command buffer from another Context", error);
    if (foreign_command) foreign_command->discard();

    BufferRef foreign_logits = foreign_context->buffers().allocate(
        "foreign sampler logits", sizeof(float) * 4u, error);
    SamplerBuffers foreign_buffer_views = valid_buffers;
    foreign_buffer_views.logits = foreign_logits;
    auto foreign_buffer_command =
        context.make_command_buffer("foreign sampler buffer", error);
    const bool rejects_foreign_buffer =
        foreign_logits && foreign_buffer_command &&
        !encode_sampler_greedy(context, *foreign_buffer_command, 1u, 4u,
                               foreign_buffer_views, error) &&
        error.find("not owned by this Context") != std::string::npos;
    check(rejects_foreign_buffer,
          "sampler rejects a buffer from another Context", error);
    if (foreign_buffer_command) foreign_buffer_command->discard();
    const bool shared_foreign_buffer = expect_shared_sampler_rejected(
        context, "sampler shared foreign buffer", "not owned by this Context",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_sampler_greedy(context, encoder, 1u, 4u,
                                         foreign_buffer_views, encode_error);
        });

    const long page_size = getpagesize();
    void* alias_memory = nullptr;
    if (page_size <= 0 ||
        posix_memalign(&alias_memory, static_cast<std::size_t>(page_size),
                       static_cast<std::size_t>(page_size)) != 0) {
        check(false, "allocate sampler no-copy alias fixture");
        return false;
    }
    std::shared_ptr<void> alias_lifetime(alias_memory, std::free);
    BufferRef alias_logits = context.buffers().wrap_no_copy(
        "sampler alias logits", alias_memory, static_cast<std::size_t>(page_size),
        alias_lifetime, error);
    BufferRef alias_history = context.buffers().wrap_no_copy(
        "sampler alias history", alias_memory, static_cast<std::size_t>(page_size),
        alias_lifetime, error);
    if (!alias_logits || !alias_history) {
        check(false, "create sampler no-copy alias fixtures", error);
        return false;
    }
    check(alias_logits.handle != alias_history.handle,
          "sampler no-copy alias fixtures use distinct Metal handles");
    alias_logits.length = sizeof(float) * 4u;
    alias_history.length = sizeof(std::int32_t);
    SamplerBuffers distinct_handle_aliases{
        alias_logits, alias_history, valid_buffers.rows, valid_buffers.candidates,
        valid_buffers.selected, valid_buffers.output};
    auto alias_command = context.make_command_buffer("sampler no-copy aliases", error);
    if (!alias_command) {
        check(false, "sampler no-copy alias command buffer", error);
        return false;
    }
    const bool rejected_distinct_handle_aliases =
        !encode_sampler_greedy(context, *alias_command, 1u, 4u,
                               distinct_handle_aliases, error) &&
        error.find("views overlap") != std::string::npos;
    check(rejected_distinct_handle_aliases,
          "sampler encoder rejects aliased no-copy views", error);
    alias_command->discard();
    const bool shared_distinct_handle_aliases = expect_shared_sampler_rejected(
        context, "sampler shared no-copy aliases", "views overlap",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_sampler_greedy(context, encoder, 1u, 4u,
                                         distinct_handle_aliases, encode_error);
        });

    auto kpart_command = context.make_command_buffer("sampler insufficient kpart", error);
    if (!kpart_command) {
        check(false, "sampler kpart command buffer", error);
        return false;
    }
    const bool rejected_small_kpart =
        !encode_sampler_sampled(context, *kpart_command, 1u, 4u, valid_buffers,
                                false, error, 1u) &&
        error.find("smaller than the requested top-k") != std::string::npos;
    check(rejected_small_kpart, "sampler encoder rejects insufficient kpart", error);
    kpart_command->discard();

    auto merge_kpart_command =
        context.make_command_buffer("sampler merge insufficient kpart", error);
    if (!merge_kpart_command) {
        check(false, "sampler merge kpart command buffer", error);
        return false;
    }
    const bool rejected_small_merge_kpart =
        !encode_sampler_split_merge(context, *merge_kpart_command, 1u, 4u,
                                    valid_buffers.rows, valid_buffers.candidates,
                                    valid_buffers.selected, 1u, error) &&
        error.find("smaller than the requested top-k") != std::string::npos;
    check(rejected_small_merge_kpart,
          "sampler merge encoder rejects insufficient kpart", error);
    merge_kpart_command->discard();

    bool discarded_prior_sampler_dispatch = !concurrent_dispatch_mode();
    if (concurrent_dispatch_mode()) {
        auto prior_dispatch_command = context.make_command_buffer(
            "sampler split dispatch then rejected merge", error);
        if (!prior_dispatch_command) {
            check(false, "sampler prior-dispatch command buffer", error);
            return false;
        }
        bool split_part_encoded = false;
        bool merge_encoded = true;
        bool group_closed = false;
        std::string merge_error;
        const bool segment_encoded = Encoder::encode(
            context, *prior_dispatch_command,
            "sampler prior split part is discarded after merge rejection",
            [&](Encoder& encoder, std::string& body_error) {
                if (!encoder.group_begin(body_error)) return false;
                split_part_encoded = encode_sampler_split_part(
                    context, encoder, 1u, 4u, valid_buffers, false, 2u,
                    body_error);
                if (!split_part_encoded) return false;
                merge_encoded = encode_sampler_split_merge(
                    context, encoder, 1u, 4u, valid_buffers.rows,
                    valid_buffers.candidates, valid_buffers.selected, 2u,
                    merge_error);
                std::string group_error;
                group_closed = encoder.group_end(group_error);
                body_error = !merge_encoded
                    ? merge_error
                    : !group_closed
                        ? group_error
                        : "sampler merge unexpectedly succeeded in an open group";
                return false;
            },
            error);
        const bool command_discarded =
            prior_dispatch_command->unsafe_native_handle() == nullptr;
        const bool refused_and_discarded = split_part_encoded && !merge_encoded &&
            group_closed && !segment_encoded && command_discarded &&
            error.find("ordered concurrent-group boundary") != std::string::npos;
        check(refused_and_discarded,
              "shared merge rejection discards earlier split-part dispatch", error);
        prior_dispatch_command->discard();
        discarded_prior_sampler_dispatch = refused_and_discarded;
    }

    row.top_k = 1;
    std::memcpy(valid_buffers.rows.contents, &row, sizeof(row));
    auto enqueued_command = context.make_command_buffer("sampler enqueued command", error);
    if (!enqueued_command || !enqueued_command->enqueue(error)) {
        check(false, "sampler command buffer enqueue for state validation", error);
        if (enqueued_command) enqueued_command->discard();
        return false;
    }
    const bool rejected_enqueued_command =
        !encode_sampler_greedy(context, *enqueued_command, 1u, 4u,
                               valid_buffers, error) &&
        error.find("before its command buffer is enqueued or committed") !=
            std::string::npos;
    check(rejected_enqueued_command,
          "sampler encoder rejects already-enqueued command buffer", error);
    const bool drained = submit_and_wait(enqueued_command,
                                         "sampler enqueued command", error);
    check(drained, "sampler enqueued validation command is safely drained", error);

    return rejected_shape && rejected_invalid_subset &&
           rejected_short_buffer && shared_short_buffer &&
           rejected_misaligned_buffer && shared_misaligned_buffer &&
           rejected_overlap && shared_overlap && rejected_small_kpart &&
           rejected_small_merge_kpart && discarded_prior_sampler_dispatch &&
           rejected_distinct_handle_aliases && shared_distinct_handle_aliases &&
           rejects_forged_cpu_metadata && shared_forged_cpu_metadata &&
           rejects_forged_gpu_metadata && shared_forged_gpu_metadata &&
           rejects_foreign_command && rejects_foreign_buffer &&
           shared_foreign_buffer && rejected_enqueued_command && drained;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_sampler_parity_test strata.metallib "
                     "[--coupled] [--seed N] [--input-set random|edge|fixture]\n");
        return 2;
    }

    const bool coupled_only = argc >= 3 && std::string_view(argv[2]) == "--coupled";
    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, coupled_only ? 3 : 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    if (options.input_set == twin::InputSet::real) {
        std::fprintf(stderr, "metal_sampler_parity_test has no model-backed real input set\n");
        return 77;
    }
    const bool reference_rng_is_known =
        reference_philox_uniform(0, 0) == 0.9584041237831116f &&
        reference_philox_uniform(1, 0) == 0.3436594605445862f &&
        reference_philox_uniform(123, (std::uint64_t{1} << 32) + 7) == 0.7804952263832092f &&
        reference_philox_uniform(0x123456789abcdef0ULL, 0xfedcba9876543210ULL) ==
            0.9908959865570068f;
    check(reference_rng_is_known, "independent upstream Philox reference known answers");
    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;

    bool passed = test_sampler_encoder_validation(*context);
    if (passed) {
        if (coupled_only) {
            passed = test_coupled_chain(*context, options) &&
                     test_coupled_chain(*context, options, true);
        } else if (options.input_set_explicit) {
            passed = test_greedy_matrix(*context, options) &&
                     test_sampler_matrix(*context, options) &&
                     test_coupled_chain(*context, options) &&
                     test_coupled_chain(*context, options, true);
        } else {
            constexpr std::array input_sets{
                twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
            for (const twin::InputSet input_set : input_sets) {
                twin::Options case_options = options;
                case_options.input_set = input_set;
                case_options.input_set_explicit = true;
                std::printf("sampler input set: %s\n", twin::input_set_name(input_set));
                passed = test_greedy_matrix(*context, case_options) &&
                         test_sampler_matrix(*context, case_options);
                if (!passed) break;
            }
            if (passed) passed = test_sampler_semantic_fixtures(*context, options);
            if (passed) {
                passed = test_coupled_chain(*context, options) &&
                         test_coupled_chain(*context, options, true);
            }
        }
    }
    if (passed && !coupled_only) {
        passed = test_sampler_shared_large_sizes(*context, options);
    }
    check(passed && g_failures == 0, "Metal sampler parity");
    std::printf("metal_sampler_parity_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
