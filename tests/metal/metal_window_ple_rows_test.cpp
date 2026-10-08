// Device-free tests for assembling Window token history into PLE row indices.
#include "strata/metal/window_ple_rows.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::metal;

Geometry supported_geometry() {
    Geometry geometry;
    geometry.family = "qwen4exp";
    geometry.n_layers = 48u;
    geometry.n_qsa_layers = 12u;
    geometry.n_kv_head = 2u;
    geometry.head_dim = 256u;
    geometry.n_embd = 2'560u;
    geometry.vocab = 248'320u;
    return geometry;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::array<std::uint32_t, kWindowPleRowsPerToken> reference_rows(
    std::span<const std::uint32_t> tokens,
    std::span<const std::int32_t> predecessors) {
    require(tokens.size() <= kWindowPleMaxTokens,
            "reference token rows fit the fixed Window capacity");
    require(predecessors.size() == tokens.size() * 2u,
            "reference predecessor matrix has two entries per token");

    std::array<std::int32_t, kWindowPleMaxTokens> token_ids{};
    std::array<std::int32_t, kWindowPleMaxTokens * 2u> prev{};
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        token_ids[row] = static_cast<std::int32_t>(tokens[row]);
        prev[row * 2u] = predecessors[row * 2u];
        prev[row * 2u + 1u] = predecessors[row * 2u + 1u];
    }

    std::array<std::uint32_t,
               kWindowPleMaxTokens * kWindowPleRowsPerToken> expected{};
    strata::kernels::ngram_rows(
        token_ids.data(), prev.data(), static_cast<int>(tokens.size()),
        strata::kernels::ple_artifact_consts(), expected.data());
    std::array<std::uint32_t, kWindowPleRowsPerToken> result{};
    for (std::size_t head = 0; head < result.size(); ++head) {
        result[head] = expected[head];
    }
    return result;
}

void require_token_rows(const WindowPleRows& actual, std::uint32_t row,
                        const std::array<std::uint32_t,
                                         kWindowPleRowsPerToken>& expected) {
    const auto actual_rows = actual.rows_for(row);
    require(actual_rows.size() == expected.size(),
            "each in-range token row exposes exactly sixteen PLE rows");
    for (std::size_t head = 0; head < expected.size(); ++head) {
        require(actual_rows[head] == expected[head],
                "PLE row IDs match the explicit predecessor oracle");
    }
}

void require_history(const WindowPleTokenHistory& history,
                     std::span<const std::uint32_t> expected) {
    const auto actual = history.committed_suffix();
    require(actual.size() == expected.size(),
            "committed token suffix has the expected length");
    require(std::equal(actual.begin(), actual.end(), expected.begin()),
            "committed token suffix retains the expected IDs oldest first");
}

void require_same_rows(const WindowPleRows& actual,
                       const WindowPleRows& expected) {
    require(actual.token_count() == expected.token_count(),
            "integrated row builder preserves the candidate token count");
    require(std::equal(actual.all_rows().begin(), actual.all_rows().end(),
                       expected.all_rows().begin(), expected.all_rows().end()),
            "integrated row IDs equal explicit construction from the suffix");
}

void test_empty_and_partial_history() {
    std::string error = "stale error";
    WindowPleRows rows;
    const Geometry geometry = supported_geometry();
    const std::array<std::uint32_t, 1> token_zero{0u};
    require(build_window_ple_rows(token_zero, {}, geometry, rows, error),
            "token ID zero is valid with no committed history: " + error);
    require(error.empty(), "successful construction clears the error string");
    require(rows.token_count() == 1u,
            "T=1 publishes exactly one token row");
    require(rows.all_rows().size() == kWindowPleRowsPerToken,
            "the packed token-major view covers exactly T rows");
    require(rows.rows_for(1u).empty(),
            "row lookup at token_count is rejected");
    require(rows.rows_for(kWindowPleMaxTokens).empty(),
            "row lookup beyond fixed storage is rejected");

    const std::array<std::int32_t, 2> no_history{
        strata::kernels::TOKEN_NULL, strata::kernels::TOKEN_NULL};
    const auto expected_zero = reference_rows(token_zero, no_history);
    const std::array<std::uint32_t, kWindowPleRowsPerToken> pinned_zero{
        768795u, 32189829u, 51641306u, 71063218u, 85886852u, 104959504u,
        134316676u, 154856127u, 173280301u, 194832882u, 202840854u,
        222492280u, 256138797u, 271010889u, 296240796u, 304432235u};
    require(expected_zero == pinned_zero,
            "empty-history token-zero hash matches the artifact fixture");
    require_token_rows(rows, 0u, pinned_zero);

    const std::array<std::uint32_t, 1> one_history{17u};
    const std::array<std::uint32_t, 1> one_token{41u};
    require(build_window_ple_rows(one_token, one_history, geometry, rows,
                                  error),
            "partial committed history is accepted: " + error);
    const std::array<std::int32_t, 2> partial_predecessors{
        strata::kernels::TOKEN_NULL, 17};
    require_token_rows(rows, 0u,
                       reference_rows(one_token, partial_predecessors));

    const std::array<std::uint32_t, 2> eos_history{7u, 248'044u};
    require(build_window_ple_rows(token_zero, eos_history, geometry, rows,
                                  error),
            "EOS in committed history is accepted: " + error);
    const std::array<std::int32_t, 2> eos_predecessors{7, 248'044};
    require_token_rows(rows, 0u,
                       reference_rows(token_zero, eos_predecessors));
}

void test_all_window_sizes_and_eos() {
    const Geometry geometry = supported_geometry();
    const std::array<std::uint32_t, kWindowPleMaxTokens> tokens{
        2u, 3u, 248'044u, 5u, 6u, 7u, 8u, 9u};
    const std::array<std::uint32_t, 2> committed{31u, 0u};
    const std::array<std::array<std::int32_t, 2>, kWindowPleMaxTokens>
        explicit_predecessors{{
            {{31, 0}},
            {{0, 2}},
            {{2, 3}},
            {{3, 248'044}},
            {{248'044, 5}},
            {{5, 6}},
            {{6, 7}},
            {{7, 8}},
        }};
    for (std::uint32_t count = 1; count <= kWindowPleMaxTokens; ++count) {
        WindowPleRows rows;
        std::string error;
        require(build_window_ple_rows(
                    std::span<const std::uint32_t>(tokens.data(), count),
                    committed, geometry, rows, error),
                "build every supported Window size: " + error);
        require(rows.token_count() == count,
                "reported token count matches T");
        require(rows.all_rows().size() ==
                    static_cast<std::size_t>(count) *
                        kWindowPleRowsPerToken,
                "packed token-major view has the exact active extent");
        for (std::uint32_t row = 0; row < count; ++row) {
            const auto expected = reference_rows(
                std::span<const std::uint32_t>(&tokens[row], 1u),
                std::span<const std::int32_t>(
                    explicit_predecessors[row].data(), 2u));
            require_token_rows(rows, row, expected);
        }
        require(rows.rows_for(count).empty(),
                "row lookup immediately beyond T is empty");
    }

    WindowPleRows rows;
    std::string error;
    require(build_window_ple_rows(tokens, committed, geometry, rows, error),
            "build the full T=8 EOS fixture: " + error);
    for (std::uint32_t row = 0; row < kWindowPleMaxTokens; ++row) {
        const auto expected = reference_rows(
            std::span<const std::uint32_t>(&tokens[row], 1u),
            std::span<const std::int32_t>(
                explicit_predecessors[row].data(), 2u));
        require_token_rows(rows, row, expected);
    }
}

void test_transactional_rejection_and_id_limits() {
    WindowPleRows rows;
    const Geometry geometry = supported_geometry();
    const std::array<std::uint32_t, 2> valid_tokens{11u, 12u};
    const std::array<std::uint32_t, 2> valid_history{8u, 9u};
    std::string error;
    require(build_window_ple_rows(valid_tokens, valid_history, geometry,
                                  rows, error),
            "build baseline result: " + error);
    const auto saved_first = rows.rows_for(0u);
    const auto saved_second = rows.rows_for(1u);
    const std::array<std::uint32_t, kWindowPleRowsPerToken> first_before = [&] {
        std::array<std::uint32_t, kWindowPleRowsPerToken> copy{};
        std::copy(saved_first.begin(), saved_first.end(), copy.begin());
        return copy;
    }();
    const std::array<std::uint32_t, kWindowPleRowsPerToken> second_before = [&] {
        std::array<std::uint32_t, kWindowPleRowsPerToken> copy{};
        std::copy(saved_second.begin(), saved_second.end(), copy.begin());
        return copy;
    }();

    const auto rejects_without_mutation =
        [&](std::span<const std::uint32_t> tokens,
            std::span<const std::uint32_t> history,
            const Geometry& candidate_geometry,
            const std::string& expected_error) {
            error.clear();
            require(!build_window_ple_rows(tokens, history,
                                           candidate_geometry, rows, error),
                    "invalid input must be rejected");
            require(error == expected_error,
                    "invalid input reports its specific validation failure");
            require(rows.token_count() == 2u,
                    "failed construction preserves the published token count");
            require_token_rows(rows, 0u, first_before);
            require_token_rows(rows, 1u, second_before);
        };

    const std::array<std::uint32_t, 0> no_tokens{};
    rejects_without_mutation(no_tokens, {}, geometry,
                             "Window PLE row building requires 1..8 tokens");
    const std::array<std::uint32_t, 9> too_many{};
    rejects_without_mutation(too_many, {}, geometry,
                             "Window PLE row building requires 1..8 tokens");
    const std::array<std::uint32_t, 3> too_much_history{1u, 2u, 3u};
    rejects_without_mutation(valid_tokens, too_much_history, geometry,
                             "committed PLE history must contain at most two token IDs");

    const std::array<std::uint32_t, 1> invalid_token{248'320u};
    rejects_without_mutation(invalid_token, {}, geometry,
                             "Window token ID is outside the model vocabulary");
    const std::array<std::uint32_t, 1> invalid_history{248'320u};
    rejects_without_mutation(valid_tokens, invalid_history, geometry,
                             "committed history token ID is outside the model vocabulary");

    Geometry wrong_vocabulary = geometry;
    wrong_vocabulary.vocab = 248'321u;
    rejects_without_mutation(
        valid_tokens, {}, wrong_vocabulary,
        "Window PLE rows require supported qwen4exp runtime geometry");
    Geometry wrong_family = geometry;
    wrong_family.family = "unsupported";
    rejects_without_mutation(
        valid_tokens, {}, wrong_family,
        "Window PLE rows require supported qwen4exp runtime geometry");
}

void test_committed_token_history_and_speculative_acceptance() {
    const Geometry geometry = supported_geometry();
    WindowPleTokenHistory history;
    std::string error = "stale error";
    const std::array<std::uint32_t, 0> empty_applied_prefix{};
    require(history.replace_committed_history(empty_applied_prefix, geometry,
                                              error),
            "initialize an empty history for a one-token prompt: " + error);
    require_history(history, empty_applied_prefix);
    const std::array<std::uint32_t, 1> single_prompt_pending_token{77u};
    WindowPleRows integrated;
    WindowPleRows explicit_rows;
    require(history.build_rows(single_prompt_pending_token, geometry,
                               integrated, error),
            "build the first decode row of a one-token prompt: " + error);
    require(build_window_ple_rows(single_prompt_pending_token,
                                  empty_applied_prefix, geometry,
                                  explicit_rows, error),
            "build the one-token prompt from empty applied history: " + error);
    require_same_rows(integrated, explicit_rows);
    require_history(history, empty_applied_prefix);
    require(history.commit_accepted_prefix(single_prompt_pending_token, 1u,
                                           geometry, error),
            "commit the pending one-token prompt decode: " + error);
    require_history(history, single_prompt_pending_token);

    const std::array<std::uint32_t, 3> prompt_prefix_applied{1u, 2u, 3u};
    require(history.replace_committed_history(prompt_prefix_applied, geometry,
                                              error),
            "initialize only from prompt tokens already applied to state: " +
                error);
    const std::array<std::uint32_t, 2> prompt_suffix{2u, 3u};
    require_history(history, prompt_suffix);
    require(error.empty(), "successful initialization clears prior errors");

    const std::array<std::uint32_t, 1> pending_prompt_token{4u};
    require(history.build_rows(pending_prompt_token, geometry, integrated,
                               error),
            "build the first decode row for the pending prompt token: " +
                error);
    require(build_window_ple_rows(pending_prompt_token, prompt_suffix,
                                  geometry, explicit_rows, error),
            "build the first decode row from the applied prompt prefix: " +
                error);
    require_same_rows(integrated, explicit_rows);
    require_history(history, prompt_suffix);
    require(history.commit_accepted_prefix(pending_prompt_token, 1u, geometry,
                                           error),
            "advance prompt history only after its first decode commit: " +
                error);
    const std::array<std::uint32_t, 2> after_prompt_decode{3u, 4u};
    require_history(history, after_prompt_decode);

    const std::array<std::uint32_t, 3> speculative_window{21u, 22u, 23u};
    require(history.build_rows(speculative_window, geometry, integrated,
                               error),
            "build rows from session-owned committed history: " + error);
    require(build_window_ple_rows(speculative_window, after_prompt_decode,
                                  geometry,
                                  explicit_rows, error),
            "build explicit rows for the same committed suffix: " + error);
    require_same_rows(integrated, explicit_rows);
    require_history(history, after_prompt_decode);

    require(history.commit_accepted_prefix(speculative_window, 2u, geometry,
                                           error),
            "commit only the accepted speculative prefix: " + error);
    const std::array<std::uint32_t, 2> accepted_suffix{21u, 22u};
    require_history(history, accepted_suffix);

    const std::array<std::uint32_t, 2> next_window{24u, 25u};
    require(history.build_rows(next_window, geometry, integrated, error),
            "build the next window after accepted-prefix commit: " + error);
    require(build_window_ple_rows(next_window, accepted_suffix, geometry,
                                  explicit_rows, error),
            "build the next window from the explicit accepted suffix: " +
                error);
    require_same_rows(integrated, explicit_rows);

    require(history.commit_accepted_prefix(next_window, 0u, geometry, error),
            "a zero-token acceptance is a valid no-op: " + error);
    require_history(history, accepted_suffix);

    const std::array<std::uint32_t, kWindowPleMaxTokens> full_window{
        30u, 31u, 32u, 33u, 34u, 35u, 36u, 37u};
    require(history.commit_accepted_prefix(full_window, full_window.size(),
                                           geometry, error),
            "commit all accepted rows in a T=8 window: " + error);
    const std::array<std::uint32_t, 2> full_suffix{36u, 37u};
    require_history(history, full_suffix);

    const std::array<std::uint32_t, 2> one_token_window{38u, 39u};
    require(history.commit_accepted_prefix(one_token_window, 1u, geometry,
                                           error),
            "advance the suffix by exactly one accepted token: " + error);
    const std::array<std::uint32_t, 2> one_token_suffix{37u, 38u};
    require_history(history, one_token_suffix);
}

void test_token_history_transactional_rejection_and_reset() {
    const Geometry geometry = supported_geometry();
    WindowPleTokenHistory history;
    std::string error;
    const std::array<std::uint32_t, 2> initial{8u, 9u};
    require(history.replace_committed_history(initial, geometry, error),
            "establish transaction baseline: " + error);

    const auto rejects_without_mutation =
        [&](auto&& operation, const std::string& expected_error) {
            error.clear();
            require(!operation(), "invalid token-history operation is rejected");
            require(error == expected_error,
                    "invalid token-history operation reports its failure");
            require_history(history, initial);
        };

    const std::array<std::uint32_t, 2> invalid_prompt{10u, 248'320u};
    rejects_without_mutation(
        [&] {
            return history.replace_committed_history(invalid_prompt, geometry,
                                                     error);
        },
        "committed history token ID is outside the model vocabulary");

    Geometry wrong_geometry = geometry;
    wrong_geometry.family = "unsupported";
    rejects_without_mutation(
        [&] {
            return history.replace_committed_history(initial, wrong_geometry,
                                                     error);
        },
        "committed PLE token history requires supported qwen4exp runtime geometry");

    const std::array<std::uint32_t, 2> valid_window{11u, 12u};
    rejects_without_mutation(
        [&] {
            return history.commit_accepted_prefix(valid_window, 3u, geometry,
                                                   error);
        },
        "accepted token prefix exceeds the speculative window");

    const std::array<std::uint32_t, 2> invalid_window{11u, 248'320u};
    rejects_without_mutation(
        [&] {
            return history.commit_accepted_prefix(invalid_window, 1u,
                                                   geometry, error);
        },
        "Window token ID is outside the model vocabulary");

    rejects_without_mutation(
        [&] {
            WindowPleRows rows;
            return history.build_rows(invalid_window, geometry, rows, error);
        },
        "Window token ID is outside the model vocabulary");

    history.reset();
    require_history(history, {});
    const std::array<std::uint32_t, 1> first_token{0u};
    require(history.commit_accepted_prefix(first_token, 1u, geometry, error),
            "a reset history accepts its first committed token: " + error);
    require_history(history, first_token);

    history.reset();
    require_history(history, {});
    require(history.replace_committed_history({}, geometry, error),
            "an empty committed prompt initializes empty PLE history: " + error);
    require_history(history, {});
}

}  // namespace

int main() {
    try {
        test_empty_and_partial_history();
        test_all_window_sizes_and_eos();
        test_transactional_rejection_and_id_limits();
        test_committed_token_history_and_speculative_acceptance();
        test_token_history_transactional_rejection_and_reset();
    } catch (const std::exception& e) {
        std::cerr << "metal_window_ple_rows_test: " << e.what() << '\n';
        return 1;
    }
    std::cout << "metal_window_ple_rows_test: all checks passed\n";
    return 0;
}
