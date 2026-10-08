// Device-free assembly of Window predecessor rows for the artifact PLE hash.
#include "strata/metal/window_ple_rows.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strata::metal {

std::uint32_t WindowPleRows::token_count() const noexcept {
    return token_count_;
}

std::span<const std::uint32_t> WindowPleRows::all_rows() const noexcept {
    return {row_ids_.data(),
            static_cast<std::size_t>(token_count_) *
                kWindowPleRowsPerToken};
}

std::span<const std::uint32_t> WindowPleRows::rows_for(
    std::uint32_t token_row) const noexcept {
    if (token_row >= token_count_ || token_row >= kWindowPleMaxTokens) {
        return {};
    }
    const std::size_t offset =
        static_cast<std::size_t>(token_row) * kWindowPleRowsPerToken;
    return {row_ids_.data() + offset, kWindowPleRowsPerToken};
}

bool build_window_ple_rows(
    std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> committed_history,
    const Geometry& geometry, WindowPleRows& output, std::string& error) {
    const auto fail = [&error](const char* message) {
        error = message;
        return false;
    };

    if (tokens.empty() || tokens.size() > kWindowPleMaxTokens) {
        return fail("Window PLE row building requires 1..8 tokens");
    }
    if (committed_history.size() > 2u) {
        return fail("committed PLE history must contain at most two token IDs");
    }
    if (!is_qwen4exp_runtime_geometry(geometry)) {
        return fail("Window PLE rows require supported qwen4exp runtime geometry");
    }
    for (const std::uint32_t token : tokens) {
        if (token >= geometry.vocab) {
            return fail("Window token ID is outside the model vocabulary");
        }
    }
    for (const std::uint32_t token : committed_history) {
        if (token >= geometry.vocab) {
            return fail("committed history token ID is outside the model vocabulary");
        }
    }

    const kernels::PleConsts constants = kernels::ple_artifact_consts();
    for (int head = 0; head < kernels::PLE_N_HEADS; ++head) {
        const std::uint64_t offset = constants.offset[head];
        const std::uint64_t vocabulary = constants.vocab[head];
        if (vocabulary == 0u ||
            offset > kernels::PLE_TABLE_ROWS ||
            vocabulary > kernels::PLE_TABLE_ROWS - offset ||
            offset > std::numeric_limits<std::uint32_t>::max() ||
            vocabulary >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::uint32_t>::max()) -
                    offset) {
            return fail("PLE artifact hash constants exceed the table geometry");
        }
    }

    std::array<std::int32_t, kWindowPleMaxTokens> token_ids{};
    std::array<std::int32_t, kWindowPleMaxTokens * 2u> predecessors{};
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        token_ids[row] = static_cast<std::int32_t>(tokens[row]);
    }

    std::array<std::int32_t, 2> history{
        kernels::TOKEN_NULL, kernels::TOKEN_NULL};
    if (committed_history.size() == 1u) {
        history[1] = static_cast<std::int32_t>(committed_history[0]);
    } else if (committed_history.size() == 2u) {
        history[0] = static_cast<std::int32_t>(committed_history[0]);
        history[1] = static_cast<std::int32_t>(committed_history[1]);
    }
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        predecessors[row * 2u] = history[0];
        predecessors[row * 2u + 1u] = history[1];
        history[0] = history[1];
        history[1] = token_ids[row];
    }

    WindowPleRows candidate;
    candidate.token_count_ = static_cast<std::uint32_t>(tokens.size());
    kernels::ngram_rows(token_ids.data(), predecessors.data(),
                        static_cast<int>(tokens.size()), constants,
                        candidate.row_ids_.data());
    for (std::size_t row = 0;
         row < tokens.size() * kWindowPleRowsPerToken; ++row) {
        if (candidate.row_ids_[row] >= kernels::PLE_TABLE_ROWS) {
            return fail("PLE artifact hash produced a row outside the table");
        }
    }

    output = candidate;
    error.clear();
    return true;
}

bool WindowPleTokenHistory::replace_committed_history(
    std::span<const std::uint32_t> committed_prefix,
    const Geometry& geometry, std::string& error) {
    const auto fail = [&error](const char* message) {
        error = message;
        return false;
    };

    if (!is_qwen4exp_runtime_geometry(geometry)) {
        return fail(
            "committed PLE token history requires supported qwen4exp runtime geometry");
    }
    for (const std::uint32_t token : committed_prefix) {
        if (token >= geometry.vocab) {
            return fail("committed history token ID is outside the model vocabulary");
        }
    }

    std::array<std::uint32_t, 2> candidate{};
    const std::size_t count = std::min(committed_prefix.size(), candidate.size());
    for (std::size_t index = 0; index < count; ++index) {
        candidate[index] =
            committed_prefix[committed_prefix.size() - count + index];
    }
    committed_ids_ = candidate;
    committed_count_ = static_cast<std::uint32_t>(count);
    error.clear();
    return true;
}

void WindowPleTokenHistory::reset() noexcept {
    committed_ids_.fill(0u);
    committed_count_ = 0u;
}

std::span<const std::uint32_t>
WindowPleTokenHistory::committed_suffix() const noexcept {
    return {committed_ids_.data(), committed_count_};
}

bool WindowPleTokenHistory::build_rows(
    std::span<const std::uint32_t> tokens, const Geometry& geometry,
    WindowPleRows& output, std::string& error) const {
    return build_window_ple_rows(tokens, committed_suffix(), geometry, output,
                                 error);
}

bool WindowPleTokenHistory::commit_accepted_prefix(
    std::span<const std::uint32_t> window_tokens, std::size_t accepted_count,
    const Geometry& geometry, std::string& error) {
    if (accepted_count > window_tokens.size()) {
        error = "accepted token prefix exceeds the speculative window";
        return false;
    }

    // Reuse the row builder's validation contract before publishing state.
    // This also rejects malformed tokens in the speculative suffix rather
    // than allowing an invalid candidate window to reach the commit boundary.
    WindowPleRows validated_rows;
    if (!build_rows(window_tokens, geometry, validated_rows, error)) {
        return false;
    }

    auto candidate = committed_ids_;
    std::uint32_t candidate_count = committed_count_;
    for (std::size_t index = 0; index < accepted_count; ++index) {
        if (candidate_count < candidate.size()) {
            candidate[candidate_count++] = window_tokens[index];
        } else {
            candidate[0] = candidate[1];
            candidate[1] = window_tokens[index];
        }
    }

    committed_ids_ = candidate;
    committed_count_ = candidate_count;
    error.clear();
    return true;
}

}  // namespace strata::metal
