// Device-free assembly of Window predecessor rows for the artifact PLE hash.
#include "strata/metal/window_ple_rows.hpp"

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

}  // namespace strata::metal
