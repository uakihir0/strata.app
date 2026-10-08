// Device-free preparation of the PLE table row indices for a Window.
#pragma once

#include "strata/kernels/ngram.hpp"
#include "strata/metal/geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace strata::metal {

inline constexpr std::uint32_t kWindowPleMaxTokens = 8u;
inline constexpr std::uint32_t kWindowPleRowsPerToken =
    static_cast<std::uint32_t>(kernels::PLE_N_HEADS);

class WindowPleRows {
public:
    std::uint32_t token_count() const noexcept;
    std::span<const std::uint32_t> all_rows() const noexcept;
    std::span<const std::uint32_t> rows_for(
        std::uint32_t token_row) const noexcept;

private:
    friend bool build_window_ple_rows(
        std::span<const std::uint32_t> tokens,
        std::span<const std::uint32_t> committed_history,
        const Geometry& geometry, WindowPleRows& output, std::string& error);

    std::uint32_t token_count_ = 0;
    std::array<std::uint32_t,
               kWindowPleMaxTokens * kWindowPleRowsPerToken> row_ids_{};
};

/// Build token-major PLE row IDs from a Window's tokens and up to two
/// committed predecessors (oldest first). A failed build leaves output
/// unchanged. The returned rows are indices only; PLE I/O and dispatch have
/// separate ownership and lifetimes.
bool build_window_ple_rows(
    std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> committed_history,
    const Geometry& geometry, WindowPleRows& output, std::string& error);

/// Session-owned suffix of committed token IDs used to prepare later PLE rows.
/// The owning Window/session must call commit_accepted_prefix only after its
/// GPU state commit succeeds; speculative tokens do not mutate this history.
class WindowPleTokenHistory {
public:
    /// Replace state from token IDs already applied to model state, retaining
    /// only their last two IDs. For prompt startup, exclude the prompt's last
    /// token when it is still pending as the first T=1 decode window. Invalid
    /// input leaves the previous state intact.
    bool replace_committed_history(
        std::span<const std::uint32_t> committed_prefix,
        const Geometry& geometry, std::string& error);

    /// Clear this helper when used independently. An owning session-state
    /// object must reset the GPU and host state through its own reset method.
    void reset() noexcept;

    /// Return up to two committed IDs in oldest-first order.
    std::span<const std::uint32_t> committed_suffix() const noexcept;

    /// Build rows using this object's committed suffix without changing it.
    bool build_rows(std::span<const std::uint32_t> tokens,
                    const Geometry& geometry, WindowPleRows& output,
                    std::string& error) const;

    /// Append only the accepted prefix of a validated 1..8-token window.
    /// Call only after the corresponding GPU state commit succeeds.
    /// Invalid input leaves the committed suffix intact.
    bool commit_accepted_prefix(
        std::span<const std::uint32_t> window_tokens,
        std::size_t accepted_count, const Geometry& geometry,
        std::string& error);

private:
    std::uint32_t committed_count_ = 0;
    std::array<std::uint32_t, 2> committed_ids_{};
};

}  // namespace strata::metal
