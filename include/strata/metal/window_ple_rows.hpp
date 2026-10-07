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

}  // namespace strata::metal
