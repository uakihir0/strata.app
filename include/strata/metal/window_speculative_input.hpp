// Host preparation for speculative multi-token FullMode Window segments.
#pragma once

#include "strata/metal/session_state.hpp"
#include "strata/metal/window.hpp"
#include "strata/metal/window_params.hpp"
#include "strata/metal/window_scratch.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace strata::metal {

struct WindowSpeculativePreparedInput {
    BufferRef params;
    BufferRef ple_input;
};

/// Build WindowParams for 1..8 speculative token rows and, when PLE is
/// enabled, stage the corresponding PLE table rows derived from committed
/// session history. Preparation does not advance PLE history. The parameter
/// slot and PLE scratch region must be idle with respect to earlier GPU work.
bool prepare_window_speculative_input(
    SessionStateStorage& session_state, const Geometry& geometry,
    FullModeStorage& full_mode, WindowParamsRing& params_ring,
    WindowScratchStorage& scratch, std::uint32_t params_slot,
    std::uint32_t position, std::span<const std::uint32_t> tokens,
    std::span<const WindowSamplerSettings> samplers, std::uint32_t seed_lo,
    std::uint32_t seed_hi, bool ple_enabled,
    WindowSpeculativePreparedInput& output, std::string& error);

}  // namespace strata::metal
