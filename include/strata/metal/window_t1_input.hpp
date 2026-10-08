// Host preparation for the production-shaped T=1 FullMode Window segment.
#pragma once

#include "strata/metal/session_state.hpp"
#include "strata/metal/window.hpp"
#include "strata/metal/window_params.hpp"
#include "strata/metal/window_scratch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

struct WindowT1PreparedInput {
    BufferRef params;
    BufferRef ple_input;
};

/// Build the T=1 parameter block and, when enabled, stage PLE rows derived
/// from the session's committed token suffix. Preparation never advances that
/// suffix. The selected parameter slot and PLE scratch region must be idle
/// with respect to earlier GPU work.
bool prepare_window_t1_input(
    SessionStateStorage& session_state, const Geometry& geometry,
    FullModeStorage& full_mode, WindowParamsRing& params_ring,
    WindowScratchStorage& scratch, std::uint32_t params_slot,
    std::uint32_t position, std::uint32_t token,
    const WindowSamplerSettings& sampler, std::uint32_t seed_lo,
    std::uint32_t seed_hi, bool ple_enabled, WindowT1PreparedInput& output,
    std::string& error);

}  // namespace strata::metal
