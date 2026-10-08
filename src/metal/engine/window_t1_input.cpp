// T=1 Window parameter and optional PLE input preparation.
#include "strata/metal/window_t1_input.hpp"

#include "strata/metal/window_ple_stage.hpp"

#include <array>

namespace strata::metal {

bool prepare_window_t1_input(
    SessionStateStorage& session_state, const Geometry& geometry,
    FullModeStorage& full_mode, WindowParamsRing& params_ring,
    WindowScratchStorage& scratch, std::uint32_t params_slot,
    std::uint32_t position, std::uint32_t token,
    const WindowSamplerSettings& sampler, std::uint32_t seed_lo,
    std::uint32_t seed_hi, bool ple_enabled, WindowT1PreparedInput& output,
    std::string& error) {
    output = {};
    error.clear();

    if (!is_qwen4exp_runtime_geometry(geometry)) {
        error = "T=1 Window input requires supported qwen4exp runtime geometry";
        return false;
    }
    if (token >= geometry.vocab) {
        error = "T=1 Window token ID is outside the model vocabulary";
        return false;
    }
    if (params_slot >= kWindowParamsRingSlotCount) {
        error = "T=1 Window parameter slot is out of range";
        return false;
    }
    if (!params_ring.is_valid()) {
        error = "T=1 Window parameter ring is unavailable";
        return false;
    }

    const std::array<std::uint32_t, 1> tokens{token};
    const std::array<WindowSamplerSettings, 1> samplers{sampler};
    WindowParams validated_params{};
    if (!build_window_params(position, tokens, samplers, 0u, seed_lo, seed_hi,
                             validated_params, error)) {
        return false;
    }

    BufferRef staged_ple_input;
    if (ple_enabled) {
        WindowPleRows rows;
        if (!session_state.build_ple_rows(tokens, geometry, rows, error)) {
            return false;
        }
        strata::kernels::PleTableReader* reader = full_mode.ple_reader();
        if (reader == nullptr || !reader->is_open()) {
            error = "T=1 PLE input requires an open FullMode PLE table";
            return false;
        }
        if (!stage_window_ple_input(*reader, rows, scratch, staged_ple_input,
                                    error)) {
            return false;
        }
    }

    BufferRef params = params_ring.build_slot(
        params_slot, position, tokens, samplers, 0u, seed_lo, seed_hi, error);
    if (!params) {
        return false;
    }

    output.params = params;
    output.ple_input = staged_ple_input;
    error.clear();
    return true;
}

}  // namespace strata::metal
