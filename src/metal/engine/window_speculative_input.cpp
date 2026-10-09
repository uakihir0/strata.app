// Host preparation for speculative multi-token Window parameters and PLE rows.
#include "strata/metal/window_speculative_input.hpp"

#include "strata/metal/window_ple_stage.hpp"

#include <vector>

namespace strata::metal {

bool prepare_window_speculative_input(
    SessionStateStorage& session_state, const Geometry& geometry,
    FullModeStorage& full_mode, WindowParamsRing& params_ring,
    WindowScratchStorage& scratch, std::uint32_t params_slot,
    std::uint32_t position, std::span<const std::uint32_t> tokens,
    std::span<const WindowSamplerSettings> samplers, std::uint32_t seed_lo,
    std::uint32_t seed_hi, bool ple_enabled,
    WindowSpeculativePreparedInput& output, std::string& error) {
    output = {};
    error.clear();

    if (!is_qwen4exp_runtime_geometry(geometry)) {
        error =
            "speculative Window input requires supported qwen4exp runtime geometry";
        return false;
    }
    if (tokens.empty() || tokens.size() > kWindowMaxRows) {
        error = "speculative Window input requires 1..8 token rows";
        return false;
    }
    if (samplers.size() != tokens.size()) {
        error =
            "speculative Window sampler row count must match the token row count";
        return false;
    }
    // PLE staging writes into caller-owned Shared scratch. Snapshot these
    // spans first because callers may have sourced them from that same region.
    const std::vector<std::uint32_t> token_rows(tokens.begin(), tokens.end());
    const std::vector<WindowSamplerSettings> sampler_rows(samplers.begin(),
                                                           samplers.end());
    for (const std::uint32_t token : token_rows) {
        if (token >= geometry.vocab) {
            error =
                "speculative Window token ID is outside the model vocabulary";
            return false;
        }
    }
    if (params_slot >= kWindowParamsRingSlotCount) {
        error = "speculative Window parameter slot is out of range";
        return false;
    }
    if (!params_ring.is_valid()) {
        error = "speculative Window parameter ring is unavailable";
        return false;
    }

    WindowParams validated_params{};
    if (!build_window_params(position, token_rows, sampler_rows, 0u, seed_lo,
                             seed_hi, validated_params, error)) {
        return false;
    }

    BufferRef staged_ple_input;
    if (ple_enabled) {
        WindowPleRows rows;
        if (!session_state.build_ple_rows(tokens, geometry, rows, error)) {
            return false;
        }
        strata::kernels::PleTableReader* const reader =
            full_mode.ple_reader();
        if (reader == nullptr || !reader->is_open()) {
            error = "speculative PLE input requires an open FullMode PLE table";
            return false;
        }
        if (!stage_window_ple_input(*reader, rows, scratch, staged_ple_input,
                                    error)) {
            return false;
        }
    }

    BufferRef params = params_ring.build_slot(
        params_slot, position, token_rows, sampler_rows, 0u, seed_lo, seed_hi,
        error);
    if (!params) {
        return false;
    }

    output.params = params;
    output.ple_input = staged_ple_input;
    error.clear();
    return true;
}

}  // namespace strata::metal
