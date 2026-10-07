// src/metal/engine/window_params.cpp - host metadata for one Metal decode window.
#include "strata/metal/window_params.hpp"

#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using strata::metal::WindowParams;
using strata::metal::WindowSampRow;
using strata::metal::WindowStepRow;

static_assert(strata::metal::kWindowMaxRows == VERIFY_MAX_T);
static_assert(sizeof(WindowStepRow) == sizeof(::StepRow));
static_assert(offsetof(WindowStepRow, pos) == offsetof(::StepRow, pos));
static_assert(offsetof(WindowStepRow, n_kv) == offsetof(::StepRow, n_kv));
static_assert(offsetof(WindowStepRow, n_bid) == offsetof(::StepRow, n_bid));
static_assert(offsetof(WindowStepRow, width) == offsetof(::StepRow, width));
static_assert(sizeof(WindowSampRow) == sizeof(::SampRow));
static_assert(offsetof(WindowSampRow, temperature) == offsetof(::SampRow, temperature));
static_assert(offsetof(WindowSampRow, top_p) == offsetof(::SampRow, top_p));
static_assert(offsetof(WindowSampRow, min_p) == offsetof(::SampRow, min_p));
static_assert(offsetof(WindowSampRow, penalty_repeat) == offsetof(::SampRow, penalty_repeat));
static_assert(offsetof(WindowSampRow, penalty_freq) == offsetof(::SampRow, penalty_freq));
static_assert(offsetof(WindowSampRow, penalty_present) == offsetof(::SampRow, penalty_present));
static_assert(offsetof(WindowSampRow, top_k) == offsetof(::SampRow, top_k));
static_assert(offsetof(WindowSampRow, min_keep) == offsetof(::SampRow, min_keep));
static_assert(offsetof(WindowSampRow, greedy) == offsetof(::SampRow, greedy));
static_assert(offsetof(WindowSampRow, counter) == offsetof(::SampRow, counter));
static_assert(offsetof(WindowSampRow, hist_off) == offsetof(::SampRow, hist_off));
static_assert(offsetof(WindowSampRow, hist_len) == offsetof(::SampRow, hist_len));
static_assert(sizeof(WindowParams) == sizeof(::WindowParams));
static_assert(std::is_standard_layout_v<WindowParams>);
static_assert(std::is_trivially_copyable_v<WindowParams>);
static_assert(offsetof(WindowParams, T) == offsetof(::WindowParams, T));
static_assert(offsetof(WindowParams, nc) == offsetof(::WindowParams, nc));
static_assert(offsetof(WindowParams, pos0) == offsetof(::WindowParams, pos0));
static_assert(offsetof(WindowParams, n_keep) == offsetof(::WindowParams, n_keep));
static_assert(offsetof(WindowParams, seed_lo) == offsetof(::WindowParams, seed_lo));
static_assert(offsetof(WindowParams, seed_hi) == offsetof(::WindowParams, seed_hi));
static_assert(offsetof(WindowParams, tokens) == offsetof(::WindowParams, tokens));
static_assert(offsetof(WindowParams, step) == offsetof(::WindowParams, step));
static_assert(offsetof(WindowParams, samp) == offsetof(::WindowParams, samp));

}  // namespace

namespace strata::metal {

bool build_window_params(std::uint32_t pos0,
                         std::span<const std::uint32_t> tokens,
                         std::span<const WindowSamplerSettings> samplers,
                         std::uint32_t n_keep,
                         std::uint32_t seed_lo,
                         std::uint32_t seed_hi,
                         WindowParams& output,
                         std::string& error) {
    if (tokens.empty() || tokens.size() > kWindowMaxRows) {
        error = "window row count must be between one and eight";
        return false;
    }
    if (samplers.size() != tokens.size()) {
        error = "window sampler row count must match the token row count";
        return false;
    }
    if (n_keep > tokens.size()) {
        error = "window keep count exceeds its row count";
        return false;
    }

    const std::uint64_t end = static_cast<std::uint64_t>(pos0) + tokens.size();
    if (end > static_cast<std::uint64_t>(INT32_MAX)) {
        error = "window positions exceed the signed StepRow range";
        return false;
    }

    WindowParams next{};
    next.T = static_cast<std::uint32_t>(tokens.size());
    next.nc = 1u;
    while (next.nc < next.T) next.nc <<= 1u;
    next.pos0 = pos0;
    next.n_keep = n_keep;
    next.seed_lo = seed_lo;
    next.seed_hi = seed_hi;

    for (std::uint32_t row = 0; row < next.T; ++row) {
        const std::uint32_t position = pos0 + row;
        const std::int32_t n_kv = static_cast<std::int32_t>(position + 1u);
        const WindowSamplerSettings& sampler = samplers[row];
        next.tokens[row] = tokens[row];
        next.step[row] = {
            static_cast<std::int32_t>(position),
            n_kv,
            n_kv / 4,
            std::min(n_kv, static_cast<std::int32_t>(kWindowMaxSelectedCells)),
        };
        next.samp[row] = {
            sampler.temperature,
            sampler.top_p,
            sampler.min_p,
            sampler.penalty_repeat,
            sampler.penalty_freq,
            sampler.penalty_present,
            sampler.top_k,
            sampler.min_keep,
            sampler.greedy,
            position,
            sampler.hist_off,
            sampler.hist_len,
        };
    }

    output = next;
    error.clear();
    return true;
}

}  // namespace strata::metal
