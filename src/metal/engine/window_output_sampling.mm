// Ordered FullMode Window logits and sampler candidate composition.
#include "strata/metal/window_output_sampling.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/sampler_dispatch.hpp"
#include "strata/metal/window_output_head.hpp"
#include "strata/metal/window_output_projection.hpp"
#include "strata/metal/window_params.hpp"
#include "strata/metal/session_state.hpp"
#include "../kernels/common/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::size_t kSamplerCandidateCapacity = 64u;
constexpr std::size_t kSamplerCandidateBytes = sizeof(SampCandidate);
static_assert(sizeof(WindowSampRow) == sizeof(::SampRow));

bool validate_sampler_row(const WindowSampRow& row, std::string& error) {
    if (!std::isfinite(row.temperature) || !std::isfinite(row.top_p) ||
        !std::isfinite(row.min_p) ||
        !std::isfinite(row.penalty_repeat) ||
        !std::isfinite(row.penalty_freq) ||
        !std::isfinite(row.penalty_present)) {
        error = "Window sampler row contains a non-finite setting";
        return false;
    }
    const double smallest_finite_temperature =
        1.0 / static_cast<double>(std::numeric_limits<float>::max());
    if (row.temperature > 0.0f &&
        static_cast<double>(row.temperature) <
            smallest_finite_temperature) {
        error = "Window sampler temperature is too small for finite scaling";
        return false;
    }
    if (row.top_p < 0.0f || row.top_p > 1.0f ||
        row.min_p < 0.0f || row.min_p > 1.0f) {
        error = "Window sampler top_p and min_p must be in [0, 1]";
        return false;
    }
    if (row.penalty_repeat <= 0.0f) {
        error = "Window sampler repeat penalty must be greater than zero";
        return false;
    }
    if (row.top_k < 0 || row.min_keep < 1 ||
        row.min_keep > static_cast<std::int32_t>(kSamplerCandidateCapacity)) {
        error = "Window sampler top_k or min_keep is outside its supported range";
        return false;
    }
    if (row.greedy != 0 && row.greedy != 1) {
        error = "Window sampler greedy must be zero or one";
        return false;
    }
    return true;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_ranges_overlap(BufferRef left, std::size_t left_bytes,
                           BufferRef right, std::size_t right_bytes) {
    return (left.handle == right.handle &&
            address_ranges_overlap(left.offset, left_bytes,
                                   right.offset, right_bytes)) ||
           address_ranges_overlap(
               reinterpret_cast<std::uintptr_t>(left.contents), left_bytes,
               reinterpret_cast<std::uintptr_t>(right.contents), right_bytes) ||
           address_ranges_overlap(left.gpu_address, left_bytes,
                                  right.gpu_address, right_bytes);
}

bool get_scratch_view(Context& context, WindowScratchStorage& scratch,
                      WindowScratchRegion region, const char* name,
                      BufferRef& output, std::string& error) {
    if (!scratch.view(region, output, error) ||
        !context.buffers().owns(output)) {
        error = std::string("Window output sampling cannot resolve scratch ") +
                name + ": " + error;
        return false;
    }
    return true;
}

bool encode_output_sampling(
    Context& context, Encoder& encoder, const FullModeStorage& full_mode,
    WindowScratchStorage& scratch, BufferRef residual,
    BufferRef window_params, WindowOutputSamplingResult& output,
    std::string& error) {
    output = {};
    error.clear();
    if (!context.buffers().owns(window_params) ||
        window_params.contents == nullptr ||
        window_params.length < sizeof(WindowParams)) {
        error = "Window output sampling requires a complete host-visible WindowParams view";
        return false;
    }

    WindowParams params{};
    std::memcpy(&params, window_params.contents, sizeof(params));
    if (params.T == 0u || params.T > kWindowMaxRows) {
        error = "Window output sampling row count must be in 1..8";
        return false;
    }
    std::uint32_t expected_nc = 1u;
    while (expected_nc < params.T) expected_nc <<= 1u;
    if (params.nc != expected_nc) {
        error = "Window output sampling WindowParams.nc does not match T";
        return false;
    }
    for (std::uint32_t row_index = 0u; row_index < params.T; ++row_index) {
        if (!validate_sampler_row(params.samp[row_index], error)) return false;
    }

    const std::size_t sampler_rows_bytes =
        static_cast<std::size_t>(params.T) * sizeof(WindowSampRow);
    BufferRef active_rows =
        window_params.slice(offsetof(WindowParams, T), sizeof(params.T));
    BufferRef sampler_rows =
        window_params.slice(offsetof(WindowParams, samp), sampler_rows_bytes);
    BufferRef history;
    BufferRef candidates;
    BufferRef selected;
    BufferRef sampler_output;
    BufferRef normalized;
    BufferRef low_rank;
    BufferRef gate;
    BufferRef mixed;
    BufferRef q8_1;
    BufferRef logits;
    BufferRef matvec_sink;
    const WindowScratchLayout& layout = scratch.layout();
    if (!scratch.is_valid() || !active_rows || !sampler_rows ||
        !get_scratch_view(context, scratch, layout.sampler_history,
                          "sampler_history", history, error) ||
        !get_scratch_view(context, scratch, layout.sampler_candidates,
                          "sampler_candidates", candidates, error) ||
        !get_scratch_view(context, scratch, layout.sampler_selection,
                          "sampler_selection", selected, error) ||
        !get_scratch_view(context, scratch, layout.sampler_output,
                          "sampler_output", sampler_output, error) ||
        !get_scratch_view(context, scratch, layout.head_normalized,
                          "head_normalized", normalized, error) ||
        !get_scratch_view(context, scratch, layout.head_low_rank,
                          "head_low_rank", low_rank, error) ||
        !get_scratch_view(context, scratch, layout.head_gate,
                          "head_gate", gate, error) ||
        !get_scratch_view(context, scratch, layout.head_mixed,
                          "head_mixed", mixed, error) ||
        !get_scratch_view(context, scratch, layout.head_q8_1,
                          "head_q8_1", q8_1, error) ||
        !get_scratch_view(context, scratch, layout.head_logits,
                          "head_logits", logits, error) ||
        !get_scratch_view(context, scratch, layout.matvec_sink,
                          "matvec_sink", matvec_sink, error)) {
        if (error.empty()) {
            error = "Window output sampling scratch or parameters are invalid";
        }
        return false;
    }

    const std::size_t active_residual_bytes =
        static_cast<std::size_t>(params.T) * kSessionHyperconnections *
        kWindowOutputInputWidth *
        sizeof(float);
    const std::size_t selected_bytes =
        static_cast<std::size_t>(params.T) * kSamplerCandidateCapacity *
        kSamplerCandidateBytes;
    if (residual.length < active_residual_bytes ||
        !context.buffers().owns(residual)) {
        error = "Window output sampling residual is undersized or unowned";
        return false;
    }
    if (buffer_ranges_overlap(residual, active_residual_bytes, candidates,
                              candidates.length) ||
        buffer_ranges_overlap(residual, active_residual_bytes, selected,
                              selected_bytes)) {
        error = "Window output sampling residual overlaps sampler output scratch";
        return false;
    }

    const std::array output_regions{
        normalized, low_rank, gate, mixed, q8_1, logits, matvec_sink,
        candidates, selected, sampler_output,
    };
    for (const BufferRef& region : output_regions) {
        if (buffer_ranges_overlap(window_params, sizeof(WindowParams), region,
                                  region.length)) {
            error = "Window output sampling parameters overlap writable scratch";
            return false;
        }
    }

    WindowOutputHeadResult head;
    const ReadOnlyBufferRef active_rows_read_only =
        context.buffers().read_only(active_rows);
    if (!active_rows_read_only ||
        !encode_window_output_head(
            context, encoder, full_mode, scratch, params.T,
            {residual, active_rows_read_only}, head, error)) {
        if (error.empty()) {
            error = "Window output sampling cannot bind active-row metadata";
        }
        return false;
    }

    const SamplerBuffers sampler_buffers{
        head.logits, history, sampler_rows, candidates, selected,
        sampler_output};
    if (!encode_sampler_sampled(context, encoder, params.T,
                                kWindowOutputVocabularySize, sampler_buffers,
                                false, error)) {
        error = "Window sampler candidate encoding failed: " + error;
        return false;
    }

    BufferRef active_candidates = selected.slice(0u, selected_bytes);
    if (!active_candidates) {
        error = "Window output sampling cannot publish active candidate rows";
        return false;
    }
    output = {head.logits, active_candidates, params.T};
    error.clear();
    return true;
}

}  // namespace

bool encode_window_output_sampling(
    Context& context, Encoder& encoder, const FullModeStorage& full_mode,
    WindowScratchStorage& scratch, BufferRef residual, BufferRef window_params,
    WindowOutputSamplingResult& output, std::string& error) {
    return encode_output_sampling(context, encoder, full_mode, scratch,
                                  residual, window_params, output, error);
}

bool encode_window_output_sampling(
    Context& context, CommandBuffer& command_buffer,
    const FullModeStorage& full_mode, WindowScratchStorage& scratch,
    BufferRef residual, BufferRef window_params,
    WindowOutputSamplingResult& output, std::string& error) {
    output = {};
    error.clear();
    return Encoder::encode(
        context, command_buffer, "Window FullMode output sampling",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_output_sampling(
                context, encoder, full_mode, scratch, residual, window_params,
                output, encode_error);
        },
        error);
}

bool sample_window_output_tail(
    Context& context, const CommandBuffer& command_buffer,
    BufferRef window_params, const WindowOutputSamplingResult& encoded,
    std::vector<TailPick>& output, std::string& error) {
    error.clear();
    if (!command_buffer.belongs_to(context)) {
        error = "Window output sampling command buffer belongs to another Context";
        return false;
    }
    if (!context.buffers().owns(window_params) ||
        window_params.contents == nullptr ||
        window_params.length < sizeof(WindowParams) ||
        !context.buffers().owns(encoded.logits) ||
        !context.buffers().owns(encoded.selected_candidates) ||
        encoded.rows == 0u || encoded.rows > kWindowMaxRows) {
        error = "Window output sampling tail inputs are invalid";
        return false;
    }

    WindowParams params{};
    std::memcpy(&params, window_params.contents, sizeof(params));
    if (params.T == 0u || params.T > kWindowMaxRows) {
        error = "Window output sampling row count must be in 1..8";
        return false;
    }
    std::uint32_t expected_nc = 1u;
    while (expected_nc < params.T) expected_nc <<= 1u;
    if (params.T != encoded.rows || params.nc != expected_nc ||
        encoded.logits.length <
            static_cast<std::size_t>(encoded.rows) *
                kWindowOutputVocabularySize * sizeof(float) ||
        encoded.selected_candidates.length <
            static_cast<std::size_t>(encoded.rows) *
                kSamplerCandidateCapacity * kSamplerCandidateBytes) {
        error = "Window output sampling tail extents do not match WindowParams";
        return false;
    }
    for (std::uint32_t row_index = 0u; row_index < encoded.rows; ++row_index) {
        if (!validate_sampler_row(params.samp[row_index], error)) return false;
    }
    if (!command_buffer.wait_until_completed(error)) {
        if (error.empty()) {
            error = "Window output sampling command did not complete successfully";
        }
        return false;
    }

    std::vector<TailPick> picks;
    picks.reserve(encoded.rows);
    const auto* const selected =
        static_cast<const SampCandidate*>(
            encoded.selected_candidates.contents);
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(params.seed_hi) << 32u) | params.seed_lo;
    for (std::uint32_t row_index = 0u; row_index < encoded.rows; ++row_index) {
        const WindowSampRow& row = params.samp[row_index];
        const std::uint32_t requested_k =
            row.top_k > 0 && row.top_k < 64
                ? static_cast<std::uint32_t>(row.top_k)
                : 64u;
        const int k = static_cast<int>(
            std::min(kWindowOutputVocabularySize, requested_k));
        const std::span<const SampCandidate> candidates_for_row(
            selected + static_cast<std::size_t>(row_index) *
                           kSamplerCandidateCapacity,
            static_cast<std::size_t>(k));
        for (const SampCandidate& candidate : candidates_for_row) {
            if (candidate.id < 0 ||
                static_cast<std::uint32_t>(candidate.id) >=
                    kWindowOutputVocabularySize) {
                error = "Window output sampler produced an out-of-range candidate ID";
                return false;
            }
            if (!std::isfinite(std::bit_cast<float>(candidate.logit_bits))) {
                error = "Window output sampler produced a non-finite candidate logit";
                return false;
            }
        }

        strata::kernels::SamplerParams sampler;
        sampler.top_k = row.top_k;
        sampler.top_p = row.top_p;
        sampler.min_p = row.min_p;
        sampler.temperature = row.temperature;
        sampler.min_keep = row.min_keep;
        sampler.penalty_repeat = row.penalty_repeat;
        sampler.penalty_freq = row.penalty_freq;
        sampler.penalty_present = row.penalty_present;
        sampler.seed = seed;
        sampler.counter = row.counter;
        sampler.greedy = row.greedy != 0;
        const std::optional<TailPick> pick =
            sampler_tail(candidates_for_row, k, sampler, row.counter);
        if (!pick) {
            error = "Window output sampler CPU tail rejected its GPU candidates";
            return false;
        }
        if (!std::isfinite(pick->probability) ||
            pick->probability < 0.0f || pick->probability > 1.0f) {
            error = "Window output sampler CPU tail produced an invalid probability";
            return false;
        }
        picks.push_back(*pick);
    }
    output = std::move(picks);
    error.clear();
    return true;
}

}  // namespace strata::metal
