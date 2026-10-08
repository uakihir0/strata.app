// FullMode Window output head followed by sampler candidate selection.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/sampler_tail.hpp"
#include "strata/metal/window_scratch.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::metal {

class CommandBuffer;
class Encoder;
class FullModeStorage;

struct WindowOutputSamplingResult {
    BufferRef logits;
    BufferRef selected_candidates;
    std::uint32_t rows = 0;
};

/// Resolve the FullMode output head, then encode the GPU top-k split and merge
/// using sampler rows from the built WindowParams slot and sampler_history from
/// WindowScratchStorage. The history bytes selected by each SampRow must be
/// prepared by the caller before encoding. The returned candidate rows become
/// readable after command completion; call sample_window_output_tail() to run
/// the CPU FP64 tail and wait for successful completion.
///
/// Keep the FullMode owner, scratch, WindowParams, residual and sampler history
/// alive and unchanged through command completion. The caller-owned form may
/// fail after head dispatches were encoded; propagate false from the enclosing
/// Encoder::encode body to discard the entire segment.
bool encode_window_output_sampling(
    Context& context, Encoder& encoder, const FullModeStorage& full_mode,
    WindowScratchStorage& scratch, BufferRef residual, BufferRef window_params,
    WindowOutputSamplingResult& output, std::string& error);

/// Standalone transactional form. Any failure discards both the head and
/// sampler candidate-selection stages.
bool encode_window_output_sampling(
    Context& context, CommandBuffer& command_buffer,
    const FullModeStorage& full_mode, WindowScratchStorage& scratch,
    BufferRef residual, BufferRef window_params,
    WindowOutputSamplingResult& output, std::string& error);

/// Wait for the producing command buffer, then run the existing CPU sampling
/// tail on each GPU-produced top-k row. `window_params` and candidate scratch
/// must remain unchanged until this call returns.
bool sample_window_output_tail(
    Context& context, const CommandBuffer& command_buffer,
    BufferRef window_params, const WindowOutputSamplingResult& encoded,
    std::vector<TailPick>& output, std::string& error);

}  // namespace strata::metal
