// FullMode resolution and ordered composition of the Window output head.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/window.hpp"
#include "strata/metal/window_scratch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class CommandBuffer;
class Encoder;

struct WindowOutputHeadBuffers {
    BufferRef residual;
    ReadOnlyBufferRef active_rows;
};

struct WindowOutputHeadResult {
    BufferRef logits;
};

/// Resolve the live output_hc_* and output.weight tensors, run the final GR
/// head mixer, and project its mixed rows to vocabulary logits using named
/// Window scratch. The returned logits view covers only the active rows.
/// Keep FullModeStorage, WindowScratchStorage, residual and active-row metadata
/// alive and unchanged until command completion. The caller-owned form may fail
/// after dispatches have been encoded; propagate false from the enclosing
/// Encoder::encode body to discard the entire segment.
bool encode_window_output_head(
    Context& context, Encoder& encoder, const FullModeStorage& full_mode,
    WindowScratchStorage& scratch, std::uint32_t rows,
    const WindowOutputHeadBuffers& buffers, WindowOutputHeadResult& output,
    std::string& error);

/// Standalone transactional form. Any failure discards the complete head
/// mixer and vocabulary-projection segment.
bool encode_window_output_head(
    Context& context, CommandBuffer& command_buffer,
    const FullModeStorage& full_mode, WindowScratchStorage& scratch,
    std::uint32_t rows, const WindowOutputHeadBuffers& buffers,
    WindowOutputHeadResult& output, std::string& error);

}  // namespace strata::metal
