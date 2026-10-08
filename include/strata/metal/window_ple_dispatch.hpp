// Ordered Window entry encoding for the layer-0 GR write and PLE block.
#pragma once

#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/ple_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

/// Inputs for the l=1 Window entry when PLE is enabled. The previous FFN
/// write updates `ple.post.residual` in place before the PLE block reads and
/// updates that same residual. `previous_block_out` and `previous_inject`
/// are layer 0's FFN outputs.
struct WindowPLEEntryBuffers {
    BufferRef previous_block_out;
    BufferRef previous_inject;
    PLEBlockBuffers ple;
};

/// Encode layer 0's GR post write followed by the PLE block for rows in 1..8.
/// In concurrent dispatch mode, call outside an open group so the Encoder can
/// insert barriers between the dependent stages. Propagate false from an
/// enclosing Encoder::encode body to discard the whole segment.
bool encode_window_ple_entry(
    Context& context, Encoder& encoder, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowPLEEntryBuffers& buffers,
    std::string& error);

/// Standalone transactional form. Any validation or encoding failure discards
/// the command buffer segment, including the preceding GR post dispatch.
bool encode_window_ple_entry(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowPLEEntryBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
