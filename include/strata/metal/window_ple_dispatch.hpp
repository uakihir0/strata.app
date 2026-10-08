// Ordered Window entry encoding for the layer-0 GR write and PLE block.
#pragma once

#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/moe_dispatch.hpp"
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

/// Connect a no-write routed/shared MoE result and its producing layer-0 GR
/// read to the layer-1 PLE entry. The PLE residual must be the same view as
/// the previous GR residual; the coordinator takes the pending MoE output
/// and injection directly from their producer structures.
struct WindowMoePLEEntryBuffers {
    MoeRoutedSharedFFNBuffers previous_ffn;
    GRReadBuffers previous_gr_read;
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

/// Encode the PLE entry using `previous_ffn.shared.output` and
/// `previous_gr_read.inject_out` as the pending layer-0 write. Call after the
/// no-write routed/shared encoder in the same ordered Encoder segment, and
/// propagate false from the enclosing Encoder::encode body.
bool encode_window_moe_ple_entry(
    Context& context, Encoder& encoder, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowMoePLEEntryBuffers& buffers,
    std::string& error);

/// Standalone transactional form. Any validation or encoding failure discards
/// the command buffer segment, including the preceding GR post dispatch.
bool encode_window_ple_entry(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowPLEEntryBuffers& buffers,
    std::string& error);

/// Standalone form for the typed pending-MoE handoff. The producer command
/// buffer dependency must be registered with
/// `consumer.depends_on(producer, error)`, and the producer must complete
/// successfully before the consumer command buffer is submitted.
bool encode_window_moe_ple_entry(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowMoePLEEntryBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
