// Exclusive selection of the T=1 Window pending-MoE consumer.
#pragma once

#include "strata/metal/window_dispatch.hpp"
#include "strata/metal/window_ple_dispatch.hpp"

#include <cstdint>
#include <string>
#include <variant>

namespace strata::metal {

struct WindowMoePendingPLEConsumer {
    WindowMoePLEEntryBuffers buffers;
    MatVecWeightFormat key_format = MatVecWeightFormat::q2_0;
    std::uint32_t compiled_columns = 0u;
    PLEHistoryUpdate history_update = PLEHistoryUpdate::defer;
};

/// A value of this type contains exactly one branch's consumer buffers.
using WindowMoePendingConsumer =
    std::variant<WindowMoePendingPLEConsumer,
                 WindowMoeAttentionEntryBuffers>;

/// The buffers for one T=1 FullMode step from token embedding through the
/// selected pending-MoE consumer. `streams` must be the exact layer-0
/// attention residual view. `embedding` is dead after the stream broadcast
/// and may be reused by a later stage in this ordered segment; the only
/// permitted overlap with a later-stage view is an exact alias of
/// `layer0.ffn.shared.output`.
struct WindowT1FullModeStepBuffers {
    BufferRef embedding;
    BufferRef streams;
    WindowT1MoeLayer0FullBuffers layer0;
    WindowMoePendingConsumer consumer;
};

/// Encode the one consumer selected by the Window pending-write rule:
/// `ple_enabled` selects PLE, otherwise it selects the next attention read.
/// The branch tag and buffer variant must agree. The selected typed consumer
/// runs in this Encoder segment; callers must propagate false from the
/// enclosing Encoder::encode body so the producer and consumer are discarded
/// together.
bool encode_window_moe_pending_consumer(
    Context& context, Encoder& encoder, std::uint32_t rows,
    bool ple_enabled, const WindowMoePendingConsumer& consumer,
    std::string& error);

/// Resolve `token_embd.weight` from FullModeStorage, then encode the T=1
/// token gather, GR stream broadcast, full-mode layer-0 MoE producer and
/// exactly one selected pending consumer in that order. The token embedding
/// tensor must have shape [2560, vocabulary_size] and use a supported
/// embedding format. `buffers.layer0.producer.router.layer0.params` supplies
/// the parameter block shared by all stages.
///
/// Propagate false from the enclosing Encoder::encode body so a late failure
/// discards every earlier dispatch in the step. This is a composable T=1
/// segment, not the complete Window scheduler or a model-parity result.
bool encode_window_t1_full_mode_step(
    Context& context, const FullModeStorage& full_mode, Encoder& encoder,
    std::uint32_t vocabulary_size, std::uint32_t experts,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    bool ple_enabled, const WindowT1FullModeStepBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
