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

}  // namespace strata::metal
