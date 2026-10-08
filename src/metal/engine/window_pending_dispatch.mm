// Exclusive selection of the pending layer-0 MoE consumer.
#include "strata/metal/window_pending_dispatch.hpp"

namespace strata::metal {

bool encode_window_moe_pending_consumer(
    Context& context, Encoder& encoder, std::uint32_t rows,
    bool ple_enabled, const WindowMoePendingConsumer& consumer,
    std::string& error) {
    error.clear();
    if (rows != 1u) {
        error = "T=1 Window pending-MoE consumer requires exactly one row";
        return false;
    }

    const auto* ple = std::get_if<WindowMoePendingPLEConsumer>(&consumer);
    const auto* attention =
        std::get_if<WindowMoeAttentionEntryBuffers>(&consumer);
    if (ple_enabled != (ple != nullptr) ||
        ple_enabled == (attention != nullptr)) {
        error = "Window pending-MoE consumer does not match the selected PLE branch";
        return false;
    }

    if (ple != nullptr) {
        return encode_window_moe_ple_entry(
            context, encoder, rows, ple->key_format, ple->compiled_columns,
            ple->history_update, ple->buffers, error);
    }
    return encode_window_moe_attention_entry(
        context, encoder, rows, *attention, error);
}

}  // namespace strata::metal
