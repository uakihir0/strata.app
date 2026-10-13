// src/metal/engine/mtp_full_layer_dispatch.cpp - ordered runtime MTP stages 1–20.
#include "strata/metal/mtp_runtime_storage.hpp"

#include "strata/metal/encoder.hpp"

namespace strata::metal {

bool encode_mtp_full_layer(Context& context, Encoder& encoder,
                           const MtpRuntimeStorage& runtime,
                           const MtpFullLayerSources& sources,
                           std::string& error) {
    error.clear();
    if (!runtime.is_valid()) {
        error = "MTP full-layer encoding needs live runtime storage";
        return false;
    }
    if (sources.input_projection.token_source !=
            MtpInputTokenSource::selected_token ||
        !sources.input_projection.selected_token) {
        error =
            "MTP full-layer encoding requires a GPU-produced selected token";
        return false;
    }

    const std::uint32_t params_slot =
        sources.input_projection.params_slot;
    MtpInputProjectionBuffers input_projection;
    MtpAddStreamsBuffers add_streams;
    MtpAttentionReadBuffers attention_read;
    MtpAttentionKvProjectionBuffers attention_kv_projection;
    MtpAttentionPassBuffers attention_pass;
    MtpFeedForwardReadBuffers feed_forward_read;
    MtpRouterBuffers router;
    MtpRoutedExpertsBuffers routed_experts;
    MtpSharedFFNWriteBuffers shared_ffn_write;
    MtpFinalMixerBuffers final_mixer;
    const MtpAttentionRuntimeMode attention_mode{sources.kv_rot};

    // Resolve the complete runtime view set before recording any MTP stage.
    // The individual encoders still validate formats, active extents, aliases,
    // parameter values, and Context ownership immediately before dispatch.
    if (!runtime.bind_input_projection(context, sources.input_projection,
                                       input_projection, error) ||
        !runtime.bind_add_streams(context, params_slot, add_streams, error) ||
        !runtime.bind_attention_read(context, params_slot, attention_read,
                                     error) ||
        !runtime.bind_attention_kv_projection(
            context, params_slot, attention_kv_projection, error) ||
        !runtime.bind_attention_pass(context, params_slot, attention_mode,
                                     sources.attention_window, attention_pass,
                                     error) ||
        !runtime.bind_feed_forward_read(context, params_slot,
                                        feed_forward_read, error) ||
        !runtime.bind_router(context, params_slot, router, error) ||
        !runtime.bind_routed_experts(context, routed_experts, error) ||
        !runtime.bind_shared_ffn_write(context, params_slot, shared_ffn_write,
                                       error) ||
        !runtime.bind_final_mixer(context, params_slot, final_mixer, error)) {
        if (error.empty()) error = "MTP full-layer runtime binding failed";
        return false;
    }

    if (!encode_mtp_input_projection(context, encoder, input_projection,
                                     error) ||
        !encode_mtp_add_streams(context, encoder, add_streams, error) ||
        !encode_mtp_attention_read(context, encoder, attention_read, error) ||
        !encode_mtp_attention_kv_projection(
            context, encoder, attention_kv_projection, error) ||
        !encode_mtp_attention_pass(context, encoder, attention_pass, error) ||
        !encode_mtp_feed_forward_read(context, encoder, feed_forward_read,
                                      error) ||
        !encode_mtp_router(context, encoder, router, error) ||
        !encode_mtp_routed_experts(context, encoder, routed_experts, error) ||
        !encode_mtp_shared_ffn_write(context, encoder, shared_ffn_write,
                                     error) ||
        !encode_mtp_final_mixer(context, encoder, final_mixer, error)) {
        return false;
    }

    error.clear();
    return true;
}

bool encode_mtp_full_layer(Context& context, CommandBuffer& command_buffer,
                           const MtpRuntimeStorage& runtime,
                           const MtpFullLayerSources& sources,
                           std::string& error) {
    error.clear();
    return Encoder::encode(
        context, command_buffer, "MTP full layer stages 1–20",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_full_layer(context, encoder, runtime, sources,
                                         encode_error);
        },
        error);
}

}  // namespace strata::metal
