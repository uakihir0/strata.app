// Exclusive selection of the pending layer-0 MoE consumer.
#include "strata/metal/window_pending_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/window.hpp"

#include <array>
#include <initializer_list>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kWindowEmbeddingWidth = 2'560u;
constexpr std::uint32_t kGgmlTypeQ3K = 11u;
constexpr std::uint32_t kGgmlTypeIQ3S = 21u;
constexpr std::uint32_t kGgmlTypeIQ4XS = 23u;
constexpr std::uint32_t kGgmlTypeF32 = 0u;
constexpr std::uint32_t kGgmlTypeF16 = 1u;
constexpr std::uint32_t kGgmlTypeQ8_0 = 8u;
constexpr std::uint32_t kGgmlTypeBf16 = 30u;
constexpr std::uint32_t kGgmlTypeQ2_0 = 42u;

bool same_view(BufferRef left, BufferRef right) {
    return left && right && left.handle == right.handle &&
           left.contents == right.contents &&
           left.gpu_address == right.gpu_address &&
           left.length == right.length && left.offset == right.offset;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool views_overlap(BufferRef left, BufferRef right) {
    const bool same_buffer_overlap =
        left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.length,
                               right.offset, right.length);
    const bool cpu_backing_overlap =
        left.contents != nullptr && right.contents != nullptr &&
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.contents), left.length,
            reinterpret_cast<std::uintptr_t>(right.contents), right.length);
    const bool gpu_backing_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        address_ranges_overlap(left.gpu_address, left.length,
                               right.gpu_address, right.length);
    return same_buffer_overlap || cpu_backing_overlap || gpu_backing_overlap;
}

bool views_overlap(BufferRef left, ReadOnlyBufferRef right) {
    const auto* const right_contents = right.contents();
    const bool cpu_backing_overlap =
        left.contents != nullptr && right_contents != nullptr &&
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.contents), left.length,
            reinterpret_cast<std::uintptr_t>(right_contents), right.length());
    const bool gpu_backing_overlap =
        left.gpu_address != 0u && right.gpu_address() != 0u &&
        address_ranges_overlap(left.gpu_address, left.length,
                               right.gpu_address(), right.length());
    return cpu_backing_overlap || gpu_backing_overlap;
}

void append_view(BufferRef view, std::vector<BufferRef>& writable,
                 std::vector<ReadOnlyBufferRef>&) {
    writable.push_back(view);
}

void append_view(ReadOnlyBufferRef view, std::vector<BufferRef>&,
                 std::vector<ReadOnlyBufferRef>& readonly) {
    readonly.push_back(view);
}

void append_gr_views(const GRReadBuffers& buffers,
                     std::vector<BufferRef>& writable,
                     std::vector<ReadOnlyBufferRef>& readonly) {
    for (BufferRef view :
         {buffers.residual, buffers.block_out, buffers.inject_prev,
          buffers.norm, buffers.normalized, buffers.rs, buffers.hc_down,
          buffers.hc_inject, buffers.lo, buffers.inject_out, buffers.hc_up,
          buffers.mixed}) {
        append_view(view, writable, readonly);
    }
}

void append_gdn_views(const WindowGDNBlockBuffers& buffers,
                      std::vector<BufferRef>& writable,
                      std::vector<ReadOnlyBufferRef>& readonly) {
    for (BufferRef view :
         {buffers.mixed, buffers.history, buffers.state, buffers.qkv_weights,
          buffers.conv_weights, buffers.alpha, buffers.beta_weights,
          buffers.dt_bias, buffers.a, buffers.gate_weights, buffers.gamma,
          buffers.output_weights, buffers.xq, buffers.qkv, buffers.h,
          buffers.gate, buffers.beta, buffers.z, buffers.y, buffers.yq,
          buffers.block_out, buffers.sink, buffers.active_rows}) {
        append_view(view, writable, readonly);
    }
}

void append_routed_shared_views(
    const MoeRoutedSharedFFNBuffers& buffers,
    std::vector<BufferRef>& writable,
    std::vector<ReadOnlyBufferRef>& readonly) {
    const MoeRoutedFFNBuffers& routed = buffers.routed;
    for (ReadOnlyBufferRef view :
         {routed.addresses, routed.input_activations, routed.plans}) {
        append_view(view, writable, readonly);
    }
    for (BufferRef view :
         {routed.gate, routed.up, routed.hidden, routed.down_activations,
          routed.parts}) {
        append_view(view, writable, readonly);
    }

    const MoeSharedFFNBuffers& shared = buffers.shared;
    for (ReadOnlyBufferRef view :
         {shared.input_activations, shared.mixed, shared.gate_weights,
          shared.up_weights, shared.down_weights, shared.shared_gate_weights,
          shared.active_rows, shared.routed_parts, shared.route_weights}) {
        append_view(view, writable, readonly);
    }
    for (BufferRef view :
         {shared.gate, shared.up, shared.hidden, shared.down_activations,
          shared.shared_output, shared.shared_gate_logits, shared.output,
          shared.gate_sink, shared.up_sink, shared.down_sink,
          shared.shared_gate_sink}) {
        append_view(view, writable, readonly);
    }
}

void append_fullmode_layer0_views(
    const WindowT1MoeLayer0FullBuffers& buffers,
    std::vector<BufferRef>& writable,
    std::vector<ReadOnlyBufferRef>& readonly) {
    const WindowT1MoeRouterPrefixBuffers& router =
        buffers.producer.router;
    append_view(router.layer0.params, writable, readonly);
    append_gr_views(router.layer0.attention, writable, readonly);
    append_gdn_views(router.layer0.gdn, writable, readonly);
    append_gr_views(router.layer0.ffn, writable, readonly);
    append_view(router.router_tensor, writable, readonly);
    for (BufferRef view : {router.router_logits, router.route_ids,
                           router.route_weights, router.router_matvec_sink,
                           buffers.producer.usage,
                           buffers.producer.plans}) {
        append_view(view, writable, readonly);
    }
    append_view(buffers.producer.resident, writable, readonly);
    append_routed_shared_views(buffers.ffn, writable, readonly);
}

void append_ple_views(const PLEBlockBuffers& buffers,
                      std::vector<BufferRef>& writable,
                      std::vector<ReadOnlyBufferRef>& readonly) {
    for (BufferRef view :
         {buffers.input_f32, buffers.key_q8_1, buffers.matvec_sink,
          buffers.post.key, buffers.post.residual, buffers.post.query,
          buffers.post.ple_norm, buffers.post.value, buffers.post.gate,
          buffers.post.gated, buffers.post.history}) {
        append_view(view, writable, readonly);
    }
    for (ReadOnlyBufferRef view :
         {buffers.key_weights, buffers.value_weights, buffers.active_rows,
          buffers.post.conv1d, buffers.post.key_gamma,
          buffers.post.query_gamma, buffers.post.conv_gamma}) {
        append_view(view, writable, readonly);
    }
}

void append_pending_consumer_views(
    const WindowMoePendingConsumer& consumer,
    std::vector<BufferRef>& writable,
    std::vector<ReadOnlyBufferRef>& readonly) {
    if (const auto* ple =
            std::get_if<WindowMoePendingPLEConsumer>(&consumer)) {
        append_routed_shared_views(ple->buffers.previous_ffn, writable,
                                   readonly);
        append_gr_views(ple->buffers.previous_gr_read, writable, readonly);
        append_ple_views(ple->buffers.ple, writable, readonly);
        return;
    }
    const auto* attention =
        std::get_if<WindowMoeAttentionEntryBuffers>(&consumer);
    if (attention == nullptr) return;
    append_routed_shared_views(attention->previous_ffn, writable, readonly);
    append_gr_views(attention->previous_gr_read, writable, readonly);
    append_gr_views(attention->attention, writable, readonly);
}

bool validate_embedding_liveness(
    BufferRef embedding, const WindowT1MoeLayer0FullBuffers& layer0,
    const WindowMoePendingConsumer& consumer, std::string& error) {
    std::vector<BufferRef> writable;
    std::vector<ReadOnlyBufferRef> readonly;
    append_fullmode_layer0_views(layer0, writable, readonly);
    append_pending_consumer_views(consumer, writable, readonly);

    const BufferRef pending_output = layer0.ffn.shared.output;
    for (BufferRef view : writable) {
        if (!views_overlap(embedding, view)) continue;
        if (same_view(embedding, pending_output) &&
            same_view(view, pending_output)) {
            continue;
        }
        error =
            "T=1 embedding scratch overlaps a later FullMode producer or consumer buffer";
        return false;
    }
    for (ReadOnlyBufferRef view : readonly) {
        if (views_overlap(embedding, view)) {
            error =
                "T=1 embedding scratch overlaps a later read-only FullMode input";
            return false;
        }
    }
    error.clear();
    return true;
}

bool embedding_format(std::uint32_t ggml_type, EmbedRowsFormat& format) {
    switch (ggml_type) {
        case kGgmlTypeQ3K:
            format = EmbedRowsFormat::q3_k;
            return true;
        case kGgmlTypeIQ3S:
            format = EmbedRowsFormat::iq3_s;
            return true;
        case kGgmlTypeIQ4XS:
            format = EmbedRowsFormat::iq4_xs;
            return true;
        default:
            return false;
    }
}

bool resolve_ple_model_weights(const FullModeStorage& full_mode,
                               WindowMoePendingConsumer& consumer,
                               std::string& error) {
    auto* const ple = std::get_if<WindowMoePendingPLEConsumer>(&consumer);
    if (ple == nullptr) {
        error = "PLE model-weight resolution requires the selected PLE branch";
        return false;
    }

    FullModeTensorView key;
    if (!full_mode.dense_tensor_view("blk.1.ple_key.weight", key, error)) {
        return false;
    }
    if (key.shape != std::vector<std::uint64_t>{2'560u, 10'240u}) {
        error = "FullMode layer-1 PLE key must have shape [2560, 10240]";
        return false;
    }
    switch (key.ggml_type) {
        case kGgmlTypeQ2_0:
            ple->key_format = MatVecWeightFormat::q2_0;
            break;
        case kGgmlTypeQ8_0:
            ple->key_format = MatVecWeightFormat::q8_0;
            break;
        case kGgmlTypeBf16:
            ple->key_format = MatVecWeightFormat::bf16;
            break;
        default:
            error = "FullMode layer-1 PLE key uses an unsupported native weight format";
            return false;
    }
    ple->buffers.ple.key_weights = key.buffer;

    FullModeTensorView value;
    if (!full_mode.dense_tensor_view("blk.1.ple_value.weight", value,
                                     error)) {
        return false;
    }
    if (value.ggml_type != kGgmlTypeBf16 ||
        value.shape != std::vector<std::uint64_t>{2'560u, 2'560u}) {
        error = "FullMode layer-1 PLE value must be BF16 with shape [2560, 2560]";
        return false;
    }
    ple->buffers.ple.value_weights = value.buffer;

    struct TensorExpectation {
        const char* name;
        std::vector<std::uint64_t> shape;
        std::uint32_t ggml_type;
        ReadOnlyBufferRef PLEPostProjectionBuffers::*member;
        const char* description;
    };
    const std::array<TensorExpectation, 4> post_weights{{
        {"blk.1.ple_norm_key.weight", {10'240u}, kGgmlTypeF32,
         &PLEPostProjectionBuffers::key_gamma, "key normalization"},
        {"blk.1.ple_norm_query.weight", {10'240u}, kGgmlTypeF32,
         &PLEPostProjectionBuffers::query_gamma, "query normalization"},
        {"blk.1.ple_norm_conv.weight", {10'240u}, kGgmlTypeF32,
         &PLEPostProjectionBuffers::conv_gamma, "convolution normalization"},
        {"blk.1.ple_conv1d.weight", {4u, 10'240u}, kGgmlTypeF16,
         &PLEPostProjectionBuffers::conv1d, "convolution"},
    }};
    for (const TensorExpectation& expected : post_weights) {
        FullModeTensorView tensor;
        if (!full_mode.dense_tensor_view(expected.name, tensor, error)) {
            return false;
        }
        if (tensor.ggml_type != expected.ggml_type ||
            tensor.shape != expected.shape) {
            error = "FullMode layer-1 PLE " +
                    std::string(expected.description) +
                    " tensor has an unsupported type or shape";
            return false;
        }
        ple->buffers.ple.post.*(expected.member) = tensor.buffer;
    }
    error.clear();
    return true;
}

}  // namespace

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

bool encode_window_t1_full_mode_step(
    Context& context, const FullModeStorage& full_mode, Encoder& encoder,
    std::uint32_t vocabulary_size, std::uint32_t experts,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    bool ple_enabled, const WindowT1FullModeStepBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "T=1 FullMode Window step requires an ordered dispatch boundary";
        return false;
    }

    const BufferRef params =
        buffers.layer0.producer.router.layer0.params;
    if (!same_view(buffers.streams,
                   buffers.layer0.producer.router.layer0.attention.residual)) {
        error = "T=1 FullMode Window streams must be the layer-0 attention residual view";
        return false;
    }

    const bool has_ple =
        std::holds_alternative<WindowMoePendingPLEConsumer>(
            buffers.consumer);
    if (ple_enabled != has_ple) {
        error = "Window pending-MoE consumer does not match the selected PLE branch";
        return false;
    }
    WindowMoePendingConsumer consumer = buffers.consumer;
    if (ple_enabled &&
        !resolve_ple_model_weights(full_mode, consumer, error)) {
        return false;
    }

    FullModeTensorView token_embedding;
    if (!full_mode.dense_tensor_view("token_embd.weight", token_embedding,
                                     error)) {
        return false;
    }
    if (token_embedding.shape.size() != 2u ||
        token_embedding.shape[0] != kWindowEmbeddingWidth ||
        token_embedding.shape[1] != vocabulary_size) {
        error = "FullMode token embedding must have shape [2560, vocabulary_size]";
        return false;
    }
    EmbedRowsFormat format{};
    if (!embedding_format(token_embedding.ggml_type, format)) {
        error = "FullMode token embedding uses an unsupported row format";
        return false;
    }
    if (!validate_embedding_liveness(buffers.embedding, buffers.layer0,
                                     consumer, error)) {
        return false;
    }

    const WindowEmbeddingBuffers embedding_buffers{
        params, token_embedding.buffer, buffers.embedding, buffers.streams};
    if (!encode_window_embedding(context, encoder, vocabulary_size, format,
                                 embedding_buffers, error)) {
        return false;
    }
    if (!encode_window_t1_moe_layer0_full_mode(
            context, full_mode, encoder, vocabulary_size, experts,
            gate_up_format, down_format, buffers.layer0, error)) {
        return false;
    }
    return encode_window_moe_pending_consumer(
        context, encoder, 1u, ple_enabled, consumer, error);
}

bool encode_window_t1_full_mode_step(
    Context& context, const FullModeStorage& full_mode,
    CommandBuffer& command_buffer, std::uint32_t vocabulary_size,
    std::uint32_t experts, MoeExpertFormat gate_up_format,
    MoeDownFormat down_format, bool ple_enabled,
    const WindowT1FullModeStepBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window T=1 FullMode token-to-consumer step",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_full_mode_step(
                context, full_mode, encoder, vocabulary_size, experts,
                gate_up_format, down_format, ple_enabled, buffers,
                encode_error);
        },
        error);
}

}  // namespace strata::metal
