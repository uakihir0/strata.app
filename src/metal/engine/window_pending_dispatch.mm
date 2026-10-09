// Exclusive selection of the pending layer-0 MoE consumer.
#include "strata/metal/window_pending_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/window.hpp"

#include <array>
#include <cstring>
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
constexpr std::uint32_t kMaxWindowRows = 8u;

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

bool validate_fullmode_step_consumer_aliases(
    const WindowMoeLayer0FullModeStepBuffers& buffers,
    const WindowMoePendingConsumer& consumer, std::string& error) {
    std::vector<BufferRef> producer_views;
    std::vector<ReadOnlyBufferRef> producer_readonly;
    append_fullmode_layer0_views(buffers.layer0, producer_views,
                                 producer_readonly);

    std::vector<BufferRef> producer_writes;
    std::vector<BufferRef> producer_reusable_scratch;
    const WindowT1MoeRouterPrefixBuffers& router =
        buffers.layer0.producer.router;
    const GRReadBuffers& attention = router.layer0.attention;
    const WindowGDNBlockBuffers& gdn = router.layer0.gdn;
    const GRReadBuffers& ffn = router.layer0.ffn;
    producer_reusable_scratch.insert(
        producer_reusable_scratch.end(),
        {attention.block_out, attention.normalized, attention.rs,
         attention.lo, attention.inject_out, attention.mixed, gdn.xq,
         gdn.qkv, gdn.h,
         gdn.gate, gdn.beta, gdn.z, gdn.y, gdn.yq, gdn.block_out,
         gdn.sink, ffn.normalized, ffn.rs, ffn.lo, ffn.inject_prev,
         ffn.mixed,
         router.router_logits, router.route_ids, router.route_weights,
         router.router_matvec_sink, buffers.layer0.producer.plans});
    producer_writes.insert(producer_writes.end(),
                           {attention.normalized, attention.rs,
                            attention.lo, attention.inject_out,
                            attention.mixed, gdn.history, gdn.state,
                            gdn.xq, gdn.qkv, gdn.h, gdn.gate, gdn.beta,
                            gdn.z, gdn.y, gdn.yq, gdn.block_out,
                            gdn.sink, ffn.normalized, ffn.rs, ffn.lo,
                            ffn.inject_out, ffn.mixed,
                            router.router_logits, router.route_ids,
                            router.route_weights,
                            router.router_matvec_sink,
                            buffers.layer0.producer.usage,
                            buffers.layer0.producer.plans});
    const MoeRoutedSharedFFNBuffers& moe = buffers.layer0.ffn;
    producer_writes.insert(
        producer_writes.end(),
        {moe.routed.gate, moe.routed.up, moe.routed.hidden,
         moe.routed.down_activations, moe.routed.parts,
         moe.shared.gate, moe.shared.up, moe.shared.hidden,
         moe.shared.down_activations, moe.shared.shared_output,
         moe.shared.shared_gate_logits, moe.shared.output,
         moe.shared.gate_sink, moe.shared.up_sink,
         moe.shared.down_sink, moe.shared.shared_gate_sink});

    struct ConsumerReadHandoff {
        BufferRef view;
        BufferRef allowed_producer_output;
    };
    std::vector<ConsumerReadHandoff> consumer_reads;
    std::vector<ReadOnlyBufferRef> consumer_readonly;
    std::vector<BufferRef> consumer_writes;
    BufferRef consumer_residual;
    if (const auto* ple =
            std::get_if<WindowMoePendingPLEConsumer>(&consumer)) {
        const WindowMoePLEEntryBuffers& entry = ple->buffers;
        const PLEBlockBuffers& block = entry.ple;
        consumer_reads.insert(
            consumer_reads.end(),
            {{entry.previous_ffn.shared.output,
              buffers.layer0.ffn.shared.output},
             {entry.previous_gr_read.residual, {}},
             {entry.previous_gr_read.inject_out,
              buffers.layer0.producer.router.layer0.ffn.inject_out},
             {block.input_f32, {}},
             {block.post.history, {}}});
        consumer_readonly.insert(
            consumer_readonly.end(),
            {block.key_weights, block.value_weights, block.active_rows,
             block.post.conv1d, block.post.key_gamma,
             block.post.query_gamma, block.post.conv_gamma});
        consumer_residual = block.post.residual;
        consumer_writes.insert(
            consumer_writes.end(),
            {block.key_q8_1, block.matvec_sink, block.post.key,
             block.post.residual, block.post.query, block.post.ple_norm,
             block.post.value, block.post.gate, block.post.gated});
    } else if (const auto* attention_entry =
                   std::get_if<WindowMoeAttentionEntryBuffers>(&consumer)) {
        consumer_reads.insert(
            consumer_reads.end(),
            {{attention_entry->previous_ffn.shared.output,
              buffers.layer0.ffn.shared.output},
             {attention_entry->previous_gr_read.residual, {}},
             {attention_entry->previous_gr_read.inject_out,
              buffers.layer0.producer.router.layer0.ffn.inject_out},
             {attention_entry->attention.norm, {}},
             {attention_entry->attention.hc_down, {}},
             {attention_entry->attention.hc_inject, {}},
             {attention_entry->attention.hc_up, {}}});
        consumer_residual = attention_entry->attention.residual;
        consumer_writes.insert(
            consumer_writes.end(),
            {attention_entry->attention.residual,
             attention_entry->attention.normalized,
             attention_entry->attention.rs,
             attention_entry->attention.lo,
             attention_entry->attention.inject_out,
             attention_entry->attention.mixed});
    } else {
        error = "Window FullMode pending consumer payload is invalid";
        return false;
    }

    const BufferRef expected_residual = ffn.residual;
    const auto is_reusable_producer_scratch = [&](BufferRef view) {
        return std::any_of(
            producer_reusable_scratch.begin(),
            producer_reusable_scratch.end(),
            [&](BufferRef scratch) { return same_view(view, scratch); });
    };
    for (std::size_t output_index = 0u;
         output_index < consumer_writes.size(); ++output_index) {
        const BufferRef output = consumer_writes[output_index];
        for (std::size_t producer_index = 0u;
             producer_index < producer_views.size(); ++producer_index) {
            const BufferRef producer_view = producer_views[producer_index];
            const bool allowed_residual_handoff =
                same_view(output, consumer_residual) &&
                same_view(output, expected_residual) &&
                same_view(producer_view, expected_residual);
            const bool allowed_reuse =
                same_view(output, producer_view) &&
                is_reusable_producer_scratch(producer_view);
            if (views_overlap(output, producer_view) &&
                !allowed_residual_handoff && !allowed_reuse) {
                error = "Window FullMode consumer writable view overlaps producer storage outside declared handoffs or dead scratch";
                return false;
            }
        }
        for (ReadOnlyBufferRef producer_view : producer_readonly) {
            const bool allowed_reuse =
                views_overlap(output, producer_view) &&
                is_reusable_producer_scratch(output);
            if (views_overlap(output, producer_view) && !allowed_reuse) {
                error = "Window FullMode consumer writable views overlap producer read-only storage";
                return false;
            }
        }
    }

    const auto validate_consumer_buffer_read =
        [&](const ConsumerReadHandoff& read) {
            for (BufferRef producer_output : producer_writes) {
                const bool allowed_handoff =
                    read.allowed_producer_output &&
                    same_view(read.view, read.allowed_producer_output) &&
                    same_view(producer_output,
                              read.allowed_producer_output);
                if (views_overlap(read.view, producer_output) &&
                    !allowed_handoff) {
                    error = "Window FullMode consumer read-only inputs overlap producer outputs outside declared handoffs";
                    return false;
                }
            }
            return true;
        };
    for (const ConsumerReadHandoff& read : consumer_reads) {
        if (!validate_consumer_buffer_read(read)) return false;
    }
    for (ReadOnlyBufferRef input : consumer_readonly) {
        for (BufferRef producer_output : producer_writes) {
            if (views_overlap(producer_output, input)) {
                error = "Window FullMode consumer read-only inputs overlap producer outputs outside declared handoffs";
                return false;
            }
        }
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
    if (rows == 0u || rows > kMaxWindowRows) {
        error = "Window pending-MoE consumer rows must be in 1..8";
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

bool encode_window_moe_layer0_full_mode_step(
    Context& context, const FullModeStorage& full_mode, Encoder& encoder,
    std::uint32_t vocabulary_size, std::uint32_t experts,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    bool ple_enabled, const WindowMoeLayer0FullModeStepBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window FullMode producer/consumer step requires ordered dispatch boundaries";
        return false;
    }

    const auto* const ple =
        std::get_if<WindowMoePendingPLEConsumer>(&buffers.consumer);
    const auto* const attention =
        std::get_if<WindowMoeAttentionEntryBuffers>(&buffers.consumer);
    if (ple_enabled != (ple != nullptr) ||
        ple_enabled == (attention != nullptr)) {
        error = "Window pending-MoE consumer does not match the selected PLE branch";
        return false;
    }

    const GRReadBuffers& producer_ffn =
        buffers.layer0.producer.router.layer0.ffn;
    const BufferRef pending_output = buffers.layer0.ffn.shared.output;
    const MoeRoutedSharedFFNBuffers* consumer_ffn = nullptr;
    const GRReadBuffers* consumer_previous_gr = nullptr;
    if (ple != nullptr) {
        consumer_ffn = &ple->buffers.previous_ffn;
        consumer_previous_gr = &ple->buffers.previous_gr_read;
    } else {
        consumer_ffn = &attention->previous_ffn;
        consumer_previous_gr = &attention->previous_gr_read;
    }
    if (!same_view(consumer_ffn->shared.output, pending_output)) {
        error = "Window FullMode consumer must use this producer's pending MoE output";
        return false;
    }
    if (!same_view(consumer_previous_gr->residual,
                   producer_ffn.residual) ||
        !same_view(consumer_previous_gr->inject_out,
                   producer_ffn.inject_out)) {
        error = "Window FullMode consumer must use this producer's FFN residual and injection";
        return false;
    }
    if (ple_enabled && ple->history_update != PLEHistoryUpdate::defer) {
        error = "speculative FullMode PLE consumer must defer committed history advancement";
        return false;
    }
    if (!validate_fullmode_step_consumer_aliases(buffers, buffers.consumer,
                                                 error)) {
        return false;
    }

    WindowMoePendingConsumer consumer = buffers.consumer;
    if (ple_enabled && !resolve_ple_model_weights(full_mode, consumer, error)) {
        return false;
    }
    if (!encode_window_moe_layer0_full_mode(
            context, full_mode, encoder, vocabulary_size, experts,
            gate_up_format, down_format, buffers.layer0, error)) {
        return false;
    }

    WindowParams params{};
    std::memcpy(&params,
                buffers.layer0.producer.router.layer0.params.contents,
                sizeof(params));
    if (auto* const selected_ple =
            std::get_if<WindowMoePendingPLEConsumer>(&consumer)) {
        selected_ple->compiled_columns = params.nc;
    }
    return encode_window_moe_pending_consumer(
        context, encoder, params.T, ple_enabled, consumer, error);
}

bool encode_window_moe_layer0_full_mode_step(
    Context& context, const FullModeStorage& full_mode,
    CommandBuffer& command_buffer, std::uint32_t vocabulary_size,
    std::uint32_t experts, MoeExpertFormat gate_up_format,
    MoeDownFormat down_format, bool ple_enabled,
    const WindowMoeLayer0FullModeStepBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer,
        "Window speculative FullMode producer and pending consumer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_moe_layer0_full_mode_step(
                context, full_mode, encoder, vocabulary_size, experts,
                gate_up_format, down_format, ple_enabled, buffers,
                encode_error);
        },
        error);
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
