// Ordered Prefill QSA projected-row layer stages.
#include "strata/metal/prefill_qsa_layer_dispatch.hpp"

#include "strata/metal/encoder.hpp"

#include <string>
#include <type_traits>
#include <variant>

namespace strata::metal {
namespace {

bool same_origin(BufferRef left, BufferRef right) {
    return left && right && left.contents == right.contents &&
           left.gpu_address == right.gpu_address &&
           left.offset == right.offset;
}

bool same_origin(BufferRef left, ReadOnlyBufferRef right) {
    return left && right && left.contents == right.contents() &&
           left.gpu_address == right.gpu_address() &&
           left.offset == right.offset();
}

bool preparation_feeds_tail(
    QsaAttentionMode mode, const PrefillQsaLayerBuffers& buffers,
    std::string& error) {
    const auto& preparation = buffers.preparation;
    const auto& tail = buffers.tail;
    if (!same_origin(preparation.key_norm_rope.steps,
                     tail.selection.steps) ||
        !same_origin(preparation.key_norm_rope.steps,
                     tail.attention.attention_chunk.steps)) {
        error = "Prefill QSA layer StepRows must feed selection and attention";
        return false;
    }
    if (!same_origin(preparation.query_norm_rope.output,
                     tail.attention.attention_chunk.query)) {
        error = "Prefill QSA layer qcur must feed attention";
        return false;
    }
    if (!same_origin(preparation.indexer_query_norm_rope.output,
                     tail.selection.indexer_queries)) {
        error = "Prefill QSA layer indexer query must feed selection";
        return false;
    }
    if (!same_origin(preparation.indexer_append.pooled,
                     tail.selection.pooled) ||
        !same_origin(preparation.indexer_append.dead,
                     tail.selection.dead)) {
        error = "Prefill QSA layer indexer state must feed selection";
        return false;
    }
    if (!same_origin(preparation.query_norm_rope.input,
                     tail.output.gate.q_full)) {
        error = "Prefill QSA layer Qf must feed the output gate";
        return false;
    }

    const bool kv_matches = std::visit(
        [&](const auto& append) {
            using Append = std::decay_t<decltype(append)>;
            if constexpr (std::is_same_v<Append, QsaKvAppendF16Buffers>) {
                return mode == QsaAttentionMode::f16 &&
                       same_origin(append.key_pool,
                                   tail.attention.attention_chunk.key_data) &&
                       same_origin(append.value_pool,
                                   tail.attention.attention_chunk.value_data);
            } else if constexpr (std::is_same_v<Append,
                                                QsaKvAppendI8Buffers>) {
                return mode == QsaAttentionMode::i8 &&
                       same_origin(append.key_codes,
                                   tail.attention.attention_chunk.key_data) &&
                       same_origin(append.key_scales,
                                   tail.attention.attention_chunk.key_scales) &&
                       same_origin(append.value_codes,
                                   tail.attention.attention_chunk.value_data) &&
                       same_origin(append.value_scales,
                                   tail.attention.attention_chunk.value_scales);
            } else if constexpr (std::is_same_v<Append,
                                                QsaKvAppendQ4Buffers>) {
                return mode == QsaAttentionMode::q4 &&
                       same_origin(append.key_pool,
                                   tail.attention.attention_chunk.key_data) &&
                       same_origin(append.value_pool,
                                   tail.attention.attention_chunk.value_data);
            } else {
                return mode == QsaAttentionMode::k8v4 &&
                       same_origin(append.key_codes,
                                   tail.attention.attention_chunk.key_data) &&
                       same_origin(append.key_scales,
                                   tail.attention.attention_chunk.key_scales) &&
                       same_origin(append.value_pool,
                                   tail.attention.attention_chunk.value_data);
            }
        },
        preparation.kv_append);
    if (!kv_matches) {
        error = "Prefill QSA layer KV append outputs must feed attention";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_qsa_layer(
    Context& context, Encoder& encoder, const PrefillParams& params,
    QsaAttentionMode attention_mode, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv, PrefillNativeWeightFormat output_format,
    const PrefillQsaLayerBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA layer requires an ordered concurrent-group "
                "boundary";
        return false;
    }
    if (!preparation_feeds_tail(attention_mode, buffers, error)) return false;

    PrefillQsaAttentionTransform attention_transform =
        PrefillQsaAttentionTransform::none;
    if (!encode_prefill_qsa_prepare(
            context, encoder, params, attention_mode, cell_capacity,
            rope_zero_cos, rope_zero_sin, rotate_uncompressed_kv,
            buffers.preparation, error, &attention_transform)) {
        return false;
    }
    return encode_prefill_qsa_tail(
        context, encoder, params, attention_mode, cell_capacity,
        output_format, buffers.tail, error, selected_backend,
        attention_transform);
}

bool encode_prefill_qsa_layer(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, QsaAttentionMode attention_mode,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv, PrefillNativeWeightFormat output_format,
    const PrefillQsaLayerBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend) {
    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    const bool encoded = Encoder::encode(
        context, command_buffer, "Prefill QSA layer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_layer(
                context, encoder, params, attention_mode, cell_capacity,
                rope_zero_cos, rope_zero_sin, rotate_uncompressed_kv,
                output_format, buffers, encode_error, &selected);
        },
        error);
    if (encoded && selected_backend != nullptr) {
        *selected_backend = selected;
    }
    return encoded;
}

}  // namespace strata::metal
