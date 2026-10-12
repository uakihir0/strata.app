// Ordered Prefill QSA projected-row layer stages.
#include "strata/metal/prefill_qsa_layer_dispatch.hpp"

#include "strata/metal/encoder.hpp"

#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
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

enum class PipelineHandoff : std::uint8_t {
    none,
    key,
    value,
    query_full,
    indexer_key,
    indexer_query,
};

struct PipelineRange {
    void* handle = nullptr;
    std::uintptr_t cpu_address = 0u;
    std::uint64_t gpu_address = 0u;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
    PipelineHandoff handoff = PipelineHandoff::none;
};

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool ranges_overlap(const PipelineRange& left, const PipelineRange& right) {
    const bool same_allocation =
        left.handle != nullptr && left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.bytes, right.offset,
                               right.bytes);
    return same_allocation ||
           address_ranges_overlap(left.cpu_address, left.bytes,
                                  right.cpu_address, right.bytes) ||
           address_ranges_overlap(left.gpu_address, left.bytes,
                                  right.gpu_address, right.bytes);
}

bool same_origin(const PipelineRange& left, const PipelineRange& right) {
    return left.cpu_address != 0u &&
           left.cpu_address == right.cpu_address &&
           left.gpu_address == right.gpu_address &&
           left.offset == right.offset;
}

PipelineRange make_pipeline_range(
    BufferRef view, std::size_t bytes, std::string_view name,
    PipelineHandoff handoff = PipelineHandoff::none) {
    return {view.handle, reinterpret_cast<std::uintptr_t>(view.contents),
            view.gpu_address, view.offset, bytes, name, handoff};
}

PipelineRange make_pipeline_range(
    ReadOnlyBufferRef view, std::size_t bytes, std::string_view name,
    PipelineHandoff handoff = PipelineHandoff::none) {
    return {nullptr, reinterpret_cast<std::uintptr_t>(view.contents()),
            view.gpu_address(), view.offset(), bytes, name, handoff};
}

bool checked_product(std::initializer_list<std::size_t> factors,
                     std::size_t& result) {
    result = 1u;
    for (const std::size_t factor : factors) {
        if (factor != 0u &&
            result > std::numeric_limits<std::size_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

bool projection_writes_are_disjoint_from_layer_inputs(
    const PrefillParams& params, QsaAttentionMode mode,
    std::uint32_t cell_capacity, PrefillNativeWeightFormat output_format,
    const PrefillQsaLayerPipelineBuffers& buffers, std::string& error) {
    constexpr std::size_t kHeadDim = 256u;
    constexpr std::size_t kKvHeads = 2u;
    constexpr std::size_t kQueryHeads = 24u;
    constexpr std::size_t kIndexerHeadDim = 128u;
    constexpr std::size_t kIndexerHeads = 4u;
    constexpr std::size_t kQ4BlockBytes = 18u;
    if (cell_capacity == 0u || cell_capacity > 262144u ||
        cell_capacity % 4u != 0u) {
        error = "Prefill QSA preparation cell capacity is invalid";
        return false;
    }

    PrefillQsaProjectionBufferSizes projection_sizes;
    if (!prefill_qsa_projection_buffer_sizes(
            params.T, projection_sizes, buffers.projection, error)) {
        return false;
    }
    const PrefillGemmMPPParams output_params{
        params.T, 2560u, 6144u, 2560u, 0u, false};
    PrefillNativeProjectionBufferSizes output_sizes;
    if (!prefill_native_projection_buffer_sizes(
            output_params, output_format, output_sizes, error)) {
        return false;
    }

    std::size_t kv_values_bytes = 0u;
    std::size_t query_full_bytes = 0u;
    std::size_t query_bytes = 0u;
    std::size_t indexer_key_bytes = 0u;
    std::size_t indexer_query_bytes = 0u;
    std::size_t step_bytes = 0u;
    std::size_t rope_bytes = 0u;
    std::size_t indexer_tail_bytes = 0u;
    std::size_t indexer_dead_bytes = 0u;
    std::size_t indexer_pool_bytes = 0u;
    if (!checked_product({params.T, kKvHeads, kHeadDim, sizeof(float)},
                         kv_values_bytes) ||
        !checked_product({params.T, kQueryHeads, 2u * kHeadDim,
                          sizeof(float)},
                         query_full_bytes) ||
        !checked_product({params.T, kQueryHeads, kHeadDim, sizeof(float)},
                         query_bytes) ||
        !checked_product({params.T, kIndexerHeadDim, sizeof(float)},
                         indexer_key_bytes) ||
        !checked_product({params.T, kIndexerHeads, kIndexerHeadDim,
                          sizeof(float)},
                         indexer_query_bytes) ||
        !checked_product({params.T, sizeof(StepRow)}, step_bytes) ||
        !checked_product({cell_capacity, 64u, sizeof(float)}, rope_bytes) ||
        !checked_product({3u, kIndexerHeadDim, sizeof(float)},
                         indexer_tail_bytes) ||
        !checked_product({kIndexerHeadDim, sizeof(float)},
                         indexer_dead_bytes) ||
        !checked_product({cell_capacity / 4u + 2u, kIndexerHeadDim,
                          sizeof(float)},
                         indexer_pool_bytes)) {
        error = "Prefill QSA pipeline active range overflows host address "
                "space";
        return false;
    }

    std::size_t key_data_bytes = 0u;
    std::size_t key_scales_bytes = 0u;
    std::size_t value_data_bytes = 0u;
    std::size_t value_scales_bytes = 0u;
    switch (mode) {
        case QsaAttentionMode::f16:
            if (!checked_product({cell_capacity, kKvHeads, kHeadDim,
                                  sizeof(std::uint16_t)},
                                 key_data_bytes)) {
                error = "Prefill QSA pipeline KV range overflows";
                return false;
            }
            value_data_bytes = key_data_bytes;
            break;
        case QsaAttentionMode::i8:
            if (!checked_product({cell_capacity, kKvHeads, kHeadDim},
                                 key_data_bytes) ||
                !checked_product({cell_capacity, kKvHeads, 4u,
                                  sizeof(std::uint16_t)},
                                 key_scales_bytes)) {
                error = "Prefill QSA pipeline KV range overflows";
                return false;
            }
            value_data_bytes = key_data_bytes;
            value_scales_bytes = key_scales_bytes;
            break;
        case QsaAttentionMode::q4:
            if (!checked_product({cell_capacity, kKvHeads, kHeadDim / 32u,
                                  kQ4BlockBytes},
                                 key_data_bytes)) {
                error = "Prefill QSA pipeline KV range overflows";
                return false;
            }
            value_data_bytes = key_data_bytes;
            break;
        case QsaAttentionMode::k8v4:
            if (!checked_product({cell_capacity, kKvHeads, kHeadDim},
                                 key_data_bytes) ||
                !checked_product({cell_capacity, kKvHeads, 4u,
                                  sizeof(std::uint16_t)},
                                 key_scales_bytes) ||
                !checked_product({cell_capacity, kKvHeads, kHeadDim / 32u,
                                  kQ4BlockBytes},
                                 value_data_bytes)) {
                error = "Prefill QSA pipeline KV range overflows";
                return false;
            }
            break;
        default:
            error = "Prefill QSA pipeline attention mode is invalid";
            return false;
    }

    const auto& layer = buffers.layer;
    const auto& preparation = layer.preparation;
    const auto& tail = layer.tail;
    BufferRef append_key{};
    BufferRef append_value{};
    BufferRef append_steps{};
    std::visit(
        [&](const auto& append) {
            append_key = append.keys;
            append_value = append.values;
            append_steps = append.steps;
        },
        preparation.kv_append);
    std::array<PipelineRange, 40u> inputs{};
    std::size_t input_count = 0u;
    const auto add_input = [&](BufferRef view, std::size_t bytes,
                               std::string_view name,
                               PipelineHandoff handoff =
                                   PipelineHandoff::none) {
        if (input_count == inputs.size()) return false;
        inputs[input_count++] =
            make_pipeline_range(view, bytes, name, handoff);
        return true;
    };
    const auto add_read_input = [&](ReadOnlyBufferRef view,
                                    std::size_t bytes,
                                    std::string_view name,
                                    PipelineHandoff handoff =
                                        PipelineHandoff::none) {
        if (input_count == inputs.size()) return false;
        inputs[input_count++] =
            make_pipeline_range(view, bytes, name, handoff);
        return true;
    };
    const bool inputs_collected =
        add_input(preparation.key_norm_rope.input, kv_values_bytes,
                  "key normalization input", PipelineHandoff::key) &&
        add_input(preparation.key_norm_rope.gamma, kHeadDim * sizeof(float),
                  "attn_k_norm gamma") &&
        add_input(preparation.key_norm_rope.steps, step_bytes,
                  "key normalization StepRows") &&
        add_input(preparation.key_norm_rope.rope, rope_bytes,
                  "key normalization RoPE") &&
        add_input(append_key, kv_values_bytes, "KV append K input",
                  PipelineHandoff::key) &&
        add_input(append_value, kv_values_bytes, "KV append V input",
                  PipelineHandoff::value) &&
        add_input(append_steps, step_bytes, "KV append StepRows") &&
        add_input(preparation.query_norm_rope.input, query_full_bytes,
                  "query normalization input",
                  PipelineHandoff::query_full) &&
        add_input(preparation.query_norm_rope.gamma,
                  kHeadDim * sizeof(float), "attn_q_norm gamma") &&
        add_input(preparation.query_norm_rope.steps, step_bytes,
                  "query normalization StepRows") &&
        add_input(preparation.query_norm_rope.rope, rope_bytes,
                  "query normalization RoPE") &&
        add_input(preparation.indexer_query_norm_rope.input,
                  indexer_query_bytes, "indexer query normalization input",
                  PipelineHandoff::indexer_query) &&
        add_input(preparation.indexer_query_norm_rope.gamma,
                  kIndexerHeadDim * sizeof(float), "indexer.q_norm gamma") &&
        add_input(preparation.indexer_query_norm_rope.steps, step_bytes,
                  "indexer query StepRows") &&
        add_input(preparation.indexer_query_norm_rope.rope, rope_bytes,
                  "indexer query RoPE") &&
        add_input(preparation.indexer_append.raw_keys, indexer_key_bytes,
                  "indexer append K input",
                  PipelineHandoff::indexer_key) &&
        add_input(preparation.indexer_append.gamma,
                  kIndexerHeadDim * sizeof(float), "indexer.k_norm gamma") &&
        add_input(preparation.indexer_append.rope, rope_bytes,
                  "indexer append RoPE") &&
        add_input(preparation.indexer_append.tail, indexer_tail_bytes,
                  "indexer tail state") &&
        add_input(preparation.indexer_append.dead, indexer_dead_bytes,
                  "indexer dead state") &&
        add_input(preparation.indexer_append.pooled, indexer_pool_bytes,
                  "indexer pooled state") &&
        add_input(preparation.indexer_append.block_position,
                  sizeof(std::int32_t), "indexer block position") &&
        add_input(tail.selection.pooled, indexer_pool_bytes,
                  "selection pooled input") &&
        add_input(tail.selection.dead, indexer_dead_bytes,
                  "selection dead input") &&
        add_input(tail.selection.indexer_queries, indexer_query_bytes,
                  "selection indexer queries",
                  PipelineHandoff::indexer_query) &&
        add_input(tail.selection.steps, step_bytes, "selection StepRows") &&
        add_input(tail.attention.attention_chunk.query, query_bytes,
                  "attention qcur input") &&
        add_input(tail.attention.attention_chunk.steps, step_bytes,
                  "attention StepRows") &&
        add_input(tail.attention.attention_chunk.key_data, key_data_bytes,
                  "attention K cache input") &&
        add_input(tail.attention.attention_chunk.key_scales,
                  key_scales_bytes,
                  "attention K scales input") &&
        add_input(tail.attention.attention_chunk.value_data,
                  value_data_bytes, "attention V cache input") &&
        add_input(tail.attention.attention_chunk.value_scales,
                  value_scales_bytes, "attention V scales input") &&
        add_read_input(tail.output.gate.q_full, query_full_bytes,
                       "attention gate Qf input",
                       PipelineHandoff::query_full) &&
        add_read_input(tail.output.native_weights, output_sizes.weights_bytes,
                       "attn_output native weights");
    if (!inputs_collected) {
        error = "Prefill QSA pipeline input range table is full";
        return false;
    }

    const std::array<PipelineRange, 6u> writes{
        make_pipeline_range(
            buffers.projection.dequantized_f16,
            projection_sizes.dequantized_f16_bytes,
            "dequantized F16 scratch"),
        make_pipeline_range(buffers.projection.key,
                            projection_sizes.attn_k.y_f32_bytes,
                            "projected K", PipelineHandoff::key),
        make_pipeline_range(buffers.projection.value,
                            projection_sizes.attn_v.y_f32_bytes,
                            "projected V", PipelineHandoff::value),
        make_pipeline_range(buffers.projection.query_full,
                            projection_sizes.attn_q.y_f32_bytes,
                            "projected Qf", PipelineHandoff::query_full),
        make_pipeline_range(buffers.projection.indexer_key,
                            projection_sizes.indexer_k_output_bytes,
                            "projected indexer K",
                            PipelineHandoff::indexer_key),
        make_pipeline_range(buffers.projection.indexer_query,
                            projection_sizes.indexer_q_output_bytes,
                            "projected indexer Q",
                            PipelineHandoff::indexer_query)};
    for (const PipelineRange& write : writes) {
        for (std::size_t input_index = 0u; input_index < input_count;
             ++input_index) {
            const PipelineRange& input = inputs[input_index];
            if (write.handoff != PipelineHandoff::none &&
                write.handoff == input.handoff &&
                same_origin(write, input)) {
                continue;
            }
            if (ranges_overlap(write, input)) {
                error = "Prefill QSA pipeline " + std::string(write.name) +
                        " overlaps live layer input " +
                        std::string(input.name);
                return false;
            }
        }
    }
    return true;
}

bool projection_feeds_layer(
    const PrefillQsaLayerPipelineBuffers& buffers, std::string& error) {
    const auto& projection = buffers.projection;
    const auto& layer = buffers.layer;
    const auto& preparation = layer.preparation;
    if (!same_origin(projection.key,
                     preparation.key_norm_rope.input)) {
        error = "Prefill QSA pipeline projected K must feed key "
                "normalization";
        return false;
    }
    if (!same_origin(projection.value,
                     std::visit([](const auto& append) {
                         return append.values;
                     }, preparation.kv_append))) {
        error = "Prefill QSA pipeline projected V must feed KV append";
        return false;
    }
    if (!same_origin(projection.query_full,
                     preparation.query_norm_rope.input)) {
        error = "Prefill QSA pipeline projected Qf must feed query "
                "normalization";
        return false;
    }
    if (!same_origin(projection.indexer_key,
                     preparation.indexer_append.raw_keys)) {
        error = "Prefill QSA pipeline projected indexer K must feed "
                "indexer append";
        return false;
    }
    if (!same_origin(
            projection.indexer_query,
            preparation.indexer_query_norm_rope.input)) {
        error = "Prefill QSA pipeline projected indexer Q must feed "
                "query normalization";
        return false;
    }
    const bool key_append_matches = std::visit(
        [&](const auto& append) {
            return same_origin(projection.key, append.keys);
        },
        preparation.kv_append);
    if (!key_append_matches) {
        error = "Prefill QSA pipeline projected K must feed KV append";
        return false;
    }
    return true;
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

bool encode_prefill_qsa_layer_pipeline(
    Context& context, Encoder& encoder, const PrefillParams& params,
    PrefillBF16X2Mode projection_mode, QsaAttentionMode attention_mode,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv, PrefillNativeWeightFormat output_format,
    const PrefillQsaLayerPipelineBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA layer pipeline requires an ordered "
                "concurrent-group boundary";
        return false;
    }
    if (!projection_feeds_layer(buffers, error)) return false;
    if (!projection_writes_are_disjoint_from_layer_inputs(
            params, attention_mode, cell_capacity, output_format, buffers,
            error)) {
        return false;
    }
    if (!encode_prefill_qsa_projection(
            context, encoder, params.T, projection_mode,
            buffers.projection, error)) {
        return false;
    }
    return encode_prefill_qsa_layer(
        context, encoder, params, attention_mode, cell_capacity,
        rope_zero_cos, rope_zero_sin, rotate_uncompressed_kv, output_format,
        buffers.layer, error, selected_backend);
}

bool encode_prefill_qsa_layer_pipeline(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, PrefillBF16X2Mode projection_mode,
    QsaAttentionMode attention_mode, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv, PrefillNativeWeightFormat output_format,
    const PrefillQsaLayerPipelineBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend) {
    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    const bool encoded = Encoder::encode(
        context, command_buffer, "Prefill QSA complete layer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_layer_pipeline(
                context, encoder, params, projection_mode, attention_mode,
                cell_capacity, rope_zero_cos, rope_zero_sin,
                rotate_uncompressed_kv, output_format, buffers,
                encode_error, &selected);
        },
        error);
    if (encoded && selected_backend != nullptr) {
        *selected_backend = selected;
    }
    return encoded;
}

}  // namespace strata::metal
