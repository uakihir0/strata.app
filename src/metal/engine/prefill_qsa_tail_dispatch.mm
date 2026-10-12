// Ordered Prefill QSA selection, attention and output tail.
#include "strata/metal/prefill_qsa_tail_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_native_projection.hpp"
#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kPrefillSelectionGroupRows = 256u;
constexpr std::uint32_t kPrefillAttentionGroupRows = 32u;
constexpr std::uint32_t kQsaAttentionChunkCells = 64u;
constexpr std::uint32_t kQsaAttentionMaxCells = 262144u;
constexpr std::uint32_t kModelWidth = 2560u;
constexpr std::uint32_t kAttentionProjectionWidth = 6144u;

struct ActiveRange {
    const void* cpu_address = nullptr;
    std::uint64_t gpu_address = 0u;
    void* handle = nullptr;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
    bool writable = false;
};

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes, Address right,
                    std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool overlaps(const ActiveRange& left, const ActiveRange& right) {
    if (left.handle != nullptr && left.handle == right.handle &&
        ranges_overlap(left.offset, left.bytes, right.offset, right.bytes)) {
        return true;
    }
    const bool cpu_overlap =
        left.cpu_address != nullptr && right.cpu_address != nullptr &&
        ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.cpu_address), left.bytes,
            reinterpret_cast<std::uintptr_t>(right.cpu_address),
            right.bytes);
    const bool gpu_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address, left.bytes, right.gpu_address,
                       right.bytes);
    return cpu_overlap || gpu_overlap;
}

template <class Left, class Right>
bool same_origin(Left left, Right right) {
    if (!left || !right) return false;
    return left.contents == right.contents() &&
           left.gpu_address == right.gpu_address() &&
           left.offset == right.offset();
}

bool same_origin(BufferRef left, BufferRef right) {
    return left && right && left.contents == right.contents &&
           left.gpu_address == right.gpu_address &&
           left.offset == right.offset;
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

bool same_view_for_tail(BufferRef left, BufferRef right) {
    return left && right && left.handle == right.handle &&
           left.offset == right.offset && left.length == right.length;
}

bool add_range(std::array<ActiveRange, 24u>& ranges, std::size_t& count,
               BufferRef view, std::size_t bytes, std::string_view name,
               bool writable, std::string& error) {
    if (!view || bytes == 0u || bytes > view.length ||
        count == ranges.size()) {
        error = "Prefill QSA tail " + std::string(name) +
                " view is invalid or undersized";
        return false;
    }
    ranges[count++] = ActiveRange{view.contents, view.gpu_address,
                                  view.handle, view.offset, bytes, name,
                                  writable};
    return true;
}

bool add_range(std::array<ActiveRange, 24u>& ranges, std::size_t& count,
               ReadOnlyBufferRef view, std::size_t bytes,
               std::string_view name, bool writable, std::string& error) {
    if (!view || bytes == 0u || bytes > view.length() ||
        count == ranges.size()) {
        error = "Prefill QSA tail " + std::string(name) +
                " view is invalid or undersized";
        return false;
    }
    ranges[count++] = ActiveRange{view.contents(), view.gpu_address(),
                                  nullptr, view.offset(), bytes, name,
                                  writable};
    return true;
}

bool checked_projection_sizes(
    std::uint32_t rows, PrefillNativeWeightFormat format,
    PrefillNativeProjectionBufferSizes& sizes, std::string& error) {
    const PrefillGemmMPPParams params{
        rows, kModelWidth, kAttentionProjectionWidth, kModelWidth, 0u, false};
    return prefill_native_projection_buffer_sizes(params, format, sizes,
                                                  error);
}

bool preflight_tail_ranges(
    Context& context, const PrefillParams& params,
    QsaAttentionMode mode, std::uint32_t cell_capacity,
    PrefillNativeWeightFormat output_format,
    const PrefillQsaTailBuffers& buffers,
    PrefillQsaAttentionTransform attention_transform,
    std::string& error) {
    error.clear();
    if (params.T == 0u || params.T > kPrefillMaxShaderRows) {
        error = "Prefill QSA tail row count exceeds the shader-row limit";
        return false;
    }
    if (cell_capacity == 0u ||
        cell_capacity > kQsaAttentionMaxCells ||
        cell_capacity % INDEXER_BLOCK_CELLS != 0u) {
        error = "Prefill QSA tail cell capacity is invalid";
        return false;
    }
    if (mode != QsaAttentionMode::f16 &&
        mode != QsaAttentionMode::i8 &&
        mode != QsaAttentionMode::q4 &&
        mode != QsaAttentionMode::k8v4) {
        error = "Prefill QSA tail attention mode is invalid";
        return false;
    }
    if (attention_transform != PrefillQsaAttentionTransform::none &&
        attention_transform != PrefillQsaAttentionTransform::fwht256) {
        error = "Prefill QSA tail attention transform is invalid";
        return false;
    }

    if (!same_origin(buffers.selection.steps,
                     buffers.attention.attention_chunk.steps)) {
        error = "Prefill QSA tail selection and attention must share "
                "StepRows";
        return false;
    }
    if (!same_origin(buffers.selection.ids,
                     buffers.attention.attention_chunk.ids)) {
        error = "Prefill QSA tail selection IDs must feed attention";
        return false;
    }
    if (!same_origin(buffers.attention.attention_merge.output,
                     buffers.output.gate.attn)) {
        error = "Prefill QSA tail attention output must feed the output gate";
        return false;
    }
    if (attention_transform == PrefillQsaAttentionTransform::fwht256 &&
        !same_origin(buffers.attention.attention_merge.output,
                     buffers.output.attention_transform)) {
        error = "Prefill QSA tail FWHT view must match the attention output";
        return false;
    }

    if (!same_view_for_tail(
            buffers.attention.attention_chunk.scratch.partial_acc,
            buffers.attention.attention_merge.scratch.partial_acc) ||
        !same_view_for_tail(
            buffers.attention.attention_chunk.scratch.partial_m,
            buffers.attention.attention_merge.scratch.partial_m) ||
        !same_view_for_tail(
            buffers.attention.attention_chunk.scratch.partial_l,
            buffers.attention.attention_merge.scratch.partial_l)) {
        error = "Prefill QSA tail attention chunk and merge must share the "
                "same scratch views";
        return false;
    }

    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::size_t batch_rows =
        std::min<std::size_t>(params.T, kPrefillAttentionGroupRows);
    constexpr std::size_t selection_query_row_bytes =
        static_cast<std::size_t>(INDEXER_HEADS) * INDEXER_HEAD_DIM *
        sizeof(float);
    constexpr std::size_t attention_row_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM * sizeof(float);
    constexpr std::size_t step_row_bytes = sizeof(StepRow);
    constexpr std::size_t id_row_bytes =
        static_cast<std::size_t>(INDEXER_SELECTION_WIDTH_MAX) *
        sizeof(std::int32_t);

    std::size_t selection_pooled_bytes = 0u;
    std::size_t selection_dead_bytes = 0u;
    std::size_t selection_query_bytes = 0u;
    std::size_t steps_bytes = 0u;
    std::size_t selection_scores_bytes = 0u;
    std::size_t ids_bytes = 0u;
    std::size_t attention_bytes = 0u;
    std::size_t pool_rows = 0u;
    std::size_t kv_f16_bytes = 0u;
    std::size_t kv_i8_code_bytes = 0u;
    std::size_t kv_i8_scale_bytes = 0u;
    std::size_t kv_q4_bytes = 0u;
    std::uint32_t attention_chunks = 0u;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    std::size_t gate_output_bytes = 0u;
    std::size_t q_full_bytes = 0u;
    if (!checked_product({max_blocks, INDEXER_HEAD_DIM, sizeof(float)},
                         selection_pooled_bytes) ||
        !checked_product({INDEXER_HEAD_DIM, sizeof(float)},
                         selection_dead_bytes) ||
        !checked_product({params.T, selection_query_row_bytes},
                         selection_query_bytes) ||
        !checked_product({params.T, step_row_bytes}, steps_bytes) ||
        !checked_product({kPrefillSelectionGroupRows, max_blocks,
                          sizeof(float)},
                         selection_scores_bytes) ||
        !checked_product({params.T, id_row_bytes}, ids_bytes) ||
        !checked_product({params.T, attention_row_bytes}, attention_bytes) ||
        !checked_product({cell_capacity, N_HEAD_KV}, pool_rows)) {
        error = "Prefill QSA tail input extent overflows host address space";
        return false;
    }
    attention_chunks =
        (INDEXER_SELECTION_WIDTH_MAX - 1u) / kQsaAttentionChunkCells + 1u;
    const std::size_t attention_slots =
        batch_rows * static_cast<std::size_t>(N_HEAD_KV) * attention_chunks;
    if (!checked_product(
            {attention_slots, N_HEAD_Q / N_HEAD_KV, HEAD_DIM, sizeof(float)},
            partial_acc_bytes) ||
        !checked_product(
            {attention_slots, N_HEAD_Q / N_HEAD_KV, sizeof(float)},
            partial_stats_bytes) ||
        !checked_product({pool_rows, HEAD_DIM, sizeof(std::uint16_t)},
                         kv_f16_bytes) ||
        !checked_product({pool_rows, HEAD_DIM}, kv_i8_code_bytes) ||
        !checked_product({pool_rows, 4u, sizeof(std::uint16_t)},
                         kv_i8_scale_bytes) ||
        !checked_product({pool_rows, 8u, sizeof(block_q4_0)}, kv_q4_bytes) ||
        !checked_product(
            {params.T, N_HEAD_Q, HEAD_DIM, sizeof(std::uint16_t)},
            gate_output_bytes) ||
        !checked_product(
            {params.T, N_HEAD_Q, 2u, HEAD_DIM, sizeof(float)},
            q_full_bytes)) {
        error = "Prefill QSA tail KV or scratch extent overflows host "
                "address space";
        return false;
    }
    static_assert(N_HEAD_Q % N_HEAD_KV == 0u);
    static_assert(sizeof(PrefillStepRow) == sizeof(StepRow));

    PrefillNativeProjectionBufferSizes projection_sizes;
    if (!checked_projection_sizes(params.T, output_format, projection_sizes,
                                  error)) {
        return false;
    }

    std::array<ActiveRange, 24u> ranges{};
    std::size_t count = 0u;
    const auto add = [&](BufferRef view, std::size_t bytes,
                         std::string_view name, bool writable) {
        if (!context.buffers().owns(view)) {
            error = "Prefill QSA tail " + std::string(name) +
                    " is not owned by this Context";
            return false;
        }
        return add_range(ranges, count, view, bytes, name, writable, error);
    };
    const auto add_read_only = [&](ReadOnlyBufferRef view, std::size_t bytes,
                                   std::string_view name) {
        if (!context.buffers().owns(view)) {
            error = "Prefill QSA tail " + std::string(name) +
                    " is not owned by this Context";
            return false;
        }
        return add_range(ranges, count, view, bytes, name, false, error);
    };

    if (!add(buffers.selection.pooled, selection_pooled_bytes,
             "selection pooled keys", false) ||
        !add(buffers.selection.dead, selection_dead_bytes,
             "selection dead key", false) ||
        !add(buffers.selection.indexer_queries, selection_query_bytes,
             "selection indexer queries", false) ||
        !add(buffers.selection.steps, steps_bytes, "selection StepRows",
             false) ||
        !add(buffers.selection.scores, selection_scores_bytes,
             "selection scores", true) ||
        !add(buffers.selection.ids, ids_bytes, "selection IDs", true) ||
        !add(buffers.attention.attention_chunk.query, attention_bytes,
             "attention queries", false) ||
        !add(buffers.attention.attention_chunk.steps, steps_bytes,
             "attention StepRows", false) ||
        !add(buffers.attention.attention_chunk.ids, ids_bytes,
             "attention IDs", false)) {
        return false;
    }

    std::size_t key_data_bytes = 0u;
    std::size_t key_scale_bytes = 0u;
    std::size_t value_data_bytes = 0u;
    std::size_t value_scale_bytes = 0u;
    switch (mode) {
        case QsaAttentionMode::f16:
            key_data_bytes = kv_f16_bytes;
            value_data_bytes = kv_f16_bytes;
            break;
        case QsaAttentionMode::i8:
            key_data_bytes = kv_i8_code_bytes;
            key_scale_bytes = kv_i8_scale_bytes;
            value_data_bytes = kv_i8_code_bytes;
            value_scale_bytes = kv_i8_scale_bytes;
            break;
        case QsaAttentionMode::q4:
            key_data_bytes = kv_q4_bytes;
            value_data_bytes = kv_q4_bytes;
            break;
        case QsaAttentionMode::k8v4:
            key_data_bytes = kv_i8_code_bytes;
            key_scale_bytes = kv_i8_scale_bytes;
            value_data_bytes = kv_q4_bytes;
            break;
    }
    if (!add(buffers.attention.attention_chunk.key_data, key_data_bytes,
             "attention key KV", false) ||
        (key_scale_bytes != 0u &&
         !add(buffers.attention.attention_chunk.key_scales, key_scale_bytes,
              "attention key scales", false)) ||
        !add(buffers.attention.attention_chunk.value_data, value_data_bytes,
             "attention value KV", false) ||
        (value_scale_bytes != 0u &&
         !add(buffers.attention.attention_chunk.value_scales,
              value_scale_bytes, "attention value scales", false)) ||
        !add(buffers.attention.attention_chunk.scratch.partial_acc,
             partial_acc_bytes, "attention partial accumulator", true) ||
        !add(buffers.attention.attention_chunk.scratch.partial_m,
             partial_stats_bytes, "attention partial maximum", true) ||
        !add(buffers.attention.attention_chunk.scratch.partial_l,
             partial_stats_bytes, "attention partial denominator", true) ||
        !add(buffers.attention.attention_merge.output, attention_bytes,
             "attention merged output", true) ||
        !add_read_only(buffers.output.gate.attn, attention_bytes,
                       "output gate attention") ||
        !add_read_only(buffers.output.gate.q_full, q_full_bytes,
                       "output gate Qf") ||
        !add(buffers.output.gate.attn_h, gate_output_bytes,
             "output gate F16", true) ||
        !add_read_only(buffers.output.native_weights,
                       projection_sizes.weights_bytes,
                       "native output weights") ||
        !add(buffers.output.dequantized_f16,
             projection_sizes.dequantized_f16_bytes,
             "dequantized output weights", true) ||
        !add(buffers.output.y_f32, projection_sizes.y_f32_bytes,
             "output projection Y", true)) {
        return false;
    }
    const bool has_fwht =
        attention_transform == PrefillQsaAttentionTransform::fwht256;
    if (has_fwht &&
        !add(buffers.output.attention_transform, attention_bytes,
             "attention FWHT view", true)) {
        return false;
    }

    const auto is_allowed_handoff = [&](const ActiveRange& left,
                                        const ActiveRange& right) {
        const auto same = [&] {
            return left.cpu_address == right.cpu_address &&
                   left.gpu_address == right.gpu_address &&
                   left.offset == right.offset &&
                   left.bytes == right.bytes;
        };
        if (same() &&
            ((left.name == "selection IDs" &&
              right.name == "attention IDs") ||
             (left.name == "attention IDs" &&
              right.name == "selection IDs") ||
             (left.name == "attention merged output" &&
              right.name == "output gate attention") ||
             (left.name == "output gate attention" &&
              right.name == "attention merged output") ||
             (has_fwht &&
              ((left.name == "attention merged output" &&
                right.name == "attention FWHT view") ||
               (left.name == "attention FWHT view" &&
                right.name == "attention merged output"))))) {
            return true;
        }
        // The gate reads the same address only after the in-place FWHT write.
        return has_fwht && same() &&
               ((left.name == "attention FWHT view" &&
                 right.name == "output gate attention") ||
                (left.name == "output gate attention" &&
                 right.name == "attention FWHT view"));
    };

    for (std::size_t i = 0u; i < count; ++i) {
        for (std::size_t j = i + 1u; j < count; ++j) {
            if (!ranges[i].writable && !ranges[j].writable) continue;
            if (!overlaps(ranges[i], ranges[j])) continue;
            if (is_allowed_handoff(ranges[i], ranges[j])) continue;
            error = "Prefill QSA tail " + std::string(ranges[i].name) +
                    " overlaps active " + std::string(ranges[j].name);
            return false;
        }
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_prefill_qsa_tail(
    Context& context, Encoder& encoder, const PrefillParams& params,
    QsaAttentionMode attention_mode, std::uint32_t cell_capacity,
    PrefillNativeWeightFormat output_format,
    const PrefillQsaTailBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend,
    PrefillQsaAttentionTransform attention_transform) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA tail requires an ordered concurrent-group "
                "boundary";
        return false;
    }
    if (!preflight_tail_ranges(context, params, attention_mode, cell_capacity,
                               output_format, buffers, attention_transform,
                               error)) {
        return false;
    }

    if (!encode_prefill_qsa_selection(context, encoder, params, cell_capacity,
                                      buffers.selection, error) ||
        !encode_prefill_qsa_attention(context, encoder, params, attention_mode,
                                      cell_capacity, buffers.attention,
                                      error)) {
        return false;
    }
    return encode_prefill_qsa_output(
        context, encoder, params.T, output_format, buffers.output, error,
        selected_backend, attention_transform);
}

bool encode_prefill_qsa_tail(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, QsaAttentionMode attention_mode,
    std::uint32_t cell_capacity,
    PrefillNativeWeightFormat output_format,
    const PrefillQsaTailBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend,
    PrefillQsaAttentionTransform attention_transform) {
    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    const bool encoded = Encoder::encode(
        context, command_buffer, "Prefill QSA tail",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_tail(
                context, encoder, params, attention_mode, cell_capacity,
                output_format, buffers, encode_error, &selected,
                attention_transform);
        },
        error);
    if (encoded && selected_backend != nullptr) {
        *selected_backend = selected;
    }
    return encoded;
}

}  // namespace strata::metal
