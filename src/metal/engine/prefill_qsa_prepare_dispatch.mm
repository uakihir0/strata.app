// Ordered Prefill QSA preparation from projected rows through cache append.
#include "strata/metal/prefill_qsa_prepare_dispatch.hpp"

#include "strata/metal/encoder.hpp"
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
#include <type_traits>
#include <variant>

namespace strata::metal {
namespace {

constexpr std::size_t kQsaHeadDim = 256u;
constexpr std::size_t kQsaKvHeads = 2u;
constexpr std::size_t kQsaQueryHeads = 24u;
constexpr std::size_t kQsaIndexerHeads = 4u;
constexpr std::size_t kQsaIndexerHeadDim = 128u;
constexpr std::size_t kIndexerBlockCells = 4u;
constexpr std::size_t kQ4BlockBytes = 18u;
constexpr std::uint32_t kQsaAttentionMaxCells = 262144u;

struct ActiveRange {
    const void* cpu_address = nullptr;
    std::uint64_t gpu_address = 0u;
    void* handle = nullptr;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
    bool writable = false;
};

struct QsaRotationPlan {
    bool keys = false;
    bool values = false;
    bool query = false;
    PrefillQsaAttentionTransform attention =
        PrefillQsaAttentionTransform::none;
};

bool make_rotation_plan(QsaAttentionMode mode, bool rotate_uncompressed_kv,
                        QsaRotationPlan& plan) {
    switch (mode) {
        case QsaAttentionMode::f16:
        case QsaAttentionMode::i8:
            plan = QsaRotationPlan{
                rotate_uncompressed_kv, rotate_uncompressed_kv,
                rotate_uncompressed_kv,
                rotate_uncompressed_kv
                    ? PrefillQsaAttentionTransform::fwht256
                    : PrefillQsaAttentionTransform::none};
            return true;
        case QsaAttentionMode::q4:
            plan = QsaRotationPlan{
                true, true, true, PrefillQsaAttentionTransform::fwht256};
            return true;
        case QsaAttentionMode::k8v4:
            plan = QsaRotationPlan{
                false, true, false, PrefillQsaAttentionTransform::fwht256};
            return true;
    }
    return false;
}

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

bool add_range(std::array<ActiveRange, 32u>& ranges, std::size_t& count,
               BufferRef view, std::size_t bytes, std::string_view name,
               bool writable, std::string& error) {
    if (!view || bytes == 0u || bytes > view.length ||
        count == ranges.size()) {
        error = "Prefill QSA preparation " + std::string(name) +
                " view is invalid or undersized";
        return false;
    }
    ranges[count++] = ActiveRange{view.contents, view.gpu_address,
                                  view.handle, view.offset, bytes, name,
                                  writable};
    return true;
}

bool validate_cross_stage_ranges(const std::array<ActiveRange, 32u>& ranges,
                                 std::size_t count, std::string& error) {
    for (std::size_t i = 0u; i < count; ++i) {
        for (std::size_t j = i + 1u; j < count; ++j) {
            if (!ranges[i].writable && !ranges[j].writable) continue;
            if (overlaps(ranges[i], ranges[j])) {
                error = "Prefill QSA preparation " +
                        std::string(ranges[i].name) +
                        " overlaps active " + std::string(ranges[j].name);
                return false;
            }
        }
    }
    return true;
}

bool mode_matches_append(QsaAttentionMode mode,
                         const PrefillQsaKvAppendBuffers& append) {
    switch (mode) {
        case QsaAttentionMode::f16:
            return std::holds_alternative<QsaKvAppendF16Buffers>(append);
        case QsaAttentionMode::i8:
            return std::holds_alternative<QsaKvAppendI8Buffers>(append);
        case QsaAttentionMode::q4:
            return std::holds_alternative<QsaKvAppendQ4Buffers>(append);
        case QsaAttentionMode::k8v4:
            return std::holds_alternative<QsaKvAppendK8V4Buffers>(append);
    }
    return false;
}

template <class Append>
BufferRef append_keys(const Append& append) {
    return append.keys;
}

template <class Append>
BufferRef append_values(const Append& append) {
    return append.values;
}

template <class Append>
BufferRef append_steps(const Append& append) {
    return append.steps;
}

bool append_bytes(QsaAttentionMode mode, std::uint32_t rows,
                  std::uint32_t cell_capacity,
                  std::size_t& current_bytes,
                  std::array<std::size_t, 4u>& output_bytes,
                  std::size_t& output_count) {
    output_count = 0u;
    if (!checked_product({static_cast<std::size_t>(rows), kQsaKvHeads,
                          kQsaHeadDim, sizeof(float)},
                         current_bytes)) {
        return false;
    }

    const std::size_t cells = static_cast<std::size_t>(cell_capacity);
    switch (mode) {
        case QsaAttentionMode::f16:
            output_count = 2u;
            return checked_product({cells, kQsaKvHeads, kQsaHeadDim,
                                    sizeof(std::uint16_t)},
                                   output_bytes[0]);
        case QsaAttentionMode::i8:
            output_count = 4u;
            return checked_product({cells, kQsaKvHeads, kQsaHeadDim},
                                   output_bytes[0]) &&
                   checked_product({cells, kQsaKvHeads, 4u,
                                    sizeof(std::uint16_t)},
                                   output_bytes[1]) &&
                   (output_bytes[2] = output_bytes[0], true) &&
                   (output_bytes[3] = output_bytes[1], true);
        case QsaAttentionMode::q4:
            output_count = 2u;
            if (!checked_product({cells, kQsaKvHeads,
                                  kQsaHeadDim / 32u, kQ4BlockBytes},
                                 output_bytes[0])) {
                return false;
            }
            output_bytes[1] = output_bytes[0];
            return true;
        case QsaAttentionMode::k8v4:
            output_count = 3u;
            return checked_product({cells, kQsaKvHeads, kQsaHeadDim},
                                   output_bytes[0]) &&
                   checked_product({cells, kQsaKvHeads, 4u,
                                    sizeof(std::uint16_t)},
                                   output_bytes[1]) &&
                   checked_product({cells, kQsaKvHeads,
                                    kQsaHeadDim / 32u, kQ4BlockBytes},
                                   output_bytes[2]);
    }
    return false;
}

bool preflight_prepare(
    const PrefillParams& params,
    QsaAttentionMode mode, std::uint32_t cell_capacity,
    bool rotate_uncompressed_kv,
    const PrefillQsaPrepareBuffers& buffers,
    QsaRotationPlan& rotation, std::string& error) {
    error.clear();
    if (params.T == 0u || params.T > kPrefillMaxShaderRows) {
        error = "Prefill QSA preparation row count exceeds the shader-row "
                "limit";
        return false;
    }
    if (cell_capacity == 0u ||
        cell_capacity > kQsaAttentionMaxCells ||
        cell_capacity % kIndexerBlockCells != 0u ||
        params.p0 > cell_capacity || params.T > cell_capacity - params.p0) {
        error = "Prefill QSA preparation cell capacity does not contain the "
                "chunk";
        return false;
    }
    const std::uint64_t end =
        static_cast<std::uint64_t>(params.p0) + params.T;
    if (end > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int32_t>::max())) {
        error = "Prefill QSA preparation positions exceed the StepRow range";
        return false;
    }
    const std::uint64_t expected_idx_f =
        params.p0 == 0u
            ? kIndexerBlockCells
            : (static_cast<std::uint64_t>(params.p0) +
               kIndexerBlockCells - 1u) &
                  ~static_cast<std::uint64_t>(kIndexerBlockCells - 1u);
    const std::uint32_t expected_idx_e = static_cast<std::uint32_t>(
        end & ~static_cast<std::uint64_t>(kIndexerBlockCells - 1u));
    if (expected_idx_f > std::numeric_limits<std::uint32_t>::max() ||
        params.idx_f != expected_idx_f || params.idx_e != expected_idx_e) {
        error = "Prefill QSA preparation indexer parameters do not match the "
                "chunk";
        return false;
    }
    if (!make_rotation_plan(mode, rotate_uncompressed_kv, rotation)) {
        error = "Prefill QSA preparation attention mode is invalid";
        return false;
    }
    if (!mode_matches_append(mode, buffers.kv_append)) {
        error = "Prefill QSA preparation KV append buffers do not match the "
                "attention mode";
        return false;
    }

    const auto& key = buffers.key_norm_rope;
    const auto& query = buffers.query_norm_rope;
    const auto& indexer_query = buffers.indexer_query_norm_rope;
    BufferRef append_key;
    BufferRef append_value;
    BufferRef append_step;
    std::visit(
        [&](const auto& append) {
            append_key = append_keys(append);
            append_value = append_values(append);
            append_step = append_steps(append);
        },
        buffers.kv_append);

    if (!same_origin(key.input, key.output) ||
        !same_origin(key.output, append_key)) {
        error = "Prefill QSA preparation key norm/RoPE output must feed the "
                "KV append in place";
        return false;
    }
    if (!same_origin(key.steps, append_step) ||
        !same_origin(key.steps, query.steps) ||
        !same_origin(key.steps, indexer_query.steps)) {
        error = "Prefill QSA preparation stages must share the same StepRows";
        return false;
    }
    if (!same_origin(indexer_query.input, indexer_query.output)) {
        error = "Prefill QSA preparation indexer-query norm/RoPE must be in "
                "place";
        return false;
    }
    if (params.T > std::numeric_limits<std::uint32_t>::max() /
                       static_cast<std::uint32_t>(kQsaQueryHeads) ||
        params.T > std::numeric_limits<std::uint32_t>::max() /
                       static_cast<std::uint32_t>(kQsaKvHeads) ||
        params.T > std::numeric_limits<std::uint32_t>::max() /
                       static_cast<std::uint32_t>(kQsaIndexerHeads)) {
        error = "Prefill QSA preparation row geometry overflows shader "
                "indices";
        return false;
    }

    std::size_t key_bytes = 0u;
    std::size_t query_input_bytes = 0u;
    std::size_t query_output_bytes = 0u;
    std::size_t indexer_query_bytes = 0u;
    std::size_t raw_key_bytes = 0u;
    std::size_t step_bytes = 0u;
    std::size_t key_gamma_bytes = 0u;
    std::size_t query_gamma_bytes = 0u;
    std::size_t indexer_gamma_bytes = 0u;
    std::size_t indexer_tail_bytes = 0u;
    std::size_t indexer_dead_bytes = 0u;
    std::size_t indexer_pool_bytes = 0u;
    std::size_t kv_input_bytes = 0u;
    std::array<std::size_t, 4u> kv_output_bytes{};
    std::size_t kv_output_count = 0u;
    if (!checked_product({static_cast<std::size_t>(params.T), kQsaKvHeads,
                          kQsaHeadDim, sizeof(float)},
                         key_bytes) ||
        !checked_product({static_cast<std::size_t>(params.T),
                          kQsaQueryHeads, 2u * kQsaHeadDim, sizeof(float)},
                         query_input_bytes) ||
        !checked_product({static_cast<std::size_t>(params.T),
                          kQsaQueryHeads, kQsaHeadDim, sizeof(float)},
                         query_output_bytes) ||
        !checked_product({static_cast<std::size_t>(params.T),
                          kQsaIndexerHeads, kQsaIndexerHeadDim,
                          sizeof(float)},
                         indexer_query_bytes) ||
        !checked_product({static_cast<std::size_t>(params.T),
                          kQsaIndexerHeadDim, sizeof(float)},
                         raw_key_bytes) ||
        !checked_product({static_cast<std::size_t>(params.T), sizeof(StepRow)},
                         step_bytes) ||
        !checked_product({kQsaHeadDim, sizeof(float)}, key_gamma_bytes) ||
        !checked_product({kQsaHeadDim, sizeof(float)}, query_gamma_bytes) ||
        !checked_product({kQsaIndexerHeadDim, sizeof(float)},
                         indexer_gamma_bytes) ||
        !checked_product({3u, kQsaIndexerHeadDim, sizeof(float)},
                         indexer_tail_bytes) ||
        !checked_product({kQsaIndexerHeadDim, sizeof(float)},
                         indexer_dead_bytes) ||
        !checked_product({static_cast<std::size_t>(cell_capacity) /
                              kIndexerBlockCells +
                              2u,
                          kQsaIndexerHeadDim, sizeof(float)},
                         indexer_pool_bytes) ||
        !append_bytes(mode, params.T, cell_capacity, kv_input_bytes,
                      kv_output_bytes, kv_output_count) ||
        kv_input_bytes != key_bytes) {
        error = "Prefill QSA preparation active extent overflows host address "
                "space";
        return false;
    }

    if (key.input.length < key_bytes ||
        append_key.length < key_bytes ||
        append_step.length < step_bytes ||
        query.steps.length < step_bytes ||
        indexer_query.steps.length < step_bytes ||
        indexer_query.input.length < indexer_query_bytes) {
        error = "Prefill QSA preparation handoff view is undersized";
        return false;
    }
    if (!key.steps || key.steps.contents == nullptr ||
        key.steps.length < step_bytes ||
        reinterpret_cast<std::uintptr_t>(key.steps.contents) %
                alignof(StepRow) !=
            0u) {
        error = "Prefill QSA preparation StepRows are invalid or misaligned";
        return false;
    }
    const auto* const step_rows =
        static_cast<const StepRow*>(key.steps.contents);
    for (std::uint32_t row = 0u; row < params.T; ++row) {
        const std::uint64_t position =
            static_cast<std::uint64_t>(params.p0) + row;
        const std::int32_t n_kv = static_cast<std::int32_t>(position + 1u);
        const StepRow expected{
            static_cast<std::int32_t>(position),
            n_kv,
            n_kv / static_cast<std::int32_t>(kIndexerBlockCells),
            std::min<std::int32_t>(n_kv, 2051)};
        if (step_rows[row].pos != expected.pos ||
            step_rows[row].n_kv != expected.n_kv ||
            step_rows[row].n_bid != expected.n_bid ||
            step_rows[row].width != expected.width) {
            error = "Prefill QSA preparation StepRows do not match the chunk";
            return false;
        }
    }

    std::array<ActiveRange, 32u> ranges{};
    std::size_t range_count = 0u;
    const auto add = [&](BufferRef view, std::size_t bytes,
                         std::string_view name, bool writable) {
        return add_range(ranges, range_count, view, bytes, name, writable,
                         error);
    };
    if (!add(key.output, key_bytes, "normalized Kc", true) ||
        !add(key.gamma, key_gamma_bytes, "attn_k_norm gamma", false) ||
        !add(key.steps, step_bytes, "shared StepRows", false) ||
        !add(key.rope, key.rope.length, "attn_k RoPE table", false) ||
        !add(append_value, kv_input_bytes, "projected Vc",
             rotation.values) ||
        !add(query.input, query_input_bytes, "projected Qf", false) ||
        !add(query.output, query_output_bytes, "normalized qcur", true) ||
        !add(query.gamma, query_gamma_bytes, "attn_q_norm gamma", false) ||
        !add(query.rope, query.rope.length, "attn_q RoPE table", false) ||
        !add(indexer_query.output, indexer_query_bytes,
             "normalized indexer q", true) ||
        !add(indexer_query.gamma, indexer_gamma_bytes,
             "indexer.q_norm gamma", false) ||
        !add(indexer_query.rope, indexer_query.rope.length,
             "indexer.q RoPE table", false) ||
        !add(buffers.indexer_append.raw_keys, raw_key_bytes,
             "projected indexer keys", false) ||
        !add(buffers.indexer_append.gamma, indexer_gamma_bytes,
             "indexer.k_norm gamma", false) ||
        !add(buffers.indexer_append.rope,
             buffers.indexer_append.rope.length,
             "indexer.k RoPE table", false)) {
        return false;
    }

    std::visit(
        [&](const auto& append) {
            using Append = std::decay_t<decltype(append)>;
            if constexpr (std::is_same_v<Append, QsaKvAppendF16Buffers>) {
                add(append.key_pool, kv_output_bytes[0], "f16 K cache", true);
                add(append.value_pool, kv_output_bytes[0], "f16 V cache",
                    true);
            } else if constexpr (std::is_same_v<Append,
                                                QsaKvAppendI8Buffers>) {
                add(append.key_codes, kv_output_bytes[0], "i8 K codes", true);
                add(append.key_scales, kv_output_bytes[1], "i8 K scales",
                    true);
                add(append.value_codes, kv_output_bytes[2], "i8 V codes",
                    true);
                add(append.value_scales, kv_output_bytes[3], "i8 V scales",
                    true);
            } else if constexpr (std::is_same_v<Append,
                                                QsaKvAppendQ4Buffers>) {
                add(append.key_pool, kv_output_bytes[0], "q4 K cache", true);
                add(append.value_pool, kv_output_bytes[1], "q4 V cache",
                    true);
            } else if constexpr (std::is_same_v<Append,
                                                QsaKvAppendK8V4Buffers>) {
                add(append.key_codes, kv_output_bytes[0], "k8v4 K codes",
                    true);
                add(append.key_scales, kv_output_bytes[1], "k8v4 K scales",
                    true);
                add(append.value_pool, kv_output_bytes[2], "k8v4 V cache",
                    true);
            }
        },
        buffers.kv_append);
    if (!error.empty()) return false;

    if (!add(buffers.indexer_append.tail, indexer_tail_bytes,
             "indexer tail", true) ||
        !add(buffers.indexer_append.dead, indexer_dead_bytes,
             "indexer dead key", true) ||
        !add(buffers.indexer_append.pooled, indexer_pool_bytes,
             "indexer pooled keys", true) ||
        !add(buffers.indexer_append.block_position, sizeof(std::int32_t),
             "indexer block position", true)) {
        return false;
    }

    if (!validate_cross_stage_ranges(ranges, range_count, error)) return false;
    return true;
}

bool encode_kv_append(Context& context, Encoder& encoder,
                      std::uint32_t token_rows,
                      std::uint32_t cell_capacity,
                      const PrefillQsaKvAppendBuffers& buffers,
                      std::string& error) {
    return std::visit(
        [&](const auto& append) {
            using Append = std::decay_t<decltype(append)>;
            if constexpr (std::is_same_v<Append, QsaKvAppendF16Buffers>) {
                return encode_prefill_qsa_kv_append_f16(
                    context, encoder, token_rows, cell_capacity, append,
                    error);
            } else if constexpr (std::is_same_v<Append,
                                                QsaKvAppendI8Buffers>) {
                return encode_prefill_qsa_kv_append_i8(
                    context, encoder, token_rows, cell_capacity, append,
                    error);
            } else if constexpr (std::is_same_v<Append,
                                                QsaKvAppendQ4Buffers>) {
                return encode_prefill_qsa_kv_append_q4(
                    context, encoder, token_rows, cell_capacity, append,
                    error);
            } else {
                return encode_prefill_qsa_kv_append_k8v4(
                    context, encoder, token_rows, cell_capacity, append,
                    error);
            }
        },
        buffers);
}

}  // namespace

bool encode_prefill_qsa_prepare(
    Context& context, Encoder& encoder, const PrefillParams& params,
    QsaAttentionMode attention_mode, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv,
    const PrefillQsaPrepareBuffers& buffers, std::string& error,
    PrefillQsaAttentionTransform* attention_transform) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA preparation requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    QsaRotationPlan rotation;
    if (!preflight_prepare(params, attention_mode, cell_capacity,
                           rotate_uncompressed_kv, buffers,
                           rotation, error)) {
        return false;
    }

    const auto& key = buffers.key_norm_rope;
    const auto& query = buffers.query_norm_rope;
    const auto& indexer_query = buffers.indexer_query_norm_rope;
    const auto encode_fwht = [&](BufferRef values, std::uint32_t rows) {
        return encode_qsa_fwht256(context, encoder, rows, values, error);
    };

    if (!encode_prefill_qsa_norm_rope(
            context, encoder, params.T, QsaNormRopeMode::key, key, error)) {
        return false;
    }
    const std::uint32_t kv_rows =
        params.T * static_cast<std::uint32_t>(kQsaKvHeads);
    if (rotation.keys && !encode_fwht(key.output, kv_rows)) return false;

    BufferRef values;
    std::visit([&](const auto& append) { values = append.values; },
               buffers.kv_append);
    if (rotation.values && !encode_fwht(values, kv_rows)) return false;
    if (!encode_kv_append(context, encoder, params.T, cell_capacity,
                          buffers.kv_append, error)) {
        return false;
    }

    if (!encode_prefill_qsa_norm_rope(
            context, encoder, params.T, QsaNormRopeMode::query, query,
            error)) {
        return false;
    }
    if (rotation.query &&
        !encode_fwht(query.output,
                     params.T *
                         static_cast<std::uint32_t>(kQsaQueryHeads))) {
        return false;
    }
    if (!encode_prefill_qsa_norm_rope(
            context, encoder, params.T, QsaNormRopeMode::indexer_query,
            indexer_query, error) ||
        !encode_prefill_qsa_indexer_append(
            context, encoder, params, cell_capacity, rope_zero_cos,
            rope_zero_sin, buffers.indexer_append, error)) {
        return false;
    }

    if (attention_transform != nullptr) {
        *attention_transform = rotation.attention;
    }
    return true;
}

bool encode_prefill_qsa_prepare(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, QsaAttentionMode attention_mode,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    bool rotate_uncompressed_kv,
    const PrefillQsaPrepareBuffers& buffers, std::string& error,
    PrefillQsaAttentionTransform* attention_transform) {
    PrefillQsaAttentionTransform selected_transform =
        PrefillQsaAttentionTransform::none;
    const bool encoded = Encoder::encode(
        context, command_buffer, "Prefill QSA preparation",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_prepare(
                context, encoder, params, attention_mode, cell_capacity,
                rope_zero_cos, rope_zero_sin, rotate_uncompressed_kv,
                buffers, encode_error, &selected_transform);
        },
        error);
    if (encoded && attention_transform != nullptr) {
        *attention_transform = selected_transform;
    }
    return encoded;
}

}  // namespace strata::metal
