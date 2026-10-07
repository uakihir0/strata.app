// src/metal/engine/kv_layout.cpp - checked, device-free QSA cache layout arithmetic.
#include "strata/metal/kv.hpp"

#include <limits>
#include <utility>

namespace strata::metal {
namespace {

constexpr std::size_t kHalfBytes = 2;
constexpr std::size_t kFloatBytes = 4;
constexpr std::size_t kQ4BlockBytes = 18;
constexpr std::size_t kQ4BlocksPerRow = kQsaKvHeadWidth / 32;
constexpr std::size_t kIndexerTailBytes = 3 * kQsaIndexerKeyWidth * kFloatBytes;
constexpr std::size_t kIndexerDeadBytes = kQsaIndexerKeyWidth * kFloatBytes;
constexpr std::size_t kRopeBytesPerCell = 32 * 2 * sizeof(float);

bool checked_add(std::size_t left, std::size_t right, std::size_t& result) {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool append_region(std::size_t bytes, std::size_t alignment, std::size_t& cursor,
                   KvRegion& region) {
    if (bytes == 0 || alignment == 0) return false;
    const std::size_t remainder = cursor % alignment;
    std::size_t offset = cursor;
    if (remainder != 0 &&
        !checked_add(cursor, alignment - remainder, offset)) {
        return false;
    }
    std::size_t end = 0;
    if (!checked_add(offset, bytes, end)) return false;
    region = {offset, bytes};
    cursor = end;
    return true;
}

bool row_bytes(KvFormat format, std::size_t& k_bytes, std::size_t& k_scale_bytes,
               std::size_t& v_bytes, std::size_t& v_scale_bytes) {
    switch (format) {
        case KvFormat::Fp16:
            k_bytes = kQsaKvHeadWidth * kHalfBytes;
            v_bytes = k_bytes;
            return true;
        case KvFormat::Int8:
            k_bytes = kQsaKvHeadWidth;
            k_scale_bytes = (kQsaKvHeadWidth / 64) * kHalfBytes;
            v_bytes = k_bytes;
            v_scale_bytes = k_scale_bytes;
            return true;
        case KvFormat::K8v4:
            k_bytes = kQsaKvHeadWidth;
            k_scale_bytes = (kQsaKvHeadWidth / 64) * kHalfBytes;
            v_bytes = kQ4BlocksPerRow * kQ4BlockBytes;
            return true;
        case KvFormat::Q4_0:
            k_bytes = kQ4BlocksPerRow * kQ4BlockBytes;
            v_bytes = k_bytes;
            return true;
    }
    return false;
}

}  // namespace

bool make_qsa_kv_layout(std::uint32_t requested_cells, KvFormat format,
                        QsaKvLayout& layout, std::string& error) {
    layout = {};
    error.clear();
    if (requested_cells == 0 || requested_cells > kKvMaximumCells) {
        error = "QSA KV capacity must be between 1 and 262144 cells";
        return false;
    }

    std::size_t k_row_bytes = 0;
    std::size_t k_scale_row_bytes = 0;
    std::size_t v_row_bytes = 0;
    std::size_t v_scale_row_bytes = 0;
    if (!row_bytes(format, k_row_bytes, k_scale_row_bytes,
                   v_row_bytes, v_scale_row_bytes)) {
        error = "unknown QSA KV format";
        return false;
    }

    const std::uint64_t capacity =
        (static_cast<std::uint64_t>(requested_cells) + kQsaKvCellsPerRowGroup - 1) /
        kQsaKvCellsPerRowGroup * kQsaKvCellsPerRowGroup;
    const std::uint64_t kv_rows = capacity * kQsaKvHeadCount;
    const std::uint64_t pooled_rows =
        capacity / kQsaKvCellsPerRowGroup + 2;
    if (capacity > kKvMaximumCells ||
        kv_rows > std::numeric_limits<std::uint32_t>::max() ||
        pooled_rows > std::numeric_limits<std::uint32_t>::max()) {
        error = "QSA KV capacity cannot be represented by the Metal layout";
        return false;
    }

    QsaKvLayout candidate;
    candidate.format = format;
    candidate.requested_cells = requested_cells;
    candidate.cell_capacity = static_cast<std::uint32_t>(capacity);
    candidate.kv_rows = static_cast<std::uint32_t>(kv_rows);
    candidate.pooled_rows_per_layer = static_cast<std::uint32_t>(pooled_rows);
    candidate.layers.resize(kQsaKvLayerCount);

    const std::size_t rows = static_cast<std::size_t>(kv_rows);
    std::size_t k_bytes = 0;
    std::size_t k_scale_bytes = 0;
    std::size_t v_bytes = 0;
    std::size_t v_scale_bytes = 0;
    if (!checked_multiply(rows, k_row_bytes, k_bytes) ||
        !checked_multiply(rows, k_scale_row_bytes, k_scale_bytes) ||
        !checked_multiply(rows, v_row_bytes, v_bytes) ||
        !checked_multiply(rows, v_scale_row_bytes, v_scale_bytes)) {
        error = "QSA KV region size overflows";
        return false;
    }

    std::size_t pooled_bytes = 0;
    if (!checked_multiply(static_cast<std::size_t>(pooled_rows),
                          kQsaIndexerKeyWidth * kFloatBytes, pooled_bytes)) {
        error = "QSA indexer pool size overflows";
        return false;
    }

    std::size_t cursor = 0;
    for (QsaKvLayerLayout& layer : candidate.layers) {
        if (!append_region(k_bytes, k_scale_row_bytes == 0 ? kHalfBytes : 1,
                           cursor, layer.k) ||
            (k_scale_bytes != 0 &&
             !append_region(k_scale_bytes, kHalfBytes, cursor, layer.k_scale)) ||
            !append_region(v_bytes, v_scale_row_bytes == 0 ? kHalfBytes : 1,
                           cursor, layer.v) ||
            (v_scale_bytes != 0 &&
             !append_region(v_scale_bytes, kHalfBytes, cursor, layer.v_scale)) ||
            !append_region(kIndexerTailBytes, kFloatBytes, cursor, layer.indexer_tail) ||
            !append_region(kIndexerDeadBytes, kFloatBytes, cursor, layer.indexer_dead) ||
            !append_region(pooled_bytes, kFloatBytes, cursor, layer.indexer_pooled) ||
            !append_region(sizeof(std::int32_t), alignof(std::int32_t),
                           cursor, layer.indexer_block_pos)) {
            error = "QSA KV allocation layout overflows";
            return false;
        }
    }

    std::size_t rope_bytes = 0;
    if (!checked_multiply(static_cast<std::size_t>(capacity), kRopeBytesPerCell, rope_bytes) ||
        !append_region(rope_bytes, alignof(float) * 2, cursor, candidate.rope)) {
        error = "QSA RoPE table size overflows";
        return false;
    }
    candidate.allocation_bytes = cursor;
    layout = std::move(candidate);
    error.clear();
    return true;
}

bool validate_main_qsa_kv_plan(const PlanResult& plan, std::uint64_t max_buffer_length,
                               QsaKvLayout& layout, std::string& error) {
    layout = {};
    if (!make_qsa_kv_layout(plan.context, plan.kv, layout, error)) return false;
    if (layout.allocation_bytes != plan.kv_bytes) {
        layout = {};
        error = "main QSA KV bytes do not match the memory plan";
        return false;
    }
    if (max_buffer_length == 0 || layout.allocation_bytes > max_buffer_length) {
        layout = {};
        error = "main QSA KV state exceeds the Metal device's maxBufferLength";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace strata::metal
