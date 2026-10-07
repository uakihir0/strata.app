// src/metal/engine/mtp_kv_layout.cpp - checked, device-free MTP drafter KV layout arithmetic.
#include "strata/metal/mtp_kv.hpp"

#include <limits>
#include <utility>

namespace strata::metal {
namespace {

constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::size_t kQ4BlockBytes = 18;
constexpr std::size_t kQ4BlocksPerRow = kQsaKvHeadWidth / 32;

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
    if (remainder != 0 && !checked_add(cursor, alignment - remainder, offset)) return false;
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
            return false;
        case KvFormat::Q4_0:
            k_bytes = kQ4BlocksPerRow * kQ4BlockBytes;
            v_bytes = k_bytes;
            return true;
    }
    return false;
}

}  // namespace

bool make_mtp_kv_layout(std::uint32_t requested_cells, KvFormat format,
                        MtpKvLayout& layout, std::string& error) {
    layout = {};
    error.clear();
    if (requested_cells == 0 || requested_cells > kKvMaximumCells) {
        error = "MTP KV capacity must be between 1 and 262144 cells";
        return false;
    }

    const KvFormat storage_format = format == KvFormat::K8v4 ? KvFormat::Int8 : format;
    std::size_t k_row_bytes = 0;
    std::size_t k_scale_row_bytes = 0;
    std::size_t v_row_bytes = 0;
    std::size_t v_scale_row_bytes = 0;
    if (!row_bytes(storage_format, k_row_bytes, k_scale_row_bytes,
                   v_row_bytes, v_scale_row_bytes)) {
        error = "unknown MTP KV format";
        return false;
    }

    const std::uint64_t capacity =
        (static_cast<std::uint64_t>(requested_cells) + kQsaKvCellsPerRowGroup - 1) /
        kQsaKvCellsPerRowGroup * kQsaKvCellsPerRowGroup;
    const std::uint64_t rows = capacity * kQsaKvHeadCount;
    if (capacity > kKvMaximumCells ||
        rows > std::numeric_limits<std::uint32_t>::max()) {
        error = "MTP KV capacity cannot be represented by the Metal layout";
        return false;
    }

    std::size_t k_bytes = 0;
    std::size_t k_scale_bytes = 0;
    std::size_t v_bytes = 0;
    std::size_t v_scale_bytes = 0;
    if (!checked_multiply(static_cast<std::size_t>(rows), k_row_bytes, k_bytes) ||
        !checked_multiply(static_cast<std::size_t>(rows), k_scale_row_bytes, k_scale_bytes) ||
        !checked_multiply(static_cast<std::size_t>(rows), v_row_bytes, v_bytes) ||
        !checked_multiply(static_cast<std::size_t>(rows), v_scale_row_bytes, v_scale_bytes)) {
        error = "MTP KV region size overflows";
        return false;
    }

    MtpKvLayout candidate;
    candidate.requested_format = format;
    candidate.storage_format = storage_format;
    candidate.requested_cells = requested_cells;
    candidate.cell_capacity = static_cast<std::uint32_t>(capacity);
    candidate.kv_rows = static_cast<std::uint32_t>(rows);

    std::size_t cursor = 0;
    const std::size_t kv_alignment =
        storage_format == KvFormat::Int8 ? 1 : kHalfBytes;
    if (!append_region(k_bytes, kv_alignment, cursor, candidate.k) ||
        (k_scale_bytes != 0 &&
         !append_region(k_scale_bytes, kHalfBytes, cursor, candidate.k_scale)) ||
        !append_region(v_bytes, kv_alignment, cursor, candidate.v) ||
        (v_scale_bytes != 0 &&
         !append_region(v_scale_bytes, kHalfBytes, cursor, candidate.v_scale))) {
        error = "MTP KV allocation layout overflows";
        return false;
    }
    candidate.allocation_bytes = cursor;
    layout = std::move(candidate);
    error.clear();
    return true;
}

}  // namespace strata::metal
