// src/metal/engine/mtp_bind_layout.cpp - checked, device-free MTP arena carving.
#include "strata/metal/mtp_bind_layout.hpp"

#include <limits>
#include <utility>

namespace strata::metal {
namespace {

constexpr std::uint64_t kF32Bytes = sizeof(float);
constexpr std::uint64_t kI32Bytes = sizeof(std::int32_t);
constexpr std::uint64_t kQ8_1BlockBytes = 36;
// At the documented 32,768-cell / 106,299-logit geometry, including the
// 106,299-entry device ID map, the normal-mode carve leaves this exact tail.
constexpr std::uint64_t kMtpBindCjkReservedBytes = 128'192;
constexpr std::uint64_t kMtpBindCjkMapPadding = 20;
constexpr std::uint64_t kMtpBindCjkLogitsPadding = 20;

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left, std::uint64_t right,
                      std::uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool align_up(std::uint64_t value, std::uint64_t alignment,
              std::uint64_t& result) {
    if (alignment == 0) return false;
    const std::uint64_t remainder = value % alignment;
    return remainder == 0
        ? (result = value, true)
        : checked_add(value, alignment - remainder, result);
}

void alignment_padding(std::uint64_t bytes, std::uint64_t& padding) {
    const std::uint64_t remainder = bytes % kMtpBindAlignment;
    padding = remainder == 0 ? 0 : kMtpBindAlignment - remainder;
}

bool append_region(MtpBindLayout& layout, MtpBindRegionId id,
                   std::uint64_t bytes, std::uint64_t& cursor) {
    if (bytes == 0) {
        layout.regions[static_cast<std::size_t>(id)] = {};
        return true;
    }
    std::uint64_t offset = 0;
    std::uint64_t end = 0;
    if (!align_up(cursor, kMtpBindAlignment, offset) ||
        !checked_add(offset, bytes, end)) {
        return false;
    }
    layout.regions[static_cast<std::size_t>(id)] = {offset, bytes};
    cursor = end;
    return true;
}

bool tensor_bytes(std::uint64_t rows, std::uint64_t columns,
                  std::uint64_t element_bytes, std::uint64_t& bytes) {
    std::uint64_t elements = 0;
    return checked_multiply(rows, columns, elements) &&
           checked_multiply(elements, element_bytes, bytes);
}

bool variable_bytes(std::uint32_t capacity, std::uint64_t& bytes) {
    if (capacity == 0 || capacity % 64 != 0) return false;
    const std::uint64_t chunks = capacity / 64;
    std::uint64_t scratch_floats = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t ids_bytes = 0;
    return checked_multiply(chunks, 24 * 258, scratch_floats) &&
           checked_add(scratch_floats, 64, scratch_floats) &&
           checked_multiply(scratch_floats, kF32Bytes, scratch_bytes) &&
           checked_multiply(capacity, kI32Bytes, ids_bytes) &&
           checked_add(scratch_bytes, ids_bytes, bytes);
}

bool expected_allocation_bytes(std::uint32_t capacity,
                               std::uint32_t draft_vocab_count,
                               std::uint64_t& bytes) {
    std::uint64_t default_variable = 0;
    std::uint64_t requested_variable = 0;
    std::uint64_t draft_vocab_bytes = 0;
    std::uint64_t output_delta = 0;
    if (!variable_bytes(kMtpDefaultBindCapacity, default_variable) ||
        !variable_bytes(capacity, requested_variable) ||
        !checked_multiply(draft_vocab_count, kI32Bytes, draft_vocab_bytes)) {
        return false;
    }

    const std::uint32_t output_count = draft_vocab_count == 0
        ? kMtpFullVocabularySize : draft_vocab_count;
    const std::uint32_t baseline_count = kMtpDraftVocabularyBaseline;
    const std::uint64_t count_delta =
        output_count > baseline_count
            ? static_cast<std::uint64_t>(output_count - baseline_count)
            : static_cast<std::uint64_t>(baseline_count - output_count);
    if (!checked_multiply(count_delta, kF32Bytes, output_delta)) return false;

    std::uint64_t total = kMtpBindBaselineBytes;
    if (output_count < baseline_count) {
        if (output_delta > total) return false;
        total -= output_delta;
    } else if (!checked_add(total, output_delta, total)) {
        return false;
    }
    if (!checked_add(total, draft_vocab_bytes, total)) return false;

    if (requested_variable < default_variable) {
        const std::uint64_t reduction = default_variable - requested_variable;
        if (reduction > total) return false;
        total -= reduction;
    } else if (!checked_add(total, requested_variable - default_variable, total)) {
        return false;
    }
    bytes = total;
    return true;
}

bool expected_reserved_bytes(std::uint32_t capacity,
                             std::uint32_t draft_vocab_count,
                             std::uint32_t output_vocab_count,
                             std::uint64_t& bytes) {
    std::uint64_t map_bytes = 0;
    std::uint64_t logits_bytes = 0;
    std::uint64_t scratch_floats = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t map_padding = 0;
    std::uint64_t logits_padding = 0;
    std::uint64_t scratch_padding = 0;
    if (!checked_multiply(draft_vocab_count, kI32Bytes, map_bytes) ||
        !checked_multiply(output_vocab_count, kF32Bytes, logits_bytes) ||
        !checked_multiply(capacity / 64, 24 * 258, scratch_floats) ||
        !checked_add(scratch_floats, 64, scratch_floats) ||
        !checked_multiply(scratch_floats, kF32Bytes, scratch_bytes)) {
        return false;
    }
    alignment_padding(map_bytes, map_padding);
    alignment_padding(logits_bytes, logits_padding);
    alignment_padding(scratch_bytes, scratch_padding);

    std::uint64_t expected = kMtpBindCjkReservedBytes;
    if (!checked_add(expected, kMtpBindCjkMapPadding, expected) ||
        !checked_add(expected, kMtpBindCjkLogitsPadding, expected) ||
        map_padding > expected) {
        return false;
    }
    expected -= map_padding;
    if (logits_padding > expected) return false;
    expected -= logits_padding;
    if (scratch_padding > expected) return false;
    expected -= scratch_padding;
    bytes = expected;
    return true;
}

bool build_regions(MtpBindLayout& layout, std::uint64_t& cursor) {
    const std::uint64_t rows = kMtpBindRows;
    const std::uint64_t hidden = 2560;
    const std::uint64_t streams = 4;
    const std::uint64_t heads = 24;
    const std::uint64_t head_width = 256;
    const std::uint64_t expert_rows = 10;
    const std::uint64_t expert_width = 640;
    std::uint64_t bytes = 0;
    cursor = 0;

    const auto append_f32 = [&](MtpBindRegionId id, std::uint64_t r,
                                std::uint64_t c) {
        std::uint64_t size = 0;
        return tensor_bytes(r, c, kF32Bytes, size) &&
               append_region(layout, id, size, cursor);
    };
    const auto append_i32 = [&](MtpBindRegionId id, std::uint64_t count) {
        std::uint64_t size = 0;
        return checked_multiply(count, kI32Bytes, size) &&
               append_region(layout, id, size, cursor);
    };

    if (!append_f32(MtpBindRegionId::mtp_rin, rows * streams, hidden) ||
        !append_i32(MtpBindRegionId::mtp_tok, 1) ||
        !append_f32(MtpBindRegionId::emb, rows, hidden) ||
        !append_f32(MtpBindRegionId::en, rows, hidden) ||
        !append_f32(MtpBindRegionId::e2, rows, hidden) ||
        !append_f32(MtpBindRegionId::hn, rows * streams, hidden) ||
        !append_f32(MtpBindRegionId::h2, rows * streams, hidden) ||
        !tensor_bytes(rows * streams, hidden / 32, kQ8_1BlockBytes, bytes) ||
        !append_region(layout, MtpBindRegionId::xq, bytes, cursor) ||
        !append_f32(MtpBindRegionId::residual, rows * streams, hidden) ||
        !append_f32(MtpBindRegionId::xn, rows, hidden * streams) ||
        !append_f32(MtpBindRegionId::lo, rows, 320) ||
        !append_f32(MtpBindRegionId::rs, rows, streams) ||
        !append_f32(MtpBindRegionId::inj, rows, streams) ||
        !append_f32(MtpBindRegionId::inj2, rows, streams) ||
        !append_f32(MtpBindRegionId::mixed, rows, hidden) ||
        !append_f32(MtpBindRegionId::gate, 1, hidden * streams) ||
        !append_f32(MtpBindRegionId::kcur, rows * 2, head_width) ||
        !append_f32(MtpBindRegionId::vcur, rows * 2, head_width) ||
        !append_f32(MtpBindRegionId::q_full, heads, head_width * 2) ||
        !append_f32(MtpBindRegionId::qcur, heads, head_width) ||
        !append_f32(MtpBindRegionId::attn, heads, head_width) ||
        !append_f32(MtpBindRegionId::attn32, heads, head_width) ||
        !append_i32(MtpBindRegionId::mtp_ids, layout.capacity)) {
        return false;
    }

    if (layout.draft_vocab_count == 0) {
        layout.regions[static_cast<std::size_t>(
            MtpBindRegionId::draft_vocab_ids)] = {};
    } else {
        if (!checked_multiply(layout.draft_vocab_count, kI32Bytes, bytes) ||
            !append_region(layout, MtpBindRegionId::draft_vocab_ids,
                           bytes, cursor)) {
            return false;
        }
    }

    const std::uint64_t scratch_floats =
        static_cast<std::uint64_t>(layout.chunk_count) * 24 * 258 + 64;
    if (!tensor_bytes(scratch_floats, 1, kF32Bytes, bytes) ||
        !append_region(layout, MtpBindRegionId::att_scratch, bytes, cursor) ||
        !append_f32(MtpBindRegionId::bo, 1, hidden) ||
        !append_f32(MtpBindRegionId::y, 1, hidden) ||
        !append_f32(MtpBindRegionId::logits_r, 1, 512) ||
        !append_i32(MtpBindRegionId::ids, 10) ||
        !append_f32(MtpBindRegionId::wts, 1, 10) ||
        !append_region(layout, MtpBindRegionId::mq, hidden, cursor) ||
        !append_f32(MtpBindRegionId::ms, 1, hidden / 32) ||
        !append_f32(MtpBindRegionId::eg, expert_rows, expert_width) ||
        !append_f32(MtpBindRegionId::eu, expert_rows, expert_width) ||
        !append_f32(MtpBindRegionId::eh, expert_rows, expert_width) ||
        !append_region(layout, MtpBindRegionId::hq,
                       expert_rows * expert_width, cursor) ||
        !append_f32(MtpBindRegionId::hs, 1, expert_rows * expert_width / 32) ||
        !append_f32(MtpBindRegionId::parts, expert_rows, hidden) ||
        !append_f32(MtpBindRegionId::sg, 1, expert_width) ||
        !append_f32(MtpBindRegionId::su, 1, expert_width) ||
        !append_f32(MtpBindRegionId::sh, 1, expert_width) ||
        !tensor_bytes(20, 1, kQ8_1BlockBytes, bytes) ||
        !append_region(layout, MtpBindRegionId::shq, bytes, cursor) ||
        !append_f32(MtpBindRegionId::shx, 1, hidden) ||
        !append_f32(MtpBindRegionId::gl, 1, 1) ||
        !append_f32(MtpBindRegionId::sample, 1, hidden) ||
        !append_f32(MtpBindRegionId::hl, 1, layout.output_vocab_count) ||
        !append_i32(MtpBindRegionId::out_ids, 1) ||
        !append_f32(MtpBindRegionId::probs, 1, 1) ||
        !append_i32(MtpBindRegionId::mtp_out, rows) ||
        !append_f32(MtpBindRegionId::mtp_prob, 1, rows)) {
        return false;
    }

    return true;
}

}  // namespace

MtpBindRegion MtpBindLayout::region(MtpBindRegionId id) const noexcept {
    const std::size_t index = static_cast<std::size_t>(id);
    return index < regions.size() ? regions[index] : MtpBindRegion{};
}

bool make_mtp_bind_layout(std::uint32_t capacity,
                          std::uint32_t draft_vocab_count,
                          MtpBindLayout& layout, std::string& error) {
    layout = {};
    error.clear();
    if (capacity == 0 || capacity > 262'144 || capacity % 64 != 0) {
        error = "MTP bind capacity must be a multiple of 64 within the context limit";
        return false;
    }
    if (draft_vocab_count > kMtpFullVocabularySize) {
        error = "MTP draft vocabulary exceeds the model vocabulary";
        return false;
    }

    MtpBindLayout candidate;
    candidate.capacity = capacity;
    candidate.draft_vocab_count = draft_vocab_count;
    candidate.output_vocab_count = draft_vocab_count == 0
        ? kMtpFullVocabularySize : draft_vocab_count;
    candidate.chunk_count = capacity / 64;
    if (!expected_allocation_bytes(capacity, draft_vocab_count,
                                   candidate.allocation_bytes)) {
        error = "MTP bind layout byte count overflows";
        return false;
    }

    std::uint64_t cursor = 0;
    if (!build_regions(candidate, cursor)) {
        error = "MTP bind layout region size or offset overflows";
        return false;
    }
    if (cursor > candidate.allocation_bytes) {
        error = "MTP bind regions exceed the geometry-derived allocation";
        return false;
    }
    candidate.used_bytes = cursor;
    candidate.reserved_bytes = candidate.allocation_bytes - cursor;
    std::uint64_t expected_reserved = 0;
    if (!expected_reserved_bytes(capacity, draft_vocab_count,
                                candidate.output_vocab_count,
                                expected_reserved) ||
        candidate.reserved_bytes != expected_reserved) {
        error = "MTP bind reserved tail does not match the documented arena contract";
        return false;
    }
    layout = std::move(candidate);
    error.clear();
    return true;
}

std::string_view mtp_bind_region_name(MtpBindRegionId id) noexcept {
    static constexpr std::array<std::string_view, kMtpBindRegionCount> names{{
        "mtp_rin", "mtp_tok", "emb", "en", "e2", "hn", "h2", "xq",
        "R", "xn", "lo", "rs", "inj", "inj2", "mixed", "gate",
        "kcur", "vcur", "q_full", "qcur", "attn", "attn32", "mtp_ids",
        "draft_vocab_ids", "att_scratch", "bo", "y", "logits_r", "ids",
        "wts", "mq", "ms", "eg", "eu", "eh", "hq", "hs", "parts", "sg",
        "su", "sh", "shq", "shx", "gl", "sample", "hl", "out_ids", "probs",
        "mtp_out", "mtp_prob",
    }};
    const std::size_t index = static_cast<std::size_t>(id);
    return index < names.size() ? names[index] : std::string_view{};
}

}  // namespace strata::metal
