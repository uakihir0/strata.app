// tests/metal/metal_mtp_bind_layout_test.cpp - device-free MTP bind arena bounds.
#include "strata/metal/mtp_bind_layout.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace strata::metal;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t expected_bytes(std::uint32_t capacity,
                             std::uint32_t draft_vocab_count) {
    constexpr std::uint64_t baseline_dynamic =
        32'768ull * sizeof(std::uint32_t) +
        (512ull * 24 * 258 + 64) * sizeof(float);
    const std::uint32_t output_count = draft_vocab_count == 0
        ? kMtpFullVocabularySize : draft_vocab_count;
    const std::uint64_t dynamic =
        static_cast<std::uint64_t>(capacity) * sizeof(std::uint32_t) +
        (static_cast<std::uint64_t>(capacity / 64) * 24 * 258 + 64) *
            sizeof(float);
    const std::int64_t output_delta =
        (static_cast<std::int64_t>(output_count) -
         kMtpDraftVocabularyBaseline) * sizeof(float);
    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(kMtpBindBaselineBytes) +
        static_cast<std::int64_t>(draft_vocab_count) * sizeof(std::uint32_t) +
        output_delta +
        static_cast<std::int64_t>(dynamic) -
        static_cast<std::int64_t>(baseline_dynamic));
}

std::uint64_t alignment_padding(std::uint64_t bytes) {
    constexpr std::uint64_t alignment = kMtpBindAlignment;
    const std::uint64_t remainder = bytes % alignment;
    return remainder == 0 ? 0 : alignment - remainder;
}

std::uint64_t expected_reserved_bytes(std::uint32_t capacity,
                                      std::uint32_t draft_vocab_count) {
    const std::uint32_t output_count = draft_vocab_count == 0
        ? kMtpFullVocabularySize : draft_vocab_count;
    const std::uint64_t map_bytes =
        static_cast<std::uint64_t>(draft_vocab_count) * sizeof(std::uint32_t);
    const std::uint64_t logits_bytes =
        static_cast<std::uint64_t>(output_count) * sizeof(float);
    const std::uint64_t scratch_bytes =
        (static_cast<std::uint64_t>(capacity / 64) * 24 * 258 + 64) *
        sizeof(float);
    return 128'192ull + 20 + 20 -
           alignment_padding(map_bytes) -
           alignment_padding(logits_bytes) -
           alignment_padding(scratch_bytes);
}

void check_layout(std::uint32_t capacity, std::uint32_t vocab_count) {
    MtpBindLayout layout;
    std::string error;
    require(make_mtp_bind_layout(capacity, vocab_count, layout, error),
            "layout rejected valid capacity/vocabulary: " + error);
    require(error.empty() && layout.capacity == capacity &&
                layout.chunk_count == capacity / 64 &&
                layout.draft_vocab_count == vocab_count &&
                layout.output_vocab_count ==
                    (vocab_count == 0 ? kMtpFullVocabularySize : vocab_count),
            "layout metadata does not describe the requested MTP bind shape");
    require(layout.allocation_bytes == expected_bytes(capacity, vocab_count),
            "layout allocation does not match the geometry/planner formula");
    require(layout.used_bytes <= layout.allocation_bytes &&
                layout.reserved_bytes ==
                    layout.allocation_bytes - layout.used_bytes,
            "layout reserved tail does not cover the exact remaining arena");
    require(layout.reserved_bytes ==
                expected_reserved_bytes(capacity, vocab_count),
            "layout reserved tail differs from its shape-dependent contract");

    std::uint64_t previous_end = 0;
    for (std::size_t index = 0; index < kMtpBindRegionCount; ++index) {
        const auto id = static_cast<MtpBindRegionId>(index);
        const MtpBindRegion region = layout.region(id);
        if (region.bytes == 0) continue;
        require(region.offset % kMtpBindAlignment == 0,
                std::string(mtp_bind_region_name(id)) +
                    " does not start at a 256-byte boundary");
        require(region.offset >= previous_end &&
                    region.bytes <= layout.used_bytes - region.offset,
                std::string(mtp_bind_region_name(id)) +
                    " overlaps an earlier region or exceeds the used arena");
        previous_end = region.offset + region.bytes;
    }
    require(previous_end == layout.used_bytes,
            "used_bytes does not end at the final named region");

    const MtpBindRegion ids = layout.region(MtpBindRegionId::mtp_ids);
    const MtpBindRegion scratch = layout.region(MtpBindRegionId::att_scratch);
    const MtpBindRegion vocab = layout.region(MtpBindRegionId::draft_vocab_ids);
    const MtpBindRegion logits = layout.region(MtpBindRegionId::hl);
    require(ids.bytes == static_cast<std::uint64_t>(capacity) * 4 &&
                scratch.bytes ==
                    (static_cast<std::uint64_t>(capacity / 64) * 24 * 258 + 64) * 4 &&
                logits.bytes ==
                    static_cast<std::uint64_t>(layout.output_vocab_count) * 4,
            "capacity, attention or logits regions have incorrect byte lengths");
    require(vocab.bytes == static_cast<std::uint64_t>(vocab_count) * 4,
            "draft vocabulary map has incorrect byte length");
    require((vocab_count != 0) == (vocab.bytes != 0),
            "empty draft vocabularies do not omit the device ID table");
}

void invalid_layouts() {
    MtpBindLayout layout;
    std::string error;
    require(!make_mtp_bind_layout(0, 0, layout, error) &&
                layout.allocation_bytes == 0 && !error.empty(),
            "zero bind capacity must fail closed");
    require(!make_mtp_bind_layout(65, 0, layout, error) &&
                layout.allocation_bytes == 0,
            "non-64-aligned bind capacity must fail closed");
    require(!make_mtp_bind_layout(262'208, 0, layout, error) &&
                layout.allocation_bytes == 0,
            "bind capacity beyond the supported maximum must fail closed");
    require(!make_mtp_bind_layout(64, kMtpFullVocabularySize + 1, layout, error) &&
                layout.allocation_bytes == 0,
            "oversized draft vocabulary must fail closed");
}

}  // namespace

int main() {
    try {
        for (const std::uint32_t capacity :
             {64u, 32'768u, 65'536u, 262'144u}) {
            for (const std::uint32_t vocab_count :
                 {0u, 40'525u, 58'963u, 106'299u, 248'320u}) {
                check_layout(capacity, vocab_count);
            }
        }
        invalid_layouts();
        std::cout << "metal_mtp_bind_layout_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_mtp_bind_layout_test: " << exception.what() << '\n';
        return 1;
    }
}
