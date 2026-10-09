// Host-side construction of one Metal Prefill chunk (docs/mac/04 §6.1).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strata::metal {

inline constexpr std::uint32_t kPrefillBatchSlotCount = 6u;
inline constexpr std::uint32_t kPrefillImageTokenFlag = 0x80000000u;
inline constexpr std::uint32_t kPrefillMaxShaderRows = 419'430u;

/// Mirrors StepRow in src/metal/kernels/common/types.h.
struct PrefillStepRow {
    std::int32_t pos = 0;
    std::int32_t n_kv = 0;
    std::int32_t n_bid = 0;
    std::int32_t width = 0;
};

/// Mirrors PrefillParams in src/metal/kernels/common/types.h.
struct PrefillParams {
    std::uint32_t T = 0;
    std::uint32_t p0 = 0;
    std::uint32_t idx_f = 0;
    std::uint32_t idx_e = 0;
};

struct PrefillChunkMetadata {
    PrefillParams params;
    // Query batch sizes for slots 0..5; each slot is written into its own
    // 256-byte arena slot by the eventual Prefill executor.
    std::array<std::uint32_t, kPrefillBatchSlotCount> batch_rows{};
    std::vector<std::uint32_t> tokens;
    std::vector<PrefillStepRow> steps;
    bool has_image_rows = false;
};

/// A contiguous, unpadded range in the prompt passed to Prefill.
struct PrefillChunkRange {
    std::size_t token_offset = 0;
    std::uint32_t pos0 = 0;
    std::uint32_t rows = 0;
};

class PrefillChunkSchedule {
public:
    std::size_t token_count() const noexcept { return token_count_; }
    std::uint32_t chunk_rows() const noexcept { return chunk_rows_; }
    std::size_t chunk_count() const noexcept {
        return token_count_ == 0u || chunk_rows_ == 0u
                   ? 0u
                   : 1u + (token_count_ - 1u) / chunk_rows_;
    }

    /// Return a scheduled range; an invalid index leaves `output` unchanged.
    bool range_at(std::size_t index, PrefillChunkRange& output) const noexcept;

private:
    friend bool build_prefill_chunk_schedule(
        std::uint32_t pos0, std::size_t token_count,
        std::uint32_t chunk_rows, PrefillChunkSchedule& output,
        std::string& error);

    std::uint32_t pos0_ = 0;
    std::size_t token_count_ = 0;
    std::uint32_t chunk_rows_ = 0;
};

/// Split a prompt into non-empty ranges no larger than `chunk_rows`.
///
/// An empty prompt produces an empty schedule. Non-empty prompt positions must
/// fit the signed StepRow range used by `build_prefill_chunk_metadata`. The
/// output is unchanged on failure.
bool build_prefill_chunk_schedule(std::uint32_t pos0,
                                  std::size_t token_count,
                                  std::uint32_t chunk_rows,
                                  PrefillChunkSchedule& output,
                                  std::string& error);

/// Build the host inputs for one non-empty chunk.
///
/// Tokens are copied verbatim, including bit 31 on image rows. Position
/// metadata must fit the signed StepRow fields, and rows must fit every
/// embedding/broadcast shader's 32-bit flattened index. On failure, `output`
/// is unchanged.
bool build_prefill_chunk_metadata(std::uint32_t pos0,
                                  std::span<const std::uint32_t> tokens,
                                  PrefillChunkMetadata& output,
                                  std::string& error);

}  // namespace strata::metal
