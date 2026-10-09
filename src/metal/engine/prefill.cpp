// src/metal/engine/prefill.cpp - host metadata for one Metal Prefill chunk.
#include "strata/metal/prefill.hpp"
#include "../kernels/common/types.h"
#include "../kernels/common/params.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

static_assert(sizeof(strata::metal::PrefillStepRow) == sizeof(::StepRow));
static_assert(offsetof(strata::metal::PrefillStepRow, pos) ==
              offsetof(::StepRow, pos));
static_assert(offsetof(strata::metal::PrefillStepRow, n_kv) ==
              offsetof(::StepRow, n_kv));
static_assert(offsetof(strata::metal::PrefillStepRow, n_bid) ==
              offsetof(::StepRow, n_bid));
static_assert(offsetof(strata::metal::PrefillStepRow, width) ==
              offsetof(::StepRow, width));
static_assert(sizeof(strata::metal::PrefillParams) == sizeof(::PrefillParams));
static_assert(offsetof(strata::metal::PrefillParams, T) ==
              offsetof(::PrefillParams, T));
static_assert(offsetof(strata::metal::PrefillParams, p0) ==
              offsetof(::PrefillParams, p0));
static_assert(offsetof(strata::metal::PrefillParams, idx_f) ==
              offsetof(::PrefillParams, idx_f));
static_assert(offsetof(strata::metal::PrefillParams, idx_e) ==
              offsetof(::PrefillParams, idx_e));
static_assert(strata::metal::kPrefillMaxShaderRows ==
              (std::uint64_t{1u} << 32u) / HC_DIM);

}  // namespace

namespace strata::metal {

bool PrefillChunkSchedule::range_at(
    std::size_t index, PrefillChunkRange& output) const noexcept {
    if (index >= chunk_count()) return false;

    const std::size_t offset =
        index * static_cast<std::size_t>(chunk_rows_);
    const auto rows = static_cast<std::uint32_t>(
        std::min(token_count_ - offset,
                 static_cast<std::size_t>(chunk_rows_)));
    PrefillChunkRange next{
        offset,
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(pos0_) + offset),
        rows,
    };
    output = next;
    return true;
}

bool build_prefill_chunk_schedule(
    std::uint32_t pos0, std::size_t token_count, std::uint32_t chunk_rows,
    PrefillChunkSchedule& output, std::string& error) {
    if (chunk_rows == 0u || chunk_rows > kPrefillMaxShaderRows) {
        error = "Prefill chunk size must fit the shader row limit";
        return false;
    }

    PrefillChunkSchedule next;
    next.pos0_ = pos0;
    next.token_count_ = token_count;
    next.chunk_rows_ = chunk_rows;
    if (token_count != 0u) {
        const auto max_step_value = static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max());
        if (pos0 >= max_step_value ||
            token_count > static_cast<std::size_t>(max_step_value - pos0)) {
            error = "Prefill prompt positions exceed the signed StepRow range";
            return false;
        }
    }

    output = next;
    error.clear();
    return true;
}

bool build_prefill_chunk_metadata(std::uint32_t pos0,
                                  std::span<const std::uint32_t> tokens,
                                  PrefillChunkMetadata& output,
                                  std::string& error) {
    if (tokens.empty()) {
        error = "Prefill chunk must contain at least one token";
        return false;
    }
    if (tokens.size() > kPrefillMaxShaderRows) {
        error = "Prefill chunk row count exceeds the 32-bit shader index range";
        return false;
    }

    const std::uint64_t end =
        static_cast<std::uint64_t>(pos0) + tokens.size();
    const auto max_step_value = static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max());
    if (end > max_step_value) {
        error = "Prefill chunk positions exceed the signed StepRow range";
        return false;
    }

    const std::uint64_t idx_f =
        pos0 == 0u
            ? 4u
            : (static_cast<std::uint64_t>(pos0) + 3u) & ~std::uint64_t{3u};
    if (idx_f > std::numeric_limits<std::uint32_t>::max()) {
        error = "Prefill chunk idx_f exceeds the parameter range";
        return false;
    }

    const auto row_count = static_cast<std::uint32_t>(tokens.size());
    PrefillChunkMetadata next;
    next.params = {
        row_count,
        pos0,
        static_cast<std::uint32_t>(idx_f),
        static_cast<std::uint32_t>(end & ~std::uint64_t{3u}),
    };
    next.batch_rows = {
        8u,
        row_count % 8u,
        256u,
        row_count % 256u,
        32u,
        row_count % 32u,
    };
    next.tokens.assign(tokens.begin(), tokens.end());
    next.steps.reserve(row_count);

    for (std::uint32_t row = 0u; row < row_count; ++row) {
        const auto pos = static_cast<std::int32_t>(pos0 + row);
        const auto n_kv = static_cast<std::int32_t>(pos + 1);
        next.steps.push_back({
            pos,
            n_kv,
            n_kv / 4,
            std::min(n_kv, 2051),
        });
        next.has_image_rows |=
            (tokens[row] & kPrefillImageTokenFlag) != 0u;
    }

    output = std::move(next);
    error.clear();
    return true;
}

}  // namespace strata::metal
