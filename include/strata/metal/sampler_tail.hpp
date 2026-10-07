// CPU half of the Metal sampler (docs/mac/04-metal-kernels.md §5.12).
#pragma once

#include "strata/kernels/sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace strata::metal {

inline constexpr std::uint32_t kSamplerMaxRows = 8;
inline constexpr std::uint32_t kSamplerMaxBlocks = 64;
inline constexpr std::uint32_t kSamplerBlockVocab = 4096;
inline constexpr std::uint32_t kSamplerMaxVocab =
    kSamplerMaxBlocks * kSamplerBlockVocab;

/// One candidate written by the Metal top-k kernels. The second word is the
/// bit pattern of the candidate's float logit, as stored in an MSL int2.
struct alignas(8) SampCandidate {
    std::int32_t id;
    std::uint32_t logit_bits;
};
static_assert(sizeof(SampCandidate) == 8);
static_assert(alignof(SampCandidate) == 8);
static_assert(offsetof(SampCandidate, id) == 0);
static_assert(offsetof(SampCandidate, logit_bits) == 4);

struct TailPick {
    std::int32_t id;
    float probability;
};

/// The upstream counter-based Philox 4x32-10 draw.
float sampler_philox_uniform(std::uint64_t seed, std::uint64_t counter) noexcept;

/// Whether a window fits the fixed Metal scratch shapes.
bool sampler_shape_supported(std::uint32_t rows, std::uint32_t n_vocab) noexcept;

/// Run top_p -> min_p -> temperature -> inverse-CDF selection on a GPU-produced
/// ordered top-k list. Greedy settings return the first candidate with
/// probability 1. `counter` is the absolute draw index for this row.
/// Returns nullopt if k is outside [1, min(candidates.size(), 64)].
std::optional<TailPick> sampler_tail(std::span<const SampCandidate> candidates, int k,
                                     const strata::kernels::SamplerParams& params,
                                     std::uint64_t counter);

}  // namespace strata::metal
