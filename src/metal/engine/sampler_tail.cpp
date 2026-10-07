// src/metal/engine/sampler_tail.cpp - CPU FP64 tail of the Metal sampler.
#include "strata/metal/sampler_tail.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace strata::metal {
namespace {

constexpr std::size_t kMaxSelected = 64;

float candidate_logit(const SampCandidate& candidate) noexcept {
    return std::bit_cast<float>(candidate.logit_bits);
}

}  // namespace

float sampler_philox_uniform(std::uint64_t seed, std::uint64_t counter) noexcept {
    std::uint32_t c0 = static_cast<std::uint32_t>(counter);
    std::uint32_t c1 = static_cast<std::uint32_t>(counter >> 32);
    std::uint32_t c2 = static_cast<std::uint32_t>(seed);
    std::uint32_t c3 = static_cast<std::uint32_t>(seed >> 32);

    for (std::uint32_t round = 0; round < 10; ++round) {
        const std::uint32_t hi0 =
            static_cast<std::uint32_t>((std::uint64_t{0x9E3779B9u} * c0) >> 32);
        const std::uint32_t hi1 =
            static_cast<std::uint32_t>((std::uint64_t{0xBB67AE85u} * c2) >> 32);
        const std::uint32_t lo0 = 0x9E3779B9u * c0;
        const std::uint32_t lo1 = 0xBB67AE85u * c2;
        const std::uint32_t n0 = hi1 ^ c1 ^ round;
        const std::uint32_t n1 = lo1;
        const std::uint32_t n2 = hi0 ^ c3;
        const std::uint32_t n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    return static_cast<float>(c0 >> 8) * (1.0f / 16777216.0f);
}

bool sampler_shape_supported(std::uint32_t rows, std::uint32_t n_vocab) noexcept {
    return rows >= 1 && rows <= kSamplerMaxRows &&
           n_vocab >= 1 && n_vocab <= kSamplerMaxVocab;
}

std::optional<TailPick> sampler_tail(std::span<const SampCandidate> candidates, int k,
                                     const strata::kernels::SamplerParams& params,
                                     std::uint64_t counter) {
    if (k < 1 || static_cast<std::size_t>(k) > candidates.size() ||
        static_cast<std::size_t>(k) > kMaxSelected) {
        return std::nullopt;
    }

    if (params.greedy || params.temperature <= 0.0f) {
        return TailPick{candidates[0].id, 1.0f};
    }

    std::array<float, kMaxSelected> logits{};
    std::array<double, kMaxSelected> probabilities{};
    for (int i = 0; i < k; ++i) logits[static_cast<std::size_t>(i)] = candidate_logit(candidates[i]);

    int n_keep = k;
    float maximum = logits[0];
    for (int i = 1; i < k; ++i) maximum = std::fmax(maximum, logits[static_cast<std::size_t>(i)]);

    if (params.top_p < 1.0f) {
        for (int i = 0; i < k; ++i) {
            probabilities[static_cast<std::size_t>(i)] =
                std::exp(static_cast<double>(logits[static_cast<std::size_t>(i)]) -
                         static_cast<double>(maximum));
        }
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += probabilities[static_cast<std::size_t>(i)];
        for (int i = 0; i < k; ++i) probabilities[static_cast<std::size_t>(i)] /= sum;

        int cut = k;
        double cumulative = 0.0;
        for (int i = 0; i < k; ++i) {
            cumulative += probabilities[static_cast<std::size_t>(i)];
            if (cumulative >= static_cast<double>(params.top_p)) {
                cut = i + 1;
                break;
            }
        }
        if (cut < params.min_keep) cut = params.min_keep < k ? params.min_keep : k;
        n_keep = cut;
    }

    if (params.min_p > 0.0f) {
        const float threshold = logits[0] + std::log(params.min_p);
        for (int i = 0; i < n_keep; ++i) {
            if (logits[static_cast<std::size_t>(i)] < threshold) {
                n_keep = i;
                break;
            }
        }
    }

    const float inverse_temperature = params.temperature > 0.0f ? 1.0f / params.temperature : 0.0f;
    float scaled_maximum = logits[0] * inverse_temperature;
    for (int i = 1; i < n_keep; ++i) {
        scaled_maximum = std::fmax(scaled_maximum,
                                   logits[static_cast<std::size_t>(i)] * inverse_temperature);
    }
    for (int i = 0; i < n_keep; ++i) {
        probabilities[static_cast<std::size_t>(i)] =
            std::exp(static_cast<double>(logits[static_cast<std::size_t>(i)] * inverse_temperature) -
                     static_cast<double>(scaled_maximum));
    }
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += probabilities[static_cast<std::size_t>(i)];
    for (int i = 0; i < n_keep; ++i) probabilities[static_cast<std::size_t>(i)] /= sum;

    const float uniform = sampler_philox_uniform(params.seed, counter);
    int picked = n_keep > 0 ? n_keep - 1 : 0;
    double cumulative = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cumulative += probabilities[static_cast<std::size_t>(i)];
        if (static_cast<double>(uniform) < cumulative) {
            picked = i;
            break;
        }
    }
    return TailPick{candidates[static_cast<std::size_t>(picked)].id,
                    n_keep > 0 ? static_cast<float>(probabilities[static_cast<std::size_t>(picked)]) : 1.0f};
}

}  // namespace strata::metal
