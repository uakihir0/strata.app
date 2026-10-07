#include "strata/metal/sampler_tail.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using strata::kernels::SamplerParams;
using strata::metal::SampCandidate;
using strata::metal::TailPick;

SampCandidate candidate(std::int32_t id, float logit) {
    return {id, std::bit_cast<std::uint32_t>(logit)};
}

int failures = 0;

void check(bool condition, const char* label) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", label);
    if (!condition) ++failures;
}

bool close(float a, float b, float tolerance = 2.0e-7f) {
    return std::fabs(a - b) <= tolerance;
}

}  // namespace

int main() {
    using strata::metal::sampler_philox_uniform;
    using strata::metal::sampler_shape_supported;
    using strata::metal::sampler_tail;

    check(sampler_philox_uniform(0, 0) == 0.9584041237831116f, "Philox zero vector known answer");
    check(sampler_philox_uniform(1, 0) == 0.3436594605445862f, "Philox low seed known answer");
    check(sampler_philox_uniform(123, (std::uint64_t{1} << 32) + 7) == 0.7804952263832092f,
          "Philox high counter known answer");
    check(sampler_philox_uniform(0x123456789abcdef0ULL, 0xfedcba9876543210ULL) == 0.9908959865570068f,
          "Philox high seed and counter known answer");
    check(sampler_shape_supported(1, 1) && sampler_shape_supported(8, 262144),
          "accepts sampler row and vocabulary boundaries");
    check(!sampler_shape_supported(0, 1) && !sampler_shape_supported(9, 1),
          "rejects row counts outside the window limit");
    check(!sampler_shape_supported(1, 0) && !sampler_shape_supported(1, 262145),
          "rejects empty or over-limit vocabularies");

    const std::array uniform = {candidate(10, 0.0f), candidate(11, 0.0f),
                                candidate(12, 0.0f), candidate(13, 0.0f)};
    SamplerParams p;
    p.seed = 1;
    auto pick = sampler_tail(uniform, 4, p, 0);
    check(pick && pick->id == 11 && close(pick->probability, 0.25f), "uniform inverse-CDF selection");

    p.seed = 0;
    pick = sampler_tail(uniform, 4, p, 0);
    check(pick && pick->id == 13 && close(pick->probability, 0.25f),
          "uniform draw reaches the last inverse-CDF interval");

    const std::array half_boundary = {candidate(14, 0.0f), candidate(15, 0.0f)};
    pick = sampler_tail(half_boundary, 2, p, 26259228u);
    check(pick && pick->id == 15 && close(pick->probability, 0.5f),
          "strict inverse-CDF comparison skips an exactly equal boundary");

    const std::array ordered = {candidate(20, 0.0f), candidate(21, -1.0f),
                                candidate(22, -2.0f), candidate(23, -3.0f)};
    p = {};
    p.seed = 0;
    p.top_p = 0.7f;
    p.min_keep = 1;
    pick = sampler_tail(ordered, 4, p, 0);
    check(pick && pick->id == 21, "top_p truncates the ordered list before drawing");
    check(pick && close(pick->probability,
                        static_cast<float>(std::exp(-1.0) / (1.0 + std::exp(-1.0)))),
          "top_p result reports normalized probability");

    p.top_p = 0.01f;
    p.min_keep = 3;
    p.seed = 1;
    pick = sampler_tail(ordered, 4, p, 0);
    check(pick && pick->id == 20, "top_p applies min_keep after its cut");

    p = {};
    p.top_p = 0.75f;
    p.min_p = 0.5f;
    p.temperature = 0.25f;
    const float threshold_leader = std::log(0.5f);
    const std::array minp = {candidate(30, 0.0f), candidate(31, threshold_leader),
                             candidate(32, threshold_leader - 0.001f)};
    pick = sampler_tail(minp, 3, p, 0);
    check(pick && pick->id == 31, "min_p follows top_p and retains threshold equality");

    const std::array order_fixture = {
        candidate(40, std::log(0.4f)), candidate(41, std::log(0.3f)),
        candidate(42, std::log(0.2f)), candidate(43, std::log(0.1f))};
    p = {};
    p.seed = 0;
    p.top_p = 0.75f;
    p.min_p = 0.3f;
    pick = sampler_tail(order_fixture, 4, p, 0);
    check(pick && pick->id == 42,
          "top_p before min_p keeps the third candidate observable");

    const std::array temperature_order = {
        candidate(60, 3.0f), candidate(61, 2.55f), candidate(62, 2.5f),
        candidate(63, 2.45f), candidate(64, 2.4f), candidate(65, 2.35f),
        candidate(66, 2.3f), candidate(67, 2.25f)};
    p = {};
    p.top_p = 0.5f;
    p.temperature = 0.5f;
    p.seed = 0;
    pick = sampler_tail(temperature_order, 8, p, 0);
    check(pick && pick->id == 63,
          "top_p truncates raw logits before temperature scaling");

    p = {};
    p.top_p = 1.0f;
    p.temperature = 0.0f;
    const std::array singleton = {candidate(47, 2.0f)};
    pick = sampler_tail(singleton, 1, p, 0);
    check(pick && pick->id == 47 && pick->probability == 1.0f,
          "single candidate has probability one for nonpositive temperature");

    const std::array greedy_candidates = {
        candidate(70, 5.0f), candidate(71, 4.0f), candidate(72, 3.0f)};
    p = {};
    p.top_p = 0.1f;
    p.min_p = 0.9f;
    p.temperature = 0.0f;
    pick = sampler_tail(greedy_candidates, 3, p, 0);
    check(pick && pick->id == 70 && pick->probability == 1.0f,
          "greedy tail returns the first of multiple candidates");

    p.greedy = true;
    p.temperature = 1.0f;
    pick = sampler_tail(greedy_candidates, 3, p, 0);
    check(pick && pick->id == 70 && pick->probability == 1.0f,
          "explicit greedy mode returns the first candidate");

    const std::array infinities = {
        candidate(50, std::numeric_limits<float>::infinity()), candidate(51, -1.0f)};
    p = {};
    p.top_p = 0.5f;
    pick = sampler_tail(infinities, 2, p, 0);
    check(pick && pick->id == 51 && std::isnan(pick->probability),
          "nonfinite tail preserves upstream last-entry fallback");

    check(!sampler_tail(uniform, 0, p, 0), "rejects empty selection");
    check(!sampler_tail(uniform, 5, p, 0), "rejects k larger than candidate span");
    std::array<SampCandidate, 65> too_wide{};
    check(!sampler_tail(too_wide, 65, p, 0), "rejects k larger than the fixed top-k bound");

    p = {};
    p.seed = 0x123456789abcdef0ULL;
    p.counter = 0xfedcba9876543200ULL;
    constexpr std::array<int, 8> expected_ids{12, 10, 13, 10, 12, 11, 13, 13};
    bool row_draws_match = true;
    for (std::uint64_t row = 0; row < expected_ids.size(); ++row) {
        const auto row_pick = sampler_tail(uniform, 4, p, p.counter + row);
        row_draws_match &= row_pick && row_pick->id == expected_ids[row] &&
                           close(row_pick->probability, 0.25f);
    }
    check(row_draws_match, "consecutive row counters match fixed high-seed Philox picks");

    const std::array penalized_distribution = {
        candidate(0, -1.5f), candidate(1, 0.0f)};
    p = {};
    p.top_k = 2;
    p.top_p = 1.0f;
    p.temperature = 0.7f;
    p.seed = 53;
    int token_zero_draws = 0;
    for (std::uint64_t counter = 0; counter < 8000u; ++counter) {
        const auto draw = sampler_tail(penalized_distribution, 2, p, counter);
        token_zero_draws += draw && draw->id == 0;
    }
    const double expected_share = 1.0 / (1.0 + std::exp(1.5 / 0.7));
    const double observed_share = static_cast<double>(token_zero_draws) / 8000.0;
    check(std::fabs(observed_share - expected_share) < 0.014,
          "8000 deterministic draws match the once-penalized distribution");

    int history_rows_wrong = 0;
    int history_rows_checked = 0;
    for (const int tail_size : {0, 1, 5, 63, 64, 65, 300}) {
        for (const int rows : {1, 3, 8}) {
            for (const int history_size : {1, 4, 64, 100}) {
                std::vector<std::int32_t> consumed(static_cast<std::size_t>(tail_size));
                std::vector<std::int32_t> window(static_cast<std::size_t>(rows));
                for (int i = 0; i < tail_size; ++i) {
                    consumed[static_cast<std::size_t>(i)] = 1000 + i;
                }
                for (int row = 0; row < rows; ++row) {
                    window[static_cast<std::size_t>(row)] = 5000 + row;
                }
                std::vector<std::int32_t> actual(
                    static_cast<std::size_t>(rows) * history_size, 12345);
                strata::kernels::penalty_rows(
                    consumed.data(), tail_size, window.data(), rows, history_size, actual.data());
                for (int row = 0; row < rows; ++row) {
                    std::vector<std::int32_t> sequence(consumed);
                    sequence.insert(sequence.end(), window.begin(), window.begin() + row + 1);
                    std::vector<std::int32_t> expected(
                        static_cast<std::size_t>(history_size), -1);
                    const int take = static_cast<int>(std::min<std::size_t>(
                        static_cast<std::size_t>(history_size), sequence.size()));
                    for (int i = 0; i < take; ++i) {
                        expected[static_cast<std::size_t>(history_size - take + i)] =
                            sequence[sequence.size() - static_cast<std::size_t>(take) +
                                     static_cast<std::size_t>(i)];
                    }
                    const auto begin = actual.begin() +
                        static_cast<std::ptrdiff_t>(row) * history_size;
                    if (!std::equal(expected.begin(), expected.end(), begin)) {
                        if (history_rows_wrong < 8) {
                            std::printf("fixture 15 mismatch tail=%d rows=%d h=%d row=%d\n",
                                        tail_size, rows, history_size, row);
                        }
                        ++history_rows_wrong;
                    }
                    ++history_rows_checked;
                }

                std::vector<std::int32_t> legacy(
                    static_cast<std::size_t>(history_size), -1);
                const int take_first_row = static_cast<int>(std::min<std::int64_t>(
                    history_size, static_cast<std::int64_t>(tail_size) + 1));
                for (int i = 0; i < take_first_row - 1; ++i) {
                    legacy[static_cast<std::size_t>(history_size - take_first_row + i)] =
                        consumed[static_cast<std::size_t>(
                            tail_size - (take_first_row - 1) + i)];
                }
                legacy.back() = window[0];
                if (!std::equal(legacy.begin(), legacy.end(), actual.begin())) {
                    if (history_rows_wrong < 8) {
                        std::printf("fixture 15 legacy mismatch tail=%d rows=%d h=%d\n",
                                    tail_size, rows, history_size);
                    }
                    ++history_rows_wrong;
                }
                ++history_rows_checked;
            }
        }
    }
    std::printf("fixture 15: %d mismatches in %d rows\n",
                history_rows_wrong, history_rows_checked);
    check(history_rows_wrong == 0 && history_rows_checked == 420,
          "upstream fixture 15 penalty-history rows");

    std::printf("sampler_tail: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
