#include "oracles/moe_route.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace twin {
namespace {

constexpr std::uint32_t kWarpSize = 32u;
constexpr std::uint32_t kTopK = 10u;
constexpr float kMinimumSelectedSum = 6.103515625e-5f;

void butterfly_sum(std::array<tf, kWarpSize>& values) {
    for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0; lane < kWarpSize; ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
}

}  // namespace

bool moe_route_reference(std::span<const float> logits, std::uint32_t rows,
                         std::uint32_t experts, bool divide_weights,
                         std::vector<std::int32_t>& ids,
                         std::vector<tf>& weights, std::string& err) {
    if (rows == 0u || (experts != 256u && experts != 512u)) {
        err = "MoE router shape is invalid";
        return false;
    }
    const std::uint64_t expected_logits =
        static_cast<std::uint64_t>(rows) * experts;
    const std::uint64_t expected_outputs =
        static_cast<std::uint64_t>(rows) * kTopK;
    if (expected_logits != logits.size() ||
        expected_outputs > std::numeric_limits<std::size_t>::max()) {
        err = "MoE router buffer size is invalid";
        return false;
    }
    if (std::any_of(logits.begin(), logits.end(),
                    [](float value) { return !std::isfinite(value); })) {
        err = "MoE router oracle accepts finite logits only";
        return false;
    }

    ids.assign(static_cast<std::size_t>(expected_outputs), -1);
    weights.assign(static_cast<std::size_t>(expected_outputs), tf{});
    const std::uint32_t vectors_per_lane = experts / kWarpSize;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t row_offset = static_cast<std::size_t>(row) * experts;
        std::array<std::array<tf, 16>, kWarpSize> values{};
        std::array<float, kWarpSize> lane_maxima{};
        for (std::uint32_t lane = 0; lane < kWarpSize; ++lane) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::uint32_t i = 0; i < vectors_per_lane; ++i) {
                const float value = logits[row_offset + lane + kWarpSize * i];
                values[lane][i] = tf{value};
                maximum = std::fmax(maximum, value);
            }
            lane_maxima[lane] = maximum;
        }
        for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
            const auto previous = lane_maxima;
            for (std::uint32_t lane = 0; lane < kWarpSize; ++lane) {
                lane_maxima[lane] =
                    std::fmax(previous[lane], previous[lane ^ offset]);
            }
        }
        const tf maximum{lane_maxima[0]};

        std::array<tf, kWarpSize> lane_sums{};
        for (std::uint32_t lane = 0; lane < kWarpSize; ++lane) {
            for (std::uint32_t i = 0; i < vectors_per_lane; ++i) {
                values[lane][i] = accurate_exp(values[lane][i] - maximum);
                lane_sums[lane] = lane_sums[lane] + values[lane][i];
            }
        }
        butterfly_sum(lane_sums);
        const tf reciprocal = tf{1.0f} / lane_sums[0];
        for (std::uint32_t lane = 0; lane < kWarpSize; ++lane) {
            for (std::uint32_t i = 0; i < vectors_per_lane; ++i) {
                values[lane][i] = values[lane][i] * reciprocal;
                if (std::isnan(values[lane][i].value)) {
                    values[lane][i] =
                        tf{-std::numeric_limits<float>::max()};
                }
            }
        }

        std::array<tf, kWarpSize> selected_sums{};
        std::array<tf, kTopK> selected_values{};
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            std::array<tf, kWarpSize> best_values{};
            std::array<std::int32_t, kWarpSize> best_experts{};
            for (std::uint32_t lane = 0; lane < kWarpSize; ++lane) {
                tf best = values[lane][0];
                std::int32_t expert = static_cast<std::int32_t>(lane);
                for (std::uint32_t i = 1; i < vectors_per_lane; ++i) {
                    if (values[lane][i].value > best.value) {
                        best = values[lane][i];
                        expert = static_cast<std::int32_t>(
                            lane + kWarpSize * i);
                    }
                }
                best_values[lane] = best;
                best_experts[lane] = expert;
            }
            for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
                const auto previous_values = best_values;
                const auto previous_experts = best_experts;
                for (std::uint32_t lane = 0; lane < kWarpSize; ++lane) {
                    const std::uint32_t other_lane = lane ^ offset;
                    const tf other = previous_values[other_lane];
                    const std::int32_t other_expert =
                        previous_experts[other_lane];
                    if (other.value > previous_values[lane].value ||
                        (other.value == previous_values[lane].value &&
                         other_expert < previous_experts[lane])) {
                        best_values[lane] = other;
                        best_experts[lane] = other_expert;
                    }
                }
            }
            const tf best = best_values[0];
            const std::uint32_t expert =
                static_cast<std::uint32_t>(best_experts[0]);
            ids[static_cast<std::size_t>(row) * kTopK + rank] =
                static_cast<std::int32_t>(expert);
            selected_values[rank] = best;
            selected_sums[expert & 31u] =
                selected_sums[expert & 31u] + best;
            values[expert & 31u][expert / kWarpSize] =
                tf{-std::numeric_limits<float>::infinity()};
        }

        butterfly_sum(selected_sums);
        const tf selected_sum{
            std::fmax(selected_sums[0].value, kMinimumSelectedSum),
            selected_sums[0].value > kMinimumSelectedSum
                ? selected_sums[0].magnitude
                : static_cast<double>(kMinimumSelectedSum)};
        const tf inverse_selected_sum = tf{1.0f} / selected_sum;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const tf value = selected_values[rank];
            weights[static_cast<std::size_t>(row) * kTopK + rank] =
                divide_weights ? value / selected_sum
                               : value * inverse_selected_sum;
        }
    }

    err.clear();
    return true;
}

}  // namespace twin
