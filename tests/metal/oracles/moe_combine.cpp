#include "oracles/moe_combine.hpp"

#include <algorithm>
#include <cmath>

namespace twin {

bool moe_combine_reference(std::span<const float> parts,
                           std::span<const float> weights,
                           std::span<const float> shared,
                           std::span<const float> gate, std::uint32_t rows,
                           std::vector<tf>& output, std::string& err) {
    constexpr std::size_t kTopK = 10u;
    constexpr std::size_t kWidth = 2560u;
    if (rows == 0u || rows > 8u ||
        parts.size() != static_cast<std::size_t>(rows) * kTopK * kWidth ||
        weights.size() != static_cast<std::size_t>(rows) * kTopK ||
        shared.size() != static_cast<std::size_t>(rows) * kWidth ||
        gate.size() != rows) {
        err = "MoE combine oracle shape is invalid";
        return false;
    }
    const auto finite = [](float value) { return std::isfinite(value); };
    if (!std::all_of(parts.begin(), parts.end(), finite) ||
        !std::all_of(weights.begin(), weights.end(), finite) ||
        !std::all_of(shared.begin(), shared.end(), finite) ||
        !std::all_of(gate.begin(), gate.end(), finite)) {
        err = "MoE combine oracle accepts finite inputs only";
        return false;
    }

    output.resize(static_cast<std::size_t>(rows) * kWidth);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t part_base =
            static_cast<std::size_t>(row) * kTopK * kWidth;
        const std::size_t weight_base =
            static_cast<std::size_t>(row) * kTopK;
        const std::size_t output_base =
            static_cast<std::size_t>(row) * kWidth;
        const tf gate_scale =
            tf{1.0f} / (tf{1.0f} + accurate_exp(tf{-gate[row]}));
        for (std::size_t column = 0u; column < kWidth; ++column) {
            tf sum = tf{parts[part_base + column]} *
                     tf{weights[weight_base]};
            for (std::size_t rank = 1u; rank < kTopK; ++rank) {
                sum = fma(
                    tf{parts[part_base + rank * kWidth + column]},
                    tf{weights[weight_base + rank]}, sum);
            }
            const tf shared_term =
                tf{shared[output_base + column]} * gate_scale;
            output[output_base + column] = sum + shared_term;
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
