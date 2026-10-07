#include "oracles/moe_swiglu.hpp"

#include <algorithm>
#include <cmath>

namespace twin {

bool moe_swiglu_reference(std::span<const float> gate,
                         std::span<const float> up,
                         std::vector<tf>& output, std::string& err) {
    constexpr std::size_t kExpertWidth = 640u;
    if (gate.empty() || gate.size() != up.size() ||
        gate.size() % kExpertWidth != 0u) {
        err = "MoE SwiGLU shape is invalid";
        return false;
    }
    if (std::any_of(gate.begin(), gate.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(up.begin(), up.end(),
                    [](float value) { return !std::isfinite(value); })) {
        err = "MoE SwiGLU oracle accepts finite inputs only";
        return false;
    }

    output.resize(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const tf g{gate[i]};
        const tf denominator = tf{1.0f} + accurate_exp(-g);
        output[i] = (g / denominator) * tf{up[i]};
    }
    err.clear();
    return true;
}

}  // namespace twin
