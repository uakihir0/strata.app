#pragma once

#include "strata/metal/tiered.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

struct MoePlanParams {
    std::uint32_t rows = 0;
    std::uint32_t n_expert = 0;
    std::uint32_t tiered = 0;
    std::uint32_t layer = 0;
};
static_assert(sizeof(MoePlanParams) == 16);

bool moe_plan_reference(std::span<const std::int32_t> ids,
                        std::uint32_t rows, std::uint32_t experts,
                        bool tiered, std::uint32_t layer,
                        std::span<const std::int32_t> resident,
                        std::span<const float> initial_usage,
                        strata::metal::TieredMoePlan& plan,
                        std::vector<float>& expected_usage,
                        std::string& err);

}  // namespace twin
