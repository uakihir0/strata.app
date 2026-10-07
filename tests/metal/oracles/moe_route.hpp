#pragma once

#include "oracles/common.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool moe_route_reference(std::span<const float> logits, std::uint32_t rows,
                         std::uint32_t experts, bool divide_weights,
                         std::vector<std::int32_t>& ids,
                         std::vector<tf>& weights, std::string& err);

}  // namespace twin
