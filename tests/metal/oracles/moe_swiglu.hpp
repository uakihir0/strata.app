#pragma once

#include "oracles/common.hpp"

#include <span>
#include <string>
#include <vector>

namespace twin {

bool moe_swiglu_reference(std::span<const float> gate,
                         std::span<const float> up,
                         std::vector<tf>& output, std::string& err);

}  // namespace twin
