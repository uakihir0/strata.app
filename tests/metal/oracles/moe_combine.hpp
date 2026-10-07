#pragma once

#include "oracles/common.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool moe_combine_reference(std::span<const float> parts,
                           std::span<const float> weights,
                           std::span<const float> shared,
                           std::span<const float> gate, std::uint32_t rows,
                           std::vector<tf>& output, std::string& err);

}  // namespace twin
