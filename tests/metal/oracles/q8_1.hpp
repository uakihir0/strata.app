#pragma once

#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool quantize_q8_1_reference(std::span<const float> input, std::uint32_t n,
                             std::uint32_t rows, std::vector<Q81>& output,
                             std::string& err);

}  // namespace twin
