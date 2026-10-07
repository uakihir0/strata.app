// include/strata/metal/expert_profile.hpp - device-free STRP ranking reader.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::metal {

bool read_expert_profile(const std::string& path, std::uint32_t n_layers, std::uint32_t n_expert,
                         std::vector<std::pair<std::uint32_t, std::uint32_t>>& ranked,
                         std::string& error);

}  // namespace strata::metal
