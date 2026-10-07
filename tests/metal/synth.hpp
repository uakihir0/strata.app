#pragma once

#include "twin.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace twin {

/// Deterministic scalar inputs for synthetic kernel twins; real tensors are loaded by model twins.
std::vector<float> synthesize_floats(InputSet set, std::size_t count,
                                     std::mt19937_64& rng, std::string& err);

}  // namespace twin
