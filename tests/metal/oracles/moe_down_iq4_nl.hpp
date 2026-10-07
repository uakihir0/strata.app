#pragma once

#include "oracles/common.hpp"
#include "strata/metal/tiered.hpp"
#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool moe_down_iq4_nl_reference(
    std::span<const block_iq4_nl> weights,
    std::span<const Q81> activations,
    std::span<const std::int32_t> expert_slots,
    const strata::metal::TieredMoePlan& plan, std::uint32_t rows,
    std::vector<tf>& output, std::string& err);

}  // namespace twin
