#pragma once

#include "oracles/common.hpp"
#include "oracles/iq3_xxs_mv.hpp"
#include "strata/metal/tiered.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool moe_gu_iq3_xxs_reference(
    std::span<const block_iq3_xxs> gate_weights,
    std::span<const block_iq3_xxs> up_weights,
    std::span<const Q81> activations,
    std::span<const std::int32_t> expert_slots,
    const strata::metal::TieredMoePlan& plan, std::uint32_t rows,
    std::vector<tf>& gate, std::vector<tf>& up, std::string& err);

}  // namespace twin
