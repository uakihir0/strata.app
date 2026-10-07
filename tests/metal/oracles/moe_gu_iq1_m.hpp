#pragma once

#include "oracles/iq1_m_mv.hpp"
#include "oracles/common.hpp"
#include "strata/metal/tiered.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool moe_gu_iq1_m_reference(
    std::span<const block_iq1_m> gate_weights,
    std::span<const block_iq1_m> up_weights,
    std::span<const Q81> activations,
    std::span<const std::int32_t> expert_slots,
    const strata::metal::TieredMoePlan& plan, std::uint32_t rows,
    std::vector<tf>& gate, std::vector<tf>& up, std::string& err);

}  // namespace twin
