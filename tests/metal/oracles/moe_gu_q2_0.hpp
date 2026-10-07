// Scalar oracle for the grouped Q2_0 routed-expert gate/up kernel.
#pragma once

#include "strata/metal/tiered.hpp"
#include "../../../src/metal/kernels/common/types.h"
#include "oracles/common.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool moe_gu_q2_0_reference(
    std::span<const block_q2_0> gate_weights,
    std::span<const block_q2_0> up_weights,
    std::span<const Q81> activations,
    std::span<const std::int32_t> expert_slots,
    const strata::metal::TieredMoePlan& plan, std::uint32_t rows,
    std::vector<tf>& gate, std::vector<tf>& up, std::string& err);

}  // namespace twin
