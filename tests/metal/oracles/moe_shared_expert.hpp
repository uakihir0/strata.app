#pragma once

#include "oracles/q8_0_mv.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

struct MoeSharedExpertOutput {
    std::vector<Q81> mixed_q8_1;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::vector<Q81> hidden_q8_1;
    std::vector<float> shared;
    std::vector<float> gate_logits;
};

bool moe_shared_expert_reference(
    std::span<const float> mixed,
    std::span<const block_q8_0> gate_weights,
    std::span<const block_q8_0> up_weights,
    std::span<const block_q8_0> down_weights,
    std::span<const std::uint16_t> gate_input_weights,
    std::uint32_t rows, std::uint32_t nc, MoeSharedExpertOutput& output,
    std::string& err);

}  // namespace twin
