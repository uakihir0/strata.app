#include "oracles/moe_shared_expert.hpp"

#include "oracles/bf16_mv.hpp"
#include "oracles/moe_swiglu.hpp"
#include "oracles/q8_1.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace twin {

bool moe_shared_expert_reference(
    std::span<const float> mixed,
    std::span<const block_q8_0> gate_weights,
    std::span<const block_q8_0> up_weights,
    std::span<const block_q8_0> down_weights,
    std::span<const std::uint16_t> gate_input_weights,
    std::uint32_t rows, std::uint32_t nc, MoeSharedExpertOutput& output,
    std::string& err) {
    constexpr std::uint32_t kResidualWidth = 2560u;
    constexpr std::uint32_t kExpertWidth = 640u;
    if (rows == 0u || rows > nc || nc > 8u ||
        mixed.size() != static_cast<std::size_t>(nc) * kResidualWidth) {
        err = "shared expert oracle shape is invalid";
        return false;
    }

    std::vector<Q81> mixed_q8;
    if (!quantize_q8_1_reference(mixed, kResidualWidth, nc, mixed_q8, err)) {
        return false;
    }
    output.mixed_q8_1 = mixed_q8;

    std::vector<float> gate_full;
    std::vector<float> up_full;
    if (!q8_0_matvec_reference(gate_weights, mixed_q8, kResidualWidth,
                               kExpertWidth, nc, rows, gate_full, err) ||
        !q8_0_matvec_reference(up_weights, mixed_q8, kResidualWidth,
                               kExpertWidth, nc, rows, up_full, err)) {
        return false;
    }

    const std::size_t hidden_count =
        static_cast<std::size_t>(rows) * kExpertWidth;
    output.gate.assign(gate_full.begin(), gate_full.begin() + hidden_count);
    output.up.assign(up_full.begin(), up_full.begin() + hidden_count);
    std::vector<tf> hidden_tf;
    if (!moe_swiglu_reference(
            std::span<const float>(gate_full.data(), hidden_count),
            std::span<const float>(up_full.data(), hidden_count), hidden_tf,
            err)) {
        return false;
    }
    output.hidden.resize(hidden_count);
    for (std::size_t i = 0u; i < hidden_count; ++i) {
        output.hidden[i] = hidden_tf[i].value;
    }
    std::vector<float> hidden_padded(static_cast<std::size_t>(nc) *
                                         kExpertWidth,
                                     0.0f);
    for (std::size_t i = 0u; i < hidden_count; ++i) {
        hidden_padded[i] = hidden_tf[i].value;
    }

    std::vector<Q81> hidden_q8;
    if (!quantize_q8_1_reference(hidden_padded, kExpertWidth, nc, hidden_q8,
                                 err)) {
        return false;
    }
    output.hidden_q8_1 = hidden_q8;

    std::vector<float> down_full;
    if (!q8_0_matvec_reference(down_weights, hidden_q8, kExpertWidth,
                               kResidualWidth, nc, rows, down_full, err)) {
        return false;
    }

    std::vector<float> gate_full_precision;
    if (!bf16_matvec_reference(gate_input_weights, mixed, kResidualWidth, 1u,
                               nc, rows, 256u, gate_full_precision, err)) {
        return false;
    }

    const std::size_t shared_count =
        static_cast<std::size_t>(rows) * kResidualWidth;
    output.shared.assign(down_full.begin(),
                         down_full.begin() + shared_count);
    output.gate_logits.assign(gate_full_precision.begin(),
                              gate_full_precision.begin() + rows);
    err.clear();
    return true;
}

}  // namespace twin
