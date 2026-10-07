#include "oracles/moe_gu_q2_0.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace twin {
namespace {

constexpr std::uint32_t kTopK = 10u;
constexpr std::uint32_t kExpertWidth = 640u;
constexpr std::uint32_t kInputWidth = 2560u;
constexpr std::uint32_t kBlocksPerRow = kInputWidth / 64u;
constexpr std::uint32_t kPartsPerBlock = 2u;
constexpr std::uint32_t kMaxEntries = 80u;

int integer_dot(const block_q2_0& weight, std::uint32_t qs_half,
                const Q81& activation) {
    int sum = 0;
    const std::uint32_t byte_base = qs_half * 8u;
    for (std::uint32_t word = 0; word < 4u; ++word) {
        for (std::uint32_t byte_in_word = 0; byte_in_word < 2u;
             ++byte_in_word) {
            const std::uint32_t packed =
                weight.qs[byte_base + 2u * word + byte_in_word];
            const std::uint32_t activation_base =
                8u * word + 4u * byte_in_word;
            for (std::uint32_t code_lane = 0; code_lane < 4u;
                 ++code_lane) {
                const int code =
                    static_cast<int>((packed >> (2u * code_lane)) & 0x03u) -
                    1;
                sum += code *
                       static_cast<int>(
                           activation.qs[activation_base + code_lane]);
            }
        }
    }
    return sum;
}

float butterfly_sum(std::array<float, 32> values) {
    for (std::size_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::size_t lane = 0; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values[0];
}

}  // namespace

bool moe_gu_q2_0_reference(
    std::span<const block_q2_0> gate_weights,
    std::span<const block_q2_0> up_weights,
    std::span<const Q81> activations,
    std::span<const std::int32_t> expert_slots,
    const strata::metal::TieredMoePlan& plan, std::uint32_t rows,
    std::vector<tf>& gate, std::vector<tf>& up, std::string& err) {
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kExpertWidth) * kBlocksPerRow;
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    if (rows == 0u || rows > 8u || plan.n_entries != entry_count ||
        plan.n_groups == 0u || plan.n_groups > plan.n_entries ||
        plan.n_entries > kMaxEntries ||
        expert_slots.empty() ||
        std::any_of(expert_slots.begin(), expert_slots.end(),
                    [](std::int32_t slot) { return slot < -1; }) ||
        gate_weights.empty() ||
        gate_weights.size() % weight_blocks_per_expert != 0u ||
        up_weights.size() != gate_weights.size() ||
        activations.size() !=
            static_cast<std::size_t>(rows) * (kInputWidth / 32u)) {
        err = "grouped Q2_0 MoE oracle shape is invalid";
        return false;
    }
    const std::size_t expert_slots_count =
        gate_weights.size() / weight_blocks_per_expert;
    if (std::any_of(expert_slots.begin(), expert_slots.end(),
                    [expert_slots_count](std::int32_t slot) {
                        return slot >= 0 &&
                               static_cast<std::size_t>(slot) >=
                                   expert_slots_count;
                    })) {
        err = "grouped Q2_0 MoE oracle expert slot is out of range";
        return false;
    }
    if (plan.grp_start[0] != 0 ||
        plan.grp_start[plan.n_groups] !=
            static_cast<std::int32_t>(entry_count)) {
        err = "grouped Q2_0 MoE oracle plan endpoints are invalid";
        return false;
    }

    gate.assign(entry_count * kExpertWidth, 0.0f);
    up.assign(entry_count * kExpertWidth, 0.0f);
    for (std::uint32_t group = 0; group < plan.n_groups; ++group) {
        const std::int32_t start = plan.grp_start[group];
        const std::int32_t end = plan.grp_start[group + 1u];
        const std::int32_t expert = plan.grp_e[group];
        if (start < 0 || end <= start ||
            end > static_cast<std::int32_t>(entry_count) || expert < 0 ||
            static_cast<std::size_t>(expert) >= expert_slots.size()) {
            err = "grouped Q2_0 MoE oracle plan group is invalid";
            return false;
        }
        const std::int32_t slot = expert_slots[static_cast<std::size_t>(expert)];
        if (slot < 0) {
            err = "grouped Q2_0 MoE oracle plan names an unmapped expert";
            return false;
        }
        for (std::uint32_t role = 0; role < 2u; ++role) {
            const std::span<const block_q2_0> weights =
                role == 0u ? gate_weights : up_weights;
            std::vector<tf>& output = role == 0u ? gate : up;
            const std::size_t expert_base =
                static_cast<std::size_t>(slot) * weight_blocks_per_expert;
            for (std::uint32_t row = 0; row < kExpertWidth; ++row) {
                for (std::uint32_t first =
                         static_cast<std::uint32_t>(start);
                     first < static_cast<std::uint32_t>(end); first += 4u) {
                    const std::uint32_t count = std::min(
                        4u, static_cast<std::uint32_t>(end) - first);
                    std::array<std::array<float, 4>, 32> lane_sums{};
                    for (std::uint32_t lane = 0; lane < 32u; ++lane) {
                        for (std::uint32_t k = lane;
                             k < kBlocksPerRow * kPartsPerBlock; k += 32u) {
                            const std::uint32_t block = k / kPartsPerBlock;
                            const std::uint32_t part = k % kPartsPerBlock;
                            const std::size_t weight_index =
                                expert_base +
                                static_cast<std::size_t>(row) *
                                    kBlocksPerRow +
                                block;
                            const block_q2_0& weight = weights[weight_index];
                            for (std::uint32_t column = 0; column < count;
                                 ++column) {
                                const std::uint32_t entry = first + column;
                                const std::int32_t token = plan.ent_tok[entry];
                                if (token < 0 ||
                                    static_cast<std::uint32_t>(token) >= rows) {
                                    err =
                                        "grouped Q2_0 MoE oracle token is out of range";
                                    return false;
                                }
                                const std::size_t activation_index =
                                    static_cast<std::size_t>(token) *
                                        (kInputWidth / 32u) +
                                    block * kPartsPerBlock + part;
                                const Q81& activation =
                                    activations[activation_index];
                                const int dot =
                                    integer_dot(weight, part, activation);
                                const std::uint32_t ds =
                                    static_cast<std::uint32_t>(activation.ds);
                                const float d2 =
                                    strata::kernels::f32_from_f16(weight.d);
                                const float d8 = strata::kernels::f32_from_f16(
                                    static_cast<std::uint16_t>(ds & 0xffffu));
                                lane_sums[lane][column] +=
                                    ((d2 * d8) * static_cast<float>(dot));
                            }
                        }
                    }
                    for (std::uint32_t column = 0; column < count; ++column) {
                        std::array<float, 32> partials{};
                        for (std::uint32_t lane = 0; lane < 32u; ++lane) {
                            partials[lane] = lane_sums[lane][column];
                        }
                        output[static_cast<std::size_t>(first + column) *
                                   kExpertWidth +
                               row] = butterfly_sum(partials);
                    }
                }
            }
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
