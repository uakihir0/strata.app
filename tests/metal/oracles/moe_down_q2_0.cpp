#include "oracles/moe_down_q2_0.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace twin {
namespace {

constexpr std::uint32_t kTopK = 10u;
constexpr std::uint32_t kDownWidth = 2560u;
constexpr std::uint32_t kInputWidth = 640u;
constexpr std::uint32_t kBlocksPerRow = kInputWidth / 64u;
constexpr std::uint32_t kPartsPerBlock = 2u;

int integer_dot(const block_q2_0& weight, std::uint32_t qs_half,
                const Q81& activation) {
    int sum = 0;
    const std::uint32_t byte_base = qs_half * 8u;
    for (std::uint32_t word = 0u; word < 4u; ++word) {
        for (std::uint32_t byte_in_word = 0u; byte_in_word < 2u;
             ++byte_in_word) {
            const std::uint32_t packed =
                weight.qs[byte_base + 2u * word + byte_in_word];
            const std::uint32_t activation_base =
                8u * word + 4u * byte_in_word;
            for (std::uint32_t code_lane = 0u; code_lane < 4u; ++code_lane) {
                const int code =
                    static_cast<int>((packed >> (2u * code_lane)) & 0x03u) -
                    1;
                sum += code * static_cast<int>(
                                   activation.qs[activation_base + code_lane]);
            }
        }
    }
    return sum;
}

tf butterfly_sum(std::array<tf, 32> values) {
    for (std::size_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::size_t lane = 0u; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values[0];
}

}  // namespace

bool moe_down_q2_0_reference(
    std::span<const block_q2_0> weights,
    std::span<const Q81> activations,
    std::span<const std::int32_t> expert_slots,
    const strata::metal::TieredMoePlan& plan, std::uint32_t rows,
    std::vector<tf>& output, std::string& err) {
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kDownWidth) * kBlocksPerRow;
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    if (rows == 0u || rows > 8u || plan.n_entries != entry_count ||
        plan.n_groups == 0u || plan.n_groups > plan.n_entries ||
        entry_count > strata::metal::kTieredMoePlanGroups ||
        expert_slots.empty() || weights.empty() ||
        weights.size() % weight_blocks_per_expert != 0u ||
        activations.size() !=
            entry_count * (kInputWidth / 32u)) {
        err = "grouped Q2_0 MoE down oracle shape is invalid";
        return false;
    }
    const std::size_t slot_count = weights.size() / weight_blocks_per_expert;
    if (std::any_of(expert_slots.begin(), expert_slots.end(),
                    [slot_count](std::int32_t slot) {
                        return slot < -1 ||
                               (slot >= 0 &&
                                static_cast<std::size_t>(slot) >= slot_count);
                    })) {
        err = "grouped Q2_0 MoE down oracle expert slot is invalid";
        return false;
    }
    if (plan.grp_start[0] != 0 ||
        plan.grp_start[plan.n_groups] !=
            static_cast<std::int32_t>(entry_count)) {
        err = "grouped Q2_0 MoE down oracle plan endpoints are invalid";
        return false;
    }
    std::vector<bool> destination_seen(entry_count, false);
    for (std::uint32_t entry = 0u; entry < entry_count; ++entry) {
        const std::int32_t destination = plan.ent_dst[entry];
        if (destination < 0 ||
            static_cast<std::size_t>(destination) >= entry_count ||
            destination_seen[static_cast<std::size_t>(destination)]) {
            err = "grouped Q2_0 MoE down oracle destination map is invalid";
            return false;
        }
        destination_seen[static_cast<std::size_t>(destination)] = true;
    }

    output.assign(entry_count * kDownWidth, tf{});
    for (std::uint32_t group = 0u; group < plan.n_groups; ++group) {
        const std::int32_t start = plan.grp_start[group];
        const std::int32_t end = plan.grp_start[group + 1u];
        const std::int32_t expert = plan.grp_e[group];
        if (start < 0 || end <= start ||
            end > static_cast<std::int32_t>(entry_count) || expert < 0 ||
            static_cast<std::size_t>(expert) >= expert_slots.size()) {
            err = "grouped Q2_0 MoE down oracle plan group is invalid";
            return false;
        }
        const std::int32_t slot = expert_slots[static_cast<std::size_t>(expert)];
        if (slot < 0) {
            err = "grouped Q2_0 MoE down oracle plan names an unmapped expert";
            return false;
        }
        const std::size_t expert_base =
            static_cast<std::size_t>(slot) * weight_blocks_per_expert;
        for (std::uint32_t row = 0u; row < kDownWidth; ++row) {
            for (std::uint32_t first = static_cast<std::uint32_t>(start);
                 first < static_cast<std::uint32_t>(end); first += 4u) {
                const std::uint32_t count = std::min(
                    4u, static_cast<std::uint32_t>(end) - first);
                std::array<std::array<tf, 4>, 32> lane_sums{};
                for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                    for (std::uint32_t k = lane;
                         k < kBlocksPerRow * kPartsPerBlock; k += 32u) {
                        const std::uint32_t block = k / kPartsPerBlock;
                        const std::uint32_t part = k % kPartsPerBlock;
                        const block_q2_0& weight =
                            weights[expert_base +
                                    static_cast<std::size_t>(row) *
                                        kBlocksPerRow +
                                    block];
                        const float weight_scale =
                            strata::kernels::f32_from_f16(weight.d);
                        for (std::uint32_t column = 0u; column < count;
                             ++column) {
                            const std::size_t grouped_entry = first + column;
                            const Q81& activation = activations[
                                grouped_entry * (kInputWidth / 32u) +
                                block * kPartsPerBlock + part];
                            const int dot = integer_dot(weight, part, activation);
                            const std::uint32_t ds =
                                static_cast<std::uint32_t>(activation.ds);
                            const float activation_scale =
                                strata::kernels::f32_from_f16(
                                    static_cast<std::uint16_t>(ds & 0xffffu));
                            lane_sums[lane][column] = fma(
                                tf(weight_scale) * tf(activation_scale),
                                tf(static_cast<float>(dot)),
                                lane_sums[lane][column]);
                        }
                    }
                }
                for (std::uint32_t column = 0u; column < count; ++column) {
                    const std::size_t grouped_entry = first + column;
                    const std::size_t destination = static_cast<std::size_t>(
                        plan.ent_dst[grouped_entry]);
                    std::array<tf, 32> partials{};
                    for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                        partials[lane] = lane_sums[lane][column];
                    }
                    output[destination * kDownWidth + row] =
                        butterfly_sum(partials);
                }
            }
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
