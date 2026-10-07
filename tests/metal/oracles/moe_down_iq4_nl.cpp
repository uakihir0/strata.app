#include "oracles/moe_down_iq4_nl.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace twin {
namespace {

constexpr std::array<std::int8_t, 16> k_iq4_nl_values = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};
constexpr std::uint32_t kTopK = 10u;
constexpr std::uint32_t kDownWidth = 2560u;
constexpr std::uint32_t kInputWidth = 640u;
constexpr std::uint32_t kBlocksPerRow = kInputWidth / 32u;
constexpr std::uint32_t kPartsPerBlock = 2u;

std::array<int, 32u> decode_logical_values(const block_iq4_nl& weight) {
    std::array<int, 32u> values{};
    for (std::uint32_t position = 0u; position < values.size(); ++position) {
        const std::uint32_t byte = position & 0x0fu;
        const std::uint32_t shift = position < 16u ? 0u : 4u;
        const std::uint32_t code = (weight.qs[byte] >> shift) & 0x0fu;
        values[position] = k_iq4_nl_values[code];
    }
    return values;
}

int partial_integer_dot(const std::array<int, 32u>& values,
                        std::uint32_t part, const Q81& activation) {
    int sum = 0;
    const std::uint32_t first = part * 8u;
    for (std::uint32_t component = 0u; component < 8u; ++component) {
        const std::uint32_t low_position = first + component;
        const std::uint32_t high_position = 16u + low_position;
        sum += values[low_position] *
               static_cast<int>(activation.qs[low_position]);
        sum += values[high_position] *
               static_cast<int>(activation.qs[high_position]);
    }
    return sum;
}

}  // namespace

bool moe_down_iq4_nl_reference(
    std::span<const block_iq4_nl> weights,
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
        activations.size() != entry_count * kBlocksPerRow) {
        err = "grouped IQ4_NL MoE down oracle shape is invalid";
        return false;
    }
    const std::size_t slot_count = weights.size() / weight_blocks_per_expert;
    if (std::any_of(expert_slots.begin(), expert_slots.end(),
                    [slot_count](std::int32_t slot) {
                        return slot < -1 ||
                               (slot >= 0 &&
                                static_cast<std::size_t>(slot) >= slot_count);
                    })) {
        err = "grouped IQ4_NL MoE down oracle expert slot is invalid";
        return false;
    }
    if (plan.grp_start[0] != 0 ||
        plan.grp_start[plan.n_groups] !=
            static_cast<std::int32_t>(entry_count)) {
        err = "grouped IQ4_NL MoE down oracle plan endpoints are invalid";
        return false;
    }
    std::vector<bool> destination_seen(entry_count, false);
    for (std::uint32_t entry = 0u; entry < entry_count; ++entry) {
        const std::int32_t destination = plan.ent_dst[entry];
        if (destination < 0 ||
            static_cast<std::size_t>(destination) >= entry_count ||
            destination_seen[static_cast<std::size_t>(destination)]) {
            err = "grouped IQ4_NL MoE down oracle destination map is invalid";
            return false;
        }
        destination_seen[static_cast<std::size_t>(destination)] = true;
    }

    output.assign(entry_count * kDownWidth, tf{});
    std::uint32_t next_entry = 0u;
    for (std::uint32_t group = 0u; group < plan.n_groups; ++group) {
        const std::int32_t start = plan.grp_start[group];
        const std::int32_t end = plan.grp_start[group + 1u];
        const std::int32_t expert = plan.grp_e[group];
        if (start < 0 || static_cast<std::uint32_t>(start) != next_entry ||
            end <= start || end > static_cast<std::int32_t>(entry_count) ||
            expert < 0 ||
            static_cast<std::size_t>(expert) >= expert_slots.size()) {
            err = "grouped IQ4_NL MoE down oracle plan group is invalid";
            return false;
        }
        next_entry = static_cast<std::uint32_t>(end);
        const std::int32_t slot = expert_slots[static_cast<std::size_t>(expert)];
        if (slot < 0) {
            err =
                "grouped IQ4_NL MoE down oracle plan names an unmapped expert";
            return false;
        }
        const std::size_t expert_base =
            static_cast<std::size_t>(slot) * weight_blocks_per_expert;
        for (std::uint32_t row = 0u; row < kDownWidth; ++row) {
            for (std::uint32_t grouped_entry =
                     static_cast<std::uint32_t>(start);
                 grouped_entry < static_cast<std::uint32_t>(end);
                 ++grouped_entry) {
                tf sum{};
                for (std::uint32_t block = 0u; block < kBlocksPerRow;
                     ++block) {
                    const block_iq4_nl& weight =
                        weights[expert_base +
                                static_cast<std::size_t>(row) *
                                    kBlocksPerRow +
                                block];
                    const std::array<int, 32u> values =
                        decode_logical_values(weight);
                    const float weight_scale =
                        strata::kernels::f32_from_f16(weight.d);
                    const Q81& activation =
                        activations[grouped_entry * kBlocksPerRow + block];
                    const std::uint32_t ds =
                        static_cast<std::uint32_t>(activation.ds);
                    const float activation_scale =
                        strata::kernels::f32_from_f16(
                            static_cast<std::uint16_t>(ds & 0xffffu));
                    const tf scale =
                        tf(weight_scale) * tf(activation_scale);
                    for (std::uint32_t part = 0u;
                         part < kPartsPerBlock; ++part) {
                        const int dot =
                            partial_integer_dot(values, part, activation);
                        sum = fma(scale, tf(static_cast<float>(dot)), sum);
                    }
                }
                const std::size_t destination = static_cast<std::size_t>(
                    plan.ent_dst[grouped_entry]);
                output[destination * kDownWidth + row] = sum;
            }
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
