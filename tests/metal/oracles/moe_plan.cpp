#include "oracles/moe_plan.hpp"

#include <algorithm>
#include <vector>

namespace twin {

bool moe_plan_reference(std::span<const std::int32_t> ids,
                        std::uint32_t rows, std::uint32_t experts,
                        bool tiered, std::uint32_t layer,
                        std::span<const std::int32_t> resident,
                        std::span<const float> initial_usage,
                        strata::metal::TieredMoePlan& plan,
                        std::vector<float>& expected_usage,
                        std::string& err) {
    if (rows == 0u || rows > 8u ||
        (experts != 256u && experts != 512u) ||
        layer >= strata::metal::kTieredMaxLayers) {
        err = "MoE plan shape is invalid";
        return false;
    }
    const std::size_t entries = static_cast<std::size_t>(rows) * 10u;
    const std::size_t table_size =
        (static_cast<std::size_t>(layer) + 1u) * experts;
    if (ids.size() != entries || resident.size() < table_size ||
        initial_usage.size() < table_size) {
        err = "MoE plan table size is invalid";
        return false;
    }
    if (std::any_of(ids.begin(), ids.end(), [experts](std::int32_t expert) {
            return expert < 0 || static_cast<std::uint32_t>(expert) >= experts;
        })) {
        err = "MoE plan expert id is out of range";
        return false;
    }

    plan = {};
    expected_usage.assign(initial_usage.begin(), initial_usage.end());
    std::vector<std::int32_t> group_for_expert(experts, -1);
    std::vector<std::int32_t> group_experts;
    std::vector<std::vector<std::int32_t>> group_entries;
    group_experts.reserve(entries);
    group_entries.reserve(entries);

    for (std::size_t entry = 0; entry < entries; ++entry) {
        const std::uint32_t expert = static_cast<std::uint32_t>(ids[entry]);
        std::int32_t group = group_for_expert[expert];
        if (group < 0) {
            group = static_cast<std::int32_t>(group_experts.size());
            group_for_expert[expert] = group;
            group_experts.push_back(static_cast<std::int32_t>(expert));
            group_entries.emplace_back();
        }
        group_entries[static_cast<std::size_t>(group)].push_back(
            static_cast<std::int32_t>(entry));
    }

    plan.n_groups = static_cast<std::uint32_t>(group_experts.size());
    plan.n_entries = static_cast<std::uint32_t>(entries);
    std::uint32_t destination = 0u;
    const std::size_t layer_offset =
        static_cast<std::size_t>(layer) * experts;
    for (std::size_t group = 0; group < group_experts.size(); ++group) {
        const std::int32_t expert = group_experts[group];
        const std::size_t expert_index =
            layer_offset + static_cast<std::uint32_t>(expert);
        const bool is_resident =
            !tiered || resident[expert_index] >= 0;
        plan.grp_e[group] = expert;
        plan.grp_res[group] = is_resident ? 1 : 0;
        plan.grp_start[group] = static_cast<std::int32_t>(destination);
        if (!is_resident) ++plan.n_miss;
        for (const std::int32_t entry : group_entries[group]) {
            plan.ent_dst[destination] = entry;
            plan.ent_tok[destination] = entry / 10;
            ++destination;
        }
    }
    plan.grp_start[group_experts.size()] =
        static_cast<std::int32_t>(destination);
    plan.skip = plan.n_miss == 0u ? 1u : 0u;

    if (tiered) {
        for (const std::int32_t expert : ids) {
            expected_usage[layer_offset +
                          static_cast<std::uint32_t>(expert)] += 1.0f;
        }
    }

    err.clear();
    return true;
}

}  // namespace twin
