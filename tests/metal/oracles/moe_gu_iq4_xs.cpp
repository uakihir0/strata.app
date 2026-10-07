#include "oracles/moe_gu_iq4_xs.hpp"

#include <algorithm>

namespace twin {
namespace {

constexpr std::uint32_t kTopK = 10u;
constexpr std::uint32_t kExpertWidth = 640u;
constexpr std::uint32_t kInputWidth = 2560u;
constexpr std::uint32_t kBlocksPerRow = kInputWidth / 256u;

}  // namespace

bool moe_gu_iq4_xs_reference(
    std::span<const block_iq4_xs> gate_weights,
    std::span<const block_iq4_xs> up_weights,
    std::span<const Q81> activations,
    std::span<const std::int32_t> expert_slots,
    const strata::metal::TieredMoePlan& plan, std::uint32_t rows,
    std::vector<tf>& gate, std::vector<tf>& up, std::string& err) {
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kExpertWidth) * kBlocksPerRow;
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    if (rows == 0u || rows > 8u || plan.n_entries != entry_count ||
        plan.n_groups == 0u || plan.n_groups > plan.n_entries ||
        plan.n_entries > strata::metal::kTieredMoePlanGroups ||
        expert_slots.empty() || gate_weights.empty() ||
        gate_weights.size() % blocks_per_expert != 0u ||
        up_weights.size() != gate_weights.size() ||
        activations.size() !=
            static_cast<std::size_t>(rows) * (kInputWidth / 32u)) {
        err = "grouped IQ4_XS MoE oracle shape is invalid";
        return false;
    }
    const std::size_t slot_count = gate_weights.size() / blocks_per_expert;
    if (std::any_of(expert_slots.begin(), expert_slots.end(),
                    [slot_count](std::int32_t slot) {
                        return slot < -1 ||
                               (slot >= 0 &&
                                static_cast<std::size_t>(slot) >= slot_count);
                    })) {
        err = "grouped IQ4_XS MoE oracle expert slot is invalid";
        return false;
    }
    if (plan.grp_start[0] != 0 ||
        plan.grp_start[plan.n_groups] !=
            static_cast<std::int32_t>(entry_count)) {
        err = "grouped IQ4_XS MoE oracle plan endpoints are invalid";
        return false;
    }

    gate.assign(entry_count * kExpertWidth, tf{});
    up.assign(entry_count * kExpertWidth, tf{});
    const std::uint32_t oracle_columns =
        rows <= 1u ? 1u : (rows <= 2u ? 2u : (rows <= 4u ? 4u : 8u));
    std::vector<Q81> oracle_activations(
        static_cast<std::size_t>(oracle_columns) * (kInputWidth / 32u));
    for (std::uint32_t token = 0u; token < oracle_columns; ++token) {
        const std::uint32_t source_token = std::min(token, rows - 1u);
        std::copy_n(
            activations.begin() +
                static_cast<std::size_t>(source_token) * (kInputWidth / 32u),
            kInputWidth / 32u,
            oracle_activations.begin() +
                static_cast<std::size_t>(token) * (kInputWidth / 32u));
    }

    std::vector<tf> per_token;
    for (std::uint32_t group = 0u; group < plan.n_groups; ++group) {
        const std::int32_t start = plan.grp_start[group];
        const std::int32_t end = plan.grp_start[group + 1u];
        const std::int32_t expert = plan.grp_e[group];
        if (start < 0 || end <= start ||
            end > static_cast<std::int32_t>(entry_count) || expert < 0 ||
            static_cast<std::size_t>(expert) >= expert_slots.size()) {
            err = "grouped IQ4_XS MoE oracle plan group is invalid";
            return false;
        }
        const std::int32_t slot = expert_slots[static_cast<std::size_t>(expert)];
        if (slot < 0) {
            err = "grouped IQ4_XS MoE oracle plan names an unmapped expert";
            return false;
        }
        const std::size_t expert_base =
            static_cast<std::size_t>(slot) * blocks_per_expert;

        for (std::uint32_t role = 0u; role < 2u; ++role) {
            const std::span<const block_iq4_xs> weights =
                role == 0u ? gate_weights : up_weights;
            std::vector<tf>& output = role == 0u ? gate : up;
            if (!iq4_xs_matvec_tf_reference(
                    weights.subspan(expert_base, blocks_per_expert),
                    oracle_activations, kInputWidth, kExpertWidth,
                    oracle_columns, rows, per_token, err)) {
                err = "grouped IQ4_XS MoE row oracle failed: " + err;
                return false;
            }
            for (std::uint32_t entry = static_cast<std::uint32_t>(start);
                 entry < static_cast<std::uint32_t>(end); ++entry) {
                const std::int32_t token = plan.ent_tok[entry];
                if (token < 0 || static_cast<std::uint32_t>(token) >= rows) {
                    err = "grouped IQ4_XS MoE oracle token is out of range";
                    return false;
                }
                std::copy_n(
                    per_token.begin() +
                        static_cast<std::size_t>(token) * kExpertWidth,
                    kExpertWidth,
                    output.begin() +
                        static_cast<std::size_t>(entry) * kExpertWidth);
            }
        }
    }
    err.clear();
    return true;
}

}  // namespace twin
