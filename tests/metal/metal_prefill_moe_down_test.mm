// GPU twin for isolated Prefill Q2_0/IQ4_NL MoE down tiles (docs/mac/04 §6.6).
#include "metallib_sha256.inc"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_moe_down_dispatch.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"
#include "../../src/metal/kernels/common/types.h"
#include "twin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kHidden = 640u;
constexpr std::uint32_t kExperts = 256u;
constexpr std::uint32_t kSecondExpert = 3u;
constexpr std::uint32_t kLayer = 2u;
constexpr std::size_t kQ2BlocksPerRow = 10u;
constexpr std::size_t kIQ4NLBlocksPerRow = 20u;
constexpr std::size_t kQ2BlockBytes = 18u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint8_t kGuardPoison = 0xffu;

enum class WeightFormat { q2_0, iq4_nl };
enum class DestinationCorruption { none, out_of_range, duplicate };

constexpr std::array<std::int8_t, 16u> kIQ4NLCodebook{
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

struct PfPlan {
    std::uint32_t n_groups = 0u;
    std::uint32_t n_entries = 0u;
    std::uint32_t n_miss = 0u;
    std::uint32_t skip = 0u;
    std::uint32_t n_tiles = 0u;
    std::uint32_t bad = 0u;
    std::uint32_t pad[2]{};
    std::int32_t grp_e[512]{};
    std::int32_t grp_res[512]{};
    std::int32_t grp_start[513]{};
};
static_assert(sizeof(PfPlan) == kPrefillMoePlanBytes);

struct Tile {
    std::uint32_t group;
    std::uint32_t first_entry;
};
static_assert(sizeof(Tile) == 8u);

struct PrefillMoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(PrefillMoeExpertAddress) == 24u);

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

template <class T>
std::span<T> typed(twin::GuardedBuffer& buffer) {
    return {reinterpret_cast<T*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(T)};
}

template <class T>
std::span<const T> typed(const twin::GuardedBuffer& buffer) {
    return {reinterpret_cast<const T*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(T)};
}

bool guards_intact(const twin::GuardedBuffer& buffer) {
    const auto allocation = buffer.allocation();
    const auto payload = buffer.payload();
    const std::size_t prefix = twin::GuardedBuffer::kGuardBytes;
    const std::size_t suffix = prefix + payload.size();
    return allocation.size() >= suffix &&
           std::all_of(allocation.begin(), allocation.begin() + prefix,
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       }) &&
           std::all_of(allocation.begin() + suffix, allocation.end(),
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       });
}

bool tail_is_poison(const twin::GuardedBuffer& buffer,
                    std::size_t active_bytes) {
    const auto payload = buffer.payload();
    return active_bytes <= payload.size() &&
           std::all_of(payload.begin() + active_bytes, payload.end(),
                       [](std::uint8_t byte) { return byte == kPoison; });
}

std::uint16_t half(float value) {
    return strata::kernels::f16_from_f32(value);
}

float f32(std::uint16_t value) {
    return strata::kernels::f32_from_f16(value);
}

float route_scale(std::size_t route) {
    return static_cast<float>((route % 4u) + 1u) * 0.125f;
}

float base_value(std::size_t k) {
    const int centered = static_cast<int>(k % 31u) - 15;
    return static_cast<float>(centered) * (1.0f / 64.0f);
}

void store_code(block_q2_0& block, std::size_t k, std::uint8_t code) {
    const std::size_t byte = k / 4u;
    const std::size_t shift = 2u * (k % 4u);
    block.qs[byte] = static_cast<std::uint8_t>(
        (block.qs[byte] & static_cast<std::uint8_t>(~(3u << shift))) |
        static_cast<std::uint8_t>(code << shift));
}

void store_iq4_nl_code(block_iq4_nl& block, std::size_t k,
                       std::uint8_t code) {
    const std::size_t byte = k & 15u;
    const std::size_t shift = (k >> 4u) * 4u;
    block.qs[byte] = static_cast<std::uint8_t>(
        (block.qs[byte] & static_cast<std::uint8_t>(~(0x0fu << shift))) |
        static_cast<std::uint8_t>(code << shift));
}

void fill_q2_weights(twin::GuardedBuffer& buffer, std::uint32_t expert) {
    auto blocks = typed<block_q2_0>(buffer);
    for (std::uint32_t row = 0u; row < kWidth; ++row) {
        for (std::size_t block_index = 0u;
             block_index < kQ2BlocksPerRow; ++block_index) {
            block_q2_0& block =
                blocks[static_cast<std::size_t>(row) * kQ2BlocksPerRow +
                       block_index];
            block.d = 0x2800u;  // F16 0.03125.
            std::fill(std::begin(block.qs), std::end(block.qs), 0u);
            for (std::size_t k = 0u; k < 64u; ++k) {
                const std::size_t global_k = block_index * 64u + k;
                const std::uint8_t code = static_cast<std::uint8_t>(
                    (row * 7u + global_k * 5u + expert * 3u +
                     global_k / 64u) %
                    4u);
                store_code(block, k, code);
            }
        }
    }
}

void fill_iq4_nl_weights(twin::GuardedBuffer& buffer,
                          std::uint32_t expert) {
    auto blocks = typed<block_iq4_nl>(buffer);
    for (std::uint32_t row = 0u; row < kWidth; ++row) {
        for (std::size_t block_index = 0u;
             block_index < kIQ4NLBlocksPerRow; ++block_index) {
            block_iq4_nl& block =
                blocks[static_cast<std::size_t>(row) * kIQ4NLBlocksPerRow +
                       block_index];
            block.d = block_index % 2u == 0u
                          ? 0x3a66u  // F16 0.7998046875.
                          : 0x2800u; // F16 0.03125.
            std::fill(std::begin(block.qs), std::end(block.qs), 0u);
            for (std::size_t k = 0u; k < 32u; ++k) {
                const std::size_t global_k = block_index * 32u + k;
                const std::uint8_t code = static_cast<std::uint8_t>(
                    (row * 7u + global_k * 5u + expert * 3u +
                     global_k / 32u) %
                    16u);
                store_iq4_nl_code(block, k, code);
            }
        }
    }
}

void fill_weights(twin::GuardedBuffer& buffer, std::uint32_t expert,
                  WeightFormat format) {
    if (format == WeightFormat::q2_0) {
        fill_q2_weights(buffer, expert);
    } else {
        fill_iq4_nl_weights(buffer, expert);
    }
}

double dot_q2_oracle(std::span<const block_q2_0> weights, std::size_t row) {
    double result = 0.0;
    for (std::size_t k = 0u; k < kHidden; ++k) {
        const block_q2_0& block =
            weights[row * kQ2BlocksPerRow + k / 64u];
        const std::uint8_t packed = block.qs[(k % 64u) / 4u];
        const int code =
            static_cast<int>((packed >> (2u * (k % 4u))) & 3u) - 1;
        const float decoded =
            f32(half(f32(block.d) * static_cast<float>(code)));
        result += static_cast<double>(base_value(k)) * decoded;
    }
    return result;
}

double dot_iq4_nl_oracle(std::span<const block_iq4_nl> weights,
                         std::size_t row,
                         bool round_weights_to_f16 = true) {
    double result = 0.0;
    for (std::size_t k = 0u; k < kHidden; ++k) {
        const std::size_t within_block = k % 32u;
        const block_iq4_nl& block =
            weights[row * kIQ4NLBlocksPerRow + k / 32u];
        const std::uint8_t packed = block.qs[within_block & 15u];
        const std::size_t shift = (within_block >> 4u) * 4u;
        const std::uint8_t code =
            static_cast<std::uint8_t>((packed >> shift) & 0x0fu);
        const float dequantized =
            f32(block.d) * static_cast<float>(kIQ4NLCodebook[code]);
        const float decoded = round_weights_to_f16
                                  ? f32(half(dequantized))
                                  : dequantized;
        result += static_cast<double>(base_value(k)) * decoded;
    }
    return result;
}

double dot_oracle(twin::GuardedBuffer& weights, WeightFormat format,
                  std::size_t row) {
    return format == WeightFormat::q2_0
               ? dot_q2_oracle(typed<block_q2_0>(weights), row)
               : dot_iq4_nl_oracle(typed<block_iq4_nl>(weights), row);
}

bool encode_down(Context& context, Encoder& encoder, WeightFormat format,
                 std::uint32_t rows, const PrefillMoeDownTileBuffers& buffers,
                 std::string& error) {
    if (format == WeightFormat::q2_0) {
        return encode_prefill_moe_down_q2_0(
            context, encoder, rows, kExperts, kLayer, buffers, error);
    }
    return encode_prefill_moe_down_iq4_nl(
        context, encoder, rows, kExperts, kLayer, buffers, error);
}

bool encode_down(Context& context, CommandBuffer& command_buffer,
                 WeightFormat format, std::uint32_t rows,
                 const PrefillMoeDownTileBuffers& buffers,
                 std::string& error) {
    if (format == WeightFormat::q2_0) {
        return encode_prefill_moe_down_q2_0(
            context, command_buffer, rows, kExperts, kLayer, buffers, error);
    }
    return encode_prefill_moe_down_iq4_nl(
        context, command_buffer, rows, kExperts, kLayer, buffers, error);
}

std::size_t round_up_div(std::size_t value, std::size_t divisor) {
    return value / divisor +
           static_cast<std::size_t>(value % divisor != 0u);
}

bool run_case(Context& context, WeightFormat format, std::uint32_t rows,
              bool caller_owned,
                  bool missing_second_expert = false,
                  bool malformed_tile = false,
                  DestinationCorruption destination_corruption =
                  DestinationCorruption::none) {
    const std::size_t entries = static_cast<std::size_t>(rows) * 10u;
    const std::string format_name =
        format == WeightFormat::q2_0 ? "Q2_0" : "IQ4_NL";
    const std::string label =
        "Prefill " + format_name + " down tile T=" + std::to_string(rows) +
        (caller_owned ? " caller-owned" : " standalone") +
        (missing_second_expert ? " missing-address" : "") +
        (malformed_tile ? " malformed-tile" : "") +
        (destination_corruption == DestinationCorruption::out_of_range
             ? " out-of-range-destination"
             : destination_corruption == DestinationCorruption::duplicate
                   ? " duplicate-destination"
                   : "");
    const std::size_t route_bytes = entries * sizeof(std::int32_t);
    const std::size_t mixed_bytes =
        static_cast<std::size_t>(rows) * kWidth * sizeof(std::uint16_t);
    const std::size_t gathered_bytes =
        entries * kWidth * sizeof(std::uint16_t);
    const std::size_t hidden_bytes =
        entries * kHidden * sizeof(std::uint16_t);
    const std::size_t histogram_bytes =
        round_up_div(entries, 256u) * 512u * sizeof(std::uint32_t);
    const std::size_t grouping_tile_capacity =
        round_up_div(entries, 64u) + 512u;
    const std::size_t tile_capacity =
        round_up_div(entries, 64u) + std::min<std::size_t>(kExperts, entries);
    const std::size_t weight_blocks_per_row =
        format == WeightFormat::q2_0 ? kQ2BlocksPerRow
                                     : kIQ4NLBlocksPerRow;
    const std::size_t plan_table_bytes =
        (static_cast<std::size_t>(kLayer) + 1u) *
        kPrefillMoePlanStrideBytes;
    const std::size_t address_table_bytes =
        (static_cast<std::size_t>(kLayer) + 1u) * kExperts *
        sizeof(PrefillMoeExpertAddress);
    const std::size_t weight_bytes =
        kWidth * weight_blocks_per_row * kQ2BlockBytes;
    const std::size_t parts_bytes =
        entries * kWidth * sizeof(float);

    twin::GuardedBuffer ids, resident, mixed, histogram, plans, tiles;
    twin::GuardedBuffer source, destination, gathered, addresses;
    twin::GuardedBuffer address_snapshot, weights0, weights3, hidden;
    twin::GuardedBuffer parts, status, alias_storage;
    std::string error;
    const std::string weight0_label =
        label + " expert 0 down " + format_name;
    const std::string weight3_label =
        label + " expert 3 down " + format_name;
    const bool allocated =
        ids.allocate(context, label + " IDs", route_bytes, error) &&
        resident.allocate(context, label + " residency",
                          (static_cast<std::size_t>(kLayer) + 1u) * kExperts *
                              sizeof(std::int32_t),
                          error) &&
        mixed.allocate(context, label + " mixed input", mixed_bytes, error) &&
        histogram.allocate(context, label + " histogram", histogram_bytes,
                           error) &&
        plans.allocate(context, label + " PfPlan table", plan_table_bytes,
                       error) &&
        tiles.allocate(context, label + " tile list",
                       grouping_tile_capacity * sizeof(Tile), error) &&
        source.allocate(context, label + " source rows", route_bytes, error) &&
        destination.allocate(context, label + " destination rows",
                             route_bytes, error) &&
        gathered.allocate(context, label + " gathered input",
                          gathered_bytes, error) &&
        addresses.allocate(context, label + " expert addresses",
                           address_table_bytes, error) &&
        address_snapshot.allocate(
            context, label + " address snapshot",
            kExperts * sizeof(PrefillMoeExpertAddress), error) &&
        weights0.allocate(context, weight0_label, weight_bytes, error) &&
        weights3.allocate(context, weight3_label, weight_bytes, error) &&
        hidden.allocate(context, label + " grouped F16 hidden",
                        hidden_bytes, error) &&
        parts.allocate(context, label + " down parts", parts_bytes + 64u,
                       error) &&
        status.allocate(context, label + " down status", 64u, error) &&
        alias_storage.allocate(context, label + " alias storage",
                               parts_bytes + 64u, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    for (twin::GuardedBuffer* buffer :
         {&histogram, &plans, &tiles, &source, &destination, &gathered,
          &hidden, &parts, &status}) {
        std::fill(buffer->payload().begin(), buffer->payload().end(), kPoison);
    }
    std::fill(typed<std::int32_t>(ids).begin(),
              typed<std::int32_t>(ids).end(), 0);
    std::fill(typed<std::int32_t>(resident).begin(),
              typed<std::int32_t>(resident).end(), 0);
    std::fill(addresses.payload().begin(), addresses.payload().end(), 0u);
    std::fill(mixed.payload().begin(), mixed.payload().end(), 0u);
    fill_weights(weights0, 0u, format);
    fill_weights(weights3, kSecondExpert, format);
    if (format == WeightFormat::iq4_nl && rows == 1u && !caller_owned) {
        const auto iq4_weights = typed<block_iq4_nl>(weights0);
        const double rounded = dot_iq4_nl_oracle(iq4_weights, 0u);
        const double unrounded =
            dot_iq4_nl_oracle(iq4_weights, 0u, false);
        const double tolerance = 2.0e-5 + std::abs(rounded) * 2.0e-5;
        check(std::abs(rounded - unrounded) > tolerance,
              label + " IQ4_NL fixture detects F16 weight rounding",
              "rounded=" + std::to_string(rounded) +
                  " unrounded=" + std::to_string(unrounded));
    }

    auto route_ids = typed<std::int32_t>(ids);
    for (std::size_t route = 0u; route < entries; ++route) {
        const bool first_expert =
            entries > 64u ? route < 65u : route % 2u == 0u;
        route_ids[route] =
            first_expert ? 0 : static_cast<std::int32_t>(kSecondExpert);
    }
    std::vector<std::size_t> grouped_to_route;
    grouped_to_route.reserve(entries);
    for (const std::int32_t expert_id :
         {0, static_cast<std::int32_t>(kSecondExpert)}) {
        for (std::size_t route = 0u; route < entries; ++route) {
            if (route_ids[route] == expert_id) grouped_to_route.push_back(route);
        }
    }
    auto grouped_hidden = typed<std::uint16_t>(hidden);
    for (std::size_t grouped = 0u; grouped < entries; ++grouped) {
        const float scale = route_scale(grouped_to_route[grouped]);
        for (std::size_t k = 0u; k < kHidden; ++k) {
            grouped_hidden[grouped * kHidden + k] =
                half(base_value(k) * scale);
        }
    }

    auto address_table = typed<PrefillMoeExpertAddress>(addresses);
    for (std::uint32_t layer = 0u; layer <= kLayer; ++layer) {
        auto& expert0 = address_table[
            static_cast<std::size_t>(layer) * kExperts];
        auto& expert3 = address_table[
            static_cast<std::size_t>(layer) * kExperts + kSecondExpert];
        expert0.role[2] = weights0.view().gpu_address;
        if (!missing_second_expert) {
            expert3.role[2] = weights3.view().gpu_address;
        }
    }

    const auto plan_slot = plans.view().slice(
        static_cast<std::size_t>(kLayer) * kPrefillMoePlanStrideBytes,
        kPrefillMoePlanStrideBytes);
    const PrefillMoeGroupingBuffers grouping{
        context.buffers().read_only(ids.view()),
        context.buffers().read_only(resident.view()),
        context.buffers().read_only(mixed.view()),
        histogram.view(),
        plan_slot,
        tiles.view(),
        source.view(),
        destination.view(),
        gathered.view(),
    };
    const std::array<PrefillMoeWeightStorageView, 2u> weight_storage{{
        {context.buffers().read_only(weights0.view())},
        {context.buffers().read_only(weights3.view())},
    }};
    const PrefillMoeDownTileBuffers down{
        context.buffers().read_only(addresses.view()),
        address_snapshot.view(),
        weight_storage,
        nullptr,
        context.buffers().read_only(hidden.view().slice(0u, hidden_bytes)),
        context.buffers().read_only(plan_slot),
        context.buffers().read_only(
            tiles.view().slice(0u, tile_capacity * sizeof(Tile))),
        context.buffers().read_only(
            destination.view().slice(0u, route_bytes)),
        parts.view(),
        status.view(),
    };

    if (caller_owned) {
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) return false;
        const bool encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_moe_grouping(
                           context, encoder, rows, kExperts, false, kLayer,
                           grouping, encode_error) &&
                       encode_down(context, encoder, format, rows, down,
                                   encode_error);
            },
            error);
        check(encoded, label + " composed encode", error);
        if (!encoded) return false;
        const bool completed =
            command->commit(error) && command->wait_until_completed(error);
        check(completed, label + " GPU completion", error);
        if (!completed) return false;
    } else {
        auto grouping_command =
            context.make_command_buffer(label + " grouping", error);
        check(grouping_command != nullptr,
              label + " grouping command buffer", error);
        if (!grouping_command) return false;
        bool completed =
            encode_prefill_moe_grouping(
                context, *grouping_command, rows, kExperts, false, kLayer,
                grouping, error) &&
            grouping_command->commit(error) &&
            grouping_command->wait_until_completed(error);
        check(completed, label + " grouping completion", error);
        if (!completed) return false;

        if (malformed_tile) {
            auto tile_values = typed<Tile>(tiles);
            tile_values[0].first_entry += 1u;
        }
        if (destination_corruption != DestinationCorruption::none) {
            auto destination_values = typed<std::int32_t>(destination);
            destination_values[1] =
                destination_corruption == DestinationCorruption::out_of_range
                    ? static_cast<std::int32_t>(entries)
                    : destination_values[0];
        }

        auto down_command =
            context.make_command_buffer(label + " down", error);
        check(down_command != nullptr, label + " down command buffer", error);
        if (!down_command) return false;
        completed =
            encode_down(context, *down_command, format, rows, down, error) &&
            down_command->commit(error) &&
            down_command->wait_until_completed(error);
        check(completed, label + " down completion", error);
        if (!completed) return false;
    }

    const auto* plan = reinterpret_cast<const PfPlan*>(plan_slot.contents);
    const std::size_t first_group_entries = static_cast<std::size_t>(
        std::count(route_ids.begin(), route_ids.end(), 0));
    const bool plan_valid =
        plan->bad == 0u && plan->n_entries == entries &&
        plan->n_groups == 2u && plan->grp_e[0] == 0 &&
        plan->grp_e[1] == static_cast<std::int32_t>(kSecondExpert) &&
        plan->grp_start[0] == 0 &&
        plan->grp_start[1] ==
            static_cast<std::int32_t>(first_group_entries) &&
        plan->grp_start[2] == static_cast<std::int32_t>(entries) &&
        plan->n_tiles ==
            round_up_div(first_group_entries, 64u) +
                round_up_div(entries - first_group_entries, 64u);
    check(plan_valid, label + " grouping plan has two ordered expert groups");

    const std::uint32_t actual_status = typed<std::uint32_t>(status)[0];
    const bool invalid_destination =
        destination_corruption != DestinationCorruption::none;
    const std::uint32_t expected_status =
        malformed_tile || invalid_destination
            ? 2u
            : (missing_second_expert ? 1u : 0u);
    check(actual_status == expected_status,
          label + " stage status reports missing weight or malformed tile",
          "actual=" + std::to_string(actual_status) +
              " expected=" + std::to_string(expected_status));

    auto actual_parts = typed<float>(parts);
    bool outputs_match = true;
    double max_abs_error = 0.0;
    std::vector<double> base_output(2u * kWidth);
    for (std::size_t output = 0u; output < kWidth; ++output) {
        base_output[output] = dot_oracle(weights0, format, output);
        base_output[kWidth + output] = dot_oracle(weights3, format, output);
    }
    const auto actual_destinations = typed<std::int32_t>(destination);
    for (std::size_t grouped = 0u; grouped < entries; ++grouped) {
        const std::size_t route = grouped_to_route[grouped];
        const std::int32_t actual_route = actual_destinations[grouped];
        if (!invalid_destination &&
            actual_route != static_cast<std::int32_t>(route)) {
            outputs_match = false;
            continue;
        }
        if (invalid_destination) {
            for (std::size_t output = 0u; output < kWidth; ++output) {
                if (actual_parts[route * kWidth + output] != 0.0f) {
                    outputs_match = false;
                }
            }
            continue;
        }
        const std::size_t expert_slot =
            route_ids[route] == 0 ? 0u : 1u;
        for (std::size_t output = 0u; output < kWidth; ++output) {
            const float actual =
                actual_parts[route * kWidth + output];
            const double expected =
                malformed_tile || invalid_destination ||
                        (missing_second_expert && expert_slot == 1u)
                    ? 0.0
                    : base_output[expert_slot * kWidth + output] *
                          static_cast<double>(route_scale(route));
            const double abs_error =
                std::abs(static_cast<double>(actual) - expected);
            max_abs_error = std::max(max_abs_error, abs_error);
            const double tolerance = 2.0e-5 + std::abs(expected) * 2.0e-5;
            if (!std::isfinite(actual) || abs_error > tolerance) {
                outputs_match = false;
            }
        }
    }
    check(outputs_match,
          label + " route-restored F32 down parts match independent scalar "
                  "format-specific dot oracle",
          "max_abs_error=" + std::to_string(max_abs_error));

    const bool tails_ok =
        tail_is_poison(parts, parts_bytes) &&
        tail_is_poison(status, sizeof(std::uint32_t));
    const bool guards_ok =
        guards_intact(ids) && guards_intact(resident) && guards_intact(mixed) &&
        guards_intact(histogram) && guards_intact(plans) && guards_intact(tiles) &&
        guards_intact(source) && guards_intact(destination) &&
        guards_intact(gathered) && guards_intact(addresses) &&
        guards_intact(address_snapshot) && guards_intact(weights0) &&
        guards_intact(weights3) && guards_intact(hidden) &&
        guards_intact(parts) && guards_intact(status);
    check(tails_ok, label + " output and status tails remain poisoned");
    check(guards_ok, label + " all fixture guards remain intact");

    const auto before_alias = std::vector<std::uint8_t>(
        alias_storage.payload().begin(), alias_storage.payload().end());
    auto alias_down = down;
    alias_down.hidden = context.buffers().read_only(
        alias_storage.view().slice(0u, hidden_bytes));
    alias_down.parts = alias_storage.view().slice(0u, parts_bytes);
    auto alias_command =
        context.make_command_buffer(label + " alias rejection", error);
    check(alias_command != nullptr,
          label + " alias rejection command buffer", error);
    if (!alias_command) return false;
    const bool alias_rejected =
        !encode_down(context, *alias_command, format, rows, alias_down, error) &&
        error.find("overlaps") != std::string::npos &&
        alias_command->unsafe_native_handle() == nullptr &&
        std::equal(before_alias.begin(), before_alias.end(),
                   alias_storage.payload().begin());
    check(alias_rejected, label + " rejects active hidden/output alias", error);

    return plan_valid && actual_status == expected_status && outputs_match &&
           tails_ok && guards_ok && alias_rejected;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_moe_down_test strata.metallib\n");
        return 2;
    }
    std::string error;
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;

    bool passed = true;
    for (const WeightFormat format :
         std::array{WeightFormat::q2_0, WeightFormat::iq4_nl}) {
        passed &= run_case(*context, format, 1u, false);
        passed &= run_case(*context, format, 7u, true);
        passed &= run_case(*context, format, 7u, false, true);
        passed &= run_case(*context, format, 7u, false, false, true);
        passed &= run_case(*context, format, 7u, false, false, false,
                           DestinationCorruption::out_of_range);
        passed &= run_case(*context, format, 7u, false, false, false,
                           DestinationCorruption::duplicate);
    }
    std::printf("%s Prefill down tile GPU twin (%d failures)\n",
                passed ? "PASS" : "FAIL", g_failures);
    return passed && g_failures == 0 ? 0 : 1;
}
