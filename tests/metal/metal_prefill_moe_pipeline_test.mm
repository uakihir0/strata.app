// Composed full-mode Prefill MoE router, grouping and gather twin
// (docs/mac/04 §6.6).
#include "metallib_sha256.inc"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"
#include "strata/metal/prefill_moe_pipeline.hpp"
#include "twin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kTopK = 10u;
constexpr std::uint32_t kExperts = 512u;
constexpr std::size_t kPlanStride = 6400u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint8_t kGuardPoison = 0xffu;

struct PfPlan {
    std::uint32_t n_groups = 0u;
    std::uint32_t n_entries = 0u;
    std::uint32_t n_miss = 0u;
    std::uint32_t skip = 0u;
    std::uint32_t n_tiles = 0u;
    std::uint32_t bad = 0u;
    std::uint32_t pad[2]{};
    std::int32_t grp_e[kExperts]{};
    std::int32_t grp_res[kExperts]{};
    std::int32_t grp_start[kExperts + 1u]{};
};
static_assert(sizeof(PfPlan) == kPrefillMoePlanBytes);

struct Tile {
    std::uint32_t group;
    std::uint32_t first_entry;
};
static_assert(sizeof(Tile) == 8u);

struct ExpectedGrouping {
    PfPlan plan{};
    std::vector<std::uint32_t> histogram;
    std::vector<Tile> tiles;
    std::vector<std::int32_t> source_rows;
    std::vector<std::int32_t> destination_rows;
    std::vector<std::uint16_t> gathered_rows;
};

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
    const std::size_t suffix = twin::GuardedBuffer::kGuardBytes +
                               payload.size();
    return allocation.size() >= suffix &&
           std::all_of(allocation.begin(),
                       allocation.begin() +
                           twin::GuardedBuffer::kGuardBytes,
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       }) &&
           std::all_of(allocation.begin() + suffix, allocation.end(),
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       });
}

std::size_t round_up_div(std::size_t value, std::size_t divisor) {
    return value / divisor +
           static_cast<std::size_t>(value % divisor != 0u);
}

ExpectedGrouping make_expected_grouping(
    std::uint32_t rows, std::uint32_t experts,
    std::span<const std::int32_t> ids,
    std::span<const std::uint16_t> mixed_h) {
    const std::size_t entries = static_cast<std::size_t>(rows) * kTopK;
    const std::size_t count_tiles = round_up_div(entries, 256u);
    ExpectedGrouping expected;
    expected.plan.n_entries = static_cast<std::uint32_t>(entries);
    std::fill_n(expected.plan.grp_e, kExperts, -1);
    std::fill_n(expected.plan.grp_res, kExperts, -1);
    expected.histogram.resize(count_tiles * kExperts);
    expected.source_rows.reserve(entries);
    expected.destination_rows.reserve(entries);
    std::vector<std::vector<std::uint32_t>> grouped(experts);

    for (std::size_t entry = 0u; entry < entries; ++entry) {
        const std::int32_t expert = ids[entry];
        if (expert >= 0 && static_cast<std::uint32_t>(expert) < experts) {
            ++expected.histogram[(entry / 256u) * kExperts +
                                 static_cast<std::uint32_t>(expert)];
            grouped[static_cast<std::size_t>(expert)].push_back(
                static_cast<std::uint32_t>(entry));
        } else {
            ++expected.plan.bad;
        }
    }

    std::uint32_t expert_offset = 0u;
    for (std::uint32_t expert = 0u; expert < experts; ++expert) {
        std::uint32_t prefix = 0u;
        for (std::size_t tile = 0u; tile < count_tiles; ++tile) {
            auto& cell =
                expected.histogram[tile * kExperts + expert];
            const std::uint32_t count = cell;
            cell = expert_offset + prefix;
            prefix += count;
        }
        expert_offset +=
            static_cast<std::uint32_t>(grouped[expert].size());
    }

    std::uint32_t grouped_entries = 0u;
    for (std::uint32_t expert = 0u; expert < experts; ++expert) {
        const auto& expert_entries = grouped[expert];
        if (expert_entries.empty()) continue;
        const std::uint32_t group = expected.plan.n_groups++;
        expected.plan.grp_e[group] = static_cast<std::int32_t>(expert);
        expected.plan.grp_res[group] = 1;
        expected.plan.grp_start[group] =
            static_cast<std::int32_t>(grouped_entries);
        for (const std::uint32_t entry : expert_entries) {
            expected.source_rows.push_back(
                static_cast<std::int32_t>(entry / kTopK));
            expected.destination_rows.push_back(
                static_cast<std::int32_t>(entry));
        }
        const std::size_t expert_tiles =
            round_up_div(expert_entries.size(), 64u);
        for (std::size_t tile = 0u; tile < expert_tiles; ++tile) {
            expected.tiles.push_back(
                {group, grouped_entries +
                            static_cast<std::uint32_t>(tile * 64u)});
        }
        grouped_entries +=
            static_cast<std::uint32_t>(expert_entries.size());
        expected.plan.grp_start[group + 1u] =
            static_cast<std::int32_t>(grouped_entries);
    }
    expected.plan.n_tiles =
        static_cast<std::uint32_t>(expected.tiles.size());
    expected.plan.skip = expected.plan.n_miss == 0u ? 1u : 0u;
    expected.gathered_rows.resize(
        static_cast<std::size_t>(grouped_entries) * kWidth);
    for (std::size_t entry = 0u; entry < grouped_entries; ++entry) {
        const std::size_t source =
            static_cast<std::size_t>(expected.source_rows[entry]) * kWidth;
        std::copy_n(mixed_h.begin() + source, kWidth,
                    expected.gathered_rows.begin() + entry * kWidth);
    }
    return expected;
}

struct CaseBuffers {
    twin::GuardedBuffer x_hi;
    twin::GuardedBuffer x_lo;
    twin::GuardedBuffer router_weights;
    twin::GuardedBuffer logits;
    twin::GuardedBuffer route_ids;
    twin::GuardedBuffer route_weights;
    twin::GuardedBuffer resident;
    twin::GuardedBuffer mixed_h;
    twin::GuardedBuffer histogram;
    twin::GuardedBuffer plan;
    twin::GuardedBuffer tiles;
    twin::GuardedBuffer source_rows;
    twin::GuardedBuffer destination_rows;
    twin::GuardedBuffer gathered_rows;

    PrefillMoeRouteGatherBuffers views(Context& context) {
        return {
            {context.buffers().read_only(x_hi.view()),
             context.buffers().read_only(x_lo.view()),
             context.buffers().read_only(router_weights.view()),
             logits.view()},
            route_ids.view(),
            route_weights.view(),
            context.buffers().read_only(resident.view()),
            context.buffers().read_only(mixed_h.view()),
            histogram.view(),
            plan.view(),
            tiles.view(),
            source_rows.view(),
            destination_rows.view(),
            gathered_rows.view(),
        };
    }
};

bool allocate_case(Context& context, CaseBuffers& b, std::uint32_t rows,
                   std::uint32_t experts, const std::string& label,
                   std::string& error) {
    const std::size_t entries = static_cast<std::size_t>(rows) * kTopK;
    const std::size_t histogram_bytes =
        round_up_div(entries, 256u) * kExperts * sizeof(std::uint32_t);
    const std::size_t tile_capacity =
        round_up_div(entries, 64u) + kExperts;
    return b.x_hi.allocate(context, label + " router high",
                           static_cast<std::size_t>(rows) * kWidth *
                               sizeof(std::uint16_t),
                           error) &&
           b.x_lo.allocate(context, label + " router low",
                           static_cast<std::size_t>(rows) * kWidth *
                               sizeof(std::uint16_t),
                           error) &&
           b.router_weights.allocate(
               context, label + " router weights",
               static_cast<std::size_t>(experts) * kWidth *
                   sizeof(std::uint16_t),
               error) &&
           b.logits.allocate(context, label + " logits",
                             static_cast<std::size_t>(rows) * experts *
                                 sizeof(float),
                             error) &&
           b.route_ids.allocate(context, label + " route IDs",
                                entries * sizeof(std::int32_t), error) &&
           b.route_weights.allocate(context, label + " route weights",
                                    entries * sizeof(float), error) &&
           b.resident.allocate(context, label + " full-mode sentinel",
                               sizeof(std::int32_t), error) &&
           b.mixed_h.allocate(context, label + " mixed F16",
                              static_cast<std::size_t>(rows) * kWidth *
                                  sizeof(std::uint16_t),
                              error) &&
           b.histogram.allocate(context, label + " histogram",
                                histogram_bytes, error) &&
           b.plan.allocate(context, label + " PfPlan", kPlanStride, error) &&
           b.tiles.allocate(context, label + " tiles",
                            tile_capacity * sizeof(Tile), error) &&
           b.source_rows.allocate(context, label + " source rows",
                                  entries * sizeof(std::int32_t), error) &&
           b.destination_rows.allocate(
               context, label + " destination rows",
               entries * sizeof(std::int32_t), error) &&
           b.gathered_rows.allocate(
               context, label + " gathered rows",
               entries * kWidth * sizeof(std::uint16_t), error);
}

void initialize_case(CaseBuffers& b, std::uint32_t rows,
                     std::uint32_t experts) {
    auto x_hi = typed<std::uint16_t>(b.x_hi);
    auto x_lo = typed<std::uint16_t>(b.x_lo);
    auto router_weights = typed<std::uint16_t>(b.router_weights);
    auto mixed_h = typed<std::uint16_t>(b.mixed_h);
    std::fill(x_hi.begin(), x_hi.end(), 0u);
    std::fill(x_lo.begin(), x_lo.end(), 0u);
    std::fill(router_weights.begin(), router_weights.end(), 0u);
    constexpr std::array<std::uint32_t, 6u> active_k{
        0u, 17u, 128u, 511u, 1024u, 2559u};
    constexpr std::array<float, 6u> high_values{
        0.5f, 0.75f, 0.125f, 1.0f, 0.375f, 0.25f};
    constexpr std::array<float, 6u> low_values{
        0.015625f, 0.0078125f, 0.0234375f,
        0.00390625f, 0.01171875f, 0.01953125f};
    constexpr std::array<float, 6u> weight_values{
        0.25f, 0.5f, 0.75f, 0.125f, 0.375f, 0.625f};
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const float row_scale = 1.0f + static_cast<float>(row) * 0.125f;
        for (std::size_t item = 0u; item < active_k.size(); ++item) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kWidth + active_k[item];
            x_hi[index] = strata::kernels::bf16_from_f32(
                high_values[item] * row_scale);
            x_lo[index] = strata::kernels::bf16_from_f32(
                low_values[item] * (1.0f + static_cast<float>(row) * 0.25f));
        }
    }
    for (std::uint32_t expert = 0u; expert < experts; ++expert) {
        const float score =
            expert < kTopK ? static_cast<float>(kTopK - expert) : -100.0f;
        for (std::size_t item = 0u; item < active_k.size(); ++item) {
            router_weights[static_cast<std::size_t>(expert) * kWidth +
                           active_k[item]] =
                strata::kernels::bf16_from_f32(score * weight_values[item]);
        }
    }
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            const int bucket =
                static_cast<int>((column * 13u + row * 107u) % 2047u) -
                1023;
            const float value =
                static_cast<float>(row * 2u) +
                static_cast<float>(bucket) / 1024.0f;
            mixed_h[static_cast<std::size_t>(row) * kWidth + column] =
                strata::kernels::f16_from_f32(value);
        }
    }
    std::fill(b.resident.payload().begin(), b.resident.payload().end(), 0u);
    for (twin::GuardedBuffer* output :
         {&b.logits, &b.route_ids, &b.route_weights, &b.histogram, &b.plan,
          &b.tiles, &b.source_rows, &b.destination_rows, &b.gathered_rows}) {
        std::fill(output->payload().begin(), output->payload().end(), kPoison);
    }
}

bool run_case(Context& context, std::uint32_t rows, std::uint32_t experts) {
    const std::string label =
        "Prefill MoE full route/gather T=" + std::to_string(rows) +
        " experts=" + std::to_string(experts);
    CaseBuffers b;
    std::string error;
    const bool allocated =
        allocate_case(context, b, rows, experts, label, error);
    check(allocated, label + " allocate", error);
    if (!allocated) return false;
    initialize_case(b, rows, experts);

    const std::array<const twin::GuardedBuffer*, 5u> inputs{{
        &b.x_hi, &b.x_lo, &b.router_weights, &b.resident, &b.mixed_h}};
    std::array<std::vector<std::uint8_t>, inputs.size()> input_bytes;
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const auto payload = inputs[index]->payload();
        input_bytes[index].assign(payload.begin(), payload.end());
    }

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const auto buffers = b.views(context);
    const bool encoded = encode_prefill_moe_full_mode_route_gather(
        context, *command, rows, experts, 0u,
        PrefillBF16X2Mode::all_projections, buffers, error);
    check(encoded, label + " encode all six steps", error);
    const bool completed =
        encoded && command->commit(error) &&
        command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto logits = typed<float>(b.logits);
    const auto ids = typed<std::int32_t>(b.route_ids);
    const auto weights = typed<float>(b.route_weights);
    const auto x_hi = typed<std::uint16_t>(b.x_hi);
    const auto x_lo = typed<std::uint16_t>(b.x_lo);
    const auto router_weights = typed<std::uint16_t>(b.router_weights);
    std::vector<double> expected_logits(
        static_cast<std::size_t>(rows) * experts, 0.0);
    bool router_matches = true;
    bool route_matches = true;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t expert = 0u; expert < experts; ++expert) {
            double high_product = 0.0;
            double low_product = 0.0;
            for (std::uint32_t k = 0u; k < kWidth; ++k) {
                const std::size_t activation_index =
                    static_cast<std::size_t>(row) * kWidth + k;
                const std::size_t weight_index =
                    static_cast<std::size_t>(expert) * kWidth + k;
                const double weight = strata::kernels::f32_from_bf16(
                    router_weights[weight_index]);
                high_product +=
                    static_cast<double>(strata::kernels::f32_from_bf16(
                        x_hi[activation_index])) *
                    weight;
                low_product +=
                    static_cast<double>(strata::kernels::f32_from_bf16(
                        x_lo[activation_index])) *
                    weight;
            }
            expected_logits[static_cast<std::size_t>(row) * experts +
                            expert] = high_product + low_product;
            const float actual =
                logits[static_cast<std::size_t>(row) * experts + expert];
            router_matches &=
                std::isfinite(actual) &&
                std::abs(static_cast<double>(actual) -
                         expected_logits[static_cast<std::size_t>(row) *
                                             experts +
                                         expert]) <= 2.0e-4;
        }

        double selected_sum = 0.0;
        std::array<double, kTopK> exponentials{};
        for (std::uint32_t rank = 0u; rank < kTopK; ++rank) {
            exponentials[rank] =
                std::exp(expected_logits[static_cast<std::size_t>(row) *
                                             experts +
                                         rank]);
            selected_sum += exponentials[rank];
        }
        for (std::uint32_t rank = 0u; rank < kTopK; ++rank) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kTopK + rank;
            route_matches &= ids[index] == static_cast<std::int32_t>(rank);
            route_matches &=
                std::abs(static_cast<double>(weights[index]) -
                         exponentials[rank] / selected_sum) <= 3.0e-6;
        }
    }
    check(router_matches, label + " BF16X2 router logits match scalar oracle");
    check(route_matches,
          label + " divide-mode top-10 IDs and weights match oracle");

    const auto expected = make_expected_grouping(
        rows, experts, ids, typed<std::uint16_t>(b.mixed_h));
    const bool plan_matches =
        std::memcmp(b.plan.payload().data(), &expected.plan,
                    sizeof(expected.plan)) == 0;
    check(plan_matches, label + " PfPlan matches host grouping oracle");
    const auto actual_histogram = typed<std::uint32_t>(b.histogram);
    check(std::equal(expected.histogram.begin(), expected.histogram.end(),
                     actual_histogram.begin()),
          label + " histogram offsets match host grouping oracle");
    const auto actual_tiles = typed<Tile>(b.tiles);
    const bool tiles_match =
        std::equal(expected.tiles.begin(), expected.tiles.end(),
                   actual_tiles.begin(),
                   [](const Tile& left, const Tile& right) {
                       return left.group == right.group &&
                              left.first_entry == right.first_entry;
                   });
    check(tiles_match, label + " expert tiles match host grouping oracle");
    check(std::equal(expected.source_rows.begin(),
                     expected.source_rows.end(),
                     typed<std::int32_t>(b.source_rows).begin()) &&
              std::equal(expected.destination_rows.begin(),
                         expected.destination_rows.end(),
                         typed<std::int32_t>(b.destination_rows).begin()),
          label + " stable source/destination maps match host oracle");
    check(std::equal(expected.gathered_rows.begin(),
                     expected.gathered_rows.end(),
                     typed<std::uint16_t>(b.gathered_rows).begin()),
          label + " F16 gather matches host oracle");

    bool inputs_unchanged = true;
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const auto payload = inputs[index]->payload();
        inputs_unchanged &=
            std::equal(input_bytes[index].begin(), input_bytes[index].end(),
                       payload.begin());
    }
    check(inputs_unchanged, label + " inputs remain unchanged");

    const std::size_t entries = static_cast<std::size_t>(rows) * kTopK;
    const std::size_t tile_bytes = expected.tiles.size() * sizeof(Tile);
    const std::size_t route_bytes = entries * sizeof(std::int32_t);
    const std::size_t gathered_bytes =
        expected.gathered_rows.size() * sizeof(std::uint16_t);
    const bool tails_match =
        std::all_of(b.tiles.payload().begin() + tile_bytes,
                    b.tiles.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(b.source_rows.payload().begin() + route_bytes,
                    b.source_rows.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(b.destination_rows.payload().begin() + route_bytes,
                    b.destination_rows.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(b.gathered_rows.payload().begin() + gathered_bytes,
                    b.gathered_rows.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(b.plan.payload().begin() + sizeof(PfPlan),
                    b.plan.payload().end(),
                    [](std::uint8_t byte) { return byte == 0u; });
    check(tails_match, label + " inactive tails remain poisoned");

    const std::array<const twin::GuardedBuffer*, 14u> all{{
        &b.x_hi,          &b.x_lo,           &b.router_weights,
        &b.logits,        &b.route_ids,      &b.route_weights,
        &b.resident,      &b.mixed_h,        &b.histogram,
        &b.plan,          &b.tiles,          &b.source_rows,
        &b.destination_rows, &b.gathered_rows,
    }};
    const bool guards_match = std::all_of(
        all.begin(), all.end(),
        [](const twin::GuardedBuffer* buffer) {
            return guards_intact(*buffer);
        });
    check(guards_match, label + " allocation guards remain intact");

    const std::vector<std::uint8_t> logits_before(
        b.logits.payload().begin(), b.logits.payload().end());
    const std::vector<std::uint8_t> routes_before(
        b.route_ids.payload().begin(), b.route_ids.payload().end());
    const std::vector<std::uint8_t> weights_before(
        b.route_weights.payload().begin(), b.route_weights.payload().end());
    const std::vector<std::uint8_t> plan_before(
        b.plan.payload().begin(), b.plan.payload().end());
    const auto short_ids = b.route_ids.view().slice(
        0u, b.route_ids.view().length - sizeof(std::int32_t));
    auto bad_buffers = buffers;
    bad_buffers.route_ids = short_ids;
    auto rejected_command =
        context.make_command_buffer(label + " late route failure", error);
    if (!rejected_command) {
        check(false, label + " rollback command buffer", error);
        return false;
    }
    const bool rejected = !encode_prefill_moe_full_mode_route_gather(
        context, *rejected_command, rows, experts, 0u,
        PrefillBF16X2Mode::all_projections, bad_buffers, error);
    check(rejected, label + " late route validation rejects short IDs");
    const bool discarded =
        rejected_command->unsafe_native_handle() == nullptr;
    check(discarded,
          label + " late validation discards the encoded router prefix");
    const bool rollback = discarded &&
        std::equal(logits_before.begin(), logits_before.end(),
                   b.logits.payload().begin()) &&
        std::equal(routes_before.begin(), routes_before.end(),
                   b.route_ids.payload().begin()) &&
        std::equal(weights_before.begin(), weights_before.end(),
                   b.route_weights.payload().begin()) &&
        std::equal(plan_before.begin(), plan_before.end(),
                   b.plan.payload().begin());
    check(rollback, label + " later validation rolls back router work");

    auto bad_mode_command =
        context.make_command_buffer(label + " mode 0 rejection", error);
    if (!bad_mode_command) {
        check(false, label + " mode validation command buffer", error);
        return false;
    }
    const bool mode_rejected =
        !encode_prefill_moe_full_mode_route_gather(
            context, *bad_mode_command, rows, experts, 0u,
            PrefillBF16X2Mode::off, buffers, error);
    check(mode_rejected, label + " rejects mode without BF16X2 remainder");
    check(std::equal(logits_before.begin(), logits_before.end(),
                     b.logits.payload().begin()),
          label + " mode rejection happens before dispatch");

    auto alias_weights_command =
        context.make_command_buffer(label + " cross-stage alias", error);
    if (!alias_weights_command) {
        check(false, label + " cross-stage alias command buffer", error);
        return false;
    }
    auto alias_weights = buffers;
    alias_weights.route_weights = b.mixed_h.view();
    const bool input_alias_rejected =
        !encode_prefill_moe_full_mode_route_gather(
            context, *alias_weights_command, rows, experts, 0u,
            PrefillBF16X2Mode::all_projections, alias_weights, error);
    check(input_alias_rejected,
          label + " rejects route weights aliasing mixed activation");

    auto alias_output_command =
        context.make_command_buffer(label + " output alias", error);
    if (!alias_output_command) {
        check(false, label + " output alias command buffer", error);
        return false;
    }
    auto alias_output = buffers;
    alias_output.route_weights = b.histogram.view();
    const bool output_alias_rejected =
        !encode_prefill_moe_full_mode_route_gather(
            context, *alias_output_command, rows, experts, 0u,
            PrefillBF16X2Mode::all_projections, alias_output, error);
    check(output_alias_rejected,
          label + " rejects route weights aliasing grouping output");

    return router_matches && route_matches && plan_matches && tiles_match &&
           inputs_unchanged && tails_match && guards_match && rejected &&
           rollback && mode_rejected && input_alias_rejected &&
           output_alias_rejected;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_moe_pipeline_test strata.metallib\n");
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
    passed &= run_case(*context, 1u, 256u);
    passed &= run_case(*context, 7u, 512u);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill MoE pipeline test(s) failed\n",
                     g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all full-mode Prefill MoE pipeline tests passed");
    return 0;
}
