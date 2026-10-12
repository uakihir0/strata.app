// Metal twin for Prefill's stable GPU expert grouping (docs/mac/04 §6.6).
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"
#include "twin.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kLayer = 2u;
constexpr std::uint8_t kPoison = 0xa5u;

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
static_assert(offsetof(PfPlan, grp_e) == 32u);
static_assert(offsetof(PfPlan, grp_res) == 2080u);
static_assert(offsetof(PfPlan, grp_start) == 4128u);

struct Tile {
    std::uint32_t group;
    std::uint32_t first_entry;
};
static_assert(sizeof(Tile) == 8u);

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

std::size_t round_up_div(std::size_t value, std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0u);
}

enum class RoutePattern { mixed, one_expert, all_experts, tile_boundaries };

struct Expected {
    PfPlan plan{};
    std::vector<std::uint32_t> histogram;
    std::vector<Tile> tiles;
    std::vector<std::int32_t> source_rows;
    std::vector<std::int32_t> destination_rows;
    std::vector<std::uint16_t> gathered;
};

Expected make_expected(std::uint32_t rows, std::uint32_t n_experts,
                       bool tiered, std::span<const std::int32_t> ids,
                       std::span<const std::int32_t> resident,
                       std::span<const std::uint16_t> mixed_h) {
    const std::size_t entries =
        static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    const std::size_t count_tiles = round_up_div(entries, 256u);
    Expected expected;
    expected.plan.n_entries = static_cast<std::uint32_t>(entries);
    std::fill_n(expected.plan.grp_e, 512u, -1);
    std::fill_n(expected.plan.grp_res, 512u, -1);
    expected.histogram.resize(count_tiles * kPrefillMoeHistogramExperts);
    expected.source_rows.reserve(entries);
    expected.destination_rows.reserve(entries);
    std::vector<std::vector<std::uint32_t>> grouped(n_experts);

    for (std::size_t entry = 0u; entry < entries; ++entry) {
        const std::int32_t expert = ids[entry];
        if (expert >= 0 &&
            static_cast<std::uint32_t>(expert) <
                kPrefillMoeHistogramExperts) {
            ++expected.histogram[(entry / 256u) *
                                     kPrefillMoeHistogramExperts +
                                 static_cast<std::uint32_t>(expert)];
        }
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= n_experts) {
            ++expected.plan.bad;
            continue;
        }
        grouped[static_cast<std::size_t>(expert)].push_back(
            static_cast<std::uint32_t>(entry));
    }

    std::uint32_t expert_offset = 0u;
    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        std::uint32_t prefix = 0u;
        for (std::size_t tile = 0u; tile < count_tiles; ++tile) {
            auto& cell = expected.histogram[
                tile * kPrefillMoeHistogramExperts + expert];
            const std::uint32_t count = cell;
            cell = expert_offset + prefix;
            prefix += count;
        }
        expert_offset += static_cast<std::uint32_t>(grouped[expert].size());
    }

    std::uint32_t grouped_entries = 0u;
    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        const auto& expert_entries = grouped[expert];
        if (expert_entries.empty()) continue;

        const std::uint32_t group = expected.plan.n_groups++;
        expected.plan.grp_e[group] = static_cast<std::int32_t>(expert);
        const bool is_resident =
            !tiered || resident[kLayer * n_experts + expert] >= 0;
        expected.plan.grp_res[group] = is_resident ? 1 : 0;
        expected.plan.n_miss += is_resident ? 0u : 1u;
        expected.plan.grp_start[group] =
            static_cast<std::int32_t>(grouped_entries);

        for (const std::uint32_t entry : expert_entries) {
            expected.source_rows.push_back(
                static_cast<std::int32_t>(entry / kPrefillMoeTopK));
            expected.destination_rows.push_back(
                static_cast<std::int32_t>(entry));
        }

        const std::uint32_t expert_tiles = static_cast<std::uint32_t>(
            round_up_div(expert_entries.size(), 64u));
        for (std::uint32_t tile = 0u; tile < expert_tiles; ++tile) {
            expected.tiles.push_back(
                {group, grouped_entries + tile * 64u});
        }
        grouped_entries +=
            static_cast<std::uint32_t>(expert_entries.size());
        expected.plan.grp_start[group + 1u] =
            static_cast<std::int32_t>(grouped_entries);
    }
    expected.plan.n_tiles =
        static_cast<std::uint32_t>(expected.tiles.size());
    expected.plan.skip = expected.plan.n_miss == 0u ? 1u : 0u;

    expected.gathered.resize(
        static_cast<std::size_t>(grouped_entries) * kWidth);
    for (std::size_t entry = 0u; entry < grouped_entries; ++entry) {
        const std::size_t source =
            static_cast<std::size_t>(expected.source_rows[entry]) * kWidth;
        std::copy_n(mixed_h.begin() + source, kWidth,
                    expected.gathered.begin() + entry * kWidth);
    }
    return expected;
}

bool active_tail_is_poison(const twin::GuardedBuffer& buffer,
                           std::size_t active_bytes) {
    const auto payload = buffer.payload();
    return active_bytes <= payload.size() &&
           std::all_of(payload.begin() + active_bytes, payload.end(),
                       [](std::uint8_t byte) { return byte == kPoison; });
}

std::vector<std::int32_t> make_ids(std::uint32_t rows,
                                   std::uint32_t n_experts,
                                   RoutePattern pattern) {
    const std::size_t entries =
        static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    std::vector<std::int32_t> ids(entries);
    for (std::size_t index = 0u; index < entries; ++index) {
        switch (pattern) {
            case RoutePattern::mixed:
                ids[index] = static_cast<std::int32_t>(
                    (index * 73u + (index / 11u) * 29u) % n_experts);
                break;
            case RoutePattern::one_expert:
                ids[index] = 0;
                break;
            case RoutePattern::all_experts:
                ids[index] = static_cast<std::int32_t>(index % n_experts);
                break;
            case RoutePattern::tile_boundaries: {
                constexpr std::array<std::size_t, 6u> ends{
                    63u, 127u, 192u, 319u, 447u, 576u};
                std::uint32_t expert = 6u;
                for (std::uint32_t boundary = 0u; boundary < ends.size();
                     ++boundary) {
                    if (index < ends[boundary]) {
                        expert = boundary;
                        break;
                    }
                }
                ids[index] = static_cast<std::int32_t>(expert % n_experts);
                break;
            }
        }
    }
    return ids;
}

bool run_case(Context& context, std::uint32_t rows, std::uint32_t n_experts,
              RoutePattern pattern, bool tiered, bool caller_owned) {
    const std::string label =
        "Prefill MoE grouping T=" + std::to_string(rows) +
        " experts=" + std::to_string(n_experts) +
        " pattern=" + std::to_string(static_cast<unsigned>(pattern)) +
        (tiered ? " tiered" : " full");
    const std::size_t entries =
        static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    const std::size_t count_tiles = round_up_div(entries, 256u);
    const std::size_t histogram_bytes =
        count_tiles * kPrefillMoeHistogramExperts * sizeof(std::uint32_t);
    const std::size_t tile_capacity =
        round_up_div(entries, 64u) + kPrefillMoeHistogramExperts;
    const std::size_t tile_bytes = tile_capacity * sizeof(Tile);
    const std::size_t route_bytes = entries * sizeof(std::int32_t);
    const std::size_t mixed_bytes =
        static_cast<std::size_t>(rows) * kWidth * sizeof(std::uint16_t);
    const std::size_t gathered_bytes =
        entries * kWidth * sizeof(std::uint16_t);

    twin::GuardedBuffer ids_buffer, resident_buffer, mixed_buffer;
    twin::GuardedBuffer histogram_buffer, plan_buffer, tile_buffer;
    twin::GuardedBuffer source_buffer, destination_buffer, gathered_buffer;
    std::string error;
    const std::size_t plan_table_bytes =
        (static_cast<std::size_t>(kLayer) + 1u) *
        kPrefillMoePlanStrideBytes;
    const bool allocated =
        ids_buffer.allocate(context, label + " ids", route_bytes, error) &&
        resident_buffer.allocate(
            context, label + " residency",
            (static_cast<std::size_t>(kLayer) + 1u) * n_experts *
                sizeof(std::int32_t),
            error) &&
        mixed_buffer.allocate(context, label + " mixed", mixed_bytes, error) &&
        histogram_buffer.allocate(context, label + " histogram",
                                  histogram_bytes, error) &&
        plan_buffer.allocate(context, label + " plans", plan_table_bytes,
                             error) &&
        tile_buffer.allocate(context, label + " tiles", tile_bytes, error) &&
        source_buffer.allocate(context, label + " source rows", route_bytes,
                                error) &&
        destination_buffer.allocate(context, label + " destination rows",
                                    route_bytes, error) &&
        gathered_buffer.allocate(context, label + " gathered rows",
                                 gathered_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    for (twin::GuardedBuffer* buffer :
         {&histogram_buffer, &plan_buffer, &tile_buffer, &source_buffer,
          &destination_buffer, &gathered_buffer}) {
        std::fill(buffer->payload().begin(), buffer->payload().end(), kPoison);
    }
    const std::vector<std::int32_t> ids =
        make_ids(rows, n_experts, pattern);
    std::memcpy(ids_buffer.payload().data(), ids.data(), route_bytes);

    auto residents = typed<std::int32_t>(resident_buffer);
    std::fill(residents.begin(), residents.end(), -1);
    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        residents[kLayer * n_experts + expert] =
            expert % 3u == 0u ? -1 : 0;
    }

    auto mixed = typed<std::uint16_t>(mixed_buffer);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            mixed[static_cast<std::size_t>(row) * kWidth + column] =
                static_cast<std::uint16_t>(
                    0x2c00u + (row * 17u + column * 13u) % 0x1000u);
        }
    }
    const std::vector<std::uint8_t> ids_before(
        ids_buffer.payload().begin(), ids_buffer.payload().end());
    const std::vector<std::uint8_t> resident_before(
        resident_buffer.payload().begin(), resident_buffer.payload().end());
    const std::vector<std::uint8_t> mixed_before(
        mixed_buffer.payload().begin(), mixed_buffer.payload().end());
    const std::array<twin::GuardedBuffer*, 9u> all_buffers{{
        &ids_buffer, &resident_buffer, &mixed_buffer, &histogram_buffer,
        &plan_buffer, &tile_buffer, &source_buffer, &destination_buffer,
        &gathered_buffer,
    }};
    for (twin::GuardedBuffer* buffer : all_buffers) {
        buffer->capture_before_dispatch();
    }
    const auto expected = make_expected(
        rows, n_experts, tiered, ids,
        std::span<const std::int32_t>(residents.data(), residents.size()),
        std::span<const std::uint16_t>(mixed.data(), mixed.size()));

    auto plan_table = plan_buffer.view();
    auto plan_slot =
        plan_table.slice(kLayer * kPrefillMoePlanStrideBytes,
                         kPrefillMoePlanStrideBytes);
    if (!plan_slot) {
        check(false, label + " select PfPlan layer slot");
        return false;
    }
    const PrefillMoeGroupingBuffers buffers{
        context.buffers().read_only(ids_buffer.view()),
        context.buffers().read_only(resident_buffer.view()),
        context.buffers().read_only(mixed_buffer.view()),
        histogram_buffer.view(),
        plan_slot,
        tile_buffer.view(),
        source_buffer.view(),
        destination_buffer.view(),
        gathered_buffer.view(),
    };

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    bool encoded = false;
    if (caller_owned) {
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_moe_grouping(
                    context, encoder, rows, n_experts, tiered, kLayer,
                    buffers, encode_error);
            },
            error);
    } else {
        encoded = encode_prefill_moe_grouping(
            context, *command, rows, n_experts, tiered, kLayer, buffers,
            error);
    }
    check(encoded, label + " encode", error);
    if (!encoded) return false;
    const bool completed =
        command->commit(error) && command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto plan_bytes = plan_slot.contents;
    const bool plan_matches =
        std::memcmp(plan_bytes, &expected.plan, sizeof(PfPlan)) == 0;
    check(plan_matches, label + " PfPlan matches stable host oracle");

    const auto actual_histogram = typed<std::uint32_t>(histogram_buffer);
    const bool histogram_matches =
        std::equal(expected.histogram.begin(), expected.histogram.end(),
                   actual_histogram.begin());
    check(histogram_matches, label + " count/scan histogram offsets");

    const auto actual_tiles = typed<Tile>(tile_buffer);
    const bool tiles_match =
        actual_tiles.size() >= expected.tiles.size() &&
        std::equal(expected.tiles.begin(), expected.tiles.end(),
                   actual_tiles.begin(),
                   [](const Tile& left, const Tile& right) {
                       return left.group == right.group &&
                              left.first_entry == right.first_entry;
                   });
    check(tiles_match, label + " expert tile list");

    const auto actual_sources = typed<std::int32_t>(source_buffer);
    const auto actual_destinations = typed<std::int32_t>(destination_buffer);
    const bool routes_match =
        actual_sources.size() >= expected.source_rows.size() &&
        actual_destinations.size() >= expected.destination_rows.size() &&
        std::equal(expected.source_rows.begin(), expected.source_rows.end(),
                   actual_sources.begin()) &&
        std::equal(expected.destination_rows.begin(),
                   expected.destination_rows.end(),
                   actual_destinations.begin());
    check(routes_match, label + " stable src/dst ordering");

    const auto actual_gathered = typed<std::uint16_t>(gathered_buffer);
    const bool gathered_matches =
        actual_gathered.size() >= expected.gathered.size() &&
        std::equal(expected.gathered.begin(), expected.gathered.end(),
                   actual_gathered.begin());
    check(gathered_matches, label + " F16 input gather");

    bool inputs_unchanged =
        std::equal(ids_before.begin(), ids_before.end(),
                   ids_buffer.payload().begin()) &&
        std::equal(resident_before.begin(), resident_before.end(),
                   resident_buffer.payload().begin()) &&
        std::equal(mixed_before.begin(), mixed_before.end(),
                   mixed_buffer.payload().begin());
    check(inputs_unchanged, label + " leaves inputs unchanged");

    const std::size_t written_routes =
        expected.source_rows.size() * sizeof(std::int32_t);
    const std::size_t written_gathered =
        expected.gathered.size() * sizeof(std::uint16_t);
    bool tails_match =
        active_tail_is_poison(tile_buffer,
                              expected.tiles.size() * sizeof(Tile)) &&
        active_tail_is_poison(source_buffer, written_routes) &&
        active_tail_is_poison(destination_buffer, written_routes) &&
        active_tail_is_poison(gathered_buffer, written_gathered);
    const auto plan_slot_bytes = plan_slot;
    const auto plan_all_bytes = plan_buffer.payload();
    tails_match &= std::all_of(
        plan_all_bytes.begin(),
        plan_all_bytes.begin() + kLayer * kPrefillMoePlanStrideBytes,
        [](std::uint8_t byte) { return byte == kPoison; });
    tails_match &= std::all_of(
        static_cast<const std::uint8_t*>(plan_slot_bytes.contents) +
            kPrefillMoePlanBytes,
        static_cast<const std::uint8_t*>(plan_slot_bytes.contents) +
            kPrefillMoePlanStrideBytes,
        [](std::uint8_t byte) { return byte == 0u; });
    check(tails_match, label + " preserves scratch tails and clears plan padding");

    const std::array<twin::ByteRange, 1u> histogram_written{{
        {0u, histogram_bytes},
    }};
    const std::array<twin::ByteRange, 1u> plan_written{{
        {kLayer * kPrefillMoePlanStrideBytes, kPrefillMoePlanStrideBytes},
    }};
    const std::array<twin::ByteRange, 1u> tiles_written{{
        {0u, expected.tiles.size() * sizeof(Tile)},
    }};
    const std::array<twin::ByteRange, 1u> routes_written{{
        {0u, written_routes},
    }};
    const std::array<twin::ByteRange, 1u> gathered_written{{
        {0u, written_gathered},
    }};
    const std::array<twin::ByteRange, 0u> no_writes{};
    const auto unchanged_outside =
        [](const twin::GuardedBuffer& buffer,
           std::span<const twin::ByteRange> written) {
            return !buffer.first_unwritten_change(written).has_value();
        };
    const bool guards_match =
        unchanged_outside(ids_buffer, no_writes) &&
        unchanged_outside(resident_buffer, no_writes) &&
        unchanged_outside(mixed_buffer, no_writes) &&
        unchanged_outside(histogram_buffer, histogram_written) &&
        unchanged_outside(plan_buffer, plan_written) &&
        unchanged_outside(tile_buffer, tiles_written) &&
        unchanged_outside(source_buffer, routes_written) &&
        unchanged_outside(destination_buffer, routes_written) &&
        unchanged_outside(gathered_buffer, gathered_written);
    check(guards_match, label + " guards and bytes outside active writes");
    return plan_matches && histogram_matches && tiles_match && routes_match &&
           gathered_matches && inputs_unchanged && tails_match &&
           guards_match;
}

bool run_invalid_route_case(Context& context, std::uint32_t n_experts) {
    const std::string label =
        "Prefill MoE invalid IDs experts=" + std::to_string(n_experts);
    constexpr std::uint32_t rows = 7u;
    constexpr std::size_t entries = rows * kPrefillMoeTopK;
    std::vector<std::int32_t> ids(entries);
    for (std::size_t index = 0u; index < entries; ++index) {
        ids[index] = static_cast<std::int32_t>(index % n_experts);
    }
    ids[0] = -1;
    ids[1] = static_cast<std::int32_t>(n_experts);
    ids[2] = 511;
    ids[3] = 512;

    // Use the same full checked path while supplying deliberately invalid IDs.
    const std::size_t count_tiles = round_up_div(entries, 256u);
    const std::size_t histogram_bytes =
        count_tiles * kPrefillMoeHistogramExperts * sizeof(std::uint32_t);
    const std::size_t tile_bytes =
        (round_up_div(entries, 64u) + kPrefillMoeHistogramExperts) *
        sizeof(Tile);
    const std::size_t route_bytes = entries * sizeof(std::int32_t);
    const std::size_t mixed_bytes =
        static_cast<std::size_t>(rows) * kWidth * sizeof(std::uint16_t);
    const std::size_t gathered_bytes =
        entries * kWidth * sizeof(std::uint16_t);
    twin::GuardedBuffer ids_buffer, resident_buffer, mixed_buffer;
    twin::GuardedBuffer histogram_buffer, plan_buffer, tile_buffer;
    twin::GuardedBuffer source_buffer, destination_buffer, gathered_buffer;
    std::string error;
    const bool allocated =
        ids_buffer.allocate(context, label + " ids", route_bytes, error) &&
        resident_buffer.allocate(
            context, label + " residency",
            (static_cast<std::size_t>(kLayer) + 1u) * n_experts *
                sizeof(std::int32_t),
            error) &&
        mixed_buffer.allocate(context, label + " mixed", mixed_bytes, error) &&
        histogram_buffer.allocate(context, label + " histogram",
                                  histogram_bytes, error) &&
        plan_buffer.allocate(
            context, label + " plans",
            (static_cast<std::size_t>(kLayer) + 1u) *
                kPrefillMoePlanStrideBytes,
            error) &&
        tile_buffer.allocate(context, label + " tiles", tile_bytes, error) &&
        source_buffer.allocate(context, label + " source", route_bytes, error) &&
        destination_buffer.allocate(context, label + " destination",
                                    route_bytes, error) &&
        gathered_buffer.allocate(context, label + " gathered",
                                 gathered_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memset(ids_buffer.payload().data(), 0, route_bytes);
    std::memcpy(ids_buffer.payload().data(), ids.data(), route_bytes);
    auto resident = typed<std::int32_t>(resident_buffer);
    std::fill(resident.begin(), resident.end(), 0);
    auto mixed = typed<std::uint16_t>(mixed_buffer);
    std::fill(mixed.begin(), mixed.end(), 0x3400u);
    for (twin::GuardedBuffer* buffer :
         {&histogram_buffer, &plan_buffer, &tile_buffer, &source_buffer,
          &destination_buffer, &gathered_buffer}) {
        std::fill(buffer->payload().begin(), buffer->payload().end(), kPoison);
    }
    const auto expected = make_expected(
        rows, n_experts, true, ids,
        std::span<const std::int32_t>(resident.data(), resident.size()),
        std::span<const std::uint16_t>(mixed.data(), mixed.size()));
    const auto plan_slot = plan_buffer.view().slice(
        kLayer * kPrefillMoePlanStrideBytes, kPrefillMoePlanStrideBytes);
    const std::array<twin::GuardedBuffer*, 9u> all_buffers{{
        &ids_buffer, &resident_buffer, &mixed_buffer, &histogram_buffer,
        &plan_buffer, &tile_buffer, &source_buffer, &destination_buffer,
        &gathered_buffer,
    }};
    for (twin::GuardedBuffer* buffer : all_buffers) {
        buffer->capture_before_dispatch();
    }
    const PrefillMoeGroupingBuffers buffers{
        context.buffers().read_only(ids_buffer.view()),
        context.buffers().read_only(resident_buffer.view()),
        context.buffers().read_only(mixed_buffer.view()),
        histogram_buffer.view(), plan_slot, tile_buffer.view(),
        source_buffer.view(), destination_buffer.view(),
        gathered_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode_prefill_moe_grouping(
        context, *command, rows, n_experts, true, kLayer, buffers, error);
    check(encoded, label + " encode", error);
    const bool completed =
        encoded && command->commit(error) && command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;
    const bool plan_matches =
        std::memcmp(plan_slot.contents, &expected.plan, sizeof(PfPlan)) == 0;
    check(plan_matches, label + " counts invalid IDs in PfPlan.bad");

    const auto actual_histogram = typed<std::uint32_t>(histogram_buffer);
    const bool histogram_matches =
        std::equal(expected.histogram.begin(), expected.histogram.end(),
                   actual_histogram.begin());
    check(histogram_matches,
          label + " invalid routes do not corrupt histogram offsets");

    const auto actual_tiles = typed<Tile>(tile_buffer);
    const bool tiles_match =
        std::equal(expected.tiles.begin(), expected.tiles.end(),
                   actual_tiles.begin(),
                   [](const Tile& left, const Tile& right) {
                       return left.group == right.group &&
                              left.first_entry == right.first_entry;
                   });
    check(tiles_match, label + " builds tiles only for valid routes");

    const auto actual_sources = typed<std::int32_t>(source_buffer);
    const auto actual_destinations = typed<std::int32_t>(destination_buffer);
    const bool routes_match =
        std::equal(expected.source_rows.begin(), expected.source_rows.end(),
                   actual_sources.begin()) &&
        std::equal(expected.destination_rows.begin(),
                   expected.destination_rows.end(),
                   actual_destinations.begin());
    check(routes_match, label + " emits compact valid src/dst entries");

    const auto actual_gathered = typed<std::uint16_t>(gathered_buffer);
    const bool gathered_matches =
        std::equal(expected.gathered.begin(), expected.gathered.end(),
                   actual_gathered.begin());
    check(gathered_matches,
          label + " gathers only rows from valid route entries");

    const std::size_t written_routes =
        expected.source_rows.size() * sizeof(std::int32_t);
    const std::size_t written_gathered =
        expected.gathered.size() * sizeof(std::uint16_t);
    const bool tails_match =
        active_tail_is_poison(tile_buffer,
                              expected.tiles.size() * sizeof(Tile)) &&
        active_tail_is_poison(source_buffer, written_routes) &&
        active_tail_is_poison(destination_buffer, written_routes) &&
        active_tail_is_poison(gathered_buffer, written_gathered);
    check(tails_match, label + " preserves invalid-entry output tails");

    const std::array<twin::ByteRange, 1u> histogram_written{{
        {0u, histogram_bytes},
    }};
    const std::array<twin::ByteRange, 1u> plan_written{{
        {kLayer * kPrefillMoePlanStrideBytes, kPrefillMoePlanStrideBytes},
    }};
    const std::array<twin::ByteRange, 1u> tiles_written{{
        {0u, expected.tiles.size() * sizeof(Tile)},
    }};
    const std::array<twin::ByteRange, 1u> routes_written{{
        {0u, written_routes},
    }};
    const std::array<twin::ByteRange, 1u> gathered_written{{
        {0u, written_gathered},
    }};
    const std::array<twin::ByteRange, 0u> no_writes{};
    const auto unchanged_outside =
        [](const twin::GuardedBuffer& buffer,
           std::span<const twin::ByteRange> written) {
            return !buffer.first_unwritten_change(written).has_value();
        };
    const bool guards_match =
        unchanged_outside(ids_buffer, no_writes) &&
        unchanged_outside(resident_buffer, no_writes) &&
        unchanged_outside(mixed_buffer, no_writes) &&
        unchanged_outside(histogram_buffer, histogram_written) &&
        unchanged_outside(plan_buffer, plan_written) &&
        unchanged_outside(tile_buffer, tiles_written) &&
        unchanged_outside(source_buffer, routes_written) &&
        unchanged_outside(destination_buffer, routes_written) &&
        unchanged_outside(gathered_buffer, gathered_written);
    check(guards_match, label + " rejects invalid-entry writes out of range");
    return plan_matches && histogram_matches && tiles_match && routes_match &&
           gathered_matches && tails_match && guards_match;
}

bool test_disjoint_active_slices(Context& context) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t experts = 512u;
    constexpr std::size_t entries = rows * kPrefillMoeTopK;
    constexpr std::size_t route_bytes = entries * sizeof(std::int32_t);
    constexpr std::size_t extra_bytes = 16u;
    const std::string label = "Prefill MoE disjoint active slices";
    twin::GuardedBuffer routes, resident, mixed, plans, tiles;
    twin::GuardedBuffer destination, gathered;
    std::string error;
    const bool allocated =
        routes.allocate(context, label + " route arena",
                        2u * route_bytes + extra_bytes, error) &&
        resident.allocate(context, label + " residency",
                          (kLayer + 1u) * experts * sizeof(std::int32_t),
                          error) &&
        mixed.allocate(context, label + " mixed",
                       rows * kWidth * sizeof(std::uint16_t), error) &&
        plans.allocate(context, label + " plan", kPrefillMoePlanStrideBytes,
                       error) &&
        tiles.allocate(context, label + " tiles",
                       (round_up_div(entries, 64u) +
                        kPrefillMoeHistogramExperts) *
                           sizeof(Tile),
                       error) &&
        destination.allocate(context, label + " destination",
                             route_bytes, error) &&
        gathered.allocate(context, label + " gathered",
                          entries * kWidth * sizeof(std::uint16_t), error);
    check(allocated, label + " allocate", error);
    if (!allocated) return false;

    std::fill(routes.payload().begin(), routes.payload().end(), kPoison);
    std::vector<std::int32_t> ids(entries);
    for (std::size_t index = 0u; index < entries; ++index) {
        ids[index] = static_cast<std::int32_t>(index);
    }
    std::memcpy(routes.payload().data(), ids.data(), route_bytes);
    std::fill(typed<std::int32_t>(resident).begin(),
              typed<std::int32_t>(resident).end(), 0);
    auto mixed_values = typed<std::uint16_t>(mixed);
    for (std::size_t index = 0u; index < mixed_values.size(); ++index) {
        mixed_values[index] = static_cast<std::uint16_t>(0x3000u + index % 512u);
    }
    for (twin::GuardedBuffer* buffer :
         {&plans, &tiles, &destination, &gathered}) {
        std::fill(buffer->payload().begin(), buffer->payload().end(), kPoison);
    }

    const auto residents = typed<std::int32_t>(resident);
    const auto expected = make_expected(rows, experts, true, ids,
                                        residents, mixed_values);
    const auto histogram_view =
        resident.view().slice(0u, 512u * sizeof(std::uint32_t));
    const auto route_read = context.buffers().read_only(
        routes.view().slice(0u, route_bytes + extra_bytes));
    const auto source_output =
        routes.view().slice(route_bytes, route_bytes + extra_bytes);
    const PrefillMoeGroupingBuffers buffers{
        route_read,
        context.buffers().read_only(resident.view()),
        context.buffers().read_only(mixed.view()),
        histogram_view,
        plans.view(),
        tiles.view(),
        source_output,
        destination.view(),
        gathered.view(),
    };
    const std::array<twin::GuardedBuffer*, 7u> guarded{{
        &routes, &resident, &mixed, &plans, &tiles, &destination, &gathered,
    }};
    for (twin::GuardedBuffer* buffer : guarded) {
        buffer->capture_before_dispatch();
    }

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode_prefill_moe_grouping(
        context, *command, rows, experts, true, kLayer, buffers, error);
    check(encoded, label + " accepts nonoverlapping active ranges", error);
    const bool completed =
        encoded && command->commit(error) && command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto actual_sources = std::span<const std::int32_t>(
        static_cast<const std::int32_t*>(source_output.contents), entries);
    const auto actual_destinations = typed<std::int32_t>(destination);
    const auto actual_gathered = typed<std::uint16_t>(gathered);
    const auto actual_histogram = std::span<const std::uint32_t>(
        static_cast<const std::uint32_t*>(histogram_view.contents), 512u);
    const bool outputs_match =
        std::memcmp(plans.view().contents, &expected.plan, sizeof(PfPlan)) == 0 &&
        std::equal(expected.histogram.begin(), expected.histogram.end(),
                   actual_histogram.begin()) &&
        std::equal(expected.source_rows.begin(), expected.source_rows.end(),
                   actual_sources.begin()) &&
        std::equal(expected.destination_rows.begin(),
                   expected.destination_rows.end(),
                   actual_destinations.begin()) &&
        std::equal(expected.gathered.begin(), expected.gathered.end(),
                   actual_gathered.begin());
    check(outputs_match, label + " executes the sliced route views");

    const std::array<twin::ByteRange, 1u> route_writes{{
        {route_bytes, route_bytes},
    }};
    const std::array<twin::ByteRange, 1u> plan_writes{{
        {0u, kPrefillMoePlanStrideBytes},
    }};
    const std::array<twin::ByteRange, 1u> residency_histogram_writes{{
        {0u, 512u * sizeof(std::uint32_t)},
    }};
    const std::array<twin::ByteRange, 1u> tiles_writes{{
        {0u, expected.tiles.size() * sizeof(Tile)},
    }};
    const std::array<twin::ByteRange, 1u> destination_writes{{
        {0u, route_bytes},
    }};
    const std::array<twin::ByteRange, 1u> gathered_writes{{
        {0u, expected.gathered.size() * sizeof(std::uint16_t)},
    }};
    const std::array<twin::ByteRange, 0u> no_writes{};
    const bool guards_match =
        !routes.first_unwritten_change(route_writes).has_value() &&
        !resident.first_unwritten_change(residency_histogram_writes)
             .has_value() &&
        !mixed.first_unwritten_change(no_writes).has_value() &&
        !plans.first_unwritten_change(plan_writes).has_value() &&
        !tiles.first_unwritten_change(tiles_writes).has_value() &&
        !destination.first_unwritten_change(destination_writes).has_value() &&
        !gathered.first_unwritten_change(gathered_writes).has_value();
    check(guards_match, label + " preserves the read slice and guards");
    return outputs_match && guards_match;
}

bool test_validation(Context& context, Context& foreign_context) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t experts = 512u;
    constexpr std::size_t entries = rows * kPrefillMoeTopK;
    twin::GuardedBuffer ids, resident, mixed, histogram, plans, tiles;
    twin::GuardedBuffer source, destination, gathered, foreign;
    std::string error;
    const bool allocated =
        ids.allocate(context, "validation ids", entries * sizeof(std::int32_t),
                     error) &&
        resident.allocate(context, "validation resident",
                          (kLayer + 1u) * experts * sizeof(std::int32_t), error) &&
        mixed.allocate(context, "validation mixed",
                       rows * kWidth * sizeof(std::uint16_t) + 16u, error) &&
        histogram.allocate(context, "validation histogram",
                           512u * sizeof(std::uint32_t), error) &&
        plans.allocate(context, "validation plan", kPrefillMoePlanStrideBytes,
                       error) &&
        tiles.allocate(context, "validation tiles",
                       (round_up_div(entries, 64u) + 512u) * sizeof(Tile),
                       error) &&
        source.allocate(context, "validation source",
                        entries * sizeof(std::int32_t), error) &&
        destination.allocate(context, "validation destination",
                             entries * sizeof(std::int32_t), error) &&
        gathered.allocate(context, "validation gathered",
                          entries * kWidth * sizeof(std::uint16_t) + 16u,
                          error) &&
        foreign.allocate(foreign_context, "validation foreign",
                         entries * sizeof(std::int32_t), error);
    check(allocated, "Prefill MoE validation allocate", error);
    if (!allocated) return false;
    const PrefillMoeGroupingBuffers valid{
        context.buffers().read_only(ids.view()),
        context.buffers().read_only(resident.view()),
        context.buffers().read_only(mixed.view()),
        histogram.view(), plans.view(), tiles.view(), source.view(),
        destination.view(), gathered.view()};
    auto rejects = [&](std::uint32_t test_rows, std::uint32_t test_experts,
                       std::uint32_t test_layer,
                       const PrefillMoeGroupingBuffers& test_buffers,
                       std::string_view expected_error,
                       const std::string& test_label) {
        auto command = context.make_command_buffer(test_label, error);
        if (!command) {
            check(false, test_label + " command buffer", error);
            return false;
        }
        const bool result = encode_prefill_moe_grouping(
            context, *command, test_rows, test_experts, false, test_layer,
            test_buffers, error);
        const bool rejected =
            !result && error.find(expected_error) != std::string::npos;
        check(rejected, test_label, error);
        return rejected;
    };

    bool passed = true;
    passed &= rejects(0u, experts, kLayer, valid, "row count",
                      "rejects zero rows");
    passed &= rejects(1u, 128u, kLayer, valid, "256 or 512",
                      "rejects unsupported expert count");
    passed &= rejects(1u, experts, kPrefillMoePlanLayers, valid,
                      "outside the PfPlan", "rejects out-of-range layer");
    auto short_gather = valid;
    short_gather.gathered_rows =
        gathered.view().slice(0u, entries * kWidth * sizeof(std::uint16_t) - 2u);
    passed &= rejects(1u, experts, kLayer, short_gather, "undersized",
                      "rejects short gather buffer");
    auto misaligned_input = valid;
    misaligned_input.mixed_h = context.buffers().read_only(
        mixed.view().slice(2u, rows * kWidth * sizeof(std::uint16_t)));
    passed &= rejects(1u, experts, kLayer, misaligned_input, "misaligned",
                      "rejects unaligned uint4 gather input");
    auto misaligned_output = valid;
    misaligned_output.gathered_rows = gathered.view().slice(
        2u, entries * kWidth * sizeof(std::uint16_t));
    passed &= rejects(1u, experts, kLayer, misaligned_output, "misaligned",
                      "rejects unaligned uint4 gather output");
    auto output_input_alias = valid;
    output_input_alias.source_rows = ids.view();
    passed &= rejects(1u, experts, kLayer, output_input_alias,
                      "overlaps active route IDs",
                      "rejects route input/output alias");
    auto output_output_alias = valid;
    output_output_alias.destination_rows = source.view();
    passed &= rejects(1u, experts, kLayer, output_output_alias,
                      "source rows overlaps active destination rows",
                      "rejects writable alias");
    auto foreign_ids = valid;
    foreign_ids.ids = context.buffers().read_only(foreign.view());
    passed &= rejects(1u, experts, kLayer, foreign_ids,
                      "not owned by this Context",
                      "rejects foreign Context input");

    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "serial") {
        check(true, "open-group refusal is covered by concurrent dispatch");
        return passed;
    }

    auto command = context.make_command_buffer("Prefill MoE open group", error);
    if (!command) {
        check(false, "Prefill MoE open-group command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill MoE open group",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_moe_grouping(
                context, encoder, 1u, experts, false, kLayer, valid,
                encode_error);
        },
        error);
    const bool open_group_rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(open_group_rejected, "rejects an open concurrent dispatch group",
          error);
    return passed && open_group_rejected;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_moe_grouping_test strata.metallib\n");
        return 2;
    }
    std::string error;
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;
    ContextOptions foreign_options = options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal Context", error);
    if (!foreign_context) return 1;

    bool passed = test_validation(*context, *foreign_context);
    passed &= test_disjoint_active_slices(*context);
    const std::array<std::uint32_t, 7u> rows{
        1u, 7u, 25u, 26u, 64u, 300u, 4096u};
    for (const std::uint32_t row_count : rows) {
        const std::uint32_t experts =
            row_count == 25u || row_count == 300u ? 256u : 512u;
        passed &= run_case(*context, row_count, experts, RoutePattern::mixed,
                           false, row_count == 7u);
        passed &= run_case(*context, row_count, experts,
                           RoutePattern::one_expert, true, false);
        passed &= run_case(*context, row_count, experts,
                           RoutePattern::all_experts, true, false);
    }
    passed &= run_case(*context, 4096u, 256u, RoutePattern::all_experts, true,
                       false);
    passed &= run_case(*context, 64u, 512u, RoutePattern::tile_boundaries,
                       true, true);
    passed &= run_invalid_route_case(*context, 256u);
    passed &= run_invalid_route_case(*context, 512u);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill MoE grouping test(s) failed\n",
                     g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all Prefill MoE grouping tests passed");
    return 0;
}
