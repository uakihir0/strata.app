// Property pf_sort_eq_host_grouping (docs/mac/04 sections 6.6 and 10.4).
//
// SYNTHETIC: route ids come from generated patterns (mixed, one expert, every expert, expert tile
// boundaries, and invalid ids) and mixed_h rows are generated values. No model weights, router
// outputs or model files are read.
//
// For T = 1, 7, 25, 26, 64, 300 and 4,096 with 256 and 512 experts, the GPU grouping (count, scan,
// scatter and the F16 row gather) must equal an independent host transcription of upstream's host
// grouping loop (prefill.cpp:1583-1615, as docs/mac/04 section 6.6 describes it). The comparison is
// exact (class B):
//   * the histogram offsets of every expert column below n_experts, for every tile;
//   * every PfPlan field, including the padding after the 6,180 bytes of the struct and the layer's
//     slot size;
//   * the expert tile list;
//   * the stable src and dst maps of the valid entries (dst is the inverse of the slot);
//   * the F16 gather of the valid entries.
// Invalid ids (outside [0, n_experts)) are counted in PfPlan.bad and take no slot. Their src and
// dst entries, gathered rows and unused tile entries must stay poison, and every byte outside the
// declared writes must be unchanged. The inputs must be unchanged. Tiered residency is checked on
// the mixed and invalid patterns.
//
// Negative controls: after the comparisons, a swap of two adjacent sorted entries must fail the
// stable map check, and a one-count change in the histogram must fail the histogram check.
//
// The serial and concurrent CTests run the same oracle under STRATA_METAL_DISPATCH. Histogram columns at
// or past n_experts are raw counts that the scan does not convert, so they are not part of the plan and
// are not compared.

#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"
#include "twin.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kLayer = 5u;
constexpr std::uint32_t kTileEntries = 256u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::array<std::uint32_t, 7> kRowCounts{1u, 7u, 25u, 26u, 64u, 300u, 4096u};
constexpr std::array<std::uint32_t, 2> kExpertCounts{256u, 512u};

// The C++ mirror of the PfPlan struct that the grouping kernel writes (common/types.h).
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

enum class Pattern { mixed, one_expert, all_experts, tile_boundaries, invalid_mixed };

const char* pattern_name(Pattern pattern) {
    switch (pattern) {
        case Pattern::mixed: return "mixed";
        case Pattern::one_expert: return "one-expert";
        case Pattern::all_experts: return "all-experts";
        case Pattern::tile_boundaries: return "tile-boundaries";
        case Pattern::invalid_mixed: return "invalid-ids";
    }
    return "unknown";
}

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

template <class T>
std::span<T> typed(twin::GuardedBuffer& buffer) {
    const auto bytes = buffer.payload();
    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
}

template <class T>
std::span<const T> typed(const twin::GuardedBuffer& buffer) {
    const auto bytes = buffer.payload();
    return {reinterpret_cast<const T*>(bytes.data()), bytes.size() / sizeof(T)};
}

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0u);
}

bool valid_id(std::int32_t id, std::uint32_t n_experts) {
    return id >= 0 && static_cast<std::uint32_t>(id) < n_experts;
}

std::vector<std::int32_t> make_ids(std::uint32_t rows, std::uint32_t n_experts,
                                   Pattern pattern) {
    const std::size_t entries = std::size_t{rows} * kPrefillMoeTopK;
    std::vector<std::int32_t> ids(entries);
    for (std::size_t index = 0u; index < entries; ++index) {
        const auto mixed_id = static_cast<std::int32_t>(
            (index * 73u + (index / 11u) * 29u) % n_experts);
        std::int32_t id = 0;
        switch (pattern) {
            case Pattern::mixed:
                id = mixed_id;
                break;
            case Pattern::one_expert:
                id = 0;
                break;
            case Pattern::all_experts:
                id = static_cast<std::int32_t>(index % n_experts);
                break;
            case Pattern::tile_boundaries: {
                // Expert 0 fills one 64-entry tile exactly, expert 1 fills the next, expert 2
                // runs one entry into a second tile, expert 5 crosses two tile boundaries, and
                // expert 6 takes the rest.
                constexpr std::array<std::size_t, 6u> ends{63u, 127u, 192u, 319u, 447u, 576u};
                std::int32_t expert = 6;
                for (std::size_t boundary = 0u; boundary < ends.size(); ++boundary) {
                    if (index <= ends[boundary]) {
                        expert = static_cast<std::int32_t>(boundary);
                        break;
                    }
                }
                id = expert % static_cast<std::int32_t>(n_experts);
                break;
            }
            case Pattern::invalid_mixed: {
                if (index % 7u != 3u) {
                    id = mixed_id;
                    break;
                }
                // Negative, the first id past the expert range, one past that (still inside
                // the 512-column histogram for 256 experts), and an id past the histogram.
                switch ((index / 7u) % 4u) {
                    case 0u: id = -1; break;
                    case 1u: id = static_cast<std::int32_t>(n_experts); break;
                    case 2u: id = static_cast<std::int32_t>(n_experts + 1u); break;
                    default: id = std::numeric_limits<std::int32_t>::min(); break;
                }
                break;
            }
        }
        ids[index] = id;
    }
    return ids;
}

// Upstream's host grouping loop (prefill.cpp:1583-1615), as docs/mac/04 section 6.6 describes it.
// The exclusive prefix of the per-expert counts, in id order, gives each expert's first slot. A
// stable fill then walks the entries in order and gives each valid entry the next free slot of its
// expert. A histogram row is the fill cursor at the start of its tile.
struct Expected {
    PfPlan plan{};
    std::vector<std::uint32_t> histogram;
    std::vector<Tile> tiles;
    std::vector<std::int32_t> src;
    std::vector<std::int32_t> dst;
};

Expected host_grouping(std::uint32_t rows, std::uint32_t n_experts, bool tiered,
                       std::span<const std::int32_t> ids,
                       std::span<const std::int32_t> resident_row) {
    const std::size_t entries = std::size_t{rows} * kPrefillMoeTopK;
    const std::size_t tile_count = ceil_div(entries, kTileEntries);
    Expected expected;
    expected.plan.n_entries = static_cast<std::uint32_t>(entries);
    std::fill_n(expected.plan.grp_e, 512u, -1);
    std::fill_n(expected.plan.grp_res, 512u, -1);

    std::vector<std::uint32_t> count(n_experts, 0u);
    for (std::size_t i = 0u; i < entries; ++i) {
        if (valid_id(ids[i], n_experts)) {
            ++count[static_cast<std::size_t>(ids[i])];
        } else {
            ++expected.plan.bad;
        }
    }
    std::vector<std::uint32_t> first(n_experts, 0u);
    std::uint32_t total = 0u;
    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        first[expert] = total;
        total += count[expert];
    }

    std::vector<std::uint32_t> cursor = first;
    expected.histogram.assign(tile_count * kPrefillMoeHistogramExperts, 0u);
    expected.src.assign(total, -1);
    expected.dst.assign(total, -1);
    for (std::size_t tile = 0u; tile < tile_count; ++tile) {
        for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
            expected.histogram[tile * kPrefillMoeHistogramExperts + expert] =
                cursor[expert];
        }
        const std::size_t end = std::min(entries, (tile + 1u) * kTileEntries);
        for (std::size_t i = tile * kTileEntries; i < end; ++i) {
            if (!valid_id(ids[i], n_experts)) continue;
            const std::uint32_t slot = cursor[static_cast<std::size_t>(ids[i])]++;
            expected.src[slot] = static_cast<std::int32_t>(i / kPrefillMoeTopK);
            expected.dst[slot] = static_cast<std::int32_t>(i);
        }
    }

    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        if (count[expert] == 0u) continue;
        const std::uint32_t group = expected.plan.n_groups++;
        expected.plan.grp_e[group] = static_cast<std::int32_t>(expert);
        const bool resident = !tiered || resident_row[expert] >= 0;
        expected.plan.grp_res[group] = resident ? 1 : 0;
        expected.plan.n_miss += resident ? 0u : 1u;
        expected.plan.grp_start[group] = static_cast<std::int32_t>(first[expert]);
        const std::size_t tiles = ceil_div(count[expert], 64u);
        for (std::size_t j = 0u; j < tiles; ++j) {
            expected.tiles.push_back(
                {group, first[expert] + static_cast<std::uint32_t>(64u * j)});
        }
    }
    expected.plan.grp_start[expected.plan.n_groups] = static_cast<std::int32_t>(total);
    expected.plan.n_tiles = static_cast<std::uint32_t>(expected.tiles.size());
    expected.plan.skip = expected.plan.n_miss == 0u ? 1u : 0u;
    return expected;
}

// The first index at which the GPU and the expected values differ, if any.
template <class T>
std::optional<std::size_t> first_mismatch(std::span<const T> actual,
                                          std::span<const T> expected) {
    if (actual.size() < expected.size()) return expected.size();
    for (std::size_t i = 0u; i < expected.size(); ++i) {
        if (!(actual[i] == expected[i])) return i;
    }
    return std::nullopt;
}

bool run_case(Context& context, const std::string& dispatch, std::uint32_t rows,
              std::uint32_t n_experts, Pattern pattern, bool tiered,
              bool caller_owned) {
    const std::string label =
        "Prefill MoE sort " + dispatch + " T=" + std::to_string(rows) +
        " experts=" + std::to_string(n_experts) + " " + pattern_name(pattern) +
        (tiered ? " tiered" : " full") + (caller_owned ? " caller-owned" : "");
    const std::size_t entries = std::size_t{rows} * kPrefillMoeTopK;
    const std::size_t tile_count = ceil_div(entries, kTileEntries);
    const std::size_t histogram_bytes =
        tile_count * kPrefillMoeHistogramExperts * sizeof(std::uint32_t);
    const std::size_t tile_capacity = ceil_div(entries, 64u) + kPrefillMoeHistogramExperts;
    const std::size_t tile_bytes = tile_capacity * sizeof(Tile);
    const std::size_t route_bytes = entries * sizeof(std::int32_t);
    const std::size_t mixed_bytes = std::size_t{rows} * kWidth * sizeof(std::uint16_t);
    const std::size_t gathered_bytes = entries * kWidth * sizeof(std::uint16_t);
    const std::size_t plan_table_bytes =
        (std::size_t{kLayer} + 1u) * kPrefillMoePlanStrideBytes;
    const std::size_t resident_bytes =
        (std::size_t{kLayer} + 1u) * n_experts * sizeof(std::int32_t);

    twin::GuardedBuffer ids_buffer, resident_buffer, mixed_buffer, histogram_buffer;
    twin::GuardedBuffer plan_buffer, tile_buffer, source_buffer, destination_buffer;
    twin::GuardedBuffer gathered_buffer;
    std::string error;
    const bool allocated =
        ids_buffer.allocate(context, label + " ids", route_bytes, error) &&
        resident_buffer.allocate(context, label + " residency", resident_bytes, error) &&
        mixed_buffer.allocate(context, label + " mixed", mixed_bytes, error) &&
        histogram_buffer.allocate(context, label + " histogram", histogram_bytes, error) &&
        plan_buffer.allocate(context, label + " plans", plan_table_bytes, error) &&
        tile_buffer.allocate(context, label + " tiles", tile_bytes, error) &&
        source_buffer.allocate(context, label + " source rows", route_bytes, error) &&
        destination_buffer.allocate(context, label + " destination rows", route_bytes,
                                    error) &&
        gathered_buffer.allocate(context, label + " gathered rows", gathered_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    for (twin::GuardedBuffer* buffer :
         {&histogram_buffer, &plan_buffer, &tile_buffer, &source_buffer,
          &destination_buffer, &gathered_buffer}) {
        std::fill(buffer->payload().begin(), buffer->payload().end(), kPoison);
    }
    const std::vector<std::int32_t> ids = make_ids(rows, n_experts, pattern);
    std::memcpy(ids_buffer.payload().data(), ids.data(), route_bytes);

    auto residents = typed<std::int32_t>(resident_buffer);
    std::fill(residents.begin(), residents.end(), -1);
    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        // Every third expert is not resident; the rest are. Only the layer's row is read.
        residents[kLayer * n_experts + expert] = expert % 3u == 0u ? -1 : 0;
    }

    auto mixed = typed<std::uint16_t>(mixed_buffer);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            mixed[std::size_t{row} * kWidth + column] = static_cast<std::uint16_t>(
                0x2c00u + (row * 17u + column * 13u) % 0x1000u);
        }
    }

    const std::vector<std::uint8_t> ids_before(ids_buffer.payload().begin(),
                                               ids_buffer.payload().end());
    const std::vector<std::uint8_t> resident_before(resident_buffer.payload().begin(),
                                                    resident_buffer.payload().end());
    const std::vector<std::uint8_t> mixed_before(mixed_buffer.payload().begin(),
                                                 mixed_buffer.payload().end());
    for (twin::GuardedBuffer* buffer :
         {&ids_buffer, &resident_buffer, &mixed_buffer, &histogram_buffer, &plan_buffer,
          &tile_buffer, &source_buffer, &destination_buffer, &gathered_buffer}) {
        buffer->capture_before_dispatch();
    }

    const auto residents_const = typed<std::int32_t>(
        static_cast<const twin::GuardedBuffer&>(resident_buffer));
    const Expected expected = host_grouping(
        rows, n_experts, tiered, ids,
        residents_const.subspan(kLayer * n_experts, n_experts));
    const auto mixed_const =
        typed<std::uint16_t>(static_cast<const twin::GuardedBuffer&>(mixed_buffer));

    const PrefillMoeGroupingBuffers buffers{
        context.buffers().read_only(ids_buffer.view()),
        context.buffers().read_only(resident_buffer.view()),
        context.buffers().read_only(mixed_buffer.view()),
        histogram_buffer.view(),
        plan_buffer.view().slice(kLayer * kPrefillMoePlanStrideBytes,
                                 kPrefillMoePlanStrideBytes),
        tile_buffer.view(),
        source_buffer.view(),
        destination_buffer.view(),
        gathered_buffer.view(),
    };
    if (!buffers.plans) {
        check(false, label + " select PfPlan layer slot");
        return false;
    }

    bool encoded = false;
    if (caller_owned) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_moe_grouping(context, encoder, rows, n_experts,
                                                   tiered, kLayer, buffers, encode_error);
            },
            error);
        encoded = encoded && command->commit(error) && command->wait_until_completed(error);
    } else {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        encoded = encode_prefill_moe_grouping(context, *command, rows, n_experts, tiered,
                                              kLayer, buffers, error) &&
                  command->commit(error) && command->wait_until_completed(error);
    }
    check(encoded, label + " encode and GPU completion", error);
    if (!encoded) return false;

    bool passed = true;
    const auto plan_bytes = plan_buffer.payload().subspan(
        kLayer * kPrefillMoePlanStrideBytes, kPrefillMoePlanStrideBytes);
    const bool plan_matches =
        std::memcmp(plan_bytes.data(), &expected.plan, sizeof(PfPlan)) == 0;
    std::string plan_detail;
    if (!plan_matches) {
        const auto* gpu = reinterpret_cast<const PfPlan*>(plan_bytes.data());
        plan_detail = "gpu n_groups " + std::to_string(gpu->n_groups) + " expected " +
                      std::to_string(expected.plan.n_groups) + ", gpu bad " +
                      std::to_string(gpu->bad) + " expected " +
                      std::to_string(expected.plan.bad) + ", gpu n_tiles " +
                      std::to_string(gpu->n_tiles) + " expected " +
                      std::to_string(expected.plan.n_tiles);
    }
    check(plan_matches, label + " PfPlan fields", plan_detail);
    passed &= plan_matches;

    const bool padding_zero = std::all_of(
        plan_bytes.begin() + kPrefillMoePlanBytes, plan_bytes.end(),
        [](std::uint8_t byte) { return byte == 0u; });
    check(padding_zero, label + " PfPlan padding is zero");
    passed &= padding_zero;

    // The histogram: every tile and every expert column below n_experts. Columns at or past
    // n_experts are not offsets and are not compared.
    const auto histogram = typed<std::uint32_t>(histogram_buffer);
    const auto first_histogram_mismatch =
        [&](const std::vector<std::uint32_t>& want) -> std::optional<std::size_t> {
        for (std::size_t tile = 0u; tile < tile_count; ++tile) {
            for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
                const std::size_t index = tile * kPrefillMoeHistogramExperts + expert;
                if (histogram[index] != want[index]) return index;
            }
        }
        return std::nullopt;
    };
    const auto histogram_mismatch = first_histogram_mismatch(expected.histogram);
    check(!histogram_mismatch, label + " count and scan histogram offsets",
          histogram_mismatch ? "first mismatch at " + std::to_string(*histogram_mismatch)
                             : std::string());
    passed &= !histogram_mismatch;

    const auto tiles = typed<Tile>(tile_buffer);
    const bool tiles_match =
        tiles.size() >= expected.tiles.size() &&
        std::equal(expected.tiles.begin(), expected.tiles.end(), tiles.begin(),
                   [](const Tile& left, const Tile& right) {
                       return left.group == right.group &&
                              left.first_entry == right.first_entry;
                   });
    check(tiles_match, label + " expert tile list");
    passed &= tiles_match;

    const auto sources = typed<std::int32_t>(source_buffer);
    const auto destinations = typed<std::int32_t>(destination_buffer);
    const auto source_mismatch =
        first_mismatch<std::int32_t>(sources, std::span<const std::int32_t>(expected.src));
    const auto destination_mismatch =
        first_mismatch<std::int32_t>(destinations, std::span<const std::int32_t>(expected.dst));
    const bool routes_match = !source_mismatch && !destination_mismatch;
    check(routes_match, label + " stable src and dst maps",
          routes_match ? std::string()
                       : "first mismatch in " +
                             std::string(source_mismatch ? "src" : "dst") + " at " +
                             std::to_string(source_mismatch ? *source_mismatch
                                                            : *destination_mismatch));
    passed &= routes_match;

    // Valid sorted entries are gathered from the rows that their src map names.
    const auto gathered = typed<std::uint16_t>(gathered_buffer);
    bool gather_matches = gathered.size() >= expected.src.size() * kWidth;
    std::size_t gather_index = 0u;
    for (std::size_t slot = 0u; gather_matches && slot < expected.src.size(); ++slot) {
        const std::size_t source_row = static_cast<std::size_t>(expected.src[slot]);
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            const std::size_t index = slot * kWidth + column;
            if (gathered[index] != mixed_const[source_row * kWidth + column]) {
                gather_matches = false;
                gather_index = index;
                break;
            }
        }
    }
    check(gather_matches, label + " F16 gather of valid entries",
          gather_matches ? std::string()
                         : "first mismatch at " + std::to_string(gather_index));
    passed &= gather_matches;

    const bool inputs_unchanged =
        std::equal(ids_before.begin(), ids_before.end(), ids_buffer.payload().begin()) &&
        std::equal(resident_before.begin(), resident_before.end(),
                   resident_buffer.payload().begin()) &&
        std::equal(mixed_before.begin(), mixed_before.end(), mixed_buffer.payload().begin());
    check(inputs_unchanged, label + " inputs unchanged");
    passed &= inputs_unchanged;

    // Declared writes. Everything else, including the poison tails past the valid entries, must
    // be unchanged from its pre-dispatch bytes.
    const std::size_t valid = expected.src.size();
    const std::array<twin::ByteRange, 1> histogram_written{{{0u, histogram_bytes}}};
    const std::array<twin::ByteRange, 1> plan_written{
        {{kLayer * kPrefillMoePlanStrideBytes, kPrefillMoePlanStrideBytes}}};
    const std::array<twin::ByteRange, 1> tiles_written{
        {{0u, expected.tiles.size() * sizeof(Tile)}}};
    const std::array<twin::ByteRange, 1> routes_written{{{0u, valid * sizeof(std::int32_t)}}};
    const std::array<twin::ByteRange, 1> gathered_written{
        {{0u, valid * kWidth * sizeof(std::uint16_t)}}};
    const std::array<twin::ByteRange, 0u> no_writes{};
    const auto unchanged_outside = [](const twin::GuardedBuffer& buffer,
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
    check(guards_match, label + " guards and poison tails outside the declared writes");
    passed &= guards_match;

    // Negative controls: the same comparator must reject a misordered slot and a wrong count. The
    // dst map is swapped because the two entries of one token share a src row, so a src swap
    // would not change the map.
    if (valid >= 2u) {
        std::vector<std::int32_t> swapped(destinations.begin(),
                                          destinations.begin() + valid);
        std::swap(swapped[0], swapped[1]);
        const bool swap_detected =
            first_mismatch<std::int32_t>(std::span<const std::int32_t>(swapped),
                                         std::span<const std::int32_t>(expected.dst))
                .has_value();
        check(swap_detected, label + " negative control: a swapped dst pair fails the map");
        passed &= swap_detected;
    }
    {
        // The same comparator must reject a histogram with one offset off by one.
        std::vector<std::uint32_t> wrong = expected.histogram;
        wrong[0] += 1u;
        check(first_histogram_mismatch(wrong).has_value(),
              label + " negative control: a one-count histogram change fails the offsets");
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: pf_sort_eq_host_grouping strata.metallib\n");
        return 2;
    }
    const char* dispatch_env = std::getenv("STRATA_METAL_DISPATCH");
    const std::string dispatch =
        dispatch_env != nullptr && dispatch_env[0] != '\0' ? dispatch_env : "serial";

    std::string error;
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;

    for (const std::uint32_t rows : kRowCounts) {
        for (const std::uint32_t experts : kExpertCounts) {
            for (const Pattern pattern :
                 {Pattern::mixed, Pattern::one_expert, Pattern::all_experts,
                  Pattern::tile_boundaries, Pattern::invalid_mixed}) {
                run_case(*context, dispatch, rows, experts, pattern, false, false);
            }
            // Residency: the mixed and invalid patterns, with the 256-expert cases in the
            // caller-owned Encoder form.
            for (const Pattern pattern : {Pattern::mixed, Pattern::invalid_mixed}) {
                run_case(*context, dispatch, rows, experts, pattern, true,
                         experts == 256u);
            }
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill MoE sort check(s) failed\n", g_failures);
        return 1;
    }
    std::puts("all Prefill MoE sort checks passed");
    return 0;
}
