// Standalone Metal MoE router twin (docs/mac/04-metal-kernels.md §5.10).
#include "oracles/moe_route.hpp"
#include "oracles/moe_plan.hpp"
#include "oracles/moe_swiglu.hpp"
#include "oracles/moe_gu_q2_0.hpp"
#include "oracles/moe_gu_iq1_m.hpp"
#include "oracles/moe_gu_iq2_xxs.hpp"
#include "oracles/moe_gu_iq2_xs.hpp"
#include "oracles/moe_gu_iq2_s.hpp"
#include "oracles/moe_gu_iq3_xxs.hpp"
#include "oracles/moe_gu_iq3_s.hpp"
#include "oracles/moe_gu_iq4_xs.hpp"
#include "oracles/moe_down_q2_0.hpp"
#include "oracles/moe_down_iq4_nl.hpp"
#include "oracles/moe_combine.hpp"
#include "oracles/moe_shared_expert.hpp"
#include "oracles/iq1_m_tool_fixture.hpp"
#include "oracles/iq2_xxs_moe_fixture.hpp"
#include "oracles/iq2_xs_moe_fixture.hpp"
#include "oracles/iq2_s_moe_fixture.hpp"
#include "oracles/iq3_xxs_moe_fixture.hpp"
#include "oracles/iq3_s_moe_fixture.hpp"
#include "oracles/iq4_xs_moe_fixture.hpp"
#include "oracles/bf16_mv.hpp"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"
#include "metallib_sha256.inc"
#include "synth.hpp"
#include "twin.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/moe_dispatch.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint16_t iq1m_fixture_scale_bits(
    const std::array<std::uint8_t, 56u>& block) {
    std::uint16_t bits = 0u;
    for (std::uint32_t word = 0; word < 4u; ++word) {
        bits |= static_cast<std::uint16_t>(
            ((block[49u + 2u * word] >> 4u) & 0x0fu) << (4u * word));
    }
    return bits;
}

constexpr bool iq1m_fixture_scales_are_one() {
    for (const auto& block : twin::kIQ1MToolFixture) {
        if (iq1m_fixture_scale_bits(block) != 0x3c00u) return false;
    }
    return true;
}
constexpr bool iq1m_fixture_covers_local_scales() {
    std::uint8_t seen = 0u;
    for (const auto& block : twin::kIQ1MToolFixture) {
        for (std::uint32_t word = 0; word < 4u; ++word) {
            const std::uint16_t packed =
                static_cast<std::uint16_t>(block[48u + 2u * word]) |
                (static_cast<std::uint16_t>(block[49u + 2u * word]) << 8u);
            for (std::uint32_t scale = 0; scale < 4u; ++scale) {
                const std::uint32_t code =
                    (packed >> (3u * scale)) & 0x07u;
                seen |= static_cast<std::uint8_t>(1u << code);
            }
        }
    }
    return seen == 0xffu;
}
static_assert(iq1m_fixture_scales_are_one(),
              "IQ1_M tool fixture blocks must retain a 1.0 scale");
static_assert(iq1m_fixture_covers_local_scales(),
              "IQ1_M tool fixture must cover all local scale codes");

constexpr std::uint32_t kRows = 4u;
constexpr std::uint32_t kTopK = 10u;
constexpr std::uint32_t kMoeExpertWidth = 640u;
constexpr std::uint32_t kMoeInputWidth = 2560u;
constexpr std::uint32_t kMoeQ2BlocksPerRow = kMoeInputWidth / 64u;
constexpr std::uint32_t kMoeIQ1MBlocksPerRow = kMoeInputWidth / 256u;
constexpr std::uint32_t kMoeIQ2XXSBlocksPerRow = kMoeInputWidth / 256u;
constexpr std::uint32_t kMoeIQ2XSBlocksPerRow = kMoeInputWidth / 256u;
constexpr std::uint32_t kMoeIQ2SBlocksPerRow = kMoeInputWidth / 256u;
constexpr std::uint32_t kMoeIQ3XXSBlocksPerRow = kMoeInputWidth / 256u;
constexpr std::uint32_t kMoeIQ3SBlocksPerRow = kMoeInputWidth / 256u;
constexpr std::uint32_t kMoeIQ4XSBlocksPerRow = kMoeInputWidth / 256u;
constexpr std::uint32_t kMoeDownInputWidth = 640u;
constexpr std::uint32_t kMoeDownWidth = 2560u;
constexpr std::uint32_t kMoeDownQ2BlocksPerRow = kMoeDownInputWidth / 64u;
constexpr std::uint32_t kMoeDownIQ4NLBlocksPerRow = kMoeDownInputWidth / 32u;
constexpr std::array<std::uint32_t, 16u> kMoeIQ4NLPart0Positions = {
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
    16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u};
constexpr std::array<std::uint32_t, 16u> kMoeIQ4NLPart1Positions = {
    8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    24u, 25u, 26u, 27u, 28u, 29u, 30u, 31u};
constexpr std::int32_t kMoeIQ4NLProbePart0Dot = -1800 * 127;
constexpr std::int32_t kMoeIQ4NLProbePart1Dot = 1035 * 127;
constexpr std::uint32_t kMoeActiveQ2Experts = 10u;
constexpr std::uint32_t kMoeGroupedLayer = 3u;
constexpr std::array<std::uint32_t, kTopK> kEqualWeightBits = {
    0x3dcccccd, 0x3dcccccd, 0x3dcccccd, 0x3dcccccd, 0x3dcccccd,
    0x3dcccccd, 0x3dcccccd, 0x3dcccccd, 0x3dcccccd, 0x3dcccccd};
// Independent scalar-F32 lane/warp calculation for the nonuniform row:
// logits 80, 79.5, ..., 75.5 at the ten selected experts and -80 elsewhere.
constexpr std::array<std::uint32_t, kTopK> kNonuniformMultiplyWeightBits = {
    0x3ecad2ab, 0x3e760962, 0x3e153a91, 0x3db5060e, 0x3d5b97c0,
    0x3d053094, 0x3ca1913c, 0x3c43fdb8, 0x3bedbfcc, 0x3b9033c2};
constexpr std::array<std::uint32_t, kTopK> kNonuniformDivideWeightBits = {
    0x3ecad2aa, 0x3e760961, 0x3e153a90, 0x3db5060d, 0x3d5b97c0,
    0x3d053093, 0x3ca1913c, 0x3c43fdb7, 0x3bedbfcb, 0x3b9033c2};
constexpr std::uint32_t kModeProbeExperts = 512u;
constexpr std::uint32_t kModeProbeSelectedBits = 0x3dcccccd;
constexpr std::uint32_t kModeProbeSumBits = 0x3e99999a;
constexpr std::uint32_t kModeProbeMultiplyBits = 0x3eaaaaab;
constexpr std::uint32_t kModeProbeDivideBits = 0x3eaaaaaa;
constexpr std::uint32_t kPlanLayer = 3u;
constexpr std::size_t kPlanUsageTail = 8u;

struct MoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(MoeExpertAddress) == 24u);

struct MoeGroupedParams {
    std::uint32_t n_expert = 0;
    std::uint32_t layer = 0;
};
static_assert(sizeof(MoeGroupedParams) == 8u);

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

std::vector<float> make_logits(std::uint32_t experts, twin::InputSet set,
                               std::uint64_t seed) {
    std::vector<float> logits(static_cast<std::size_t>(kRows) * experts);
    std::mt19937_64 rng(seed);
    for (std::uint32_t row = 0; row < kRows; ++row) {
        float* row_logits = logits.data() + static_cast<std::size_t>(row) * experts;
        if (set == twin::InputSet::random ||
            (set == twin::InputSet::fixture && row == 3u)) {
            std::vector<std::uint32_t> ranks(experts);
            std::iota(ranks.begin(), ranks.end(), 0u);
            std::shuffle(ranks.begin(), ranks.end(), rng);
            for (std::uint32_t expert = 0; expert < experts; ++expert) {
                row_logits[expert] =
                    static_cast<float>(ranks[expert]) * 0.125f -
                    static_cast<float>(experts) * 0.0625f;
            }
            continue;
        }

        if (row == 0u) {
            std::fill_n(row_logits, experts, 0.0f);
        } else if (row == 1u) {
            std::fill_n(row_logits, experts, -16.0f);
            const std::uint32_t start = experts - 20u;
            std::fill_n(row_logits + start, 12u, 8.0f);
        } else if (row == 2u) {
            std::fill_n(row_logits, experts, -80.0f);
            for (std::uint32_t rank = 0; rank < 11u; ++rank) {
                const std::uint32_t expert =
                    rank % 2u == 0u ? experts - 1u - rank / 2u
                                    : rank / 2u;
                row_logits[expert] = 80.0f - 0.5f * static_cast<float>(rank);
            }
        } else {
            for (std::uint32_t expert = 0; expert < experts; ++expert) {
                row_logits[expert] = (expert & 1u) == 0u ? 100.0f : -100.0f;
            }
        }
    }
    return logits;
}

bool test_route(Context& context, std::uint32_t experts,
                std::span<const float> logits, std::uint32_t rows,
                bool divide_weights, twin::InputSet set,
                const twin::Options& options,
                std::vector<std::int32_t>* actual_ids_out = nullptr,
                std::vector<float>* actual_weights_out = nullptr) {
    const std::string label =
        "MoE route experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " div=" + std::to_string(divide_weights ? 1 : 0) +
        " set=" + twin::input_set_name(set);
    std::vector<std::int32_t> expected_ids;
    std::vector<twin::tf> expected_weights;
    std::string err;
    check(twin::moe_route_reference(logits, rows, experts, divide_weights,
                                    expected_ids, expected_weights, err),
          label + " CPU route oracle", err);
    if (expected_ids.size() != static_cast<std::size_t>(rows) * kTopK ||
        expected_weights.size() != expected_ids.size()) {
        check(false, label + " oracle output shape");
        return false;
    }

    const std::size_t ids_bytes = expected_ids.size() * sizeof(std::int32_t);
    const std::size_t weights_bytes =
        expected_weights.size() * sizeof(float);
    const std::size_t logits_bytes = logits.size_bytes();
    twin::GuardedBuffer input;
    twin::GuardedBuffer ids;
    twin::GuardedBuffer weights;
    check(input.allocate(context, label + " logits", logits_bytes, err),
          label + " logits buffer", err);
    check(ids.allocate(context, label + " IDs", ids_bytes, err),
          label + " IDs buffer", err);
    check(weights.allocate(context, label + " weights", weights_bytes, err),
          label + " weights buffer", err);
    if (!input.view() || !ids.view() || !weights.view()) return false;

    std::memcpy(input.payload().data(), logits.data(), logits_bytes);
    const std::span<const std::uint8_t> expected_id_bytes(
        reinterpret_cast<const std::uint8_t*>(expected_ids.data()), ids_bytes);
    check(ids.poison_bytes_against(expected_id_bytes, err),
          label + " IDs poison", err);
    check(weights.poison_floats_against(
              expected_weights, twin::Comparison::units, err),
          label + " weights poison", err);
    input.capture_before_dispatch();
    ids.capture_before_dispatch();
    weights.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    const bool dispatched = command &&
        encode_moe_route(
            context, *command, rows, experts, divide_weights,
            {context.buffers().read_only(input.view()), ids.view(),
             weights.view()},
            err) &&
        command->commit(err) && command->wait_until_completed(err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto id_failure =
        twin::compare_bytes(ids.payload(), expected_id_bytes);
    if (id_failure) {
        twin::report_failure(label.c_str(), options, rows, *id_failure);
    }
    check(!id_failure, label + " exact route IDs");

    const auto* actual_weight_data =
        reinterpret_cast<const float*>(weights.payload().data());
    const std::span<const float> actual_weight_span(
        actual_weight_data, expected_weights.size());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto weight_failure =
        twin::compare_floats(actual_weight_span, expected_weights, tolerance);
    if (weight_failure) {
        twin::report_failure(label.c_str(), options, rows, *weight_failure);
    }
    check(!weight_failure, label + " route weights U(16)");

    const auto* actual_ids_data =
        reinterpret_cast<const std::int32_t*>(ids.payload().data());
    bool ids_valid = true;
    bool weights_normalized = true;
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::array<bool, 512> seen{};
        float weight_sum = 0.0f;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::int32_t expert = actual_ids_data[index];
            ids_valid &= expert >= 0 &&
                         static_cast<std::uint32_t>(expert) < experts;
            if (expert >= 0 && static_cast<std::uint32_t>(expert) < experts) {
                ids_valid &= !seen[static_cast<std::uint32_t>(expert)];
                seen[static_cast<std::uint32_t>(expert)] = true;
            }
            weight_sum += actual_weight_data[index];
        }
        weights_normalized &=
            std::isfinite(weight_sum) && std::fabs(weight_sum - 1.0f) < 2.0e-5f;
    }
    check(ids_valid, label + " IDs are unique and in range");
    check(weights_normalized, label + " selected weights sum to one");

    const std::array<twin::ByteRange, 1> ids_written = {{{0u, ids_bytes}}};
    const std::array<twin::ByteRange, 1> weights_written = {
        {{0u, weights_bytes}}};
    check(!input.first_unwritten_change({}),
          label + " logits and guards remain unchanged");
    check(!ids.first_unwritten_change(ids_written),
          label + " IDs guards remain unchanged");
    check(!weights.first_unwritten_change(weights_written),
          label + " weights guards remain unchanged");

    if (actual_ids_out != nullptr) {
        actual_ids_out->assign(actual_ids_data,
                               actual_ids_data + expected_ids.size());
    }
    if (actual_weights_out != nullptr) {
        actual_weights_out->assign(actual_weight_data,
                                   actual_weight_data +
                                       expected_weights.size());
    }
    return !id_failure && !weight_failure && ids_valid && weights_normalized;
}

bool test_route_dispatch_validation(Context& context) {
    constexpr std::string_view label = "MoE route encoder validation";
    std::string err;
    twin::GuardedBuffer logits;
    twin::GuardedBuffer ids;
    twin::GuardedBuffer weights;
    twin::GuardedBuffer alias;
    twin::GuardedBuffer logits_output_overlap;
    check(logits.allocate(context, "MoE route validation logits",
                          512u * sizeof(float), err),
          "MoE route validation logits buffer", err);
    check(ids.allocate(context, "MoE route validation IDs",
                       10u * sizeof(std::int32_t), err),
          "MoE route validation IDs buffer", err);
    check(weights.allocate(context, "MoE route validation weights",
                           10u * sizeof(float), err),
          "MoE route validation weights buffer", err);
    check(alias.allocate(context, "MoE route validation aliased outputs",
                         10u * sizeof(float), err),
          "MoE route validation alias buffer", err);
    check(logits_output_overlap.allocate(
              context, "MoE route validation logits-output overlap",
              512u * sizeof(float) + 10u * sizeof(float), err),
          "MoE route validation logits-output overlap buffer", err);
    if (!logits.view() || !ids.view() || !weights.view() || !alias.view() ||
        !logits_output_overlap.view()) {
        return false;
    }
    std::fill(logits.payload().begin(), logits.payload().end(), 0u);
    std::fill(ids.payload().begin(), ids.payload().end(), 0xa5u);
    std::fill(weights.payload().begin(), weights.payload().end(), 0x5au);
    std::fill(alias.payload().begin(), alias.payload().end(), 0x3cu);
    std::fill(logits_output_overlap.payload().begin(),
              logits_output_overlap.payload().end(), 0u);
    const std::vector<std::uint8_t> ids_before(ids.payload().begin(),
                                               ids.payload().end());
    const std::vector<std::uint8_t> weights_before(weights.payload().begin(),
                                                   weights.payload().end());
    const std::vector<std::uint8_t> alias_before(alias.payload().begin(),
                                                alias.payload().end());
    const std::vector<std::uint8_t> overlap_before(
        logits_output_overlap.payload().begin(),
        logits_output_overlap.payload().end());
    const ReadOnlyBufferRef logits_view =
        context.buffers().read_only(logits.view());
    if (!logits_view) {
        check(false, std::string(label) + " read-only logits view");
        return false;
    }

    const auto reject = [&](const std::string& case_label,
                            std::uint32_t rows, std::uint32_t experts,
                            MoeRouteBuffers buffers,
                            std::string_view error_fragment) {
        auto command = context.make_command_buffer(case_label, err);
        const bool rejected = command && !encode_moe_route(
            context, *command, rows, experts, false, buffers, err);
        const bool discarded = command &&
            command->unsafe_native_handle() == nullptr;
        const bool error_matches =
            err.find(error_fragment) != std::string::npos;
        check(rejected && discarded && error_matches,
              case_label + " rejects and discards transaction", err);
        return rejected && discarded && error_matches;
    };

    const bool invalid_experts = reject(
        "reject unsupported MoE expert count", 1u, 128u,
        {logits_view, ids.view(), weights.view()}, "256 or 512 experts");
    const bool zero_rows = reject(
        "reject zero MoE route rows", 0u, 512u,
        {logits_view, ids.view(), weights.view()}, "positive rows");
    const bool short_logits = reject(
        "reject undersized MoE route logits", 1u, 512u,
        {logits_view.slice(0u, sizeof(float)), ids.view(), weights.view()},
        "undersized");
    const bool aliased_outputs = reject(
        "reject overlapping MoE route outputs", 1u, 512u,
        {logits_view, alias.view(), alias.view()}, "overlap");
    const BufferRef overlap_base = logits_output_overlap.view();
    const BufferRef overlap_logits =
        overlap_base.slice(0u, 512u * sizeof(float));
    const BufferRef partial_output =
        overlap_base.slice(1024u, 10u * sizeof(float));
    const bool exact_logits_ids = reject(
        "reject exact MoE logits and IDs alias", 1u, 512u,
        {logits_view, logits.view(), weights.view()}, "overlap");
    const bool exact_logits_weights = reject(
        "reject exact MoE logits and weights alias", 1u, 512u,
        {logits_view, ids.view(), logits.view()}, "overlap");
    const bool partial_logits_ids = reject(
        "reject partial MoE logits and IDs alias", 1u, 512u,
        {context.buffers().read_only(overlap_logits), partial_output,
         weights.view()},
        "overlap");
    const bool partial_logits_weights = reject(
        "reject partial MoE logits and weights alias", 1u, 512u,
        {context.buffers().read_only(overlap_logits), ids.view(),
         partial_output},
        "overlap");
    const auto reject_caller_owned = [&](const std::string& case_label,
                                         bool reject_inside_group) {
        std::fill(ids.payload().begin(), ids.payload().end(), 0xa5u);
        std::fill(weights.payload().begin(), weights.payload().end(), 0x5au);
        auto command = context.make_command_buffer(case_label, err);
        const bool discarded = command && !Encoder::encode(
            context, *command, case_label,
            [&](Encoder& encoder, std::string& body_error) {
                if (!encode_moe_route(
                        context, encoder, 1u, 512u, false,
                        {logits_view, ids.view(), weights.view()},
                        body_error)) {
                    return false;
                }
                if (reject_inside_group && !encoder.group_begin(body_error)) {
                    return false;
                }
                return encode_moe_route(
                    context, encoder, 1u,
                    reject_inside_group ? 512u : 128u, false,
                    {logits_view, ids.view(), weights.view()}, body_error);
            },
            err);
        const bool command_discarded = command &&
            command->unsafe_native_handle() == nullptr;
        const bool outputs_unchanged =
            std::equal(ids_before.begin(), ids_before.end(),
                       ids.payload().begin(), ids.payload().end()) &&
            std::equal(weights_before.begin(), weights_before.end(),
                       weights.payload().begin(), weights.payload().end());
        const std::string_view expected_error = reject_inside_group
            ? "ordered dispatch boundary"
            : "256 or 512 experts";
        const bool error_matches = err.find(expected_error) != std::string::npos;
        check(discarded && command_discarded && outputs_unchanged &&
                  error_matches,
              case_label + " propagates failure and discards earlier work", err);
        return discarded && command_discarded && outputs_unchanged &&
               error_matches;
    };
    const bool caller_owned_late_failure = reject_caller_owned(
        "reject caller-owned late MoE route failure", false);
    const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    const bool concurrent_mode = dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent";
    const bool caller_owned_open_group = !concurrent_mode ||
        reject_caller_owned(
            "reject caller-owned MoE route in concurrent group", true);
    const bool outputs_unchanged =
        std::equal(ids_before.begin(), ids_before.end(), ids.payload().begin(),
                   ids.payload().end()) &&
        std::equal(weights_before.begin(), weights_before.end(),
                   weights.payload().begin(), weights.payload().end()) &&
        std::equal(alias_before.begin(), alias_before.end(),
                   alias.payload().begin(), alias.payload().end()) &&
        std::equal(overlap_before.begin(), overlap_before.end(),
                   logits_output_overlap.payload().begin(),
                   logits_output_overlap.payload().end());
    check(outputs_unchanged,
          std::string(label) + " leaves rejected outputs unchanged");
    return invalid_experts && zero_rows && short_logits && aliased_outputs &&
           exact_logits_ids && exact_logits_weights && partial_logits_ids &&
           partial_logits_weights && caller_owned_late_failure &&
           caller_owned_open_group && outputs_unchanged;
}

void check_literal_choices(twin::InputSet set, std::uint32_t experts,
                          std::span<const std::int32_t> actual_ids) {
    constexpr std::uint32_t rows = kRows;
    bool matches = actual_ids.size() ==
                   static_cast<std::size_t>(rows) * kTopK;
    for (std::uint32_t rank = 0; rank < kTopK && matches; ++rank) {
        matches &= actual_ids[rank] == static_cast<std::int32_t>(rank);
    }
    check(matches, "MoE router all-equal logits choose the ten lowest IDs");

    if (set != twin::InputSet::random) {
        const std::uint32_t tie_start = experts - 20u;
        bool tie_matches = actual_ids.size() ==
                           static_cast<std::size_t>(rows) * kTopK;
        for (std::uint32_t rank = 0; rank < kTopK && tie_matches; ++rank) {
            tie_matches &=
                actual_ids[kTopK + rank] ==
                static_cast<std::int32_t>(tie_start + rank);
        }
        check(tie_matches,
              "MoE router twelve-way cutoff tie selects the lower ten IDs");

        bool ordered_matches = actual_ids.size() ==
                               static_cast<std::size_t>(rows) * kTopK;
        for (std::uint32_t rank = 0; rank < kTopK && ordered_matches; ++rank) {
            const std::uint32_t expert =
                rank % 2u == 0u ? experts - 1u - rank / 2u : rank / 2u;
            ordered_matches &=
                actual_ids[2u * kTopK + rank] ==
                static_cast<std::int32_t>(expert);
        }
        check(ordered_matches,
              "MoE router orders nonuniform top-k values by descending weight");
    }

    if (set == twin::InputSet::edge) {
        bool alternating_matches = actual_ids.size() ==
                                   static_cast<std::size_t>(rows) * kTopK;
        for (std::uint32_t rank = 0;
             rank < kTopK && alternating_matches; ++rank) {
            alternating_matches &=
                actual_ids[3u * kTopK + rank] ==
                static_cast<std::int32_t>(2u * rank);
        }
        check(alternating_matches,
              "MoE router alternating ties choose lower expert IDs");
    }
}

void check_weight_golden(const std::string& label,
                         std::span<const float> actual,
                         const std::array<std::uint32_t, kTopK>& golden_bits,
                         const twin::Options& options) {
    std::array<twin::tf, kTopK> expected{};
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        expected[rank] = twin::tf{std::bit_cast<float>(golden_bits[rank])};
    }
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto failure = twin::compare_floats(actual, expected, tolerance);
    if (failure) twin::report_failure(label.c_str(), options, 1u, *failure);
    check(!failure, label);
}

bool dispatch_mode_probe(Context& context, bool divide_weights,
                         std::uint32_t golden_bits,
                         const twin::Options& options) {
    std::array<float, kModeProbeExperts> logits{};
    std::array<std::int32_t, kTopK> expected_ids{};
    std::array<float, 2u> mode_values = {
        std::bit_cast<float>(kModeProbeSelectedBits),
        std::bit_cast<float>(kModeProbeSumBits)};
    std::array<std::uint32_t, kTopK> expected_bits{};
    std::array<twin::tf, kTopK> expected_weights{};
    std::iota(expected_ids.begin(), expected_ids.end(), 0);
    expected_bits.fill(golden_bits);
    for (auto& expected : expected_weights) {
        expected = twin::tf{std::bit_cast<float>(golden_bits)};
    }

    std::string err;
    twin::GuardedBuffer logits_buffer;
    twin::GuardedBuffer ids_buffer;
    twin::GuardedBuffer weights_buffer;
    twin::GuardedBuffer mode_values_buffer;
    check(logits_buffer.allocate(context, "mode probe logits",
                                 sizeof(logits), err),
          "MoE mode probe logits buffer", err);
    check(ids_buffer.allocate(context, "mode probe IDs",
                              sizeof(expected_ids), err),
          "MoE mode probe IDs buffer", err);
    check(weights_buffer.allocate(context, "mode probe output",
                                  sizeof(expected_bits), err),
          "MoE mode probe output buffer", err);
    check(mode_values_buffer.allocate(context, "mode probe values",
                                      sizeof(mode_values), err),
          "MoE mode probe values buffer", err);
    if (!logits_buffer.view() || !ids_buffer.view() ||
        !weights_buffer.view() || !mode_values_buffer.view()) {
        return false;
    }
    std::memcpy(logits_buffer.payload().data(), logits.data(), sizeof(logits));
    std::memcpy(mode_values_buffer.payload().data(), mode_values.data(),
                sizeof(mode_values));
    const std::span<const std::uint8_t> expected_id_bytes(
        reinterpret_cast<const std::uint8_t*>(expected_ids.data()),
        sizeof(expected_ids));
    check(ids_buffer.poison_bytes_against(expected_id_bytes, err),
          "MoE mode probe IDs poison", err);
    check(weights_buffer.poison_floats_against(
              expected_weights, twin::Comparison::bits, err),
          "MoE mode probe output poison", err);
    logits_buffer.capture_before_dispatch();
    ids_buffer.capture_before_dispatch();
    weights_buffer.capture_before_dispatch();
    mode_values_buffer.capture_before_dispatch();

    const std::uint32_t divide = divide_weights ? 1u : 0u;
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {1u, 1u, 1u};
    work.threads_per_group = {32u, 1u, 1u};
    work.buffers = {{0u, logits_buffer.view()},
                    {1u, ids_buffer.view()},
                    {2u, weights_buffer.view()},
                    {4u, mode_values_buffer.view()}};
    work.parameters = &divide;
    work.parameter_bytes = sizeof(divide);
    work.parameter_index = 3u;
    check(twin::dispatch_manifest(context,
                                  "strata_moe_route_weight_mode_probe",
                                  work, err),
          divide_weights ? "MoE mode probe division dispatch"
                         : "MoE mode probe reciprocal dispatch",
          err);

    const std::span<const std::uint8_t> expected_bytes(
        reinterpret_cast<const std::uint8_t*>(expected_bits.data()),
        sizeof(expected_bits));
    const auto failure =
        twin::compare_bytes(weights_buffer.payload(), expected_bytes);
    if (failure) {
        twin::report_failure("MoE weight mode probe", options, 1u, *failure);
    }
    check(!failure,
          divide_weights ? "MoE mode probe uses true division"
                         : "MoE mode probe multiplies by reciprocal");
    const auto id_failure =
        twin::compare_bytes(ids_buffer.payload(), expected_id_bytes);
    if (id_failure) {
        twin::report_failure("MoE mode probe IDs", options, 1u, *id_failure);
    }
    check(!id_failure, "MoE mode probe route body selects the first ten IDs");
    const std::array<twin::ByteRange, 1> ids_written = {
        {{0u, sizeof(expected_ids)}}};
    const std::array<twin::ByteRange, 1> weights_written = {
        {{0u, sizeof(expected_bits)}}};
    check(!logits_buffer.first_unwritten_change({}),
          "MoE mode probe logits and guards unchanged");
    check(!ids_buffer.first_unwritten_change(ids_written),
          "MoE mode probe IDs guards unchanged");
    check(!weights_buffer.first_unwritten_change(weights_written),
          "MoE mode probe output guards unchanged");
    check(!mode_values_buffer.first_unwritten_change({}),
          "MoE mode probe mode values and guards unchanged");
    return !failure && !id_failure;
}

bool test_weight_mode_probe(Context& context, const twin::Options& options) {
    const bool multiply_ok = dispatch_mode_probe(
        context, false, kModeProbeMultiplyBits, options);
    const bool divide_ok = dispatch_mode_probe(
        context, true, kModeProbeDivideBits, options);
    return multiply_ok && divide_ok;
}

std::vector<std::int32_t> make_plan_ids(std::uint32_t rows,
                                        std::uint32_t experts,
                                        twin::InputSet set,
                                        std::uint64_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(count);
    if (set == twin::InputSet::edge) {
        std::iota(ids.begin(), ids.end(), 0);
    } else if (set == twin::InputSet::fixture && rows == 1u) {
        std::fill(ids.begin(), ids.end(),
                  static_cast<std::int32_t>(experts - 1u));
    } else if (set == twin::InputSet::fixture) {
        const std::array<std::int32_t, kTopK> pattern = {
            static_cast<std::int32_t>(experts - 1u), 0, 3,
            static_cast<std::int32_t>(experts - 1u), 2, 3, 0, 2,
            static_cast<std::int32_t>(experts - 1u), 7};
        for (std::size_t entry = 0; entry < count; ++entry) {
            ids[entry] = pattern[entry % pattern.size()];
        }
    } else {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<std::uint32_t> expert(0u, experts - 1u);
        for (std::int32_t& id : ids) {
            id = static_cast<std::int32_t>(expert(rng));
        }
    }
    return ids;
}

std::vector<std::int32_t> make_plan_resident(std::uint32_t experts) {
    const std::size_t size =
        static_cast<std::size_t>(kPlanLayer + 1u) * experts +
        kPlanUsageTail;
    std::vector<std::int32_t> resident(size, -1);
    const std::size_t layer_offset =
        static_cast<std::size_t>(kPlanLayer) * experts;
    for (std::size_t index = 0; index < layer_offset; ++index) {
        resident[index] = 0;
    }
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        if (expert % 3u == 0u) {
            resident[layer_offset + expert] =
                static_cast<std::int32_t>(expert / 3u);
        }
    }
    for (std::size_t index = layer_offset + experts;
         index < resident.size(); ++index) {
        resident[index] = 0;
    }
    // Full-mode cases must ignore this deliberately partial table.
    return resident;
}

std::vector<float> make_plan_usage(std::uint32_t experts) {
    const std::size_t size =
        static_cast<std::size_t>(kPlanLayer + 1u) * experts +
        kPlanUsageTail;
    std::vector<float> usage(size);
    for (std::size_t index = 0; index < usage.size(); ++index) {
        usage[index] =
            0.25f + static_cast<float>(index % 31u) * 0.0625f;
    }
    return usage;
}

bool test_plan_case(Context& context, std::uint32_t experts,
                    std::uint32_t rows, twin::InputSet set,
                    bool tiered, const twin::Options& options) {
    const std::string label =
        "MoE plan experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " tiered=" + std::to_string(tiered ? 1 : 0) +
        " set=" + twin::input_set_name(set);
    const std::vector<std::int32_t> ids = make_plan_ids(
        rows, experts, set, options.seed ^ (experts << rows));
    const std::vector<std::int32_t> resident = make_plan_resident(experts);
    const std::vector<float> initial_usage = make_plan_usage(experts);
    strata::metal::TieredMoePlan expected_plan{};
    std::vector<float> expected_usage;
    std::string err;
    check(twin::moe_plan_reference(
              ids, rows, experts, tiered, kPlanLayer, resident,
              initial_usage, expected_plan, expected_usage, err),
          label + " CPU plan oracle", err);
    if (expected_usage.size() != initial_usage.size()) {
        check(false, label + " oracle usage shape");
        return false;
    }

    const std::size_t plan_count = kPlanLayer + 1u;
    std::vector<strata::metal::TieredMoePlan> expected_plans(plan_count);
    expected_plans[kPlanLayer] = expected_plan;
    const std::size_t plans_bytes =
        expected_plans.size() * sizeof(expected_plans.front());
    const std::size_t ids_bytes = ids.size() * sizeof(ids.front());
    const std::size_t resident_bytes =
        resident.size() * sizeof(resident.front());
    const std::size_t usage_bytes =
        initial_usage.size() * sizeof(initial_usage.front());

    twin::GuardedBuffer ids_buffer;
    twin::GuardedBuffer resident_buffer;
    twin::GuardedBuffer usage_buffer;
    twin::GuardedBuffer plans_buffer;
    check(ids_buffer.allocate(context, label + " IDs", ids_bytes, err),
          label + " IDs buffer", err);
    check(resident_buffer.allocate(context, label + " resident table",
                                   resident_bytes, err),
          label + " resident buffer", err);
    check(usage_buffer.allocate(context, label + " usage",
                                usage_bytes, err),
          label + " usage buffer", err);
    check(plans_buffer.allocate(context, label + " plans", plans_bytes, err),
          label + " plans buffer", err);
    if (!ids_buffer.view() || !resident_buffer.view() ||
        !usage_buffer.view() || !plans_buffer.view()) {
        return false;
    }

    std::memcpy(ids_buffer.payload().data(), ids.data(), ids_bytes);
    std::memcpy(resident_buffer.payload().data(), resident.data(),
                resident_bytes);
    std::memcpy(usage_buffer.payload().data(), initial_usage.data(),
                usage_bytes);
    std::vector<std::uint8_t> initial_plans(plans_bytes, 0u);
    const std::size_t plan_offset =
        static_cast<std::size_t>(kPlanLayer) * sizeof(expected_plan);
    const auto* expected_plan_data =
        reinterpret_cast<const std::uint8_t*>(&expected_plan);
    for (std::size_t byte = 0; byte < sizeof(expected_plan); ++byte) {
        initial_plans[plan_offset + byte] =
            static_cast<std::uint8_t>(expected_plan_data[byte] ^ 0xffu);
    }
    std::memcpy(plans_buffer.payload().data(), initial_plans.data(),
                plans_bytes);
    ids_buffer.capture_before_dispatch();
    resident_buffer.capture_before_dispatch();
    usage_buffer.capture_before_dispatch();
    plans_buffer.capture_before_dispatch();

    const ReadOnlyBufferRef ids_view =
        context.buffers().read_only(ids_buffer.view());
    const ReadOnlyBufferRef resident_view =
        context.buffers().read_only(resident_buffer.view());
    auto command = context.make_command_buffer(label, err);
    const bool dispatched = ids_view && resident_view && command &&
        encode_moe_plan(
            context, *command, rows, experts, tiered, kPlanLayer,
            {ids_view, resident_view, usage_buffer.view(),
             plans_buffer.view()},
            err) &&
        command->commit(err) && command->wait_until_completed(err);
    check(dispatched, label + " validated engine encoder", err);
    if (!dispatched) return false;

    const auto* expected_plan_bytes =
        reinterpret_cast<const std::uint8_t*>(expected_plans.data());
    const std::span<const std::uint8_t> expected_plans_span(
        expected_plan_bytes, plans_bytes);
    const auto plan_failure =
        twin::compare_bytes(plans_buffer.payload(), expected_plans_span);
    if (plan_failure) {
        twin::report_failure(label.c_str(), options, rows, *plan_failure);
    }
    check(!plan_failure, label + " exact plan ABI and entries");

    const auto* expected_usage_bytes =
        reinterpret_cast<const std::uint8_t*>(expected_usage.data());
    const std::span<const std::uint8_t> expected_usage_span(
        expected_usage_bytes, usage_bytes);
    const auto usage_failure =
        twin::compare_bytes(usage_buffer.payload(), expected_usage_span);
    if (usage_failure) {
        twin::report_failure(label.c_str(), options, rows, *usage_failure);
    }
    check(!usage_failure,
          tiered ? label + " exact routed usage increments"
                 : label + " full mode leaves usage unchanged");

    std::vector<twin::ByteRange> usage_written;
    if (tiered) {
        std::vector<bool> touched(experts, false);
        for (const std::int32_t expert : ids) {
            touched[static_cast<std::uint32_t>(expert)] = true;
        }
        const std::size_t layer_offset =
            static_cast<std::size_t>(kPlanLayer) * experts;
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            if (touched[expert]) {
                usage_written.push_back(
                    {(layer_offset + expert) * sizeof(float), sizeof(float)});
            }
        }
    }
    const std::array<twin::ByteRange, 1> plans_written = {
        {{plan_offset, sizeof(expected_plan)}}};
    check(!ids_buffer.first_unwritten_change({}),
          label + " IDs and guards unchanged");
    check(!resident_buffer.first_unwritten_change({}),
          label + " resident table and guards unchanged");
    check(!usage_buffer.first_unwritten_change(usage_written),
          label + " usage writes stay within routed experts");
    check(!plans_buffer.first_unwritten_change(plans_written),
          label + " plan writes stay within the selected layer");
    return !plan_failure && !usage_failure;
}

bool test_plan_dispatch_validation(Context& context) {
    constexpr std::uint32_t kLayer = 3u;
    constexpr std::uint32_t kExperts = 512u;
    constexpr std::uint32_t kRows = 1u;
    constexpr std::size_t kIdsBytes = 80u * sizeof(std::int32_t);
    constexpr std::size_t kTableEntries = (kLayer + 1u) * kExperts;
    constexpr std::size_t kResidentBytes =
        kTableEntries * sizeof(std::int32_t);
    constexpr std::size_t kUsageBytes = kTableEntries * sizeof(float);
    constexpr std::size_t kPlansBytes =
        (kLayer + 1u) * sizeof(TieredMoePlan);
    constexpr std::string_view label = "MoE plan encoder validation";

    std::string err;
    twin::GuardedBuffer ids;
    twin::GuardedBuffer resident;
    twin::GuardedBuffer usage;
    twin::GuardedBuffer plans;
    twin::GuardedBuffer alias;
    check(ids.allocate(context, "MoE plan validation IDs", kIdsBytes + 1u,
                       err),
          "MoE plan validation IDs buffer", err);
    check(resident.allocate(context, "MoE plan validation resident",
                            kResidentBytes, err),
          "MoE plan validation resident buffer", err);
    check(usage.allocate(context, "MoE plan validation usage", kUsageBytes,
                         err),
          "MoE plan validation usage buffer", err);
    check(plans.allocate(context, "MoE plan validation plans", kPlansBytes,
                         err),
          "MoE plan validation plans buffer", err);
    check(alias.allocate(context, "MoE plan validation alias",
                         std::max(kUsageBytes, kPlansBytes), err),
          "MoE plan validation alias buffer", err);
    if (!ids.view() || !resident.view() || !usage.view() || !plans.view() ||
        !alias.view()) {
        return false;
    }

    std::fill(usage.payload().begin(), usage.payload().end(), 0x5au);
    std::fill(plans.payload().begin(), plans.payload().end(), 0xa5u);
    std::fill(alias.payload().begin(), alias.payload().end(), 0x3cu);
    const std::vector<std::uint8_t> usage_before(usage.payload().begin(),
                                                 usage.payload().end());
    const std::vector<std::uint8_t> plans_before(plans.payload().begin(),
                                                 plans.payload().end());
    const std::vector<std::uint8_t> alias_before(alias.payload().begin(),
                                                 alias.payload().end());
    const ReadOnlyBufferRef ids_view = context.buffers().read_only(ids.view());
    const ReadOnlyBufferRef resident_view =
        context.buffers().read_only(resident.view());
    if (!ids_view || !resident_view) {
        check(false, std::string(label) + " read-only input views", err);
        return false;
    }
    const MoePlanBuffers valid{ids_view, resident_view, usage.view(),
                               plans.view()};

    const auto reject = [&](const std::string& case_label,
                            std::uint32_t rows, std::uint32_t experts,
                            bool tiered, std::uint32_t layer,
                            MoePlanBuffers buffers,
                            std::string_view error_fragment) {
        auto command = context.make_command_buffer(case_label, err);
        const bool rejected = command && !encode_moe_plan(
            context, *command, rows, experts, tiered, layer, buffers, err);
        const bool discarded = command &&
            command->unsafe_native_handle() == nullptr;
        const bool error_matches =
            err.find(error_fragment) != std::string::npos;
        const bool outputs_unchanged =
            std::equal(usage_before.begin(), usage_before.end(),
                       usage.payload().begin(), usage.payload().end()) &&
            std::equal(plans_before.begin(), plans_before.end(),
                       plans.payload().begin(), plans.payload().end()) &&
            std::equal(alias_before.begin(), alias_before.end(),
                       alias.payload().begin(), alias.payload().end());
        check(rejected && discarded && error_matches && outputs_unchanged,
              case_label + " rejects before dispatch and discards", err);
        return rejected && discarded && error_matches && outputs_unchanged;
    };

    const bool zero_rows = reject(
        "reject zero-row MoE plan", 0u, kExperts, true, kLayer, valid,
        "1..8 rows");
    const bool too_many_rows = reject(
        "reject over-capacity MoE plan", 9u, kExperts, true, kLayer, valid,
        "1..8 rows");
    const bool zero_experts = reject(
        "reject zero-expert MoE plan", kRows, 0u, true,
        kLayer, valid, "1..512 experts");
    const bool over_capacity_experts = reject(
        "reject over-capacity MoE plan expert count", kRows, 513u, true,
        kLayer, valid, "1..512 experts");
    const bool invalid_layer = reject(
        "reject out-of-range MoE plan layer", kRows, kExperts, true,
        kTieredMaxLayers, valid, "valid layer");
    const bool short_ids = reject(
        "reject undersized MoE plan IDs", kRows, kExperts, true, kLayer,
        {ids_view.slice(0u, kRows * kTopK * sizeof(std::int32_t) -
                               sizeof(std::int32_t)),
         resident_view, usage.view(), plans.view()},
        "IDs");
    const bool misaligned_ids = reject(
        "reject misaligned MoE plan IDs", kRows, kExperts, true, kLayer,
        {context.buffers().read_only(ids.view().slice(
             1u, kRows * kTopK * sizeof(std::int32_t))),
         resident_view, usage.view(), plans.view()},
        "misaligned");
    const bool short_resident = reject(
        "reject undersized MoE resident table", kRows, kExperts, true,
        kLayer,
        {ids_view, resident_view.slice(0u, sizeof(std::int32_t)),
         usage.view(), plans.view()},
        "resident table");
    const bool short_usage = reject(
        "reject undersized MoE usage table in full mode", kRows, kExperts,
        false, kLayer,
        {ids_view, resident_view, usage.view().slice(0u, sizeof(float)),
         plans.view()},
        "usage table");
    const bool short_plans = reject(
        "reject undersized MoE plan output", kRows, kExperts, true, kLayer,
        {ids_view, resident_view, usage.view(),
         plans.view().slice(0u, sizeof(TieredMoePlan))},
        "plan output");

    const BufferRef alias_view = alias.view();
    const ReadOnlyBufferRef alias_read =
        context.buffers().read_only(alias_view);
    const bool ids_usage_alias = reject(
        "reject MoE plan IDs and usage alias", kRows, kExperts, true,
        kLayer,
        {context.buffers().read_only(alias_view.slice(0u, kIdsBytes)),
         resident_view, alias_view, plans.view()},
        "overlap");
    const bool resident_plan_alias = reject(
        "reject MoE plan resident and output alias", kRows, kExperts, true,
        kLayer,
        {ids_view, alias_read, usage.view(), alias_view},
        "overlap");
    const bool usage_plan_alias = reject(
        "reject MoE plan usage and output alias", kRows, kExperts, true,
        kLayer,
        {ids_view, resident_view, alias_view, alias_view},
        "overlap");
    const bool ids_plan_alias = reject(
        "reject MoE plan IDs and output alias", kRows, kExperts, true,
        kLayer,
        {context.buffers().read_only(
             alias_view.slice(0u, kRows * kTopK * sizeof(std::int32_t))),
         resident_view, usage.view(), alias_view},
        "overlap");
    const bool resident_usage_alias = reject(
        "reject MoE plan resident and usage alias", kRows, kExperts, true,
        kLayer,
        {ids_view, alias_read, alias_view, plans.view()},
        "overlap");

    auto late_failure_command =
        context.make_command_buffer("reject caller-owned late MoE plan failure",
                                    err);
    const bool late_failure = late_failure_command && !Encoder::encode(
        context, *late_failure_command, "MoE plan late failure",
        [&](Encoder& encoder, std::string& body_error) {
            if (!encode_moe_plan(context, encoder, kRows, kExperts, true,
                                 kLayer, valid, body_error)) {
                return false;
            }
            return encode_moe_plan(context, encoder, 9u, kExperts, true,
                                   kLayer, valid, body_error);
        },
        err);
    const bool late_discarded = late_failure_command &&
        late_failure_command->unsafe_native_handle() == nullptr;
    const bool after_late_unchanged =
        std::equal(usage_before.begin(), usage_before.end(),
                   usage.payload().begin(), usage.payload().end()) &&
        std::equal(plans_before.begin(), plans_before.end(),
                   plans.payload().begin(), plans.payload().end());
    check(late_failure && late_discarded && after_late_unchanged &&
              err.find("1..8 rows") != std::string::npos,
          "caller-owned late MoE plan failure rolls back earlier dispatch",
          err);

    const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    const bool concurrent_mode = dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent";
    bool open_group_rejected = true;
    if (concurrent_mode) {
        auto group_command = context.make_command_buffer(
            "reject MoE plan inside concurrent group", err);
        const bool encoded = group_command && !Encoder::encode(
            context, *group_command, "MoE plan open group",
            [&](Encoder& encoder, std::string& body_error) {
                return encoder.group_begin(body_error) &&
                    encode_moe_plan(context, encoder, kRows, kExperts, true,
                                    kLayer, valid, body_error);
            },
            err);
        const bool discarded = group_command &&
            group_command->unsafe_native_handle() == nullptr;
        open_group_rejected = encoded && discarded &&
            err.find("ordered dispatch boundary") != std::string::npos;
        check(open_group_rejected,
              "caller-owned MoE plan rejects concurrent group", err);
    }

    const bool final_unchanged =
        std::equal(usage_before.begin(), usage_before.end(),
                   usage.payload().begin(), usage.payload().end()) &&
        std::equal(plans_before.begin(), plans_before.end(),
                   plans.payload().begin(), plans.payload().end()) &&
        std::equal(alias_before.begin(), alias_before.end(),
                   alias.payload().begin(), alias.payload().end());
    check(final_unchanged, std::string(label) + " leaves outputs unchanged");
    return zero_rows && too_many_rows && zero_experts &&
           over_capacity_experts && invalid_layer && short_ids &&
           misaligned_ids && short_resident && short_usage &&
           short_plans && ids_usage_alias && resident_plan_alias &&
           usage_plan_alias && ids_plan_alias && resident_usage_alias &&
           late_failure && late_discarded &&
           after_late_unchanged && open_group_rejected && final_unchanged;
}

bool test_plan_inputs(Context& context, twin::InputSet set,
                      const twin::Options& options) {
    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            for (const bool tiered : {false, true}) {
                completed &= test_plan_case(context, experts, rows, set,
                                            tiered, options);
            }
        }
    }
    return completed;
}

void make_swiglu_inputs(std::uint32_t entries, twin::InputSet set,
                        std::uint64_t seed, std::vector<float>& gate,
                        std::vector<float>& up) {
    constexpr std::size_t kExpertWidth = 640u;
    const std::size_t count =
        static_cast<std::size_t>(entries) * kExpertWidth;
    gate.resize(count);
    up.resize(count);
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 12u> gate_values = {
            -80.0f, -20.0f, -8.0f, -2.0f, -0.5f, -0.0f,
            0.0f,   0.5f,   2.0f,  8.0f,  20.0f, 80.0f};
        constexpr std::array<float, 8u> up_values = {
            -4.0f, -1.5f, -0.25f, 0.0f, 0.125f, 0.75f, 2.0f, 8.0f};
        for (std::size_t i = 0; i < count; ++i) {
            gate[i] = gate_values[i % gate_values.size()];
            up[i] = up_values[(i * 5u + 1u) % up_values.size()];
        }
    } else if (set == twin::InputSet::fixture) {
        constexpr std::array<float, 8u> gate_values = {
            -8.0f, -2.0f, -0.5f, -0.0f, 0.5f, 2.0f, 8.0f, 0.125f};
        constexpr std::array<float, 8u> up_values = {
            1.5f, -2.0f, 0.25f, 3.0f, -4.0f, 0.5f, 2.0f, -0.75f};
        for (std::size_t i = 0; i < count; ++i) {
            gate[i] = gate_values[i % gate_values.size()];
            up[i] = up_values[i % up_values.size()];
        }
    } else {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> gate_distribution(0.0f, 2.0f);
        std::normal_distribution<float> up_distribution(0.0f, 1.0f);
        for (std::size_t i = 0; i < count; ++i) {
            gate[i] = gate_distribution(rng);
            up[i] = up_distribution(rng);
        }
    }
}

bool test_swiglu_case(Context& context, std::uint32_t rows,
                      twin::InputSet set, const twin::Options& options,
                      bool shared_expert) {
    const std::string label =
        std::string(shared_expert ? "Shared expert SwiGLU rows="
                                  : "MoE SwiGLU rows=") +
        std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::uint32_t entries = shared_expert ? rows : rows * kTopK;
    std::vector<float> gate;
    std::vector<float> up;
    make_swiglu_inputs(entries, set, options.seed ^ rows, gate, up);
    std::vector<twin::tf> expected;
    std::string err;
    check(twin::moe_swiglu_reference(gate, up, expected, err),
          label + " CPU SwiGLU oracle", err);
    if (expected.size() != gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }

    const std::size_t input_bytes = gate.size() * sizeof(float);
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    twin::GuardedBuffer output_buffer;
    check(gate_buffer.allocate(context, label + " gate", input_bytes, err),
          label + " gate buffer", err);
    check(up_buffer.allocate(context, label + " up", input_bytes, err),
          label + " up buffer", err);
    check(output_buffer.allocate(context, label + " output", input_bytes, err),
          label + " output buffer", err);
    if (!gate_buffer.view() || !up_buffer.view() || !output_buffer.view()) {
        return false;
    }
    std::memcpy(gate_buffer.payload().data(), gate.data(), input_bytes);
    std::memcpy(up_buffer.payload().data(), up.data(), input_bytes);
    check(output_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " output poison", err);
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {3u, entries, 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, gate_buffer.view()},
                    {1u, up_buffer.view()},
                    {2u, output_buffer.view()}};
    const char* kernel =
        shared_expert ? "strata_shx_swiglu" : "strata_moe_swiglu";
    const bool dispatched =
        twin::dispatch_manifest(context, kernel, work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_data =
        reinterpret_cast<const float*>(output_buffer.payload().data());
    const std::span<const float> actual(actual_data, expected.size());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto failure =
        twin::compare_floats(actual, expected, tolerance);
    if (failure) {
        twin::report_failure(label.c_str(), options, rows, *failure);
    }
    check(!failure, label + " output U(16)");
    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, input_bytes}}};
    check(!gate_buffer.first_unwritten_change({}),
          label + " gate and guards unchanged");
    check(!up_buffer.first_unwritten_change({}),
          label + " up and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " output guards unchanged");
    return !failure;
}

bool test_swiglu_inputs(Context& context, twin::InputSet set,
                        const twin::Options& options) {
    bool completed = true;
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        completed &= test_swiglu_case(context, rows, set, options, false);
    }
    return completed;
}

bool test_shared_swiglu_inputs(Context& context, twin::InputSet set,
                               const twin::Options& options) {
    bool completed = true;
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        completed &= test_swiglu_case(context, rows, set, options, true);
    }
    return completed;
}

struct SharedQ81Params {
    std::uint32_t n;
    std::uint32_t rows;
};

struct SharedMVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};

bool dispatch_shared_q8_1(Context& context, std::string_view label,
                          std::span<const float> inputs, std::uint32_t n,
                          std::uint32_t rows, const twin::Options& options,
                          std::vector<Q81>& actual) {
    std::string err;
    std::vector<Q81> expected;
    const bool oracle_ready =
        twin::quantize_q8_1_reference(inputs, n, rows, expected, err);
    check(oracle_ready, std::string(label) + " CPU Q8_1 oracle", err);
    if (!oracle_ready) return false;

    const std::size_t input_bytes = inputs.size() * sizeof(float);
    const std::size_t output_bytes = expected.size() * sizeof(Q81);
    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer output_buffer;
    check(input_buffer.allocate(context, std::string(label) + " Q8_1 input",
                                input_bytes, err),
          std::string(label) + " Q8_1 input buffer", err);
    check(output_buffer.allocate(context, std::string(label) + " Q8_1 output",
                                 output_bytes, err),
          std::string(label) + " Q8_1 output buffer", err);
    if (!input_buffer.view() || !output_buffer.view()) return false;
    std::memcpy(input_buffer.payload().data(), inputs.data(), input_bytes);
    check(output_buffer.poison_bytes_against(expected_bytes, err),
          std::string(label) + " Q8_1 output poison", err);
    input_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    const SharedQ81Params params{n, rows};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n + 255u) / 256u, rows, 1u};
    work.threads_per_group = {0u, 0u, 0u};
    work.buffers = {{0u, input_buffer.view()}, {1u, output_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 2u;
    const bool dispatched =
        twin::dispatch_manifest(context, "strata_q_q8_1", work, err);
    check(dispatched, std::string(label) + " Q8_1 Metal dispatch", err);
    if (!dispatched) return false;

    const auto failure = twin::compare_bytes(output_buffer.payload(),
                                              expected_bytes);
    if (failure) {
        twin::report_failure(std::string(label).c_str(), options, rows,
                             *failure);
    }
    check(!failure, std::string(label) + " exact Q8_1 bytes");
    const std::array<twin::ByteRange, 1> written = {{{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          std::string(label) + " Q8_1 input and guards unchanged");
    check(!output_buffer.first_unwritten_change(written),
          std::string(label) + " Q8_1 output guards unchanged");
    actual.resize(expected.size());
    std::memcpy(actual.data(), output_buffer.payload().data(), output_bytes);
    return !failure;
}

bool dispatch_shared_q8_0(
    Context& context, std::string_view label, std::span<const Q81> activations,
    std::span<const block_q8_0> weights, std::uint32_t n_in,
    std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
    const twin::Options& options, std::vector<float>& actual) {
    std::string err;
    std::vector<float> expected;
    const bool oracle_ready = twin::q8_0_matvec_reference(
        weights, activations, n_in, n_out, nc, live_rows, expected, err);
    check(oracle_ready, std::string(label) + " CPU Q8_0 oracle", err);
    if (!oracle_ready) return false;

    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q8_0);
    const std::size_t output_count =
        static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count =
        static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0u) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count,
                    sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer row_count_buffer;
    check(input_buffer.allocate(context, std::string(label) + " Q8_0 input",
                                input_bytes, err),
          std::string(label) + " Q8_0 input buffer", err);
    check(weight_buffer.allocate(context, std::string(label) + " Q8_0 weights",
                                 weight_bytes, err),
          std::string(label) + " Q8_0 weight buffer", err);
    check(output_buffer.allocate(context, std::string(label) + " Q8_0 output",
                                 output_bytes, err),
          std::string(label) + " Q8_0 output buffer", err);
    check(sink_buffer.allocate(context, std::string(label) + " Q8_0 sink",
                               sink_bytes, err),
          std::string(label) + " Q8_0 sink buffer", err);
    check(row_count_buffer.allocate(context, std::string(label) + " live rows",
                                    sizeof(live_rows), err),
          std::string(label) + " live row-count buffer", err);
    if (!input_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !row_count_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight_buffer.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count_buffer.payload().data(), &live_rows,
                sizeof(live_rows));
    check(output_buffer.poison_bytes_against(expected_output_bytes, err),
          std::string(label) + " Q8_0 output poison", err);
    check(sink_buffer.poison_bytes_against(expected_sink_bytes, err),
          std::string(label) + " Q8_0 sink poison", err);
    input_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    row_count_buffer.capture_before_dispatch();

    const SharedMVParams params{n_in, n_out};
    const std::uint32_t rows_per_group = n_in / 32u < 32u ? 4u : 1u;
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1u, 1u};
    work.threads_per_group = {32u, 4u, 1u};
    work.buffers = {{0u, input_buffer.view()},
                    {1u, weight_buffer.view()},
                    {2u, output_buffer.view()},
                    {3u, sink_buffer.view()},
                    {4u, row_count_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_q8_0_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, std::string(label) + " Q8_0 Metal dispatch", err);
    if (!dispatched) return false;

    const auto output_failure = twin::compare_bytes(
        output_buffer.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(std::string(label).c_str(), options, live_rows,
                             *output_failure);
    }
    check(!output_failure, std::string(label) + " exact Q8_0 output bytes");
    std::optional<twin::Failure> sink_failure;
    if (sink_count != 0u) {
        sink_failure = twin::compare_bytes(sink_buffer.payload(),
                                           expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(std::string(label).c_str(), options,
                                 nc - live_rows, *sink_failure);
        }
        check(!sink_failure,
              std::string(label) + " exact Q8_0 partial-batch sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input_buffer.first_unwritten_change({}),
          std::string(label) + " Q8_0 input and guards unchanged");
    check(!weight_buffer.first_unwritten_change({}),
          std::string(label) + " Q8_0 weights and guards unchanged");
    check(!row_count_buffer.first_unwritten_change({}),
          std::string(label) + " Q8_0 row count and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          std::string(label) + " Q8_0 output guards unchanged");
    check(!sink_buffer.first_unwritten_change(sink_written),
          std::string(label) + " Q8_0 sink guards unchanged");
    actual.resize(output_count);
    std::memcpy(actual.data(), output_buffer.payload().data(), output_bytes);
    return !output_failure && !sink_failure;
}

bool dispatch_shared_swiglu(
    Context& context, std::uint32_t rows, twin::InputSet set,
    const twin::Options& options, std::span<const float> gate,
    std::span<const float> up, std::span<const float> expected,
    std::vector<float>& actual) {
    const std::string label =
        "Shared expert chain SwiGLU rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    std::string err;
    if (gate.size() != static_cast<std::size_t>(rows) * kMoeExpertWidth ||
        up.size() != gate.size() || expected.size() != gate.size()) {
        check(false, label + " input shape");
        return false;
    }
    const std::size_t input_bytes = gate.size() * sizeof(float);
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    twin::GuardedBuffer output_buffer;
    check(gate_buffer.allocate(context, label + " gate", input_bytes, err),
          label + " gate buffer", err);
    check(up_buffer.allocate(context, label + " up", input_bytes, err),
          label + " up buffer", err);
    check(output_buffer.allocate(context, label + " output", input_bytes, err),
          label + " output buffer", err);
    if (!gate_buffer.view() || !up_buffer.view() || !output_buffer.view()) {
        return false;
    }
    std::memcpy(gate_buffer.payload().data(), gate.data(), input_bytes);
    std::memcpy(up_buffer.payload().data(), up.data(), input_bytes);
    std::vector<twin::tf> expected_values(expected.size());
    std::transform(expected.begin(), expected.end(), expected_values.begin(),
                   [](float value) { return twin::tf{value}; });
    check(output_buffer.poison_floats_against(
              expected_values, twin::Comparison::units, err),
          label + " output poison", err);
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {3u, rows, 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, gate_buffer.view()},
                    {1u, up_buffer.view()},
                    {2u, output_buffer.view()}};
    const bool dispatched =
        twin::dispatch_manifest(context, "strata_shx_swiglu", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* output_data =
        reinterpret_cast<const float*>(output_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto failure = twin::compare_floats(
        std::span<const float>(output_data, expected.size()), expected_values,
        tolerance);
    if (failure) twin::report_failure(label.c_str(), options, rows, *failure);
    check(!failure, label + " output U(16)");
    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, input_bytes}}};
    check(!gate_buffer.first_unwritten_change({}),
          label + " gate and guards unchanged");
    check(!up_buffer.first_unwritten_change({}),
          label + " up and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " output guards unchanged");
    actual.resize(expected.size());
    std::memcpy(actual.data(), output_buffer.payload().data(), input_bytes);
    return !failure;
}

bool dispatch_shared_bf16_gate(
    Context& context, std::uint32_t rows, std::uint32_t nc,
    twin::InputSet set, const twin::Options& options,
    std::span<const float> mixed,
    std::span<const std::uint16_t> gate_weights,
    std::span<const float> expected, std::vector<float>& actual) {
    const std::string label =
        "Shared expert BF16 gate rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    std::string err;
    std::vector<float> reference;
    const bool oracle_ready = twin::bf16_matvec_reference(
        gate_weights, mixed, kMoeInputWidth, 1u, nc, rows, 256u, reference,
        err);
    check(oracle_ready, label + " CPU BF16 oracle", err);
    if (!oracle_ready) return false;
    if (expected.size() != rows || reference.size() != nc) {
        check(false, label + " oracle output shape");
        return false;
    }
    check(!twin::compare_bytes(
              std::span<const std::uint8_t>(
                  reinterpret_cast<const std::uint8_t*>(reference.data()),
                  rows * sizeof(float)),
              std::span<const std::uint8_t>(
                  reinterpret_cast<const std::uint8_t*>(expected.data()),
                  rows * sizeof(float))),
          label + " composed BF16 oracle agrees");

    const std::size_t input_bytes = mixed.size() * sizeof(float);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(std::uint16_t);
    const std::size_t output_bytes = rows * sizeof(float);
    const std::size_t sink_count = nc - rows;
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(reference.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0u) {
        std::memcpy(expected_sink_storage.data(), reference.data() + rows,
                    sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer row_count_buffer;
    check(input_buffer.allocate(context, label + " input", input_bytes, err),
          label + " input buffer", err);
    check(weight_buffer.allocate(context, label + " weights", weight_bytes,
                                 err),
          label + " weight buffer", err);
    check(output_buffer.allocate(context, label + " output", output_bytes,
                                 err),
          label + " output buffer", err);
    check(sink_buffer.allocate(context, label + " sink", sink_bytes, err),
          label + " sink buffer", err);
    check(row_count_buffer.allocate(context, label + " live rows",
                                    sizeof(rows), err),
          label + " live row-count buffer", err);
    if (!input_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !row_count_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), mixed.data(), input_bytes);
    std::memcpy(weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(row_count_buffer.payload().data(), &rows, sizeof(rows));
    check(output_buffer.poison_bytes_against(expected_output_bytes, err),
          label + " output poison", err);
    check(sink_buffer.poison_bytes_against(expected_sink_bytes, err),
          label + " sink poison", err);
    input_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    row_count_buffer.capture_before_dispatch();

    const SharedMVParams params{kMoeInputWidth, 1u};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {1u, 1u, 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, input_buffer.view()},
                    {1u, weight_buffer.view()},
                    {2u, output_buffer.view()},
                    {3u, sink_buffer.view()},
                    {4u, row_count_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_bf16_b256_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto output_failure = twin::compare_bytes(
        output_buffer.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(label.c_str(), options, rows, *output_failure);
    }
    check(!output_failure, label + " exact BF16 output bytes");
    std::optional<twin::Failure> sink_failure;
    if (sink_count != 0u) {
        sink_failure =
            twin::compare_bytes(sink_buffer.payload(), expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(label.c_str(), options, sink_count,
                                 *sink_failure);
        }
        check(!sink_failure, label + " exact partial-batch sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " input and guards unchanged");
    check(!weight_buffer.first_unwritten_change({}),
          label + " weights and guards unchanged");
    check(!row_count_buffer.first_unwritten_change({}),
          label + " row count and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " output guards unchanged");
    check(!sink_buffer.first_unwritten_change(sink_written),
          label + " sink guards unchanged");
    actual.resize(rows);
    std::memcpy(actual.data(), output_buffer.payload().data(), output_bytes);
    return !output_failure && !sink_failure;
}

void make_shared_expert_weights(
    twin::InputSet set, std::uint64_t seed,
    std::vector<block_q8_0>& gate_weights,
    std::vector<block_q8_0>& up_weights,
    std::vector<block_q8_0>& down_weights,
    std::vector<std::uint16_t>& gate_input_weights) {
    const float gu_scale =
        set == twin::InputSet::edge
            ? 0x1p-24f
            : (set == twin::InputSet::fixture ? 0x1p-16f : 0x1p-12f);
    const float down_scale =
        set == twin::InputSet::edge ? 0x1p-18f : 0x1p-14f;
    const std::size_t gu_blocks =
        static_cast<std::size_t>(kMoeExpertWidth) * (kMoeInputWidth / 32u);
    const std::size_t down_blocks =
        static_cast<std::size_t>(kMoeDownWidth) *
        (kMoeDownInputWidth / 32u);
    gate_weights.resize(gu_blocks);
    up_weights.resize(gu_blocks);
    down_weights.resize(down_blocks);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> random_code(-127, 127);
    constexpr std::array<int, 13u> fixture_codes = {
        -127, -91, -63, -31, -7, -1, 0, 3, 19, 47, 79, 103, 127};
    auto fill_weights = [&](std::vector<block_q8_0>& weights, float scale,
                            std::uint32_t salt) {
        for (std::size_t block_index = 0u;
             block_index < weights.size(); ++block_index) {
            block_q8_0& block = weights[block_index];
            block.d = strata::kernels::f16_from_f32(scale);
            for (std::size_t lane = 0u; lane < 32u; ++lane) {
                int code = 0;
                if (set == twin::InputSet::fixture) {
                    code = fixture_codes[(lane * 5u + block_index * 3u +
                                          salt) %
                                         fixture_codes.size()];
                } else {
                    code = random_code(rng);
                    if (set == twin::InputSet::edge &&
                        ((lane + block_index + salt) % 11u == 0u)) {
                        code = (lane + block_index) % 2u == 0u ? -127 : 127;
                    }
                }
                block.qs[lane] = static_cast<std::int8_t>(code);
            }
        }
    };
    fill_weights(gate_weights, gu_scale, 1u);
    fill_weights(up_weights, gu_scale, 5u);
    fill_weights(down_weights, down_scale, 9u);

    gate_input_weights.resize(kMoeInputWidth);
    const float gate_scale =
        set == twin::InputSet::edge
            ? 0x1p-32f
            : (set == twin::InputSet::fixture ? 0x1p-12f : 0x1p-8f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (std::size_t i = 0u; i < gate_input_weights.size(); ++i) {
        const float source =
            set == twin::InputSet::fixture
                ? static_cast<float>(static_cast<int>((i * 7u) % 17u) - 8) /
                      8.0f
                : normal(rng);
        gate_input_weights[i] =
            strata::kernels::bf16_from_f32(source * gate_scale);
    }
}

bool dispatch_shared_combine(
    Context& context, std::uint32_t rows, twin::InputSet set,
    const twin::Options& options, std::span<const float> parts,
    std::span<const float> route_weights, std::span<const float> shared,
    std::span<const float> gate, std::vector<float>& actual) {
    const std::string label =
        "Shared expert chain combine rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    std::string err;
    std::vector<twin::tf> expected;
    const bool oracle_ready = twin::moe_combine_reference(
        parts, route_weights, shared, gate, rows, expected, err);
    check(oracle_ready, label + " CPU combine oracle", err);
    if (!oracle_ready) return false;

    const std::size_t parts_bytes = parts.size_bytes();
    const std::size_t weights_bytes = route_weights.size_bytes();
    const std::size_t shared_bytes = shared.size_bytes();
    const std::size_t gate_bytes = gate.size_bytes();
    const std::size_t output_bytes = expected.size() * sizeof(float);
    twin::GuardedBuffer parts_buffer;
    twin::GuardedBuffer weights_buffer;
    twin::GuardedBuffer shared_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer output_buffer;
    check(parts_buffer.allocate(context, label + " routed parts", parts_bytes,
                                err),
          label + " parts buffer", err);
    check(weights_buffer.allocate(context, label + " route weights",
                                  weights_bytes, err),
          label + " route weights buffer", err);
    check(shared_buffer.allocate(context, label + " shared output",
                                 shared_bytes, err),
          label + " shared buffer", err);
    check(gate_buffer.allocate(context, label + " gate logits", gate_bytes,
                               err),
          label + " gate buffer", err);
    check(output_buffer.allocate(context, label + " combined output",
                                 output_bytes, err),
          label + " output buffer", err);
    if (!parts_buffer.view() || !weights_buffer.view() ||
        !shared_buffer.view() || !gate_buffer.view() ||
        !output_buffer.view()) {
        return false;
    }
    std::memcpy(parts_buffer.payload().data(), parts.data(), parts_bytes);
    std::memcpy(weights_buffer.payload().data(), route_weights.data(),
                weights_bytes);
    std::memcpy(shared_buffer.payload().data(), shared.data(), shared_bytes);
    std::memcpy(gate_buffer.payload().data(), gate.data(), gate_bytes);
    check(output_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " output poison", err);
    parts_buffer.capture_before_dispatch();
    weights_buffer.capture_before_dispatch();
    shared_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {10u, rows, 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, parts_buffer.view()},
                    {1u, weights_buffer.view()},
                    {2u, shared_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, output_buffer.view()}};
    const bool dispatched =
        twin::dispatch_manifest(context, "strata_moe_combine", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* output_data =
        reinterpret_cast<const float*>(output_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto failure = twin::compare_floats(
        std::span<const float>(output_data, expected.size()), expected,
        tolerance);
    if (failure) twin::report_failure(label.c_str(), options, rows, *failure);
    check(!failure, label + " output U(16)");
    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!parts_buffer.first_unwritten_change({}),
          label + " routed parts and guards unchanged");
    check(!weights_buffer.first_unwritten_change({}),
          label + " route weights and guards unchanged");
    check(!shared_buffer.first_unwritten_change({}),
          label + " shared output and guards unchanged");
    check(!gate_buffer.first_unwritten_change({}),
          label + " gate logits and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " combined output guards unchanged");
    actual.resize(expected.size());
    std::memcpy(actual.data(), output_buffer.payload().data(), output_bytes);
    return !failure;
}

bool test_shared_expert_chain_case(
    Context& context, std::uint32_t rows, twin::InputSet set,
    const twin::Options& options, std::span<const block_q8_0> gate_weights,
    std::span<const block_q8_0> up_weights,
    std::span<const block_q8_0> down_weights,
    std::span<const std::uint16_t> gate_input_weights) {
    const std::uint32_t nc =
        rows <= 1u ? 1u : (rows <= 2u ? 2u : (rows <= 4u ? 4u : 8u));
    const std::string label =
        "Shared expert chain rows=" + std::to_string(rows) +
        " nc=" + std::to_string(nc) +
        " set=" + twin::input_set_name(set);
    std::mt19937_64 rng(options.seed ^ (static_cast<std::uint64_t>(rows) << 32u) ^
                        (static_cast<std::uint64_t>(set) << 48u));
    std::string err;
    std::vector<float> active_mixed =
        twin::synthesize_floats(set,
                                static_cast<std::size_t>(rows) * kMoeInputWidth,
                                rng, err);
    if (!err.empty()) {
        check(false, label + " input synthesis", err);
        return false;
    }
    std::vector<float> mixed(static_cast<std::size_t>(nc) * kMoeInputWidth,
                             0.0f);
    std::copy(active_mixed.begin(), active_mixed.end(), mixed.begin());

    twin::MoeSharedExpertOutput composite;
    const bool composite_ready = twin::moe_shared_expert_reference(
        mixed, gate_weights, up_weights, down_weights, gate_input_weights,
        rows, nc, composite, err);
    check(composite_ready, label + " composed scalar oracle", err);
    if (!composite_ready) return false;

    std::vector<Q81> mixed_q8_1;
    bool completed = dispatch_shared_q8_1(
        context, label + " input", mixed, kMoeInputWidth, nc, options,
        mixed_q8_1);
    if (mixed_q8_1.size() != composite.mixed_q8_1.size() ||
        twin::compare_bytes(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(mixed_q8_1.data()),
                mixed_q8_1.size() * sizeof(Q81)),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    composite.mixed_q8_1.data()),
                composite.mixed_q8_1.size() * sizeof(Q81)))) {
        check(false, label + " input quantization matches composed oracle");
        completed = false;
    }

    std::vector<float> gate;
    std::vector<float> up;
    completed &= dispatch_shared_q8_0(
        context, label + " gate/up gate", mixed_q8_1, gate_weights,
        kMoeInputWidth, kMoeExpertWidth, nc, rows, options, gate);
    completed &= dispatch_shared_q8_0(
        context, label + " gate/up up", mixed_q8_1, up_weights,
        kMoeInputWidth, kMoeExpertWidth, nc, rows, options, up);
    const twin::Tolerance activation_tolerance{
        twin::Comparison::units, 16.0, 0.0, 0u};
    std::vector<twin::tf> expected_gate(composite.gate.size());
    std::vector<twin::tf> expected_up(composite.up.size());
    std::transform(composite.gate.begin(), composite.gate.end(),
                   expected_gate.begin(),
                   [](float value) { return twin::tf{value}; });
    std::transform(composite.up.begin(), composite.up.end(),
                   expected_up.begin(),
                   [](float value) { return twin::tf{value}; });
    const auto gate_failure = twin::compare_floats(
        gate, expected_gate, activation_tolerance);
    const auto up_failure = twin::compare_floats(
        up, expected_up, activation_tolerance);
    check(!gate_failure, label + " gate equals composed Q8_0 oracle U(16)");
    check(!up_failure, label + " up equals composed Q8_0 oracle U(16)");
    completed &= !gate_failure && !up_failure;

    std::vector<float> hidden;
    completed &= dispatch_shared_swiglu(
        context, rows, set, options, gate, up, composite.hidden, hidden);
    std::vector<float> hidden_padded(static_cast<std::size_t>(nc) *
                                         kMoeDownInputWidth,
                                     0.0f);
    std::copy(hidden.begin(), hidden.end(), hidden_padded.begin());
    std::vector<Q81> hidden_q8_1;
    completed &= dispatch_shared_q8_1(
        context, label + " hidden", hidden_padded, kMoeDownInputWidth, nc,
        options, hidden_q8_1);
    const auto hidden_quant_failure = twin::compare_bytes(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(hidden_q8_1.data()),
            hidden_q8_1.size() * sizeof(Q81)),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(
                composite.hidden_q8_1.data()),
            composite.hidden_q8_1.size() * sizeof(Q81)));
    check(!hidden_quant_failure,
          label + " hidden Q8_1 matches composed pipeline exactly");
    completed &= !hidden_quant_failure;

    std::vector<float> shared;
    completed &= dispatch_shared_q8_0(
        context, label + " down", hidden_q8_1, down_weights,
        kMoeDownInputWidth, kMoeDownWidth, nc, rows, options, shared);
    const auto shared_failure = twin::compare_bytes(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(shared.data()),
            shared.size() * sizeof(float)),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(composite.shared.data()),
            composite.shared.size() * sizeof(float)));
    check(!shared_failure, label + " shared down output is exact");
    completed &= !shared_failure;

    std::vector<float> gate_logits;
    completed &= dispatch_shared_bf16_gate(
        context, rows, nc, set, options, mixed, gate_input_weights,
        composite.gate_logits, gate_logits);

    constexpr std::size_t kTopK = 10u;
    const std::size_t part_count =
        static_cast<std::size_t>(rows) * kTopK * kMoeInputWidth;
    std::vector<float> routed_parts =
        twin::synthesize_floats(set, part_count, rng, err);
    if (!err.empty()) {
        check(false, label + " routed-part synthesis", err);
        return false;
    }
    std::vector<float> route_weights(static_cast<std::size_t>(rows) * kTopK);
    constexpr std::array<float, kTopK> fixed_route = {
        0.04f, 0.06f, 0.08f, 0.10f, 0.12f,
        0.14f, 0.16f, 0.10f, 0.08f, 0.12f};
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            route_weights[static_cast<std::size_t>(row) * kTopK + rank] =
                set == twin::InputSet::random
                    ? fixed_route[(rank + row) % kTopK]
                    : (set == twin::InputSet::edge
                           ? ((rank + row) % 3u == 0u ? 0.0f
                                                      : fixed_route[rank])
                           : fixed_route[rank]);
        }
        const auto begin = route_weights.begin() + row * kTopK;
        const float total =
            std::accumulate(begin, begin + kTopK, 0.0f);
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            route_weights[static_cast<std::size_t>(row) * kTopK + rank] /=
                total;
        }
    }
    std::vector<float> combined;
    completed &= dispatch_shared_combine(
        context, rows, set, options, routed_parts, route_weights, shared,
        gate_logits, combined);
    return completed;
}

bool test_shared_expert_chain_inputs(Context& context, twin::InputSet set,
                                     const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip shared expert synthetic chain for real inputs; "
                    "model pack fixtures are unavailable\n");
        return true;
    }
    std::vector<block_q8_0> gate_weights;
    std::vector<block_q8_0> up_weights;
    std::vector<block_q8_0> down_weights;
    std::vector<std::uint16_t> gate_input_weights;
    make_shared_expert_weights(
        set, options.seed ^ (static_cast<std::uint64_t>(set) << 52u),
        gate_weights, up_weights, down_weights, gate_input_weights);
    bool completed = true;
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        completed &= test_shared_expert_chain_case(
            context, rows, set, options, gate_weights, up_weights,
            down_weights, gate_input_weights);
    }
    return completed;
}

bool test_moe_combine_case(Context& context, std::uint32_t rows,
                           twin::InputSet set,
                           const twin::Options& options) {
    constexpr std::size_t kWidth = 2560u;
    constexpr std::size_t kTopK = 10u;
    const std::string label =
        "MoE combine rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t part_count =
        static_cast<std::size_t>(rows) * kTopK * kWidth;
    const std::size_t output_count =
        static_cast<std::size_t>(rows) * kWidth;
    std::mt19937_64 rng(options.seed ^ (static_cast<std::uint64_t>(rows) << 32u) ^
                        static_cast<std::uint64_t>(set));
    std::string err;
    std::vector<float> parts =
        twin::synthesize_floats(set, part_count, rng, err);
    if (!err.empty()) {
        check(false, label + " parts synthesis", err);
        return false;
    }
    std::vector<float> shared =
        twin::synthesize_floats(set, output_count, rng, err);
    if (!err.empty()) {
        check(false, label + " shared-output synthesis", err);
        return false;
    }

    std::vector<float> weights(static_cast<std::size_t>(rows) * kTopK);
    std::vector<float> gate(rows);
    std::uniform_real_distribution<float> random_weight(0.01f, 1.0f);
    std::uniform_real_distribution<float> random_gate(-16.0f, 16.0f);
    constexpr std::array<float, kTopK> edge_weight = {
        0.0f, 1.0f, 0.0f, 0.25f, 0.5f,
        0.0f, 0.125f, 1.0f, 0.0f, 0.75f};
    constexpr std::array<float, 8u> edge_gate = {
        -80.0f, -16.0f, -1.0f, -0.0f, 0.0f, 1.0f, 16.0f, 80.0f};
    constexpr std::array<float, 8u> fixture_gate = {
        -32.0f, -8.0f, -1.0f, -0.0f, 0.0f, 1.0f, 8.0f, 32.0f};
    for (std::uint32_t row = 0u; row < rows; ++row) {
        std::array<float, kTopK> raw_weights{};
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            if (set == twin::InputSet::random) {
                raw_weights[rank] = random_weight(rng);
            } else if (set == twin::InputSet::edge) {
                raw_weights[rank] = edge_weight[rank];
            } else {
                raw_weights[rank] =
                    static_cast<float>(1u + ((rank * 7u + row * 3u) % 10u));
            }
        }
        const float total =
            std::accumulate(raw_weights.begin(), raw_weights.end(), 0.0f);
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            weights[static_cast<std::size_t>(row) * kTopK + rank] =
                raw_weights[rank] / total;
        }
        if (set == twin::InputSet::random) {
            gate[row] = random_gate(rng);
        } else if (set == twin::InputSet::edge) {
            gate[row] = edge_gate[row % edge_gate.size()];
        } else {
            gate[row] = fixture_gate[row % fixture_gate.size()];
        }
    }

    if (set == twin::InputSet::fixture) {
        constexpr float kOneMinusEpsilon = 0.5f - 0x1p-24f;
        const std::size_t row_weight_base = 0u;
        weights[row_weight_base] = 0.5f;
        weights[row_weight_base + 1u] = kOneMinusEpsilon;
        weights[row_weight_base + 2u] = 0x1p-24f;
        for (std::size_t rank = 3u; rank < kTopK; ++rank) {
            weights[row_weight_base + rank] = 0.0f;
        }
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            parts[rank * kWidth] = 0.0f;
            parts[rank * kWidth + 1u] = 0.0f;
            parts[rank * kWidth + 2u] = 0.0f;
        }
        parts[0] = -2.0f;
        parts[kWidth] = 2.0f + 0x1p-22f;
        // With weights[0] = 0.5, column 1 has this exact routed sum.
        parts[1u] = -0x1.4aaaacp+0f;
        shared[0] = 0.0f;
        shared[1] = 0x1.fp-1f;
        // Column 2 exposes the exact device sigmoid scale for column 1.
        shared[2] = 1.0f;
        gate[0] = 0x1.62e43p-1f;
    }

    std::vector<twin::tf> expected;
    const bool oracle_ready = twin::moe_combine_reference(
        parts, weights, shared, gate, rows, expected, err);
    check(oracle_ready, label + " CPU scalar oracle", err);
    if (!oracle_ready || expected.size() != output_count) return false;

    const std::size_t parts_bytes = parts.size() * sizeof(float);
    const std::size_t weights_bytes = weights.size() * sizeof(float);
    const std::size_t shared_bytes = shared.size() * sizeof(float);
    const std::size_t gate_bytes = gate.size() * sizeof(float);
    const std::size_t output_bytes = output_count * sizeof(float);
    twin::GuardedBuffer parts_buffer;
    twin::GuardedBuffer weights_buffer;
    twin::GuardedBuffer shared_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer output_buffer;
    check(parts_buffer.allocate(context, label + " routed parts",
                                parts_bytes, err),
          label + " parts buffer", err);
    check(weights_buffer.allocate(context, label + " route weights",
                                  weights_bytes, err),
          label + " route weights buffer", err);
    check(shared_buffer.allocate(context, label + " shared output",
                                 shared_bytes, err),
          label + " shared buffer", err);
    check(gate_buffer.allocate(context, label + " shared gate logits",
                               gate_bytes, err),
          label + " gate buffer", err);
    check(output_buffer.allocate(context, label + " combined output",
                                 output_bytes, err),
          label + " output buffer", err);
    if (!parts_buffer.view() || !weights_buffer.view() ||
        !shared_buffer.view() || !gate_buffer.view() ||
        !output_buffer.view()) {
        return false;
    }
    std::memcpy(parts_buffer.payload().data(), parts.data(), parts_bytes);
    std::memcpy(weights_buffer.payload().data(), weights.data(),
                weights_bytes);
    std::memcpy(shared_buffer.payload().data(), shared.data(), shared_bytes);
    std::memcpy(gate_buffer.payload().data(), gate.data(), gate_bytes);
    check(output_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " output poison", err);
    parts_buffer.capture_before_dispatch();
    weights_buffer.capture_before_dispatch();
    shared_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {10u, rows, 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, parts_buffer.view()},
                    {1u, weights_buffer.view()},
                    {2u, shared_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, output_buffer.view()}};
    const bool dispatched =
        twin::dispatch_manifest(context, "strata_moe_combine", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_data =
        reinterpret_cast<const float*>(output_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto failure = twin::compare_floats(
        std::span<const float>(actual_data, expected.size()), expected,
        tolerance);
    if (failure) {
        twin::report_failure(label.c_str(), options, rows, *failure);
    }
    check(!failure, label + " output U(16)");
    if (set == twin::InputSet::fixture) {
        constexpr float fma_golden = -0x1p-46f;
        check(std::bit_cast<std::uint32_t>(expected[0].value) ==
                      std::bit_cast<std::uint32_t>(fma_golden) &&
                  std::bit_cast<std::uint32_t>(actual_data[0]) ==
                      std::bit_cast<std::uint32_t>(fma_golden),
              label + " first-product plus fused-rank analytic golden");

        constexpr float route_sum = -0x1.4aaaacp-1f;
        constexpr float shared_value = 0x1.fp-1f;
        volatile float rounded_shared_term =
            shared_value * actual_data[2];  // column 2 exposes sigmoid_k(gate)
        const float separate_add = route_sum + rounded_shared_term;
        const float fused_add =
            std::fma(shared_value, actual_data[2], route_sum);
        check(std::bit_cast<std::uint32_t>(actual_data[1]) ==
                      std::bit_cast<std::uint32_t>(separate_add) &&
                  std::bit_cast<std::uint32_t>(separate_add) !=
                      std::bit_cast<std::uint32_t>(fused_add),
              label + " bitwise separate shared multiply/add probe");
    }

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!parts_buffer.first_unwritten_change({}),
          label + " routed parts and guards unchanged");
    check(!weights_buffer.first_unwritten_change({}),
          label + " route weights and guards unchanged");
    check(!shared_buffer.first_unwritten_change({}),
          label + " shared output and guards unchanged");
    check(!gate_buffer.first_unwritten_change({}),
          label + " shared gate and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " output guards unchanged");
    return !failure;
}

bool test_moe_combine_inputs(Context& context, twin::InputSet set,
                             const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE combine real inputs: external model tensors are "
                    "unavailable under REV-002\n");
        return true;
    }
    bool completed = true;
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        completed &= test_moe_combine_case(context, rows, set, options);
    }
    return completed;
}

std::vector<std::uint32_t> make_moe_q2_experts(std::uint32_t experts) {
    return {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, experts - 2u, experts - 1u};
}

float moe_q2_scale(twin::InputSet set, std::mt19937_64& rng,
                   std::size_t index) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 8u> scales = {
            0.0f, 0x1p-24f, 0x1p-14f, 0x1p-10f,
            0x1p-6f, 0.125f, 0.5f, 1.0f};
        return scales[index % scales.size()];
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::array<float, 8u> scales = {
            0.0f, 0.03125f, 0.0625f, 0.125f,
            0.25f, 0.5f, 0.75f, 1.0f};
        return scales[index % scales.size()];
    }
    std::uniform_real_distribution<float> exponent(-12.0f, -4.0f);
    return std::exp2(exponent(rng));
}

void make_moe_q2_weights(twin::InputSet set, std::uint64_t seed,
                         std::vector<block_q2_0>& gate,
                         std::vector<block_q2_0>& up) {
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeQ2BlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(kMoeActiveQ2Experts) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<unsigned> random_byte(0u, 255u);
    constexpr std::array<std::uint8_t, 8u> edge_bytes = {
        0x00u, 0x55u, 0xaau, 0xffu, 0x1bu, 0xe4u, 0x39u, 0xc6u};

    for (std::uint32_t slot = 0; slot < kMoeActiveQ2Experts; ++slot) {
        for (std::uint32_t role = 0; role < 2u; ++role) {
            std::vector<block_q2_0>& weights = role == 0u ? gate : up;
            for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0;
                     block < kMoeQ2BlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) * kMoeQ2BlocksPerRow +
                        block;
                    block_q2_0& weight = weights[index];
                    const std::size_t scale_index =
                        static_cast<std::size_t>(slot) * 31u +
                        static_cast<std::size_t>(role) * 17u +
                        static_cast<std::size_t>(row) * 7u + block;
                    weight.d = strata::kernels::f16_from_f32(
                        moe_q2_scale(set, rng, scale_index));
                    for (std::size_t byte = 0; byte < std::size(weight.qs);
                         ++byte) {
                        if (set == twin::InputSet::edge) {
                            weight.qs[byte] = edge_bytes[
                                (slot + role + row + block + byte) %
                                edge_bytes.size()];
                        } else if (set == twin::InputSet::fixture) {
                            weight.qs[byte] = static_cast<std::uint8_t>(
                                (slot * 29u + role * 61u + row * 7u +
                                 block * 13u + byte * 19u) &
                                0xffu);
                        } else {
                            weight.qs[byte] =
                                static_cast<std::uint8_t>(random_byte(rng));
                        }
                    }
                }
            }
        }
    }
}

void make_moe_down_q2_0_weights(twin::InputSet set, std::uint64_t seed,
                                std::vector<block_q2_0>& weights,
                                std::uint32_t active_slots =
                                    kMoeActiveQ2Experts) {
    constexpr std::array<std::uint8_t, 8u> edge_bytes = {
        0x00u, 0x55u, 0xaau, 0xffu, 0x1bu, 0xe4u, 0x39u, 0xc6u};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeDownWidth) * kMoeDownQ2BlocksPerRow;
    weights.resize(static_cast<std::size_t>(active_slots) *
                   blocks_per_expert);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<unsigned> random_byte(0u, 255u);
    for (std::uint32_t slot = 0u; slot < active_slots; ++slot) {
        for (std::uint32_t row = 0u; row < kMoeDownWidth; ++row) {
            for (std::uint32_t block = 0u;
                 block < kMoeDownQ2BlocksPerRow; ++block) {
                const std::size_t index =
                    static_cast<std::size_t>(slot) * blocks_per_expert +
                    static_cast<std::size_t>(row) *
                        kMoeDownQ2BlocksPerRow +
                    block;
                block_q2_0& weight = weights[index];
                const std::size_t scale_index =
                    static_cast<std::size_t>(slot) * 31u +
                    static_cast<std::size_t>(row) * 7u + block;
                weight.d = strata::kernels::f16_from_f32(
                    moe_q2_scale(set, rng, scale_index));
                for (std::size_t byte = 0u; byte < std::size(weight.qs);
                     ++byte) {
                    if (set == twin::InputSet::edge) {
                        weight.qs[byte] = edge_bytes[
                            (slot + row + block + byte) % edge_bytes.size()];
                    } else if (set == twin::InputSet::fixture) {
                        weight.qs[byte] = static_cast<std::uint8_t>(
                            (slot * 29u + row * 7u + block * 13u +
                             byte * 19u) &
                            0xffu);
                    } else {
                        weight.qs[byte] =
                            static_cast<std::uint8_t>(random_byte(rng));
                    }
                }
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        for (std::uint32_t slot = 0u; slot < active_slots; ++slot) {
            block_q2_0& weight =
                weights[static_cast<std::size_t>(slot) * blocks_per_expert];
            weight.qs[0] =
                static_cast<std::uint8_t>((weight.qs[0] & 0xfcu) | 0x02u);
        }
    }
}

bool make_moe_down_q2_0_activations(
    std::uint32_t entries, twin::InputSet set, std::uint64_t seed,
    std::vector<Q81>& activations, std::string& err) {
    if (entries == 0u || entries > 80u) {
        err = "MoE down activation row count is invalid";
        return false;
    }
    std::mt19937_64 rng(seed);
    std::vector<float> source =
        twin::synthesize_floats(set,
                                static_cast<std::size_t>(entries) *
                                    kMoeDownInputWidth,
                                rng, err);
    if (!err.empty()) return false;
    return twin::quantize_q8_1_reference(
        source, kMoeDownInputWidth, entries, activations, err);
}

float moe_iq4_nl_scale(twin::InputSet set, std::mt19937_64& rng,
                       std::size_t index) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 8u> scales = {
            0.0f, 0x1p-24f, 0x1p-14f, 0x1p-10f,
            0x1p-6f, 0.125f, 0.5f, 1.0f};
        return scales[index % scales.size()];
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::array<float, 8u> scales = {
            0.0f, 0.03125f, 0.0625f, 0.125f,
            0.25f, 0.5f, 0.75f, 1.0f};
        return scales[index % scales.size()];
    }
    std::uniform_real_distribution<float> exponent(-12.0f, -4.0f);
    return std::exp2(exponent(rng));
}

void set_moe_iq4_nl_code(block_iq4_nl& weight, std::uint32_t position,
                         std::uint32_t code) {
    const std::uint32_t byte = position & 0x0fu;
    const std::uint32_t shift = position < 16u ? 0u : 4u;
    const std::uint8_t mask = static_cast<std::uint8_t>(0x0fu << shift);
    weight.qs[byte] = static_cast<std::uint8_t>(
        (weight.qs[byte] & static_cast<std::uint8_t>(~mask)) |
        static_cast<std::uint8_t>(code << shift));
}

void make_moe_down_iq4_nl_weights(
    twin::InputSet set, std::uint64_t seed,
    std::vector<block_iq4_nl>& weights,
    std::uint32_t active_slots = kMoeActiveQ2Experts) {
    constexpr std::array<std::uint8_t, 8u> edge_bytes = {
        0x00u, 0x55u, 0xaau, 0xffu, 0x1bu, 0xe4u, 0x39u, 0xc6u};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeDownWidth) *
        kMoeDownIQ4NLBlocksPerRow;
    weights.resize(static_cast<std::size_t>(active_slots) *
                   blocks_per_expert);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<unsigned> random_byte(0u, 255u);
    for (std::uint32_t slot = 0u; slot < active_slots; ++slot) {
        for (std::uint32_t row = 0u; row < kMoeDownWidth; ++row) {
            for (std::uint32_t block = 0u;
                 block < kMoeDownIQ4NLBlocksPerRow; ++block) {
                const std::size_t index =
                    static_cast<std::size_t>(slot) * blocks_per_expert +
                    static_cast<std::size_t>(row) *
                        kMoeDownIQ4NLBlocksPerRow +
                    block;
                block_iq4_nl& weight = weights[index];
                const std::size_t scale_index =
                    static_cast<std::size_t>(slot) * 31u +
                    static_cast<std::size_t>(row) * 7u + block;
                weight.d = strata::kernels::f16_from_f32(
                    moe_iq4_nl_scale(set, rng, scale_index));
                for (std::size_t byte = 0u; byte < std::size(weight.qs);
                     ++byte) {
                    if (set == twin::InputSet::edge) {
                        weight.qs[byte] = edge_bytes[
                            (slot + row + block + byte) % edge_bytes.size()];
                    } else if (set == twin::InputSet::fixture) {
                        weight.qs[byte] = static_cast<std::uint8_t>(
                            (slot * 29u + row * 7u + block * 13u +
                             byte * 19u) &
                            0xffu);
                    } else {
                        weight.qs[byte] =
                            static_cast<std::uint8_t>(random_byte(rng));
                    }
                }
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        for (std::uint32_t slot = 0u; slot < active_slots; ++slot) {
            block_iq4_nl& weight =
                weights[static_cast<std::size_t>(slot) * blocks_per_expert];
            for (std::uint32_t index = 0u;
                 index < kMoeIQ4NLPart0Positions.size(); ++index) {
                const std::uint32_t code =
                    index < 14u ? 0u : (index == 14u ? 6u : 8u);
                set_moe_iq4_nl_code(
                    weight, kMoeIQ4NLPart0Positions[index], code);
            }
            for (std::uint32_t index = 0u;
                 index < kMoeIQ4NLPart1Positions.size(); ++index) {
                const std::uint32_t code =
                    index < 9u ? 15u
                               : (index == 9u
                                      ? 11u
                                      : (index == 10u
                                             ? 6u
                                             : (index < 13u ? 8u : 0u)));
                set_moe_iq4_nl_code(
                    weight, kMoeIQ4NLPart1Positions[index], code);
            }
        }
    }
}

bool make_moe_down_iq4_nl_activations(
    std::uint32_t entries, twin::InputSet set, std::uint64_t seed,
    std::vector<Q81>& activations, std::string& err) {
    if (entries == 0u || entries > 80u) {
        err = "MoE IQ4_NL down activation row count is invalid";
        return false;
    }
    std::mt19937_64 rng(seed);
    std::vector<float> source =
        twin::synthesize_floats(set,
                                static_cast<std::size_t>(entries) *
                                    kMoeDownInputWidth,
                                rng, err);
    if (!err.empty()) return false;
    return twin::quantize_q8_1_reference(
        source, kMoeDownInputWidth, entries, activations, err);
}

void pack_moe_iq1_m_scales(
    block_iq1_m& weight, std::uint16_t half_scale_bits,
    const std::array<std::uint8_t, 16u>& local_scales) {
    std::array<std::uint16_t, 4u> packed{};
    for (std::uint32_t part = 0; part < 8u; ++part) {
        const std::uint32_t word = part / 2u;
        const std::uint32_t shift = 6u * (part & 1u);
        packed[word] |= static_cast<std::uint16_t>(
            (std::uint16_t(local_scales[2u * part]) << shift) |
            (std::uint16_t(local_scales[2u * part + 1u]) << (shift + 3u)));
    }
    for (std::uint32_t word = 0; word < packed.size(); ++word) {
        packed[word] |= static_cast<std::uint16_t>(
            ((half_scale_bits >> (4u * word)) & 0x0fu) << 12u);
        weight.scales[2u * word] =
            static_cast<std::uint8_t>(packed[word] & 0xffu);
        weight.scales[2u * word + 1u] =
            static_cast<std::uint8_t>(packed[word] >> 8u);
    }
}

std::uint16_t unpack_moe_iq1_m_half_scale(const block_iq1_m& weight) {
    std::uint16_t bits = 0u;
    for (std::uint32_t word = 0; word < 4u; ++word) {
        const std::uint32_t offset = 2u * word;
        const std::uint16_t packed =
            static_cast<std::uint16_t>(weight.scales[offset]) |
            (static_cast<std::uint16_t>(weight.scales[offset + 1u]) << 8u);
        bits |= static_cast<std::uint16_t>(
            ((packed >> 12u) & 0x0fu) << (4u * word));
    }
    return bits;
}

void make_moe_iq1_m_weights(twin::InputSet set, std::uint64_t seed,
                            std::vector<block_iq1_m>& gate,
                            std::vector<block_iq1_m>& up) {
    constexpr std::array<std::uint16_t, 8u> edge_scales = {
        0x0000u, 0x0001u, 0x0400u, 0x1000u,
        0x3555u, 0x3c00u, 0x4000u, 0x4200u};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeIQ1MBlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(kMoeActiveQ2Experts) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> random_byte(0u, 255u);
    std::uniform_int_distribution<std::uint32_t> random_grid(0u, 2047u);
    std::uniform_real_distribution<float> random_log_scale(-12.0f, -4.0f);

    for (std::uint32_t slot = 0; slot < kMoeActiveQ2Experts; ++slot) {
        for (std::uint32_t role = 0; role < 2u; ++role) {
            std::vector<block_iq1_m>& weights = role == 0u ? gate : up;
            for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0;
                     block < kMoeIQ1MBlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) * kMoeIQ1MBlocksPerRow +
                        block;
                    block_iq1_m& weight = weights[index];
                    const std::size_t salt =
                        static_cast<std::size_t>(slot) * 17u +
                        static_cast<std::size_t>(role) * 29u +
                        static_cast<std::size_t>(row) * 11u + block * 7u;
                    if (set == twin::InputSet::fixture) {
                        const auto& fixture =
                            twin::kIQ1MToolFixture[
                                salt % twin::kIQ1MToolFixture.size()];
                        std::memcpy(&weight, fixture.data(), sizeof(weight));
                        continue;
                    }
                    const std::uint16_t half_scale_bits =
                        set == twin::InputSet::random
                            ? strata::kernels::f16_from_f32(
                                  std::exp2(random_log_scale(rng)))
                            : edge_scales[salt % edge_scales.size()];
                    std::array<std::uint8_t, 16u> local_scales{};
                    for (std::uint32_t part = 0; part < 8u; ++part) {
                        for (std::uint32_t codeword = 0; codeword < 4u;
                             ++codeword) {
                            const std::uint32_t grid_index =
                                set == twin::InputSet::random
                                    ? random_grid(rng)
                                    : static_cast<std::uint32_t>(
                                          salt * 31u + part * 0x11bu +
                                          codeword * 0x5du +
                                          (set == twin::InputSet::edge
                                               ? 0x7f3u
                                               : 0x35bu)) &
                                          0x7ffu;
                            const std::uint32_t delta =
                                set == twin::InputSet::random
                                    ? random_byte(rng) & 1u
                                    : static_cast<std::uint32_t>(
                                          (salt + part * 3u + codeword * 5u) &
                                          1u);
                            const std::uint32_t qhl =
                                ((grid_index >> 8u) & 0x07u) | (delta << 3u);
                            weight.qs[4u * part + codeword] =
                                static_cast<std::uint8_t>(grid_index & 0xffu);
                            const std::uint32_t qh_index =
                                2u * part + codeword / 2u;
                            const std::uint32_t qh_shift =
                                4u * (codeword & 1u);
                            weight.qh[qh_index] |=
                                static_cast<std::uint8_t>(qhl << qh_shift);
                        }
                        for (std::uint32_t half = 0; half < 2u; ++half) {
                            const std::uint32_t scale_index = 2u * part + half;
                            local_scales[scale_index] =
                                set == twin::InputSet::random
                                    ? static_cast<std::uint8_t>(
                                          random_byte(rng) & 0x07u)
                                    : static_cast<std::uint8_t>(
                                          (salt + part * 3u + half * 5u +
                                           (set == twin::InputSet::edge ? 7u
                                                                        : 1u)) &
                                          0x07u);
                        }
                    }
                    pack_moe_iq1_m_scales(weight, half_scale_bits,
                                          local_scales);
                }
            }
        }
    }
}

void make_moe_iq2_xxs_weights(twin::InputSet set, std::uint64_t seed,
                              std::vector<block_iq2_xxs>& gate,
                              std::vector<block_iq2_xxs>& up,
                              std::uint32_t active_slots =
                                  kMoeActiveQ2Experts) {
    constexpr std::array<std::uint16_t, 8u> edge_scales = {
        0x0000u, 0x0001u, 0x0400u, 0x1000u,
        0x3555u, 0x3c00u, 0x4000u, 0x4200u};
    constexpr std::array<std::uint16_t, 8u> edge_words = {
        0x0000u, 0xffffu, 0x5555u, 0xaaaau,
        0x00ffu, 0xff00u, 0x3333u, 0xccccu};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ2XXSBlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(active_slots) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> random_word(0u, 0xffffu);
    std::uniform_real_distribution<float> random_log_scale(-12.0f, -4.0f);

    for (std::uint32_t slot = 0; slot < active_slots; ++slot) {
        for (std::uint32_t role = 0; role < 2u; ++role) {
            std::vector<block_iq2_xxs>& weights =
                role == 0u ? gate : up;
            for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0;
                     block < kMoeIQ2XXSBlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) *
                            kMoeIQ2XXSBlocksPerRow +
                        block;
                    block_iq2_xxs& weight = weights[index];
                    const std::size_t salt =
                        static_cast<std::size_t>(slot) * 19u +
                        static_cast<std::size_t>(role) * 31u +
                        static_cast<std::size_t>(row) * 13u + block * 7u;
                    if (set == twin::InputSet::fixture) {
                        weight.d = strata::kernels::f16_from_f32(1.0f);
                        std::copy(twin::kIQ2XXSMoePackedWords.begin(),
                                  twin::kIQ2XXSMoePackedWords.end(),
                                  weight.qs);
                        continue;
                    }
                    weight.d = set == twin::InputSet::random
                                   ? strata::kernels::f16_from_f32(
                                         std::exp2(random_log_scale(rng)))
                                   : edge_scales[salt % edge_scales.size()];
                    for (std::size_t word = 0;
                         word < std::size(weight.qs); ++word) {
                        if (set == twin::InputSet::edge) {
                            weight.qs[word] = edge_words[
                                (salt + word * 3u) % edge_words.size()];
                        } else {
                            weight.qs[word] =
                                static_cast<std::uint16_t>(random_word(rng));
                        }
                    }
                }
            }
        }
    }
}

void make_moe_iq2_xs_weights(twin::InputSet set, std::uint64_t seed,
                             std::vector<block_iq2_xs>& gate,
                             std::vector<block_iq2_xs>& up,
                             std::uint32_t active_slots =
                                 kMoeActiveQ2Experts) {
    constexpr std::array<std::uint16_t, 8u> edge_scales = {
        0x0000u, 0x0001u, 0x0400u, 0x1000u,
        0x3555u, 0x3c00u, 0x4000u, 0x4200u};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ2XSBlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(active_slots) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> random_word(0u, 0xffffu);
    std::uniform_int_distribution<std::uint32_t> random_byte(0u, 0xffu);
    std::uniform_real_distribution<float> random_log_scale(-12.0f, -4.0f);

    for (std::uint32_t slot = 0; slot < active_slots; ++slot) {
        for (std::uint32_t role = 0; role < 2u; ++role) {
            std::vector<block_iq2_xs>& weights = role == 0u ? gate : up;
            for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0;
                     block < kMoeIQ2XSBlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) *
                            kMoeIQ2XSBlocksPerRow +
                        block;
                    block_iq2_xs& weight = weights[index];
                    const std::size_t salt =
                        static_cast<std::size_t>(slot) * 19u +
                        static_cast<std::size_t>(role) * 31u +
                        static_cast<std::size_t>(row) * 13u + block * 7u;
                    if (set == twin::InputSet::fixture) {
                        weight.d = strata::kernels::f16_from_f32(1.0f);
                        std::copy(twin::kIQ2XSMoePackedWords.begin(),
                                  twin::kIQ2XSMoePackedWords.end(),
                                  weight.qs);
                        std::copy(twin::kIQ2XSMoeScales.begin(),
                                  twin::kIQ2XSMoeScales.end(),
                                  weight.scales);
                        continue;
                    }
                    weight.d = set == twin::InputSet::random
                                   ? strata::kernels::f16_from_f32(
                                         std::exp2(random_log_scale(rng)))
                                   : edge_scales[salt % edge_scales.size()];
                    for (std::size_t word = 0;
                         word < std::size(weight.qs); ++word) {
                        weight.qs[word] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint16_t>(
                                      0x0101u * ((salt + word * 3u) & 0xffu))
                                : static_cast<std::uint16_t>(random_word(rng));
                    }
                    for (std::size_t part = 0;
                         part < std::size(weight.scales); ++part) {
                        weight.scales[part] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      (salt + part * 37u) & 0xffu)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                }
            }
        }
    }
}

void make_moe_iq2_s_weights(twin::InputSet set, std::uint64_t seed,
                            std::vector<block_iq2_s>& gate,
                            std::vector<block_iq2_s>& up,
                            std::uint32_t active_slots =
                                kMoeActiveQ2Experts) {
    constexpr std::array<std::uint16_t, 8u> edge_scales = {
        0x0000u, 0x0001u, 0x0400u, 0x1000u,
        0x3555u, 0x3c00u, 0x4000u, 0x4200u};
    constexpr std::array<std::uint8_t, 8u> edge_qh = {
        0x00u, 0xffu, 0x55u, 0xaau, 0x1bu, 0xe4u, 0x3cu, 0xc3u};
    constexpr std::array<std::uint8_t, 8u> edge_local_scales = {
        0x00u, 0x01u, 0x0fu, 0x10u, 0x7fu, 0x80u, 0xa5u, 0xffu};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ2SBlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(active_slots) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> random_byte(0u, 0xffu);
    std::uniform_real_distribution<float> random_log_scale(-12.0f, -4.0f);

    for (std::uint32_t slot = 0; slot < active_slots; ++slot) {
        for (std::uint32_t role = 0; role < 2u; ++role) {
            std::vector<block_iq2_s>& weights = role == 0u ? gate : up;
            for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0;
                     block < kMoeIQ2SBlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) *
                            kMoeIQ2SBlocksPerRow +
                        block;
                    block_iq2_s& weight = weights[index];
                    const std::size_t salt =
                        static_cast<std::size_t>(slot) * 19u +
                        static_cast<std::size_t>(role) * 31u +
                        static_cast<std::size_t>(row) * 13u + block * 7u;
                    if (set == twin::InputSet::fixture) {
                        weight.d = strata::kernels::f16_from_f32(1.0f);
                        std::copy(twin::kIQ2SMoePackedBytes.begin(),
                                  twin::kIQ2SMoePackedBytes.end(),
                                  weight.qs);
                        std::copy(twin::kIQ2SMoeHighCodes.begin(),
                                  twin::kIQ2SMoeHighCodes.end(), weight.qh);
                        std::copy(twin::kIQ2SMoeScales.begin(),
                                  twin::kIQ2SMoeScales.end(), weight.scales);
                        continue;
                    }
                    weight.d = set == twin::InputSet::random
                                   ? strata::kernels::f16_from_f32(
                                         std::exp2(random_log_scale(rng)))
                                   : edge_scales[salt % edge_scales.size()];
                    for (std::size_t byte = 0;
                         byte < std::size(weight.qs); ++byte) {
                        weight.qs[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 13u + byte * 29u)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                    for (std::size_t part = 0;
                         part < std::size(weight.qh); ++part) {
                        weight.qh[part] =
                            set == twin::InputSet::edge
                                ? edge_qh[(salt + part) % edge_qh.size()]
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                    for (std::size_t part = 0;
                         part < std::size(weight.scales); ++part) {
                        weight.scales[part] =
                            set == twin::InputSet::edge
                                ? edge_local_scales[
                                      (salt + part) % edge_local_scales.size()]
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                }
            }
        }
    }
}

void make_moe_iq3_xxs_weights(twin::InputSet set, std::uint64_t seed,
                              std::vector<block_iq3_xxs>& gate,
                              std::vector<block_iq3_xxs>& up,
                              std::uint32_t active_slots =
                                  kMoeActiveQ2Experts) {
    constexpr std::array<std::uint16_t, 8u> edge_scales = {
        0x0000u, 0x0001u, 0x0400u, 0x1000u,
        0x3555u, 0x3c00u, 0x4000u, 0x4200u};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ3XXSBlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(active_slots) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> random_byte(0u, 0xffu);
    std::uniform_real_distribution<float> random_log_scale(-12.0f, -4.0f);

    for (std::uint32_t slot = 0; slot < active_slots; ++slot) {
        for (std::uint32_t role = 0; role < 2u; ++role) {
            std::vector<block_iq3_xxs>& weights =
                role == 0u ? gate : up;
            for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0;
                     block < kMoeIQ3XXSBlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) *
                            kMoeIQ3XXSBlocksPerRow +
                        block;
                    block_iq3_xxs& weight = weights[index];
                    const std::size_t salt =
                        static_cast<std::size_t>(slot) * 19u +
                        static_cast<std::size_t>(role) * 31u +
                        static_cast<std::size_t>(row) * 13u + block * 7u;
                    if (set == twin::InputSet::fixture) {
                        weight.d = strata::kernels::f16_from_f32(1.0f);
                        std::copy(twin::kIQ3XXSMoeIndices.begin(),
                                  twin::kIQ3XXSMoeIndices.end(),
                                  weight.qs + 56u);
                        std::copy(twin::kIQ3XXSMoeAux.begin(),
                                  twin::kIQ3XXSMoeAux.end(),
                                  weight.qs + 92u);
                        continue;
                    }
                    weight.d = set == twin::InputSet::random
                                   ? strata::kernels::f16_from_f32(
                                         std::exp2(random_log_scale(rng)))
                                   : edge_scales[salt % edge_scales.size()];
                    for (std::size_t byte = 0;
                         byte < std::size(weight.qs); ++byte) {
                        weight.qs[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 13u + byte * 29u)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                }
            }
        }
    }
}

void make_moe_iq3_s_weights(twin::InputSet set, std::uint64_t seed,
                            std::vector<block_iq3_s>& gate,
                            std::vector<block_iq3_s>& up,
                            std::uint32_t active_slots =
                                kMoeActiveQ2Experts) {
    constexpr std::array<std::uint16_t, 8u> edge_scales = {
        0x0000u, 0x0001u, 0x0400u, 0x1000u,
        0x3555u, 0x3c00u, 0x4000u, 0x4200u};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeIQ3SBlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(active_slots) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> random_byte(0u, 0xffu);
    std::uniform_real_distribution<float> random_log_scale(-12.0f, -4.0f);

    for (std::uint32_t slot = 0; slot < active_slots; ++slot) {
        for (std::uint32_t role = 0; role < 2u; ++role) {
            std::vector<block_iq3_s>& weights = role == 0u ? gate : up;
            for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0;
                     block < kMoeIQ3SBlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) *
                            kMoeIQ3SBlocksPerRow +
                        block;
                    block_iq3_s& weight = weights[index];
                    const std::size_t salt =
                        static_cast<std::size_t>(slot) * 19u +
                        static_cast<std::size_t>(role) * 31u +
                        static_cast<std::size_t>(row) * 13u + block * 7u;
                    if (set == twin::InputSet::fixture) {
                        weight.d = strata::kernels::f16_from_f32(1.0f);
                        std::copy(twin::kIQ3SMoeQs.begin(),
                                  twin::kIQ3SMoeQs.end(), weight.qs);
                        std::copy(twin::kIQ3SMoeQh.begin(),
                                  twin::kIQ3SMoeQh.end(), weight.qh);
                        std::copy(twin::kIQ3SMoeSigns.begin(),
                                  twin::kIQ3SMoeSigns.end(), weight.signs);
                        std::copy(twin::kIQ3SMoeScales.begin(),
                                  twin::kIQ3SMoeScales.end(), weight.scales);
                        continue;
                    }
                    weight.d = set == twin::InputSet::random
                                   ? strata::kernels::f16_from_f32(
                                         std::exp2(random_log_scale(rng)))
                                   : edge_scales[salt % edge_scales.size()];
                    for (std::size_t byte = 0;
                         byte < std::size(weight.qs); ++byte) {
                        weight.qs[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 13u + byte * 29u)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                    for (std::size_t byte = 0;
                         byte < std::size(weight.qh); ++byte) {
                        weight.qh[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 7u + byte * 53u + 0x81u)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                    for (std::size_t byte = 0;
                         byte < std::size(weight.signs); ++byte) {
                        weight.signs[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 11u + byte * 31u + 0x49u)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                    for (std::size_t byte = 0;
                         byte < std::size(weight.scales); ++byte) {
                        weight.scales[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 3u + byte * 0x35u + 0x8bu)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                }
            }
        }
    }
}

void make_moe_iq4_xs_weights(twin::InputSet set, std::uint64_t seed,
                             std::vector<block_iq4_xs>& gate,
                             std::vector<block_iq4_xs>& up,
                             std::uint32_t active_slots =
                                 kMoeActiveQ2Experts) {
    constexpr std::array<std::uint16_t, 8u> edge_scales = {
        0x0000u, 0x0001u, 0x0400u, 0x1000u,
        0x3555u, 0x3c00u, 0x4000u, 0x4200u};
    const std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeIQ4XSBlocksPerRow;
    const std::size_t count =
        static_cast<std::size_t>(active_slots) * blocks_per_expert;
    gate.resize(count);
    up.resize(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> random_byte(0u, 0xffu);
    std::uniform_int_distribution<std::uint32_t> random_halfword(0u, 0xffffu);
    std::uniform_real_distribution<float> random_log_scale(-12.0f, -4.0f);

    for (std::uint32_t slot = 0u; slot < active_slots; ++slot) {
        for (std::uint32_t role = 0u; role < 2u; ++role) {
            std::vector<block_iq4_xs>& weights =
                role == 0u ? gate : up;
            for (std::uint32_t row = 0u; row < kMoeExpertWidth; ++row) {
                for (std::uint32_t block = 0u;
                     block < kMoeIQ4XSBlocksPerRow; ++block) {
                    const std::size_t index =
                        static_cast<std::size_t>(slot) * blocks_per_expert +
                        static_cast<std::size_t>(row) *
                            kMoeIQ4XSBlocksPerRow +
                        block;
                    block_iq4_xs& weight = weights[index];
                    const std::size_t salt =
                        static_cast<std::size_t>(slot) * 19u +
                        static_cast<std::size_t>(role) * 31u +
                        static_cast<std::size_t>(row) * 13u + block * 7u;
                    if (set == twin::InputSet::fixture) {
                        weight = twin::iq4_xs_moe_fixture();
                        continue;
                    }
                    weight.d =
                        set == twin::InputSet::random
                            ? strata::kernels::f16_from_f32(
                                  std::exp2(random_log_scale(rng)))
                            : edge_scales[salt % edge_scales.size()];
                    weight.scales_h =
                        set == twin::InputSet::edge
                            ? static_cast<std::uint16_t>(
                                  salt * 0x0317u + 0x8421u)
                            : static_cast<std::uint16_t>(
                                  random_halfword(rng));
                    for (std::size_t byte = 0;
                         byte < std::size(weight.scales_l); ++byte) {
                        weight.scales_l[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 3u + byte * 0x35u + 0x8bu)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                    for (std::size_t byte = 0;
                         byte < std::size(weight.qs); ++byte) {
                        weight.qs[byte] =
                            set == twin::InputSet::edge
                                ? static_cast<std::uint8_t>(
                                      salt * 13u + byte * 29u)
                                : static_cast<std::uint8_t>(random_byte(rng));
                    }
                }
            }
        }
    }
}

bool make_moe_q2_activations(std::uint32_t rows, twin::InputSet set,
                             std::uint64_t seed,
                             std::vector<Q81>& activations,
                             std::string& err) {
    const std::size_t input_count =
        static_cast<std::size_t>(rows) * kMoeInputWidth;
    std::mt19937_64 rng(seed);
    std::vector<float> source(input_count, 0.0f);
    if (set == twin::InputSet::random) {
        source = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) return false;
    } else {
        constexpr std::array<int, 16u> codes = {
            -127, -111, -89, -67, -43, -19, -3, 1,
            7,    23,   41,  59,  83,  101, 119, 127};
        const std::size_t blocks_per_row = kMoeInputWidth / 32u;
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::size_t block = 0; block < blocks_per_row; ++block) {
                const std::size_t base =
                    static_cast<std::size_t>(row) * kMoeInputWidth +
                    block * 32u;
                const auto install_scaled_codes =
                    [&](float step, std::uint32_t salt) {
                        for (std::size_t lane = 0; lane < 32u; ++lane) {
                            const std::size_t code_index =
                                (lane * 5u + row * 7u + block * 3u + salt) %
                                codes.size();
                            source[base + lane] =
                                step * static_cast<float>(codes[code_index]);
                        }
                    };
                if (set == twin::InputSet::edge) {
                    switch ((block + row * 3u) % 8u) {
                        case 0u:
                            for (std::size_t lane = 0; lane < 32u; ++lane) {
                                source[base + lane] =
                                    lane % 2u == 0u ? 0.0f : -0.0f;
                            }
                            break;
                        case 1u:
                            install_scaled_codes(0x1p-24f, row + 1u);
                            break;
                        case 2u:
                            install_scaled_codes(0x1p-30f, row + 3u);
                            break;
                        case 3u:
                            install_scaled_codes(0.125f, row + 5u);
                            source[base] = 127.0f * 0.125f;
                            source[base + 1u] = 1.5f * 0.125f;
                            source[base + 2u] = -1.5f * 0.125f;
                            break;
                        case 4u:
                            install_scaled_codes(0x1p-14f, row + 7u);
                            break;
                        case 5u:
                            install_scaled_codes(0x1p-4f, row + 9u);
                            source[base] = row % 2u == 0u ? 65504.0f
                                                          : -65504.0f;
                            source[base + 1u] = -source[base];
                            break;
                        case 6u:
                            install_scaled_codes(0x1p-8f, row + 11u);
                            break;
                        default:
                            install_scaled_codes(0x1p-6f, row + 13u);
                            break;
                    }
                } else {
                    switch ((block + row * 5u) % 6u) {
                        case 0u:
                            for (std::size_t lane = 0; lane < 32u; ++lane) {
                                source[base + lane] =
                                    lane % 2u == 0u ? 0.0f : -0.0f;
                            }
                            break;
                        case 1u:
                            install_scaled_codes(0x1p-24f, row + 2u);
                            break;
                        case 2u:
                            install_scaled_codes(0x1p-12f, row + 4u);
                            break;
                        case 3u:
                            install_scaled_codes(0x1p-6f, row + 6u);
                            break;
                        case 4u:
                            install_scaled_codes(0.125f, row + 8u);
                            source[base] = 127.0f * 0.125f;
                            source[base + 1u] = 1.5f * 0.125f;
                            source[base + 2u] = -1.5f * 0.125f;
                            break;
                        default:
                            install_scaled_codes(
                                std::ldexp(1.0f, -8 + int((row + block) % 3u)),
                                row + 10u);
                            break;
                    }
                }
            }
        }
        err.clear();
    }
    if (!err.empty()) return false;
    if (!twin::quantize_q8_1_reference(source, kMoeInputWidth, rows,
                                       activations, err)) {
        return false;
    }
    return true;
}

bool test_moe_gu_q2_0_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_q2_0> gate_weights,
    std::span<const block_q2_0> up_weights,
    twin::GuardedBuffer& address_buffer) {
    const std::string label =
        "MoE grouped Q2_0 gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            ids[static_cast<std::size_t>(row) * kTopK + rank] =
                static_cast<std::int32_t>(
                    active_experts[(rank + row) % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    check(twin::moe_plan_reference(ids, rows, experts, false,
                                   kMoeGroupedLayer, resident, initial_usage,
                                   plan, expected_usage, err),
          label + " plan oracle", err);
    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    const bool activations_ready = make_moe_q2_activations(
        rows, set, options.seed ^ experts ^
                      (static_cast<std::uint64_t>(set) << 48u),
        activations, err);
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    check(twin::moe_gu_q2_0_reference(
              gate_weights, up_weights, activations, expert_slots, plan,
              rows, expected_gate, expected_up, err),
          label + " CPU grouped row-dot oracle", err);
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output",
                               output_bytes, err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(),
                input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_q2_0", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_q2_0_inputs(Context& context, twin::InputSet set,
                             const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped Q2_0 real inputs: external model tensors "
                    "are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_q2_0> gate_weights;
        std::vector<block_q2_0> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_q2_weights(set, weight_seed, gate_weights, up_weights);
        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeQ2BlocksPerRow * sizeof(block_q2_0);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_q2_0);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped Q2_0 gate weights", weight_bytes, err),
              "MoE grouped Q2_0 gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped Q2_0 up weights", weight_bytes, err),
              "MoE grouped Q2_0 up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped expert address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped expert address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped Q2_0 residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_q2_0_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq1_m_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq1_m> gate_weights,
    std::span<const block_iq1_m> up_weights,
    twin::GuardedBuffer& gate_weight_buffer,
    twin::GuardedBuffer& up_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ1_M gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[static_cast<std::size_t>(row) * kTopK + rank] =
                static_cast<std::int32_t>(
                    active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    check(twin::moe_plan_reference(ids, rows, experts, false,
                                   kMoeGroupedLayer, resident, initial_usage,
                                   plan, expected_usage, err),
          label + " plan oracle", err);
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }
    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    const bool activations_ready = make_moe_q2_activations(
        rows, set, options.seed ^ experts ^
                      (static_cast<std::uint64_t>(set) << 48u),
        activations, err);
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    const bool oracle_ready = twin::moe_gu_iq1_m_reference(
        gate_weights, up_weights, activations, expert_slots, plan, rows,
        expected_gate, expected_up, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output",
                               output_bytes, err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(),
                input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq1_m", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_iq1_m_inputs(Context& context, twin::InputSet set,
                              const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ1_M real inputs: external model tensors "
                    "are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq1_m> gate_weights;
        std::vector<block_iq1_m> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_iq1_m_weights(set, weight_seed, gate_weights, up_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq1_m& weight) {
                const std::uint16_t bits =
                    unpack_moe_iq1_m_half_scale(weight);
                return bits >= 0x0c00u && bits <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(gate_weights.begin(), gate_weights.end(),
                            in_scale_range) &&
                std::all_of(up_weights.begin(), up_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ1_M scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }
        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ1MBlocksPerRow * sizeof(block_iq1_m);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_iq1_m);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped IQ1_M gate weights", weight_bytes, err),
              "MoE grouped IQ1_M gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped IQ1_M up weights", weight_bytes, err),
              "MoE grouped IQ1_M up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ1_M address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ1_M address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped IQ1_M residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_iq1_m_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, gate_weight_buffer,
                up_weight_buffer, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq1_m_max_groups(Context& context,
                                  const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    std::iota(active_experts.begin(), active_experts.end(), 0u);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] =
            static_cast<std::int32_t>(slot % kMoeActiveQ2Experts);
    }

    std::vector<block_iq1_m> gate_weights;
    std::vector<block_iq1_m> up_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_iq1_m_weights(twin::InputSet::fixture, weight_seed,
                           gate_weights, up_weights);
    const std::size_t weights_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ1MBlocksPerRow * sizeof(block_iq1_m);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(block_iq1_m);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(gate_weight_buffer.allocate(
              context, "MoE grouped IQ1_M maximum-group gate weights",
              weight_bytes, err),
          "MoE grouped IQ1_M maximum-group gate weight buffer", err);
    check(up_weight_buffer.allocate(
              context, "MoE grouped IQ1_M maximum-group up weights",
              weight_bytes, err),
          "MoE grouped IQ1_M maximum-group up weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ1_M maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ1_M maximum-group address buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    for (std::uint32_t expert_slot = 0; expert_slot < max_groups;
         ++expert_slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[expert_slot];
        const std::size_t weight_slot =
            expert_slot % kMoeActiveQ2Experts;
        addresses[expert_index].role[0] =
            gate_weight_buffer.view().gpu_address +
            weight_slot * weights_per_expert;
        addresses[expert_index].role[1] =
            up_weight_buffer.view().gpu_address +
            weight_slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, "MoE grouped IQ1_M maximum-group residency", err);
    if (!residency_ready) return false;

    return test_moe_gu_iq1_m_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, gate_weights, up_weights,
        gate_weight_buffer, up_weight_buffer, address_buffer, true);
}

bool test_moe_gu_iq2_xxs_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq2_xxs> gate_weights,
    std::span<const block_iq2_xxs> up_weights,
    twin::GuardedBuffer& gate_weight_buffer,
    twin::GuardedBuffer& up_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ2_XXS gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    check(twin::moe_plan_reference(ids, rows, experts, false,
                                   kMoeGroupedLayer, resident, initial_usage,
                                   plan, expected_usage, err),
          label + " plan oracle", err);
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        std::vector<float> source(
            static_cast<std::size_t>(rows) * kMoeInputWidth, 0.0f);
        for (std::uint32_t row = 0; row < rows; ++row) {
            source[static_cast<std::size_t>(row) * kMoeInputWidth +
                   twin::kIQ2XXSMoeOneHotPositions[1]] = 127.0f;
        }
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeInputWidth, rows, activations, err);
    } else {
        activations_ready = make_moe_q2_activations(
            rows, set, options.seed ^ experts ^
                          (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    const bool oracle_ready = twin::moe_gu_iq2_xxs_reference(
        gate_weights, up_weights, activations, expert_slots, plan, rows,
        expected_gate, expected_up, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    if (one_expert_per_entry) {
        constexpr std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ2XXSBlocksPerRow;
        bool known_answers_match = true;
        bool gate_answers_are_distinct = true;
        bool up_answers_are_distinct = true;
        std::array<std::uint32_t, 80u> expected_gate_bits{};
        std::array<std::uint32_t, 80u> expected_up_bits{};
        for (std::uint32_t entry = 0; entry < entry_count; ++entry) {
            const std::int32_t expert = plan.grp_e[entry];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const std::size_t block_index =
                static_cast<std::size_t>(slot) * weight_blocks_per_expert;
            const float gate_scale = strata::kernels::f32_from_f16(
                gate_weights[static_cast<std::size_t>(slot) *
                             weight_blocks_per_expert]
                    .d);
            const float up_scale = strata::kernels::f32_from_f16(
                up_weights[block_index].d);
            const float known_gate = std::fma(
                gate_scale, twin::kIQ2XXSMoeOneHotAnswers[1], 0.0f);
            const float known_up = std::fma(
                up_scale, twin::kIQ2XXSMoeOneHotAnswers[1], 0.0f);
            const float oracle_gate =
                expected_gate[static_cast<std::size_t>(entry) *
                              kMoeExpertWidth]
                    .value;
            const float oracle_up =
                expected_up[static_cast<std::size_t>(entry) *
                            kMoeExpertWidth]
                    .value;
            known_answers_match &=
                std::bit_cast<std::uint32_t>(oracle_gate) ==
                    std::bit_cast<std::uint32_t>(known_gate) &&
                std::bit_cast<std::uint32_t>(oracle_up) ==
                    std::bit_cast<std::uint32_t>(known_up);
            expected_gate_bits[entry] =
                std::bit_cast<std::uint32_t>(known_gate);
            expected_up_bits[entry] = std::bit_cast<std::uint32_t>(known_up);
            for (std::uint32_t previous = 0; previous < entry; ++previous) {
                gate_answers_are_distinct &=
                    expected_gate_bits[previous] != expected_gate_bits[entry];
                up_answers_are_distinct &=
                    expected_up_bits[previous] != expected_up_bits[entry];
            }
        }
        check(known_answers_match,
              label + " expert-specific gate/up one-hot oracle answers");
        check(gate_answers_are_distinct && up_answers_are_distinct,
              label + " all 80 experts have distinct gate/up one-hot answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output",
                               output_bytes, err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(),
                input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq2_xxs", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_iq2_xxs_inputs(Context& context, twin::InputSet set,
                                const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ2_XXS real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq2_xxs> gate_weights;
        std::vector<block_iq2_xxs> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_iq2_xxs_weights(set, weight_seed, gate_weights, up_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq2_xxs& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(gate_weights.begin(), gate_weights.end(),
                            in_scale_range) &&
                std::all_of(up_weights.begin(), up_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ2_XXS scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ2XXSBlocksPerRow * sizeof(block_iq2_xxs);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_iq2_xxs);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped IQ2_XXS gate weights",
                  weight_bytes, err),
              "MoE grouped IQ2_XXS gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped IQ2_XXS up weights",
                  weight_bytes, err),
              "MoE grouped IQ2_XXS up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ2_XXS address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ2_XXS address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped IQ2_XXS residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_iq2_xxs_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, gate_weight_buffer,
                up_weight_buffer, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq2_xxs_max_groups(Context& context,
                                    const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    std::iota(active_experts.begin(), active_experts.end(), 0u);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_iq2_xxs> gate_weights;
    std::vector<block_iq2_xxs> up_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_iq2_xxs_weights(twin::InputSet::fixture, weight_seed,
                             gate_weights, up_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ2XXSBlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::uint16_t unique_gate_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        const std::uint16_t unique_up_scale =
            static_cast<std::uint16_t>(0x3400u + 8u * slot);
        gate_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_gate_scale;
        up_weights[static_cast<std::size_t>(slot) *
                   weight_blocks_per_expert]
            .d = unique_up_scale;
        expert_scales_are_distinct &=
            gate_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_gate_scale &&
            up_weights[static_cast<std::size_t>(slot) *
                       weight_blocks_per_expert]
                .d == unique_up_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_gate_scale !=
                    gate_weights[static_cast<std::size_t>(slot - 1u) *
                                 weight_blocks_per_expert]
                        .d &&
                unique_up_scale !=
                    up_weights[static_cast<std::size_t>(slot - 1u) *
                               weight_blocks_per_expert]
                        .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped IQ2_XXS assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_iq2_xxs);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(block_iq2_xxs);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(gate_weight_buffer.allocate(
              context, "MoE grouped IQ2_XXS maximum-group gate weights",
              weight_bytes, err),
          "MoE grouped IQ2_XXS maximum-group gate weight buffer", err);
    check(up_weight_buffer.allocate(
              context, "MoE grouped IQ2_XXS maximum-group up weights",
              weight_bytes, err),
          "MoE grouped IQ2_XXS maximum-group up weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ2_XXS maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ2_XXS maximum-group address buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    for (std::uint32_t expert_slot = 0; expert_slot < max_groups;
         ++expert_slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[expert_slot];
        addresses[expert_index].role[0] =
            gate_weight_buffer.view().gpu_address +
            expert_slot * weights_per_expert;
        addresses[expert_index].role[1] =
            up_weight_buffer.view().gpu_address +
            expert_slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, "MoE grouped IQ2_XXS maximum-group residency",
          err);
    if (!residency_ready) return false;

    return test_moe_gu_iq2_xxs_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, gate_weights, up_weights,
        gate_weight_buffer, up_weight_buffer, address_buffer, true);
}

bool test_moe_gu_iq2_xxs_onehot_fixture(
    Context& context, const twin::Options& options) {
    constexpr std::uint32_t experts = 256u;
    constexpr std::uint32_t rows = 8u;
    constexpr std::size_t blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ2XXSBlocksPerRow;
    const std::string label = "MoE grouped IQ2_XXS fixed one-hot fixture";

    std::vector<std::int32_t> ids(rows * kTopK, 0);
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    check(plan.n_groups == 1u && plan.grp_start[1] == 80,
          label + " one group spans all 80 entries");

    std::vector<float> source(static_cast<std::size_t>(rows) *
                                  kMoeInputWidth,
                              0.0f);
    for (std::uint32_t row = 0; row < rows; ++row) {
        source[static_cast<std::size_t>(row) * kMoeInputWidth +
               twin::kIQ2XXSMoeOneHotPositions[row]] = 127.0f;
    }
    std::vector<Q81> activations;
    const bool activation_ready = twin::quantize_q8_1_reference(
        source, kMoeInputWidth, rows, activations, err);
    check(activation_ready, label + " Q8_1 one-hot quantization", err);
    if (!activation_ready) return false;

    std::vector<block_iq2_xxs> gate_weights(blocks_per_expert);
    std::vector<block_iq2_xxs> up_weights(blocks_per_expert);
    for (std::uint32_t row = 0; row < kMoeExpertWidth; ++row) {
        block_iq2_xxs& gate_block =
            gate_weights[static_cast<std::size_t>(row) *
                         kMoeIQ2XXSBlocksPerRow];
        block_iq2_xxs& up_block =
            up_weights[static_cast<std::size_t>(row) *
                       kMoeIQ2XXSBlocksPerRow];
        gate_block.d = strata::kernels::f16_from_f32(1.0f);
        up_block.d = strata::kernels::f16_from_f32(1.0f);
        std::copy(twin::kIQ2XXSMoePackedWords.begin(),
                  twin::kIQ2XXSMoePackedWords.end(), gate_block.qs);
        std::copy(twin::kIQ2XXSMoePackedWords.begin(),
                  twin::kIQ2XXSMoePackedWords.end(), up_block.qs);
    }

    std::vector<std::int32_t> expert_slots(experts, -1);
    expert_slots[0] = 0;
    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    check(twin::moe_gu_iq2_xxs_reference(
              gate_weights, up_weights, activations, expert_slots, plan, rows,
              expected_gate, expected_up, err),
          label + " independent grouped oracle", err);
    if (expected_gate.size() !=
            static_cast<std::size_t>(rows * kTopK) * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer address_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t weight_bytes =
        blocks_per_expert * sizeof(block_iq2_xxs);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(gate_weight_buffer.allocate(context, label + " gate weights",
                                     weight_bytes, err),
          label + " gate weight buffer", err);
    check(up_weight_buffer.allocate(context, label + " up weights",
                                    weight_bytes, err),
          label + " up weight buffer", err);
    check(input_buffer.allocate(context, label + " activations", input_bytes,
                                err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " plan", plan_bytes, err),
          label + " plan buffer", err);
    check(address_buffer.allocate(
              context, label + " address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          label + " address buffer", err);
    check(gate_buffer.allocate(context, label + " gate output", output_bytes,
                               err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !input_buffer.view() || !plan_buffer.view() ||
        !address_buffer.view() || !gate_buffer.view() || !up_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    const std::size_t expert_index =
        static_cast<std::size_t>(kMoeGroupedLayer) * experts;
    addresses[expert_index].role[0] =
        gate_weight_buffer.view().gpu_address;
    addresses[expert_index].role[1] = up_weight_buffer.view().gpu_address;
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, label + " residency request", err);
    if (!residency_ready) return false;
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, rows * kTopK, 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq2_xxs", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate, expected_gate.size()),
        expected_gate, tolerance);
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up, expected_up.size()), expected_up,
        tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    check(!up_failure, label + " up outputs U(16)");

    std::array<std::uint32_t, rows * kTopK> gate_known_bits{};
    std::array<std::uint32_t, rows * kTopK> up_known_bits{};
    std::array<std::uint32_t, rows * kTopK> expected_known_bits{};
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t expected_bits = std::bit_cast<std::uint32_t>(
            twin::kIQ2XXSMoeOneHotAnswers[row]);
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::uint32_t entry = row * kTopK + rank;
            gate_known_bits[entry] = std::bit_cast<std::uint32_t>(
                actual_gate[static_cast<std::size_t>(entry) *
                            kMoeExpertWidth]);
            up_known_bits[entry] = std::bit_cast<std::uint32_t>(
                actual_up[static_cast<std::size_t>(entry) *
                          kMoeExpertWidth]);
            expected_known_bits[entry] = expected_bits;
        }
    }
    const auto expected_known_bytes =
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_known_bits.data()),
            sizeof(expected_known_bits));
    check(!twin::compare_bytes(
              std::span<const std::uint8_t>(
                  reinterpret_cast<const std::uint8_t*>(gate_known_bits.data()),
                  sizeof(gate_known_bits)),
              expected_known_bytes),
          label + " exact gate one-hot answers");
    check(!twin::compare_bytes(
              std::span<const std::uint8_t>(
                  reinterpret_cast<const std::uint8_t*>(up_known_bits.data()),
                  sizeof(up_known_bits)),
              expected_known_bytes),
          label + " exact up one-hot answers");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!input_buffer.first_unwritten_change({}),
          label + " activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate output guards unchanged");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up output guards unchanged");
    return !gate_failure && !up_failure;
}

void check_independent_weight_goldens(std::uint32_t experts,
                                     bool divide_weights,
                                     std::span<const float> actual_weights,
                                     const twin::Options& options) {
    const std::size_t expected_size =
        static_cast<std::size_t>(kRows) * kTopK;
    if (actual_weights.size() != expected_size) {
        check(false, "MoE independent weight golden output shape");
        return;
    }

    check_weight_golden(
        "MoE all-equal independent weight golden U(16), experts=" +
            std::to_string(experts) + " div=" +
            std::to_string(divide_weights ? 1 : 0),
        actual_weights.first(kTopK), kEqualWeightBits, options);
    const auto nonuniform_golden =
        divide_weights ? kNonuniformDivideWeightBits
                       : kNonuniformMultiplyWeightBits;
    check_weight_golden(
        "MoE nonuniform independent weight golden U(16), experts=" +
            std::to_string(experts) + " div=" +
            std::to_string(divide_weights ? 1 : 0),
        actual_weights.subspan(2u * kTopK, kTopK), nonuniform_golden, options);
}

bool test_moe_gu_iq2_xs_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq2_xs> gate_weights,
    std::span<const block_iq2_xs> up_weights,
    twin::GuardedBuffer& gate_weight_buffer,
    twin::GuardedBuffer& up_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ2_XS gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        std::vector<float> source(
            static_cast<std::size_t>(rows) * kMoeInputWidth, 0.0f);
        for (std::uint32_t row = 0; row < rows; ++row) {
            source[static_cast<std::size_t>(row) * kMoeInputWidth +
                   twin::kIQ2XSMoeOneHotPositions[1]] = 127.0f;
        }
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeInputWidth, rows, activations, err);
    } else {
        activations_ready = make_moe_q2_activations(
            rows, set, options.seed ^ experts ^
                          (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    const bool oracle_ready = twin::moe_gu_iq2_xs_reference(
        gate_weights, up_weights, activations, expert_slots, plan, rows,
        expected_gate, expected_up, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    if (one_expert_per_entry) {
        constexpr std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ2XSBlocksPerRow;
        bool known_answers_match = true;
        bool gate_answers_are_distinct = true;
        bool up_answers_are_distinct = true;
        std::array<std::uint32_t, 80u> expected_gate_bits{};
        std::array<std::uint32_t, 80u> expected_up_bits{};
        for (std::uint32_t entry = 0; entry < entry_count; ++entry) {
            const std::int32_t expert = plan.grp_e[entry];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const std::size_t block_index =
                static_cast<std::size_t>(slot) * weight_blocks_per_expert;
            const float gate_scale = strata::kernels::f32_from_f16(
                gate_weights[block_index].d);
            const float up_scale = strata::kernels::f32_from_f16(
                up_weights[block_index].d);
            const float known_gate =
                std::fma(gate_scale, twin::kIQ2XSMoeOneHotAnswers[1], 0.0f);
            const float known_up =
                std::fma(up_scale, twin::kIQ2XSMoeOneHotAnswers[1], 0.0f);
            const float oracle_gate =
                expected_gate[static_cast<std::size_t>(entry) *
                              kMoeExpertWidth]
                    .value;
            const float oracle_up =
                expected_up[static_cast<std::size_t>(entry) *
                            kMoeExpertWidth]
                    .value;
            known_answers_match &=
                std::bit_cast<std::uint32_t>(oracle_gate) ==
                    std::bit_cast<std::uint32_t>(known_gate) &&
                std::bit_cast<std::uint32_t>(oracle_up) ==
                    std::bit_cast<std::uint32_t>(known_up);
            expected_gate_bits[entry] =
                std::bit_cast<std::uint32_t>(known_gate);
            expected_up_bits[entry] = std::bit_cast<std::uint32_t>(known_up);
            for (std::uint32_t previous = 0; previous < entry; ++previous) {
                gate_answers_are_distinct &=
                    expected_gate_bits[previous] != expected_gate_bits[entry];
                up_answers_are_distinct &=
                    expected_up_bits[previous] != expected_up_bits[entry];
            }
        }
        check(known_answers_match,
              label + " expert-specific gate/up one-hot oracle answers");
        check(gate_answers_are_distinct && up_answers_are_distinct,
              label + " all 80 experts have distinct gate/up one-hot answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output", output_bytes,
                               err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq2_xs", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_iq2_xs_inputs(Context& context, twin::InputSet set,
                               const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ2_XS real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq2_xs> gate_weights;
        std::vector<block_iq2_xs> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_iq2_xs_weights(set, weight_seed, gate_weights, up_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq2_xs& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(gate_weights.begin(), gate_weights.end(),
                            in_scale_range) &&
                std::all_of(up_weights.begin(), up_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ2_XS scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ2XSBlocksPerRow * sizeof(block_iq2_xs);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_iq2_xs);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped IQ2_XS gate weights", weight_bytes,
                  err),
              "MoE grouped IQ2_XS gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped IQ2_XS up weights", weight_bytes,
                  err),
              "MoE grouped IQ2_XS up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ2_XS address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ2_XS address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped IQ2_XS residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_iq2_xs_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, gate_weight_buffer,
                up_weight_buffer, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq2_xs_max_groups(Context& context,
                                   const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    std::iota(active_experts.begin(), active_experts.end(), 0u);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_iq2_xs> gate_weights;
    std::vector<block_iq2_xs> up_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_iq2_xs_weights(twin::InputSet::fixture, weight_seed,
                             gate_weights, up_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeIQ2XSBlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::uint16_t unique_gate_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        const std::uint16_t unique_up_scale =
            static_cast<std::uint16_t>(0x3400u + 8u * slot);
        gate_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_gate_scale;
        up_weights[static_cast<std::size_t>(slot) *
                   weight_blocks_per_expert]
            .d = unique_up_scale;
        expert_scales_are_distinct &=
            gate_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_gate_scale &&
            up_weights[static_cast<std::size_t>(slot) *
                       weight_blocks_per_expert]
                .d == unique_up_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_gate_scale !=
                    gate_weights[static_cast<std::size_t>(slot - 1u) *
                                 weight_blocks_per_expert]
                        .d &&
                unique_up_scale !=
                    up_weights[static_cast<std::size_t>(slot - 1u) *
                               weight_blocks_per_expert]
                        .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped IQ2_XS assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_iq2_xs);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(block_iq2_xs);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(gate_weight_buffer.allocate(
              context, "MoE grouped IQ2_XS maximum-group gate weights",
              weight_bytes, err),
          "MoE grouped IQ2_XS maximum-group gate weight buffer", err);
    check(up_weight_buffer.allocate(
              context, "MoE grouped IQ2_XS maximum-group up weights",
              weight_bytes, err),
          "MoE grouped IQ2_XS maximum-group up weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ2_XS maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ2_XS maximum-group address buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[slot];
        addresses[expert_index].role[0] =
            gate_weight_buffer.view().gpu_address + slot * weights_per_expert;
        addresses[expert_index].role[1] =
            up_weight_buffer.view().gpu_address + slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, "MoE grouped IQ2_XS maximum-group residency",
          err);
    if (!residency_ready) return false;

    return test_moe_gu_iq2_xs_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, gate_weights, up_weights,
        gate_weight_buffer, up_weight_buffer, address_buffer, true);
}

bool test_moe_gu_iq2_s_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq2_s> gate_weights,
    std::span<const block_iq2_s> up_weights,
    twin::GuardedBuffer& gate_weight_buffer,
    twin::GuardedBuffer& up_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ2_S gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        std::vector<float> source(
            static_cast<std::size_t>(rows) * kMoeInputWidth, 0.0f);
        for (std::uint32_t row = 0; row < rows; ++row) {
            source[static_cast<std::size_t>(row) * kMoeInputWidth +
                   twin::kIQ2SMoeOneHotPositions[7]] = 127.0f;
        }
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeInputWidth, rows, activations, err);
    } else {
        activations_ready = make_moe_q2_activations(
            rows, set, options.seed ^ experts ^
                          (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    const bool oracle_ready = twin::moe_gu_iq2_s_reference(
        gate_weights, up_weights, activations, expert_slots, plan, rows,
        expected_gate, expected_up, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    if (one_expert_per_entry) {
        constexpr std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ2SBlocksPerRow;
        bool known_answers_match = true;
        bool gate_answers_are_distinct = true;
        bool up_answers_are_distinct = true;
        std::array<std::uint32_t, 80u> expected_gate_bits{};
        std::array<std::uint32_t, 80u> expected_up_bits{};
        for (std::uint32_t entry = 0; entry < entry_count; ++entry) {
            const std::int32_t expert = plan.grp_e[entry];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const std::size_t block_index =
                static_cast<std::size_t>(slot) * weight_blocks_per_expert;
            const float gate_scale = strata::kernels::f32_from_f16(
                gate_weights[block_index].d);
            const float up_scale = strata::kernels::f32_from_f16(
                up_weights[block_index].d);
            const float known_gate =
                std::fma(gate_scale, twin::kIQ2SMoeOneHotAnswers[7], 0.0f);
            const float known_up =
                std::fma(up_scale, twin::kIQ2SMoeOneHotAnswers[7], 0.0f);
            const float oracle_gate =
                expected_gate[static_cast<std::size_t>(entry) *
                              kMoeExpertWidth]
                    .value;
            const float oracle_up =
                expected_up[static_cast<std::size_t>(entry) *
                            kMoeExpertWidth]
                    .value;
            known_answers_match &=
                std::bit_cast<std::uint32_t>(oracle_gate) ==
                    std::bit_cast<std::uint32_t>(known_gate) &&
                std::bit_cast<std::uint32_t>(oracle_up) ==
                    std::bit_cast<std::uint32_t>(known_up);
            expected_gate_bits[entry] =
                std::bit_cast<std::uint32_t>(known_gate);
            expected_up_bits[entry] = std::bit_cast<std::uint32_t>(known_up);
            for (std::uint32_t previous = 0; previous < entry; ++previous) {
                gate_answers_are_distinct &=
                    expected_gate_bits[previous] != expected_gate_bits[entry];
                up_answers_are_distinct &=
                    expected_up_bits[previous] != expected_up_bits[entry];
            }
        }
        check(known_answers_match,
              label + " expert-specific gate/up one-hot oracle answers");
        check(gate_answers_are_distinct && up_answers_are_distinct,
              label + " all 80 experts have distinct gate/up one-hot answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output", output_bytes,
                               err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq2_s", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_iq2_s_inputs(Context& context, twin::InputSet set,
                              const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ2_S real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq2_s> gate_weights;
        std::vector<block_iq2_s> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_iq2_s_weights(set, weight_seed, gate_weights, up_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq2_s& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(gate_weights.begin(), gate_weights.end(),
                            in_scale_range) &&
                std::all_of(up_weights.begin(), up_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ2_S scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ2SBlocksPerRow * sizeof(block_iq2_s);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_iq2_s);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped IQ2_S gate weights", weight_bytes,
                  err),
              "MoE grouped IQ2_S gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped IQ2_S up weights", weight_bytes, err),
              "MoE grouped IQ2_S up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ2_S address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ2_S address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped IQ2_S residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_iq2_s_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, gate_weight_buffer,
                up_weight_buffer, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq2_s_max_groups(Context& context,
                                  const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    std::iota(active_experts.begin(), active_experts.end(), 0u);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_iq2_s> gate_weights;
    std::vector<block_iq2_s> up_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_iq2_s_weights(twin::InputSet::fixture, weight_seed, gate_weights,
                           up_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeIQ2SBlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::uint16_t unique_gate_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        const std::uint16_t unique_up_scale =
            static_cast<std::uint16_t>(0x3400u + 8u * slot);
        gate_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_gate_scale;
        up_weights[static_cast<std::size_t>(slot) *
                   weight_blocks_per_expert]
            .d = unique_up_scale;
        expert_scales_are_distinct &=
            gate_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_gate_scale &&
            up_weights[static_cast<std::size_t>(slot) *
                       weight_blocks_per_expert]
                .d == unique_up_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_gate_scale !=
                    gate_weights[static_cast<std::size_t>(slot - 1u) *
                                 weight_blocks_per_expert]
                        .d &&
                unique_up_scale !=
                    up_weights[static_cast<std::size_t>(slot - 1u) *
                               weight_blocks_per_expert]
                        .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped IQ2_S assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_iq2_s);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(block_iq2_s);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(gate_weight_buffer.allocate(
              context, "MoE grouped IQ2_S maximum-group gate weights",
              weight_bytes, err),
          "MoE grouped IQ2_S maximum-group gate weight buffer", err);
    check(up_weight_buffer.allocate(
              context, "MoE grouped IQ2_S maximum-group up weights",
              weight_bytes, err),
          "MoE grouped IQ2_S maximum-group up weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ2_S maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ2_S maximum-group address buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[slot];
        addresses[expert_index].role[0] =
            gate_weight_buffer.view().gpu_address + slot * weights_per_expert;
        addresses[expert_index].role[1] =
            up_weight_buffer.view().gpu_address + slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, "MoE grouped IQ2_S maximum-group residency", err);
    if (!residency_ready) return false;

    return test_moe_gu_iq2_s_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, gate_weights, up_weights,
        gate_weight_buffer, up_weight_buffer, address_buffer, true);
}

bool test_moe_gu_iq3_xxs_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq3_xxs> gate_weights,
    std::span<const block_iq3_xxs> up_weights,
    twin::GuardedBuffer& gate_weight_buffer,
    twin::GuardedBuffer& up_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ3_XXS gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        std::vector<float> source(
            static_cast<std::size_t>(rows) * kMoeInputWidth, 0.0f);
        for (std::uint32_t row = 0; row < rows; ++row) {
            source[static_cast<std::size_t>(row) * kMoeInputWidth +
                   twin::kIQ3XXSMoeOneHotPositions[31]] = 127.0f;
        }
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeInputWidth, rows, activations, err);
    } else {
        activations_ready = make_moe_q2_activations(
            rows, set, options.seed ^ experts ^
                          (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    const bool oracle_ready = twin::moe_gu_iq3_xxs_reference(
        gate_weights, up_weights, activations, expert_slots, plan, rows,
        expected_gate, expected_up, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    if (one_expert_per_entry) {
        constexpr std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ3XXSBlocksPerRow;
        bool known_answers_match = true;
        bool gate_answers_are_distinct = true;
        bool up_answers_are_distinct = true;
        std::array<std::uint32_t, 80u> expected_gate_bits{};
        std::array<std::uint32_t, 80u> expected_up_bits{};
        for (std::uint32_t entry = 0; entry < entry_count; ++entry) {
            const std::int32_t expert = plan.grp_e[entry];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const std::size_t block_index =
                static_cast<std::size_t>(slot) * weight_blocks_per_expert;
            const float gate_scale = strata::kernels::f32_from_f16(
                gate_weights[block_index].d);
            const float up_scale = strata::kernels::f32_from_f16(
                up_weights[block_index].d);
            const float known_gate =
                std::fma(gate_scale, twin::kIQ3XXSMoeOneHotAnswers[31], 0.0f);
            const float known_up =
                std::fma(up_scale, twin::kIQ3XXSMoeOneHotAnswers[31], 0.0f);
            const float oracle_gate =
                expected_gate[static_cast<std::size_t>(entry) *
                              kMoeExpertWidth]
                    .value;
            const float oracle_up =
                expected_up[static_cast<std::size_t>(entry) *
                            kMoeExpertWidth]
                    .value;
            known_answers_match &=
                std::bit_cast<std::uint32_t>(oracle_gate) ==
                    std::bit_cast<std::uint32_t>(known_gate) &&
                std::bit_cast<std::uint32_t>(oracle_up) ==
                    std::bit_cast<std::uint32_t>(known_up);
            expected_gate_bits[entry] =
                std::bit_cast<std::uint32_t>(known_gate);
            expected_up_bits[entry] = std::bit_cast<std::uint32_t>(known_up);
            for (std::uint32_t previous = 0; previous < entry; ++previous) {
                gate_answers_are_distinct &=
                    expected_gate_bits[previous] != expected_gate_bits[entry];
                up_answers_are_distinct &=
                    expected_up_bits[previous] != expected_up_bits[entry];
            }
        }
        check(known_answers_match,
              label + " expert-specific gate/up one-hot oracle answers");
        check(gate_answers_are_distinct && up_answers_are_distinct,
              label + " all 80 experts have distinct gate/up one-hot answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output", output_bytes,
                               err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq3_xxs", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_iq3_xxs_inputs(Context& context, twin::InputSet set,
                              const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ3_XXS real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq3_xxs> gate_weights;
        std::vector<block_iq3_xxs> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_iq3_xxs_weights(set, weight_seed, gate_weights, up_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq3_xxs& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(gate_weights.begin(), gate_weights.end(),
                            in_scale_range) &&
                std::all_of(up_weights.begin(), up_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ3_XXS scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ3XXSBlocksPerRow * sizeof(block_iq3_xxs);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_iq3_xxs);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped IQ3_XXS gate weights", weight_bytes,
                  err),
              "MoE grouped IQ3_XXS gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped IQ3_XXS up weights", weight_bytes, err),
              "MoE grouped IQ3_XXS up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ3_XXS address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ3_XXS address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped IQ3_XXS residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_iq3_xxs_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, gate_weight_buffer,
                up_weight_buffer, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq3_xxs_max_groups(Context& context,
                                  const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    std::iota(active_experts.begin(), active_experts.end(), 0u);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_iq3_xxs> gate_weights;
    std::vector<block_iq3_xxs> up_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_iq3_xxs_weights(twin::InputSet::fixture, weight_seed, gate_weights,
                           up_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeIQ3XXSBlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::uint16_t unique_gate_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        const std::uint16_t unique_up_scale =
            static_cast<std::uint16_t>(0x3400u + 8u * slot);
        gate_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_gate_scale;
        up_weights[static_cast<std::size_t>(slot) *
                   weight_blocks_per_expert]
            .d = unique_up_scale;
        expert_scales_are_distinct &=
            gate_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_gate_scale &&
            up_weights[static_cast<std::size_t>(slot) *
                       weight_blocks_per_expert]
                .d == unique_up_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_gate_scale !=
                    gate_weights[static_cast<std::size_t>(slot - 1u) *
                                 weight_blocks_per_expert]
                        .d &&
                unique_up_scale !=
                    up_weights[static_cast<std::size_t>(slot - 1u) *
                               weight_blocks_per_expert]
                        .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped IQ3_XXS assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_iq3_xxs);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(block_iq3_xxs);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(gate_weight_buffer.allocate(
              context, "MoE grouped IQ3_XXS maximum-group gate weights",
              weight_bytes, err),
          "MoE grouped IQ3_XXS maximum-group gate weight buffer", err);
    check(up_weight_buffer.allocate(
              context, "MoE grouped IQ3_XXS maximum-group up weights",
              weight_bytes, err),
          "MoE grouped IQ3_XXS maximum-group up weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ3_XXS maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ3_XXS maximum-group address buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[slot];
        addresses[expert_index].role[0] =
            gate_weight_buffer.view().gpu_address + slot * weights_per_expert;
        addresses[expert_index].role[1] =
            up_weight_buffer.view().gpu_address + slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, "MoE grouped IQ3_XXS maximum-group residency", err);
    if (!residency_ready) return false;

    return test_moe_gu_iq3_xxs_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, gate_weights, up_weights,
        gate_weight_buffer, up_weight_buffer, address_buffer, true);
}

bool test_moe_gu_iq3_s_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq3_s> gate_weights,
    std::span<const block_iq3_s> up_weights,
    twin::GuardedBuffer& gate_weight_buffer,
    twin::GuardedBuffer& up_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ3_S gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        std::vector<float> source(
            static_cast<std::size_t>(rows) * kMoeInputWidth, 0.0f);
        for (std::uint32_t row = 0; row < rows; ++row) {
            source[static_cast<std::size_t>(row) * kMoeInputWidth +
                   twin::kIQ3SMoeOneHotPosition] = 127.0f;
        }
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeInputWidth, rows, activations, err);
    } else {
        activations_ready = make_moe_q2_activations(
            rows, set, options.seed ^ experts ^
                          (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    const bool oracle_ready = twin::moe_gu_iq3_s_reference(
        gate_weights, up_weights, activations, expert_slots, plan, rows,
        expected_gate, expected_up, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    if (one_expert_per_entry) {
        constexpr std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ3SBlocksPerRow;
        bool known_answers_match = true;
        bool gate_answers_are_distinct = true;
        bool up_answers_are_distinct = true;
        std::array<std::uint32_t, 80u> expected_gate_bits{};
        std::array<std::uint32_t, 80u> expected_up_bits{};
        for (std::uint32_t entry = 0; entry < entry_count; ++entry) {
            const std::int32_t expert = plan.grp_e[entry];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const std::size_t block_index =
                static_cast<std::size_t>(slot) * weight_blocks_per_expert;
            const float gate_scale = strata::kernels::f32_from_f16(
                gate_weights[block_index].d);
            const float up_scale = strata::kernels::f32_from_f16(
                up_weights[block_index].d);
            const float known_gate =
                std::fma(gate_scale, twin::kIQ3SMoeOneHotAnswer, 0.0f);
            const float known_up =
                std::fma(up_scale, twin::kIQ3SMoeOneHotAnswer, 0.0f);
            const float oracle_gate =
                expected_gate[static_cast<std::size_t>(entry) *
                              kMoeExpertWidth]
                    .value;
            const float oracle_up =
                expected_up[static_cast<std::size_t>(entry) *
                            kMoeExpertWidth]
                    .value;
            known_answers_match &=
                std::bit_cast<std::uint32_t>(oracle_gate) ==
                    std::bit_cast<std::uint32_t>(known_gate) &&
                std::bit_cast<std::uint32_t>(oracle_up) ==
                    std::bit_cast<std::uint32_t>(known_up);
            expected_gate_bits[entry] =
                std::bit_cast<std::uint32_t>(known_gate);
            expected_up_bits[entry] = std::bit_cast<std::uint32_t>(known_up);
            for (std::uint32_t previous = 0; previous < entry; ++previous) {
                gate_answers_are_distinct &=
                    expected_gate_bits[previous] != expected_gate_bits[entry];
                up_answers_are_distinct &=
                    expected_up_bits[previous] != expected_up_bits[entry];
            }
        }
        check(known_answers_match,
              label + " expert-specific gate/up one-hot oracle answers");
        check(gate_answers_are_distinct && up_answers_are_distinct,
              label + " all 80 experts have distinct gate/up one-hot answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output", output_bytes,
                               err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq3_s", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_iq3_s_inputs(Context& context, twin::InputSet set,
                              const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ3_S real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq3_s> gate_weights;
        std::vector<block_iq3_s> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_iq3_s_weights(set, weight_seed, gate_weights, up_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq3_s& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(gate_weights.begin(), gate_weights.end(),
                            in_scale_range) &&
                std::all_of(up_weights.begin(), up_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ3_S scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ3SBlocksPerRow * sizeof(block_iq3_s);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_iq3_s);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped IQ3_S gate weights", weight_bytes,
                  err),
              "MoE grouped IQ3_S gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped IQ3_S up weights", weight_bytes, err),
              "MoE grouped IQ3_S up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ3_S address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ3_S address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped IQ3_S residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_iq3_s_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, gate_weight_buffer,
                up_weight_buffer, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq3_s_max_groups(Context& context,
                                  const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    std::iota(active_experts.begin(), active_experts.end(), 0u);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_iq3_s> gate_weights;
    std::vector<block_iq3_s> up_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_iq3_s_weights(twin::InputSet::fixture, weight_seed, gate_weights,
                           up_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeIQ3SBlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::uint16_t unique_gate_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        const std::uint16_t unique_up_scale =
            static_cast<std::uint16_t>(0x3400u + 8u * slot);
        gate_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_gate_scale;
        up_weights[static_cast<std::size_t>(slot) *
                   weight_blocks_per_expert]
            .d = unique_up_scale;
        expert_scales_are_distinct &=
            gate_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_gate_scale &&
            up_weights[static_cast<std::size_t>(slot) *
                       weight_blocks_per_expert]
                .d == unique_up_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_gate_scale !=
                    gate_weights[static_cast<std::size_t>(slot - 1u) *
                                 weight_blocks_per_expert]
                        .d &&
                unique_up_scale !=
                    up_weights[static_cast<std::size_t>(slot - 1u) *
                               weight_blocks_per_expert]
                        .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped IQ3_S assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_iq3_s);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(block_iq3_s);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(gate_weight_buffer.allocate(
              context, "MoE grouped IQ3_S maximum-group gate weights",
              weight_bytes, err),
          "MoE grouped IQ3_S maximum-group gate weight buffer", err);
    check(up_weight_buffer.allocate(
              context, "MoE grouped IQ3_S maximum-group up weights",
              weight_bytes, err),
          "MoE grouped IQ3_S maximum-group up weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ3_S maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ3_S maximum-group address buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    for (std::uint32_t slot = 0; slot < max_groups; ++slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[slot];
        addresses[expert_index].role[0] =
            gate_weight_buffer.view().gpu_address + slot * weights_per_expert;
        addresses[expert_index].role[1] =
            up_weight_buffer.view().gpu_address + slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, "MoE grouped IQ3_S maximum-group residency", err);
    if (!residency_ready) return false;

    return test_moe_gu_iq3_s_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, gate_weights, up_weights,
        gate_weight_buffer, up_weight_buffer, address_buffer, true);
}

bool test_moe_gu_iq4_xs_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq4_xs> gate_weights,
    std::span<const block_iq4_xs> up_weights,
    twin::GuardedBuffer& gate_weight_buffer,
    twin::GuardedBuffer& up_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ4_XS gate/up experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t rank = 0u; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        std::vector<float> source(
            static_cast<std::size_t>(rows) * kMoeInputWidth, 0.0f);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            source[static_cast<std::size_t>(row) * kMoeInputWidth +
                   twin::kIQ4XSMoeOneHotPosition] = 127.0f;
        }
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeInputWidth, rows, activations, err);
    } else {
        activations_ready = make_moe_q2_activations(
            rows, set, options.seed ^ experts ^
                          (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        static_cast<std::size_t>(rows) * (kMoeInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }
    const bool oracle_ready = twin::moe_gu_iq4_xs_reference(
        gate_weights, up_weights, activations, expert_slots, plan, rows,
        expected_gate, expected_up, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected_gate.size() != entry_count * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, label + " oracle output shape");
        return false;
    }
    if (one_expert_per_entry) {
        constexpr std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ4XSBlocksPerRow;
        bool known_answers_match = true;
        bool gate_answers_are_distinct = true;
        bool up_answers_are_distinct = true;
        std::array<std::uint32_t, 80u> expected_gate_bits{};
        std::array<std::uint32_t, 80u> expected_up_bits{};
        for (std::uint32_t entry = 0u; entry < entry_count; ++entry) {
            const std::int32_t expert = plan.grp_e[entry];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const std::size_t block_index =
                static_cast<std::size_t>(slot) * weight_blocks_per_expert;
            const float gate_scale = strata::kernels::f32_from_f16(
                gate_weights[block_index].d);
            const float up_scale = strata::kernels::f32_from_f16(
                up_weights[block_index].d);
            const float known_gate = std::fma(
                gate_scale, twin::kIQ4XSMoeOneHotAnswer, 0.0f);
            const float known_up = std::fma(
                up_scale, twin::kIQ4XSMoeOneHotAnswer, 0.0f);
            const float oracle_gate =
                expected_gate[static_cast<std::size_t>(entry) *
                              kMoeExpertWidth]
                    .value;
            const float oracle_up =
                expected_up[static_cast<std::size_t>(entry) *
                            kMoeExpertWidth]
                    .value;
            known_answers_match &=
                std::bit_cast<std::uint32_t>(oracle_gate) ==
                    std::bit_cast<std::uint32_t>(known_gate) &&
                std::bit_cast<std::uint32_t>(oracle_up) ==
                    std::bit_cast<std::uint32_t>(known_up);
            expected_gate_bits[entry] =
                std::bit_cast<std::uint32_t>(known_gate);
            expected_up_bits[entry] =
                std::bit_cast<std::uint32_t>(known_up);
            for (std::uint32_t previous = 0u; previous < entry; ++previous) {
                gate_answers_are_distinct &=
                    expected_gate_bits[previous] != expected_gate_bits[entry];
                up_answers_are_distinct &=
                    expected_up_bits[previous] != expected_up_bits[entry];
            }
        }
        check(known_answers_match,
              label + " expert-specific gate/up one-hot oracle answers");
        check(gate_answers_are_distinct && up_answers_are_distinct,
              label + " all 80 experts have distinct gate/up one-hot answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected_gate.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(gate_buffer.allocate(context, label + " gate output", output_bytes,
                               err),
          label + " gate output buffer", err);
    check(up_buffer.allocate(context, label + " up output", output_bytes,
                             err),
          label + " up output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !gate_buffer.view() || !up_buffer.view() ||
        !gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, err),
          label + " gate output poison", err);
    check(up_buffer.poison_floats_against(
              expected_up, twin::Comparison::units, err),
          label + " up output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    up_buffer.capture_before_dispatch();
    gate_weight_buffer.capture_before_dispatch();
    up_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {160u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, gate_buffer.view()},
                    {4u, up_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_gu_iq4_xs", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_gate_data =
        reinterpret_cast<const float*>(gate_buffer.payload().data());
    const auto* actual_up_data =
        reinterpret_cast<const float*>(up_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto gate_failure = twin::compare_floats(
        std::span<const float>(actual_gate_data, expected_gate.size()),
        expected_gate, tolerance);
    if (gate_failure) {
        twin::report_failure(label.c_str(), options, rows, *gate_failure);
    }
    check(!gate_failure, label + " gate outputs U(16)");
    const auto up_failure = twin::compare_floats(
        std::span<const float>(actual_up_data, expected_up.size()),
        expected_up, tolerance);
    if (up_failure) {
        twin::report_failure(label.c_str(), options, rows, *up_failure);
    }
    check(!up_failure, label + " up outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!gate_weight_buffer.first_unwritten_change({}),
          label + " gate weights and guards unchanged");
    check(!up_weight_buffer.first_unwritten_change({}),
          label + " up weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!gate_buffer.first_unwritten_change(output_written),
          label + " gate writes stay within grouped entries");
    check(!up_buffer.first_unwritten_change(output_written),
          label + " up writes stay within grouped entries");
    return !gate_failure && !up_failure;
}

bool test_moe_gu_iq4_xs_inputs(Context& context, twin::InputSet set,
                               const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ4_XS real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq4_xs> gate_weights;
        std::vector<block_iq4_xs> up_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_iq4_xs_weights(set, weight_seed, gate_weights, up_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq4_xs& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(gate_weights.begin(), gate_weights.end(),
                            in_scale_range) &&
                std::all_of(up_weights.begin(), up_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ4_XS scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeExpertWidth) *
            kMoeIQ4XSBlocksPerRow * sizeof(block_iq4_xs);
        const std::size_t weight_bytes =
            gate_weights.size() * sizeof(block_iq4_xs);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer gate_weight_buffer;
        twin::GuardedBuffer up_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(gate_weight_buffer.allocate(
                  context, "MoE grouped IQ4_XS gate weights", weight_bytes,
                  err),
              "MoE grouped IQ4_XS gate weight buffer", err);
        check(up_weight_buffer.allocate(
                  context, "MoE grouped IQ4_XS up weights", weight_bytes, err),
              "MoE grouped IQ4_XS up weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ4_XS address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ4_XS address buffer", err);
        if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
            !address_buffer.view()) {
            return false;
        }
        std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                    weight_bytes);
        std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[0] =
                gate_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
            addresses[expert_index].role[1] =
                up_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped IQ4_XS residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_gu_iq4_xs_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, gate_weights, up_weights, gate_weight_buffer,
                up_weight_buffer, address_buffer);
        }
    }
    return completed;
}

bool test_moe_gu_iq4_xs_max_groups(Context& context,
                                   const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    std::iota(active_experts.begin(), active_experts.end(), 0u);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_iq4_xs> gate_weights;
    std::vector<block_iq4_xs> up_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_iq4_xs_weights(twin::InputSet::fixture, weight_seed,
                             gate_weights, up_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) * kMoeIQ4XSBlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        const std::uint16_t unique_gate_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        const std::uint16_t unique_up_scale =
            static_cast<std::uint16_t>(0x3400u + 8u * slot);
        gate_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_gate_scale;
        up_weights[static_cast<std::size_t>(slot) *
                   weight_blocks_per_expert]
            .d = unique_up_scale;
        expert_scales_are_distinct &=
            gate_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_gate_scale &&
            up_weights[static_cast<std::size_t>(slot) *
                       weight_blocks_per_expert]
                .d == unique_up_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_gate_scale !=
                    gate_weights[static_cast<std::size_t>(slot - 1u) *
                                 weight_blocks_per_expert]
                        .d &&
                unique_up_scale !=
                    up_weights[static_cast<std::size_t>(slot - 1u) *
                               weight_blocks_per_expert]
                        .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped IQ4_XS assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_iq4_xs);
    const std::size_t weight_bytes =
        gate_weights.size() * sizeof(block_iq4_xs);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(gate_weight_buffer.allocate(
              context, "MoE grouped IQ4_XS maximum-group gate weights",
              weight_bytes, err),
          "MoE grouped IQ4_XS maximum-group gate weight buffer", err);
    check(up_weight_buffer.allocate(
              context, "MoE grouped IQ4_XS maximum-group up weights",
              weight_bytes, err),
          "MoE grouped IQ4_XS maximum-group up weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ4_XS maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ4_XS maximum-group address buffer", err);
    if (!gate_weight_buffer.view() || !up_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                weight_bytes);
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                weight_bytes);
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[slot];
        addresses[expert_index].role[0] =
            gate_weight_buffer.view().gpu_address + slot * weights_per_expert;
        addresses[expert_index].role[1] =
            up_weight_buffer.view().gpu_address + slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, "MoE grouped IQ4_XS maximum-group residency", err);
    if (!residency_ready) return false;

    return test_moe_gu_iq4_xs_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, gate_weights, up_weights,
        gate_weight_buffer, up_weight_buffer, address_buffer, true);
}

bool test_moe_down_q2_0_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_q2_0> down_weights,
    twin::GuardedBuffer& down_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped Q2_0 down experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t rank = 0u; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    std::vector<float> source(entry_count * kMoeDownInputWidth, 0.0f);
    if (one_expert_per_entry) {
        for (std::size_t entry = 0u; entry < entry_count; ++entry) {
            source[entry * kMoeDownInputWidth] = 127.0f;
        }
    }
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeDownInputWidth,
            static_cast<std::uint32_t>(entry_count), activations, err);
    } else {
        activations_ready = make_moe_down_q2_0_activations(
            static_cast<std::uint32_t>(entry_count), set,
            options.seed ^ experts ^
                (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        entry_count * (kMoeDownInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }

    std::vector<twin::tf> expected;
    const bool oracle_ready = twin::moe_down_q2_0_reference(
        down_weights, activations, expert_slots, plan, rows, expected, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected.size() != entry_count * kMoeDownWidth) {
        check(false, label + " oracle output shape");
        return false;
    }

    if (one_expert_per_entry) {
        const std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeDownWidth) *
            kMoeDownQ2BlocksPerRow;
        bool known_answers_match = true;
        bool answers_are_distinct = true;
        std::array<std::uint32_t, 80u> expected_bits{};
        for (std::uint32_t group = 0u; group < plan.n_groups; ++group) {
            const std::uint32_t first =
                static_cast<std::uint32_t>(plan.grp_start[group]);
            const std::uint32_t end =
                static_cast<std::uint32_t>(plan.grp_start[group + 1u]);
            if (end - first != 1u) {
                known_answers_match = false;
                continue;
            }
            const std::int32_t expert = plan.grp_e[group];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const float scale = strata::kernels::f32_from_f16(
                down_weights[static_cast<std::size_t>(slot) *
                             weight_blocks_per_expert]
                    .d);
            const float known =
                std::fma(scale, 127.0f, 0.0f);
            const std::size_t destination =
                static_cast<std::size_t>(plan.ent_dst[first]);
            const float oracle = expected[destination * kMoeDownWidth].value;
            known_answers_match &=
                std::bit_cast<std::uint32_t>(oracle) ==
                std::bit_cast<std::uint32_t>(known);
            expected_bits[group] = std::bit_cast<std::uint32_t>(known);
            for (std::uint32_t previous = 0u; previous < group; ++previous) {
                answers_are_distinct &=
                    expected_bits[previous] != expected_bits[group];
            }
        }
        check(known_answers_match,
              label + " expert-specific one-hot oracle answers");
        check(answers_are_distinct,
              label + " all 80 experts have distinct down answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    plans[kMoeGroupedLayer] = plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer output_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected.size() * sizeof(float);
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(output_buffer.allocate(context, label + " down output",
                                 output_bytes, err),
          label + " down output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !output_buffer.view() || !down_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    check(output_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " down output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    down_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {320u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, output_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 4u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_down_q2_0", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_data =
        reinterpret_cast<const float*>(output_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto failure = twin::compare_floats(
        std::span<const float>(actual_data, expected.size()), expected,
        tolerance);
    if (failure) {
        twin::report_failure(label.c_str(), options, rows, *failure);
    }
    check(!failure, label + " down outputs U(16)");

    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!down_weight_buffer.first_unwritten_change({}),
          label + " down weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " down writes stay within original entries");
    return !failure;
}

bool test_moe_down_q2_0_inputs(Context& context, twin::InputSet set,
                              const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped Q2_0 down real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_q2_0> down_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_down_q2_0_weights(set, weight_seed, down_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_q2_0& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(down_weights.begin(), down_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random Q2_0 down scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeDownWidth) *
            kMoeDownQ2BlocksPerRow * sizeof(block_q2_0);
        const std::size_t weight_bytes =
            down_weights.size() * sizeof(block_q2_0);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer down_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(down_weight_buffer.allocate(
                  context, "MoE grouped Q2_0 down weights", weight_bytes, err),
              "MoE grouped Q2_0 down weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped Q2_0 down address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped Q2_0 down address buffer", err);
        if (!down_weight_buffer.view() || !address_buffer.view()) {
            return false;
        }
        std::memcpy(down_weight_buffer.payload().data(), down_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[2] =
                down_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready, "MoE grouped Q2_0 down residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_down_q2_0_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, down_weights, down_weight_buffer,
                address_buffer);
        }
    }
    return completed;
}

bool test_moe_down_q2_0_max_groups(Context& context,
                                   const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    for (std::uint32_t entry = 0u; entry < max_groups; ++entry) {
        active_experts[entry] = (37u * entry) % max_groups;
    }
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_q2_0> down_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_down_q2_0_weights(
        twin::InputSet::fixture, weight_seed, down_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeDownWidth) *
        kMoeDownQ2BlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        const std::uint16_t unique_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        down_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_scale;
        expert_scales_are_distinct &=
            down_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_scale !=
                down_weights[static_cast<std::size_t>(slot - 1u) *
                             weight_blocks_per_expert]
                    .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped Q2_0 down assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_q2_0);
    const std::size_t weight_bytes =
        down_weights.size() * sizeof(block_q2_0);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer down_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(down_weight_buffer.allocate(
              context, "MoE grouped Q2_0 down maximum-group weights",
              weight_bytes, err),
          "MoE grouped Q2_0 down maximum-group weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped Q2_0 down maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped Q2_0 down maximum-group address buffer", err);
    if (!down_weight_buffer.view() || !address_buffer.view()) {
        return false;
    }
    std::memcpy(down_weight_buffer.payload().data(), down_weights.data(),
                weight_bytes);
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[slot];
        addresses[expert_index].role[2] =
            down_weight_buffer.view().gpu_address + slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready,
          "MoE grouped Q2_0 down maximum-group residency", err);
    if (!residency_ready) return false;

    return test_moe_down_q2_0_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, down_weights, down_weight_buffer,
        address_buffer, true);
}

bool test_moe_down_iq4_nl_case(
    Context& context, std::uint32_t experts, std::uint32_t rows,
    twin::InputSet set, const twin::Options& options,
    std::span<const std::uint32_t> active_experts,
    std::span<const std::int32_t> expert_slots,
    std::span<const block_iq4_nl> down_weights,
    twin::GuardedBuffer& down_weight_buffer,
    twin::GuardedBuffer& address_buffer,
    bool one_expert_per_entry = false) {
    const std::string label =
        "MoE grouped IQ4_NL down experts=" + std::to_string(experts) +
        " rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t entry_count = static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::int32_t> ids(entry_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t rank = 0u; rank < kTopK; ++rank) {
            const std::size_t entry =
                static_cast<std::size_t>(row) * kTopK + rank;
            const std::size_t expert_index =
                one_expert_per_entry ? entry : rank + row;
            ids[entry] = static_cast<std::int32_t>(
                active_experts[expert_index % active_experts.size()]);
        }
    }
    std::vector<std::int32_t> resident(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts, 1);
    std::vector<float> initial_usage(resident.size(), 0.0f);
    TieredMoePlan plan{};
    std::vector<float> expected_usage;
    std::string err;
    const bool plan_ready = twin::moe_plan_reference(
        ids, rows, experts, false, kMoeGroupedLayer, resident, initial_usage,
        plan, expected_usage, err);
    check(plan_ready, label + " plan oracle", err);
    if (!plan_ready) return false;
    if (one_expert_per_entry) {
        check(plan.n_groups == entry_count,
              label + " creates the maximum number of groups");
        check(plan.grp_start[plan.n_groups] ==
                  static_cast<std::int32_t>(entry_count),
              label + " reaches grp_start[80]");
    }

    TieredMoePlan single_plan{};
    single_plan.n_groups = plan.n_entries;
    single_plan.n_entries = plan.n_entries;
    single_plan.n_miss = plan.n_miss;
    single_plan.skip = plan.skip;
    single_plan.grp_start[0] = 0;
    for (std::uint32_t group = 0u; group < plan.n_groups; ++group) {
        const std::uint32_t start =
            static_cast<std::uint32_t>(plan.grp_start[group]);
        const std::uint32_t end =
            static_cast<std::uint32_t>(plan.grp_start[group + 1u]);
        for (std::uint32_t entry = start; entry < end; ++entry) {
            single_plan.grp_e[entry] = plan.grp_e[group];
            single_plan.grp_res[entry] = plan.grp_res[group];
            single_plan.grp_start[entry] =
                static_cast<std::int32_t>(entry);
            single_plan.grp_start[entry + 1u] =
                static_cast<std::int32_t>(entry + 1u);
            single_plan.ent_tok[entry] = plan.ent_tok[entry];
            single_plan.ent_dst[entry] = plan.ent_dst[entry];
        }
    }

    std::vector<float> source(entry_count * kMoeDownInputWidth, 0.0f);
    if (one_expert_per_entry) {
        for (std::size_t entry = 0u; entry < entry_count; ++entry) {
            for (std::uint32_t index = 0u; index < 15u; ++index) {
                source[entry * kMoeDownInputWidth +
                       kMoeIQ4NLPart0Positions[index]] = 127.0f;
            }
            for (std::uint32_t index = 0u; index < 13u; ++index) {
                source[entry * kMoeDownInputWidth +
                       kMoeIQ4NLPart1Positions[index]] = 127.0f;
            }
        }
    }
    std::vector<Q81> activations;
    bool activations_ready = false;
    if (one_expert_per_entry) {
        activations_ready = twin::quantize_q8_1_reference(
            source, kMoeDownInputWidth,
            static_cast<std::uint32_t>(entry_count), activations, err);
    } else {
        activations_ready = make_moe_down_iq4_nl_activations(
            static_cast<std::uint32_t>(entry_count), set,
            options.seed ^ experts ^
                (static_cast<std::uint64_t>(set) << 48u),
            activations, err);
    }
    check(activations_ready, label + " Q8_1 activation oracle", err);
    if (!activations_ready) return false;
    if (activations.size() !=
        entry_count * (kMoeDownInputWidth / 32u)) {
        check(false, label + " activation output shape");
        return false;
    }

    std::vector<twin::tf> expected;
    const bool oracle_ready = twin::moe_down_iq4_nl_reference(
        down_weights, activations, expert_slots, plan, rows, expected, err);
    check(oracle_ready, label + " CPU grouped row-dot oracle", err);
    if (!oracle_ready) return false;
    if (expected.size() != entry_count * kMoeDownWidth) {
        check(false, label + " oracle output shape");
        return false;
    }

    std::array<std::uint32_t, 80u> expected_bits{};
    std::array<std::size_t, 80u> golden_destinations{};
    if (one_expert_per_entry) {
        const std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeDownWidth) *
            kMoeDownIQ4NLBlocksPerRow;
        constexpr std::array<std::int8_t, 16u> codebook = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1,    13,   25,  38,  53,  69,  89,  113};
        bool fixture_weights_match = true;
        bool fixture_activations_match = true;
        bool answers_are_distinct = true;
        bool fixture_distinguishes_split_fma = false;
        for (std::uint32_t group = 0u; group < plan.n_groups; ++group) {
            const std::uint32_t first =
                static_cast<std::uint32_t>(plan.grp_start[group]);
            const std::uint32_t end =
                static_cast<std::uint32_t>(plan.grp_start[group + 1u]);
            if (end - first != 1u) {
                fixture_weights_match = false;
                continue;
            }
            const std::int32_t expert = plan.grp_e[group];
            const std::int32_t slot =
                expert_slots[static_cast<std::size_t>(expert)];
            const block_iq4_nl& weight =
                down_weights[static_cast<std::size_t>(slot) *
                             weight_blocks_per_expert];
            for (std::uint32_t index = 0u;
                 index < kMoeIQ4NLPart0Positions.size(); ++index) {
                const std::uint32_t position =
                    kMoeIQ4NLPart0Positions[index];
                const std::uint32_t code =
                    (weight.qs[position & 0x0fu] >>
                     (position < 16u ? 0u : 4u)) &
                    0x0fu;
                const std::uint32_t expected_code =
                    index < 14u ? 0u : (index == 14u ? 6u : 8u);
                fixture_weights_match &= code == expected_code;
            }
            for (std::uint32_t index = 0u;
                 index < kMoeIQ4NLPart1Positions.size(); ++index) {
                const std::uint32_t position =
                    kMoeIQ4NLPart1Positions[index];
                const std::uint32_t code =
                    (weight.qs[position & 0x0fu] >>
                     (position < 16u ? 0u : 4u)) &
                    0x0fu;
                const std::uint32_t expected_code =
                    index < 9u ? 15u
                               : (index == 9u
                                      ? 11u
                                      : (index == 10u
                                             ? 6u
                                             : (index < 13u ? 8u : 0u)));
                fixture_weights_match &= code == expected_code;
            }

            const Q81& activation =
                activations[static_cast<std::size_t>(first) *
                            kMoeDownIQ4NLBlocksPerRow];
            const std::uint32_t ds =
                static_cast<std::uint32_t>(activation.ds);
            fixture_activations_match &=
                static_cast<std::uint16_t>(ds & 0xffffu) == 0x3c00u;
            for (std::uint32_t position = 0u; position < 32u; ++position) {
                const bool in_part0 =
                    std::find(kMoeIQ4NLPart0Positions.begin(),
                              kMoeIQ4NLPart0Positions.begin() + 15,
                              position) !=
                    kMoeIQ4NLPart0Positions.begin() + 15;
                const bool in_part1 =
                    std::find(kMoeIQ4NLPart1Positions.begin(),
                              kMoeIQ4NLPart1Positions.begin() + 13,
                              position) !=
                    kMoeIQ4NLPart1Positions.begin() + 13;
                const std::int8_t expected_q =
                    in_part0 || in_part1 ? 127 : 0;
                fixture_activations_match &=
                    activation.qs[position] == expected_q;
            }

            const float scale = strata::kernels::f32_from_f16(
                weight.d);
            const float part0 = std::fma(
                scale, static_cast<float>(kMoeIQ4NLProbePart0Dot), 0.0f);
            const float part1 = std::fma(
                scale, static_cast<float>(kMoeIQ4NLProbePart1Dot), 0.0f);
            const float known = part0 + part1;
            const float unsplit = std::fma(
                scale, static_cast<float>(kMoeIQ4NLProbePart0Dot +
                                          kMoeIQ4NLProbePart1Dot),
                0.0f);
            fixture_distinguishes_split_fma |=
                std::bit_cast<std::uint32_t>(known) !=
                std::bit_cast<std::uint32_t>(unsplit);
            const std::size_t destination =
                static_cast<std::size_t>(plan.ent_dst[first]);
            expected_bits[group] = std::bit_cast<std::uint32_t>(known);
            golden_destinations[group] = destination;
            for (std::uint32_t previous = 0u; previous < group; ++previous) {
                answers_are_distinct &=
                    expected_bits[previous] != expected_bits[group];
            }
        }
        for (std::uint32_t entry = 0u; entry < entry_count; ++entry) {
            for (std::uint32_t block = 1u;
                 block < kMoeDownIQ4NLBlocksPerRow; ++block) {
                const Q81& activation =
                    activations[static_cast<std::size_t>(entry) *
                                    kMoeDownIQ4NLBlocksPerRow +
                                block];
                const std::uint32_t ds =
                    static_cast<std::uint32_t>(activation.ds);
                fixture_activations_match &=
                    static_cast<std::uint16_t>(ds & 0xffffu) == 0u;
            }
        }
        fixture_weights_match &=
            (14 * codebook[0] + codebook[6]) * 127 ==
                kMoeIQ4NLProbePart0Dot &&
            (9 * codebook[15] + codebook[11] + codebook[6] +
             2 * codebook[8]) *
                    127 ==
                kMoeIQ4NLProbePart1Dot;
        check(fixture_weights_match,
              label + " fixed IQ4_NL codewords encode distinct partial dots");
        check(fixture_activations_match,
              label + " fixed Q8_1 probe encodes both 16-value parts");
        check(fixture_distinguishes_split_fma,
              label + " fixed probe distinguishes split from merged FMA");
        check(answers_are_distinct,
              label + " all 80 experts have distinct down answers");
    }

    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> plans{};
    std::array<TieredMoePlan, kMoeGroupedLayer + 1u> single_plans{};
    plans[kMoeGroupedLayer] = plan;
    single_plans[kMoeGroupedLayer] = single_plan;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer plan_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer single_plan_buffer;
    twin::GuardedBuffer single_output_buffer;
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t plan_bytes = sizeof(plans);
    const std::size_t output_bytes = expected.size() * sizeof(float);
    const std::array<twin::ByteRange, 1> output_written = {
        {{0u, output_bytes}}};
    check(input_buffer.allocate(context, label + " Q8_1 activations",
                                input_bytes, err),
          label + " activation buffer", err);
    check(plan_buffer.allocate(context, label + " grouped plan",
                               plan_bytes, err),
          label + " plan buffer", err);
    check(output_buffer.allocate(context, label + " down output",
                                 output_bytes, err),
          label + " down output buffer", err);
    check(single_plan_buffer.allocate(context, label + " single-entry plan",
                                      sizeof(single_plans), err),
          label + " single-entry plan buffer", err);
    check(single_output_buffer.allocate(context, label + " single-entry output",
                                        output_bytes, err),
          label + " single-entry output buffer", err);
    if (!input_buffer.view() || !plan_buffer.view() ||
        !output_buffer.view() || !single_plan_buffer.view() ||
        !single_output_buffer.view() || !down_weight_buffer.view() ||
        !address_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(), input_bytes);
    std::memcpy(plan_buffer.payload().data(), plans.data(), plan_bytes);
    std::memcpy(single_plan_buffer.payload().data(), single_plans.data(),
                sizeof(single_plans));
    check(output_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " down output poison", err);
    check(single_output_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " single-entry output poison", err);
    input_buffer.capture_before_dispatch();
    plan_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    single_plan_buffer.capture_before_dispatch();
    single_output_buffer.capture_before_dispatch();
    down_weight_buffer.capture_before_dispatch();
    address_buffer.capture_before_dispatch();

    const MoeGroupedParams params{experts, kMoeGroupedLayer};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {320u, static_cast<std::uint32_t>(entry_count), 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.buffers = {{0u, address_buffer.view()},
                    {1u, input_buffer.view()},
                    {2u, plan_buffer.view()},
                    {3u, output_buffer.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 4u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_moe_down_iq4_nl", work, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto* actual_data =
        reinterpret_cast<const float*>(output_buffer.payload().data());
    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto failure = twin::compare_floats(
        std::span<const float>(actual_data, expected.size()), expected,
        tolerance);
    if (failure) {
        twin::report_failure(label.c_str(), options, rows, *failure);
    }
    check(!failure, label + " down outputs U(16)");

    twin::Dispatch single_work = work;
    single_work.buffers[2].buffer = single_plan_buffer.view();
    single_work.buffers[3].buffer = single_output_buffer.view();
    const bool single_dispatched = twin::dispatch_manifest(
        context, "strata_moe_down_iq4_nl", single_work, err);
    check(single_dispatched,
          label + " forced single-entry Metal dispatch", err);
    if (!single_dispatched) return false;
    const auto* single_data =
        reinterpret_cast<const float*>(single_output_buffer.payload().data());
    const auto grouped_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(actual_data), output_bytes);
    const auto single_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(single_data), output_bytes);
    const auto multi_single_mismatch =
        twin::compare_bytes(grouped_bytes, single_bytes);
    if (multi_single_mismatch) {
        twin::report_failure("IQ4_NL down multi-entry equals single-entry",
                             options, rows, *multi_single_mismatch);
    }
    check(!multi_single_mismatch,
          label + " grouped entries bitwise match single-entry groups");

    if (one_expert_per_entry) {
        bool gpu_golden_answers_match = true;
        for (std::uint32_t group = 0u; group < plan.n_groups; ++group) {
            const float actual =
                actual_data[golden_destinations[group] * kMoeDownWidth];
            gpu_golden_answers_match &=
                std::bit_cast<std::uint32_t>(actual) == expected_bits[group];
        }
        check(gpu_golden_answers_match,
              label + " GPU matches all 80 fixed two-half goldens exactly");

        std::vector<block_iq4_nl> flipped_weights(
            down_weights.begin(), down_weights.end());
        const std::int32_t first_expert = plan.grp_e[0];
        const std::int32_t first_slot =
            expert_slots[static_cast<std::size_t>(first_expert)];
        if (first_slot < 0) {
            check(false, label + " negative-control expert has a weight slot");
            return false;
        }
        const std::size_t weight_blocks_per_expert =
            static_cast<std::size_t>(kMoeDownWidth) *
            kMoeDownIQ4NLBlocksPerRow;
        flipped_weights[static_cast<std::size_t>(first_slot) *
                        weight_blocks_per_expert]
            .qs[0] ^= 0x01u;

        twin::GuardedBuffer flipped_weight_buffer;
        twin::GuardedBuffer flipped_address_buffer;
        twin::GuardedBuffer flipped_output_buffer;
        const std::size_t flipped_weight_bytes =
            flipped_weights.size() * sizeof(block_iq4_nl);
        std::vector<MoeExpertAddress> flipped_addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        std::memcpy(flipped_addresses.data(), address_buffer.payload().data(),
                    flipped_addresses.size() * sizeof(MoeExpertAddress));
        check(flipped_weight_buffer.allocate(
                  context, label + " negative-control weights",
                  flipped_weight_bytes, err),
              label + " negative-control weight buffer", err);
        check(flipped_address_buffer.allocate(
                  context, label + " negative-control addresses",
                  flipped_addresses.size() * sizeof(MoeExpertAddress), err),
              label + " negative-control address buffer", err);
        check(flipped_output_buffer.allocate(
                  context, label + " negative-control output", output_bytes,
                  err),
              label + " negative-control output buffer", err);
        if (!flipped_weight_buffer.view() ||
            !flipped_address_buffer.view() ||
            !flipped_output_buffer.view()) {
            return false;
        }
        std::memcpy(flipped_weight_buffer.payload().data(),
                    flipped_weights.data(), flipped_weight_bytes);
        const std::size_t weights_per_expert =
            weight_blocks_per_expert * sizeof(block_iq4_nl);
        for (std::uint32_t expert = 0u; expert < experts; ++expert) {
            const std::int32_t slot = expert_slots[expert];
            if (slot < 0) continue;
            flipped_addresses[
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                expert]
                .role[2] =
                flipped_weight_buffer.view().gpu_address +
                static_cast<std::size_t>(slot) * weights_per_expert;
        }
        std::memcpy(flipped_address_buffer.payload().data(),
                    flipped_addresses.data(),
                    flipped_addresses.size() * sizeof(MoeExpertAddress));
        const bool flipped_residency_ready =
            context.buffers().commit_residency(err);
        check(flipped_residency_ready,
              label + " negative-control residency", err);
        if (!flipped_residency_ready) return false;
        std::memcpy(flipped_output_buffer.payload().data(),
                    single_output_buffer.payload().data(), output_bytes);
        flipped_weight_buffer.capture_before_dispatch();
        flipped_address_buffer.capture_before_dispatch();
        flipped_output_buffer.capture_before_dispatch();

        twin::Dispatch flipped_work = single_work;
        flipped_work.buffers[0].buffer = flipped_address_buffer.view();
        flipped_work.buffers[3].buffer = flipped_output_buffer.view();
        const bool flipped_dispatched = twin::dispatch_manifest(
            context, "strata_moe_down_iq4_nl", flipped_work, err);
        check(flipped_dispatched,
              label + " one-bit negative-control dispatch", err);
        if (!flipped_dispatched) return false;
        const auto* flipped_data = reinterpret_cast<const float*>(
            flipped_output_buffer.payload().data());
        const std::size_t golden_index =
            golden_destinations[0] * kMoeDownWidth;
        const float scale = strata::kernels::f32_from_f16(
            down_weights[static_cast<std::size_t>(first_slot) *
                         weight_blocks_per_expert]
                .d);
        const float flipped_part0 = std::fma(
            scale, static_cast<float>((-1777) * 127), 0.0f);
        const float flipped_part1 = std::fma(
            scale, static_cast<float>(kMoeIQ4NLProbePart1Dot), 0.0f);
        const float flipped_golden = flipped_part0 + flipped_part1;
        const std::uint32_t flipped_golden_bits =
            std::bit_cast<std::uint32_t>(flipped_golden);
        const auto flipped_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(flipped_data), output_bytes);
        const bool negative_control_detected =
            std::bit_cast<std::uint32_t>(flipped_data[golden_index]) ==
                flipped_golden_bits &&
            flipped_golden_bits != expected_bits[0] &&
            twin::compare_bytes(single_bytes, flipped_bytes).has_value();
        check(negative_control_detected,
              label + " one-bit weight flip matches its changed fixed golden");
        check(!flipped_weight_buffer.first_unwritten_change({}),
              label + " negative-control weights and guards unchanged");
        check(!flipped_address_buffer.first_unwritten_change({}),
              label + " negative-control addresses and guards unchanged");
        check(!flipped_output_buffer.first_unwritten_change(output_written),
              label + " negative-control output guards unchanged");
    }

    check(!input_buffer.first_unwritten_change({}),
          label + " Q8_1 activations and guards unchanged");
    check(!plan_buffer.first_unwritten_change({}),
          label + " plan and guards unchanged");
    check(!single_plan_buffer.first_unwritten_change({}),
          label + " single-entry plan and guards unchanged");
    check(!single_output_buffer.first_unwritten_change(output_written),
          label + " single-entry output guards unchanged");
    check(!down_weight_buffer.first_unwritten_change({}),
          label + " down weights and guards unchanged");
    check(!address_buffer.first_unwritten_change({}),
          label + " expert address table and guards unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " down writes stay within original entries");
    return !failure;
}

bool test_moe_down_iq4_nl_inputs(Context& context, twin::InputSet set,
                                 const twin::Options& options) {
    if (set == twin::InputSet::real) {
        std::printf("skip MoE grouped IQ4_NL down real inputs: external model "
                    "tensors are unavailable under REV-002\n");
        return true;
    }

    bool completed = true;
    for (const std::uint32_t experts : {512u, 256u}) {
        const std::vector<std::uint32_t> active_experts =
            make_moe_q2_experts(experts);
        std::vector<std::int32_t> expert_slots(experts, -1);
        for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
            expert_slots[active_experts[slot]] =
                static_cast<std::int32_t>(slot);
        }

        std::vector<block_iq4_nl> down_weights;
        const std::uint64_t weight_seed =
            options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
            static_cast<std::uint64_t>(set);
        make_moe_down_iq4_nl_weights(set, weight_seed, down_weights);
        if (set == twin::InputSet::random) {
            const auto in_scale_range = [](const block_iq4_nl& weight) {
                return weight.d >= 0x0c00u && weight.d <= 0x2c00u;
            };
            const bool random_scales_in_range =
                std::all_of(down_weights.begin(), down_weights.end(),
                            in_scale_range);
            check(random_scales_in_range,
                  "random IQ4_NL down scales are log-uniform in [2^-12, 2^-4]");
            if (!random_scales_in_range) return false;
        }

        const std::size_t weights_per_expert =
            static_cast<std::size_t>(kMoeDownWidth) *
            kMoeDownIQ4NLBlocksPerRow * sizeof(block_iq4_nl);
        const std::size_t weight_bytes =
            down_weights.size() * sizeof(block_iq4_nl);
        std::vector<MoeExpertAddress> addresses(
            static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
        twin::GuardedBuffer down_weight_buffer;
        twin::GuardedBuffer address_buffer;
        std::string err;
        check(down_weight_buffer.allocate(
                  context, "MoE grouped IQ4_NL down weights", weight_bytes,
                  err),
              "MoE grouped IQ4_NL down weight buffer", err);
        check(address_buffer.allocate(
                  context, "MoE grouped IQ4_NL down address table",
                  addresses.size() * sizeof(MoeExpertAddress), err),
              "MoE grouped IQ4_NL down address buffer", err);
        if (!down_weight_buffer.view() || !address_buffer.view()) {
            return false;
        }
        std::memcpy(down_weight_buffer.payload().data(), down_weights.data(),
                    weight_bytes);
        for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
            const std::size_t expert_index =
                static_cast<std::size_t>(kMoeGroupedLayer) * experts +
                active_experts[slot];
            addresses[expert_index].role[2] =
                down_weight_buffer.view().gpu_address +
                slot * weights_per_expert;
        }
        std::memcpy(address_buffer.payload().data(), addresses.data(),
                    addresses.size() * sizeof(MoeExpertAddress));
        const bool residency_ready =
            context.buffers().commit_residency(err);
        check(residency_ready,
              "MoE grouped IQ4_NL down residency request", err);
        if (!residency_ready) return false;

        for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
            completed &= test_moe_down_iq4_nl_case(
                context, experts, rows, set, options, active_experts,
                expert_slots, down_weights, down_weight_buffer,
                address_buffer);
        }
    }
    return completed;
}

bool test_moe_down_iq4_nl_max_groups(Context& context,
                                     const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t max_groups = 80u;
    std::vector<std::uint32_t> active_experts(max_groups);
    for (std::uint32_t entry = 0u; entry < max_groups; ++entry) {
        active_experts[entry] = (37u * entry) % max_groups;
    }
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<block_iq4_nl> down_weights;
    const std::uint64_t weight_seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(twin::InputSet::fixture);
    make_moe_down_iq4_nl_weights(
        twin::InputSet::fixture, weight_seed, down_weights, max_groups);
    const std::size_t weight_blocks_per_expert =
        static_cast<std::size_t>(kMoeDownWidth) *
        kMoeDownIQ4NLBlocksPerRow;
    bool expert_scales_are_distinct = true;
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        const std::uint16_t unique_scale =
            static_cast<std::uint16_t>(0x3000u + 8u * slot);
        down_weights[static_cast<std::size_t>(slot) *
                     weight_blocks_per_expert]
            .d = unique_scale;
        expert_scales_are_distinct &=
            down_weights[static_cast<std::size_t>(slot) *
                         weight_blocks_per_expert]
                .d == unique_scale;
        if (slot != 0u) {
            expert_scales_are_distinct &=
                unique_scale !=
                down_weights[static_cast<std::size_t>(slot - 1u) *
                             weight_blocks_per_expert]
                    .d;
        }
    }
    check(expert_scales_are_distinct,
          "MoE grouped IQ4_NL down assigns distinct scales to all 80 experts");
    const std::size_t weights_per_expert =
        weight_blocks_per_expert * sizeof(block_iq4_nl);
    const std::size_t weight_bytes =
        down_weights.size() * sizeof(block_iq4_nl);
    std::vector<MoeExpertAddress> addresses(
        static_cast<std::size_t>(kMoeGroupedLayer + 1u) * experts);
    twin::GuardedBuffer down_weight_buffer;
    twin::GuardedBuffer address_buffer;
    std::string err;
    check(down_weight_buffer.allocate(
              context, "MoE grouped IQ4_NL down maximum-group weights",
              weight_bytes, err),
          "MoE grouped IQ4_NL down maximum-group weight buffer", err);
    check(address_buffer.allocate(
              context, "MoE grouped IQ4_NL down maximum-group address table",
              addresses.size() * sizeof(MoeExpertAddress), err),
          "MoE grouped IQ4_NL down maximum-group address buffer", err);
    if (!down_weight_buffer.view() || !address_buffer.view()) {
        return false;
    }
    std::memcpy(down_weight_buffer.payload().data(), down_weights.data(),
                weight_bytes);
    for (std::uint32_t slot = 0u; slot < max_groups; ++slot) {
        const std::size_t expert_index =
            static_cast<std::size_t>(kMoeGroupedLayer) * experts +
            active_experts[slot];
        addresses[expert_index].role[2] =
            down_weight_buffer.view().gpu_address + slot * weights_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready,
          "MoE grouped IQ4_NL down maximum-group residency", err);
    if (!residency_ready) return false;

    return test_moe_down_iq4_nl_case(
        context, experts, 8u, twin::InputSet::fixture, options,
        active_experts, expert_slots, down_weights, down_weight_buffer,
        address_buffer, true);
}

bool test_input_set(Context& context, std::uint32_t experts,
                    twin::InputSet set, const twin::Options& options) {
    const std::vector<float> logits =
        make_logits(experts, set, options.seed ^ experts);
    bool completed = true;
    std::vector<std::int32_t> multi_ids;
    std::vector<float> multi_weights;
    for (const bool divide_weights : {false, true}) {
        completed &= test_route(context, experts, logits, kRows,
                                divide_weights, set, options, &multi_ids,
                                &multi_weights);
        if (set != twin::InputSet::random) {
            check_literal_choices(set, experts, multi_ids);
            check_independent_weight_goldens(experts, divide_weights,
                                             multi_weights, options);
        }
        for (std::uint32_t row = 0; row < kRows; ++row) {
            const std::span<const float> one_row(
                logits.data() + static_cast<std::size_t>(row) * experts,
                experts);
            std::vector<std::int32_t> single_ids;
            std::vector<float> single_weights;
            completed &= test_route(context, experts, one_row, 1u,
                                    divide_weights, set, options,
                                    &single_ids, &single_weights);
            if (single_ids.size() != kTopK ||
                single_weights.size() != kTopK ||
                multi_ids.size() != static_cast<std::size_t>(kRows) * kTopK ||
                multi_weights.size() !=
                    static_cast<std::size_t>(kRows) * kTopK) {
                check(false, "MoE route multi/single outputs available");
                completed = false;
                continue;
            }
            const auto multi_id_bytes = std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    multi_ids.data() + static_cast<std::size_t>(row) * kTopK),
                kTopK * sizeof(std::int32_t));
            const auto single_id_bytes = std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(single_ids.data()),
                kTopK * sizeof(std::int32_t));
            const auto multi_weight_bytes = std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(
                    multi_weights.data() +
                    static_cast<std::size_t>(row) * kTopK),
                kTopK * sizeof(float));
            const auto single_weight_bytes = std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(single_weights.data()),
                kTopK * sizeof(float));
            check(!twin::compare_bytes(multi_id_bytes, single_id_bytes),
                  "MoE router row IDs independent of batch width");
            check(!twin::compare_bytes(multi_weight_bytes,
                                      single_weight_bytes),
                  "MoE router row weights independent of batch width");
        }
    }
    return completed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_moe_test strata.metallib [--seed N] "
                     "[--input-set random|edge|fixture|real]\n");
        return 2;
    }

    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    if (options.input_set_explicit &&
        options.input_set == twin::InputSet::real) {
        std::printf("SKIP: metal_moe_test real tensors require the model pack "
                    "and CUDA dump fixture recorded in REV-002\n");
        return 77;
    }
    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;

    if (!test_route_dispatch_validation(*context)) ++g_failures;
    if (!test_plan_dispatch_validation(*context)) ++g_failures;

    std::vector<twin::InputSet> sets;
    if (options.input_set_explicit) {
        sets.push_back(options.input_set);
    } else {
        sets = {twin::InputSet::random, twin::InputSet::edge,
                twin::InputSet::fixture};
    }
    for (const twin::InputSet set : sets) {
        options.input_set = set;
        for (const std::uint32_t experts : {512u, 256u}) {
            if (!test_input_set(*context, experts, set, options)) {
                ++g_failures;
            }
        }
        if (!test_plan_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_swiglu_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_shared_swiglu_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_shared_expert_chain_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_combine_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_q2_0_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_iq1_m_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_iq2_xxs_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_iq2_xs_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_iq2_s_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_iq3_xxs_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_iq3_s_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_gu_iq4_xs_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_down_q2_0_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (!test_moe_down_iq4_nl_inputs(*context, set, options)) {
            ++g_failures;
        }
        if (set == twin::InputSet::fixture &&
            !test_moe_gu_iq1_m_max_groups(*context, options)) {
            ++g_failures;
        }
        if (set == twin::InputSet::fixture) {
            if (!test_moe_gu_iq2_xxs_max_groups(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_gu_iq2_xxs_onehot_fixture(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_gu_iq2_xs_max_groups(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_gu_iq2_s_max_groups(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_gu_iq3_xxs_max_groups(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_gu_iq3_s_max_groups(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_gu_iq4_xs_max_groups(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_down_q2_0_max_groups(*context, options)) {
                ++g_failures;
            }
            if (!test_moe_down_iq4_nl_max_groups(*context, options)) {
                ++g_failures;
            }
        }
    }
    if (!test_weight_mode_probe(*context, options)) {
        ++g_failures;
    }
    std::printf("metal_moe_test: %s\n",
                g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
