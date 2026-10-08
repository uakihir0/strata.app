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
#include "strata/metal/gr_dispatch.hpp"
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
#include <type_traits>
#include <utility>
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

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

bool dispatch_grouped_experts(
    Context& context, MoeExpertFormat format, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    const twin::GuardedBuffer& address_buffer,
    const twin::GuardedBuffer& input_buffer,
    const twin::GuardedBuffer& plan_buffer,
    const twin::GuardedBuffer& gate_buffer,
    const twin::GuardedBuffer& up_buffer, std::string& error,
    bool caller_owned = false) {
    const MoeGroupedExpertBuffers buffers{
        context.buffers().read_only(address_buffer.view()),
        context.buffers().read_only(input_buffer.view()),
        context.buffers().read_only(plan_buffer.view()),
        gate_buffer.view(),
        up_buffer.view(),
    };
    auto command_buffer =
        context.make_command_buffer("grouped MoE expert twin", error);
    if (!command_buffer) {
        return false;
    }
    const bool encoded = caller_owned
        ? Encoder::encode(
              context, *command_buffer, "grouped MoE expert twin",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_moe_experts(context, encoder, rows, experts,
                                            layer, format, buffers,
                                            encode_error);
              },
              error)
        : encode_moe_experts(context, *command_buffer, rows, experts, layer,
                             format, buffers, error);
    if (!encoded || !command_buffer->commit(error) ||
        !command_buffer->wait_until_completed(error)) {
        return false;
    }
    error.clear();
    return true;
}

bool dispatch_grouped_down(
    Context& context, MoeDownFormat format, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    const twin::GuardedBuffer& address_buffer,
    const twin::GuardedBuffer& input_buffer,
    const twin::GuardedBuffer& plan_buffer,
    const twin::GuardedBuffer& parts_buffer, std::string& error,
    bool caller_owned = false) {
    const MoeGroupedDownBuffers buffers{
        context.buffers().read_only(address_buffer.view()),
        context.buffers().read_only(input_buffer.view()),
        context.buffers().read_only(plan_buffer.view()),
        parts_buffer.view(),
    };
    auto command_buffer =
        context.make_command_buffer("grouped MoE down twin", error);
    if (!command_buffer) return false;
    const bool encoded = caller_owned
        ? Encoder::encode(
              context, *command_buffer, "grouped MoE down twin",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_moe_down(context, encoder, rows, experts,
                                         layer, format, buffers,
                                         encode_error);
              },
              error)
        : encode_moe_down(context, *command_buffer, rows, experts, layer,
                          format, buffers, error);
    if (!encoded || !command_buffer->commit(error) ||
        !command_buffer->wait_until_completed(error)) {
        return false;
    }
    error.clear();
    return true;
}

bool dispatch_moe_combine(
    Context& context, std::uint32_t rows,
    const twin::GuardedBuffer& parts_buffer,
    const twin::GuardedBuffer& weights_buffer,
    const twin::GuardedBuffer& shared_buffer,
    const twin::GuardedBuffer& gate_buffer,
    const twin::GuardedBuffer& output_buffer, std::string& error,
    bool caller_owned = false) {
    const MoeCombineBuffers buffers{
        context.buffers().read_only(parts_buffer.view()),
        context.buffers().read_only(weights_buffer.view()),
        context.buffers().read_only(shared_buffer.view()),
        context.buffers().read_only(gate_buffer.view()),
        output_buffer.view(),
    };
    auto command_buffer =
        context.make_command_buffer("MoE combine twin", error);
    if (!command_buffer) return false;
    const bool encoded = caller_owned
        ? Encoder::encode(
              context, *command_buffer, "MoE combine twin",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_moe_combine(context, encoder, rows, buffers,
                                            encode_error);
              },
              error)
        : encode_moe_combine(context, *command_buffer, rows, buffers, error);
    if (!encoded || !command_buffer->commit(error) ||
        !command_buffer->wait_until_completed(error)) {
        return false;
    }
    error.clear();
    return true;
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

bool test_grouped_expert_dispatch_validation(Context& context) {
    constexpr std::uint32_t kRows = 1u;
    constexpr std::uint32_t kExperts = 1u;
    constexpr std::uint32_t kLayer = 0u;
    constexpr std::size_t kAddressBytes = sizeof(MoeExpertAddress);
    constexpr std::size_t kActivationBytes = 80u * sizeof(Q81);
    constexpr std::size_t kPlanBytes = sizeof(TieredMoePlan);
    constexpr std::size_t kOutputBytes =
        kTopK * kMoeExpertWidth * sizeof(float);
    constexpr std::size_t kSafeWeightBytes =
        kMoeExpertWidth * kMoeQ2BlocksPerRow * sizeof(block_q2_0);
    constexpr std::string_view label = "grouped MoE expert encoder validation";

    std::string err;
    twin::GuardedBuffer addresses;
    twin::GuardedBuffer activations;
    twin::GuardedBuffer plans;
    twin::GuardedBuffer gate;
    twin::GuardedBuffer up;
    twin::GuardedBuffer alias;
    twin::GuardedBuffer safe_weights;
    check(addresses.allocate(context, "grouped expert validation addresses",
                             2u * kAddressBytes + 1u, err),
          "grouped expert address table buffer", err);
    check(activations.allocate(
              context, "grouped expert validation Q8_1 activations",
              2u * kActivationBytes, err),
          "grouped expert Q8_1 activation buffer", err);
    check(plans.allocate(context, "grouped expert validation plans",
                         kPlanBytes, err),
          "grouped expert plan table buffer", err);
    check(gate.allocate(context, "grouped expert validation gate",
                        2u * kOutputBytes, err),
          "grouped expert gate output buffer", err);
    check(up.allocate(context, "grouped expert validation up",
                      2u * kOutputBytes, err),
          "grouped expert up output buffer", err);
    check(alias.allocate(context, "grouped expert validation alias",
                         kOutputBytes + kAddressBytes, err),
          "grouped expert alias buffer", err);
    check(safe_weights.allocate(context, "grouped expert validation weights",
                                kSafeWeightBytes, err),
          "grouped expert safe weight buffer", err);
    if (!addresses.view() || !activations.view() || !plans.view() ||
        !gate.view() || !up.view() || !alias.view() ||
        !safe_weights.view()) {
        return false;
    }

    std::fill(safe_weights.payload().begin(), safe_weights.payload().end(), 0u);
    MoeExpertAddress safe_address{};
    safe_address.role[0] = safe_weights.view().gpu_address;
    safe_address.role[1] = safe_weights.view().gpu_address;
    const std::array safe_address_table{safe_address, safe_address};
    std::memcpy(addresses.payload().data(), safe_address_table.data(),
                sizeof(safe_address_table));
    if (!context.buffers().commit_residency(err)) {
        check(false, "grouped expert validation weight residency", err);
        return false;
    }

    std::fill(gate.payload().begin(), gate.payload().end(), 0x5au);
    std::fill(up.payload().begin(), up.payload().end(), 0xa5u);
    std::fill(alias.payload().begin(), alias.payload().end(), 0x3cu);
    std::fill(activations.payload().begin(), activations.payload().end(), 0u);
    const std::vector<std::uint8_t> gate_before(gate.payload().begin(),
                                                gate.payload().end());
    const std::vector<std::uint8_t> up_before(up.payload().begin(),
                                              up.payload().end());
    const std::vector<std::uint8_t> alias_before(alias.payload().begin(),
                                                 alias.payload().end());
    const ReadOnlyBufferRef addresses_view =
        context.buffers().read_only(addresses.view());
    const ReadOnlyBufferRef activations_view =
        context.buffers().read_only(activations.view());
    const ReadOnlyBufferRef plans_view =
        context.buffers().read_only(plans.view());
    if (!addresses_view || !activations_view || !plans_view) {
        check(false, std::string(label) + " read-only input views", err);
        return false;
    }
    const MoeGroupedExpertBuffers valid{
        addresses_view, activations_view, plans_view, gate.view(), up.view()};

    const auto outputs_unchanged = [&] {
        return std::equal(gate_before.begin(), gate_before.end(),
                          gate.payload().begin(), gate.payload().end()) &&
               std::equal(up_before.begin(), up_before.end(),
                          up.payload().begin(), up.payload().end()) &&
               std::equal(alias_before.begin(), alias_before.end(),
                          alias.payload().begin(), alias.payload().end());
    };
    const auto reject = [&](const std::string& case_label,
                            std::uint32_t rows, std::uint32_t experts,
                            std::uint32_t layer, MoeExpertFormat format,
                            MoeGroupedExpertBuffers buffers,
                            std::string_view error_fragment) {
        auto command = context.make_command_buffer(case_label, err);
        const bool rejected = command && !encode_moe_experts(
            context, *command, rows, experts, layer, format, buffers, err);
        const bool discarded =
            command && command->unsafe_native_handle() == nullptr;
        const bool error_matches =
            err.find(error_fragment) != std::string::npos;
        const bool unchanged = outputs_unchanged();
        check(rejected && discarded && error_matches && unchanged,
              case_label + " rejects before dispatch and discards", err);
        return rejected && discarded && error_matches && unchanged;
    };

    const bool zero_rows = reject(
        "reject zero-row grouped experts", 0u, kExperts, kLayer,
        MoeExpertFormat::q2_0, valid, "1..8 rows");
    const bool too_many_rows = reject(
        "reject over-capacity grouped experts", 9u, kExperts, kLayer,
        MoeExpertFormat::q2_0, valid, "1..8 rows");
    const bool zero_experts = reject(
        "reject zero-expert grouped experts", kRows, 0u, kLayer,
        MoeExpertFormat::q2_0, valid, "1..512 experts");
    const bool invalid_layer = reject(
        "reject out-of-range grouped expert layer", kRows, kExperts,
        kTieredMaxLayers, MoeExpertFormat::q2_0, valid, "valid layer");
    const bool invalid_format = reject(
        "reject unsupported grouped expert format", kRows, kExperts,
        kLayer, static_cast<MoeExpertFormat>(0xffu), valid, "unsupported");
    const bool short_addresses = reject(
        "reject undersized grouped expert address table", kRows, kExperts,
        kLayer, MoeExpertFormat::q2_0,
        {addresses_view.slice(0u, kAddressBytes - 1u), activations_view,
         plans_view, gate.view(), up.view()},
        "address table");
    const bool misaligned_addresses = reject(
        "reject misaligned grouped expert address table", kRows, kExperts,
        kLayer, MoeExpertFormat::q2_0,
        {addresses_view.slice(1u, kAddressBytes), activations_view,
         plans_view, gate.view(), up.view()},
        "misaligned");
    const bool short_activations = reject(
        "reject undersized grouped expert activations", kRows, kExperts,
        kLayer, MoeExpertFormat::q2_0,
        {addresses_view, activations_view.slice(0u, sizeof(Q81)),
         plans_view, gate.view(), up.view()},
        "activations");
    const bool short_plans = reject(
        "reject undersized grouped expert plans", kRows, kExperts, kLayer,
        MoeExpertFormat::q2_0,
        {addresses_view, activations_view,
         plans_view.slice(0u, kPlanBytes - 1u), gate.view(), up.view()},
        "plan table");
    const BufferRef alias_view = alias.view();
    const ReadOnlyBufferRef alias_read =
        context.buffers().read_only(alias_view);
    const bool aliased_outputs = reject(
        "reject overlapping grouped expert outputs", kRows, kExperts,
        kLayer, MoeExpertFormat::q2_0,
        {addresses_view, activations_view, plans_view, alias_view,
         alias_view},
        "overlap");
    const bool aliased_input_output = reject(
        "reject grouped expert input and output alias", kRows, kExperts,
        kLayer, MoeExpertFormat::q2_0,
        {alias_read, activations_view, plans_view, alias_view, up.view()},
        "overlap");
    const bool partial_input_output_alias = reject(
        "reject partial grouped expert input and output alias", kRows,
        kExperts, kLayer, MoeExpertFormat::q2_0,
        {context.buffers().read_only(
             alias_view.slice(0u, kAddressBytes)),
         activations_view, plans_view,
         alias_view.slice(kAddressBytes / 2u, kOutputBytes), up.view()},
        "overlap");

    auto late_failure_command =
        context.make_command_buffer("reject caller-owned late grouped expert failure",
                                    err);
    const bool late_failure = late_failure_command && !Encoder::encode(
        context, *late_failure_command, "grouped expert late failure",
        [&](Encoder& encoder, std::string& body_error) {
            if (!encode_moe_experts(context, encoder, kRows, kExperts, kLayer,
                                    MoeExpertFormat::q2_0, valid,
                                    body_error)) {
                return false;
            }
            return encode_moe_experts(context, encoder, 9u, kExperts, kLayer,
                                      MoeExpertFormat::q2_0, valid,
                                      body_error);
        },
        err);
    const bool late_discarded = late_failure_command &&
        late_failure_command->unsafe_native_handle() == nullptr;
    const bool after_late_unchanged = outputs_unchanged();
    check(late_failure && late_discarded && after_late_unchanged &&
              err.find("1..8 rows") != std::string::npos,
          "caller-owned late grouped expert failure rolls back earlier work",
          err);

    const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    const bool concurrent_mode = dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent";
    bool open_group_rejected = true;
    if (concurrent_mode) {
        auto group_command = context.make_command_buffer(
            "reject grouped experts inside concurrent group", err);
        const bool encoded = group_command && !Encoder::encode(
            context, *group_command, "grouped expert open group",
            [&](Encoder& encoder, std::string& body_error) {
                return encoder.group_begin(body_error) &&
                    encode_moe_experts(context, encoder, kRows, kExperts,
                                       kLayer, MoeExpertFormat::q2_0, valid,
                                       body_error);
            },
            err);
        const bool discarded =
            group_command && group_command->unsafe_native_handle() == nullptr;
        open_group_rejected = encoded && discarded &&
            err.find("ordered dispatch boundary") != std::string::npos;
        check(open_group_rejected,
              "caller-owned grouped experts rejects concurrent group", err);
    }

    const auto make_test_plan = [](std::uint32_t rows,
                                   std::int32_t expert) {
        TieredMoePlan plan{};
        plan.n_groups = 1u;
        plan.n_entries = rows * kTopK;
        plan.skip = 1u;
        plan.grp_e[0] = expert;
        plan.grp_start[1] = static_cast<std::int32_t>(plan.n_entries);
        for (std::uint32_t entry = 0; entry < plan.n_entries; ++entry) {
            plan.ent_tok[entry] = static_cast<std::int32_t>(entry / kTopK);
            plan.ent_dst[entry] = static_cast<std::int32_t>(entry);
        }
        return plan;
    };
    const auto dispatch_mismatched_plan = [&](
        const std::string& case_label, std::uint32_t rows,
        std::uint32_t experts, const TieredMoePlan& plan) {
        std::memcpy(plans.payload().data(), &plan, sizeof(plan));
        auto command = context.make_command_buffer(case_label, err);
        const bool completed = command && encode_moe_experts(
            context, *command, rows, experts, kLayer,
            MoeExpertFormat::q2_0, valid, err) &&
            command->commit(err) && command->wait_until_completed(err);
        const bool unchanged = outputs_unchanged();
        check(completed && unchanged,
              case_label + " is bounded by the GPU plan", err);
        return completed && unchanged;
    };
    const TieredMoePlan two_row_plan = make_test_plan(2u, 0);
    const TieredMoePlan wrong_expert_plan = make_test_plan(1u, 1);
    TieredMoePlan wrong_token_plan = make_test_plan(1u, 0);
    wrong_token_plan.ent_tok[0] = 1;
    TieredMoePlan wrong_group_range_plan = make_test_plan(1u, 0);
    wrong_group_range_plan.grp_start[1] =
        static_cast<std::int32_t>(wrong_group_range_plan.n_entries + 1u);
    const bool mismatched_rows = dispatch_mismatched_plan(
        "reject grouped expert plan row-count mismatch", 1u, kExperts,
        two_row_plan);
    const bool mismatched_experts = dispatch_mismatched_plan(
        "reject grouped expert plan expert-count mismatch", kRows,
        kExperts, wrong_expert_plan);
    const bool invalid_token = dispatch_mismatched_plan(
        "reject grouped expert plan token index", kRows, kExperts,
        wrong_token_plan);
    const bool invalid_group_range = dispatch_mismatched_plan(
        "reject grouped expert plan group range", kRows, kExperts,
        wrong_group_range_plan);

    const bool final_unchanged = outputs_unchanged();
    check(final_unchanged,
          std::string(label) + " leaves rejected outputs unchanged");
    return zero_rows && too_many_rows && zero_experts && invalid_layer &&
           invalid_format && short_addresses && misaligned_addresses &&
           short_activations && short_plans && aliased_outputs &&
           aliased_input_output && partial_input_output_alias &&
           late_failure && late_discarded && after_late_unchanged &&
           open_group_rejected && mismatched_rows && mismatched_experts &&
           invalid_token && invalid_group_range && final_unchanged;
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

    const MoeSwiGLUBuffers buffers{
        context.buffers().read_only(gate_buffer.view()),
        context.buffers().read_only(up_buffer.view()), output_buffer.view()};
    auto command = context.make_command_buffer(label, err);
    const bool caller_owned =
        rows == 1u && set == twin::InputSet::fixture;
    const bool encoded = command && (caller_owned
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_moe_swiglu(context, encoder, rows,
                                           shared_expert, buffers,
                                           encode_error);
              },
              err)
        : encode_moe_swiglu(context, *command, rows, shared_expert, buffers,
                            err));
    const bool dispatched = encoded && command->commit(err) &&
                            command->wait_until_completed(err);
    check(dispatched,
          label + (caller_owned ? " caller-owned engine encoder"
                                : " standalone engine encoder"),
          err);
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

bool test_swiglu_dispatch_validation(Context& context) {
    constexpr std::uint32_t kRows = 1u;
    constexpr std::size_t kRoutedBytes =
        80u * 640u * sizeof(float);
    constexpr std::size_t kOneRowRoutedBytes =
        10u * 640u * sizeof(float);
    constexpr std::size_t kSharedRowBytes = 640u * sizeof(float);
    constexpr std::uint8_t kPoison = 0xa5u;
    const std::string label = "MoE SwiGLU encoder validation";
    std::string err;

    twin::GuardedBuffer gate;
    twin::GuardedBuffer up;
    twin::GuardedBuffer output;
    if (!gate.allocate(context, label + " gate", kRoutedBytes, err) ||
        !up.allocate(context, label + " up", kRoutedBytes * 2u, err) ||
        !output.allocate(context, label + " output", kRoutedBytes + 1u,
                         err)) {
        check(false, label + " buffers", err);
        return false;
    }
    std::fill(gate.payload().begin(), gate.payload().end(), 0u);
    std::fill(up.payload().begin(), up.payload().end(), 0u);
    std::fill(output.payload().begin(), output.payload().end(), kPoison);

    const ReadOnlyBufferRef gate_view =
        context.buffers().read_only(gate.view());
    const ReadOnlyBufferRef up_view =
        context.buffers().read_only(up.view().slice(0u, kRoutedBytes));
    const MoeSwiGLUBuffers valid{gate_view, up_view, output.view()};
    const std::vector<std::uint8_t> gate_before(gate.payload().begin(),
                                                gate.payload().end());
    const std::vector<std::uint8_t> up_before(up.payload().begin(),
                                              up.payload().end());
    const std::vector<std::uint8_t> output_before(output.payload().begin(),
                                                 output.payload().end());

    const auto reject = [&](const std::string& case_label,
                            std::uint32_t rows, bool shared_expert,
                            const MoeSwiGLUBuffers& buffers,
                            std::string_view expected_error) {
        auto command = context.make_command_buffer(case_label, err);
        const bool rejected = command && !encode_moe_swiglu(
            context, *command, rows, shared_expert, buffers, err);
        const bool discarded = command &&
            command->unsafe_native_handle() == nullptr;
        const bool unchanged =
            std::equal(gate_before.begin(), gate_before.end(),
                       gate.payload().begin(), gate.payload().end()) &&
            std::equal(up_before.begin(), up_before.end(), up.payload().begin(),
                       up.payload().end()) &&
            std::equal(output_before.begin(), output_before.end(),
                       output.payload().begin(), output.payload().end());
        const bool error_matches =
            err.find(expected_error) != std::string::npos;
        check(rejected && discarded && unchanged && error_matches,
              case_label + " rejects before dispatch and discards", err);
        return rejected && discarded && unchanged && error_matches;
    };

    const bool zero_rows = reject("reject zero-row MoE SwiGLU", 0u, false,
                                  valid, "1..8 rows");
    const bool too_many_rows = reject("reject over-capacity MoE SwiGLU", 9u,
                                      false, valid, "1..8 rows");
    const MoeSwiGLUBuffers short_gate{
        gate_view.slice(0u, kOneRowRoutedBytes - sizeof(float)), up_view,
        output.view()};
    const bool undersized_gate = reject(
        "reject short MoE SwiGLU gate", kRows, false, short_gate, "gate input");
    const MoeSwiGLUBuffers short_shared_gate{
        gate_view.slice(0u, kSharedRowBytes - sizeof(float)), up_view,
        output.view()};
    const bool undersized_shared_gate = reject(
        "reject short shared-expert SwiGLU gate", kRows, true,
        short_shared_gate, "gate input");
    const MoeSwiGLUBuffers misaligned_up{
        gate_view,
        context.buffers().read_only(
            up.view().slice(1u, kOneRowRoutedBytes)),
        output.view()};
    const bool misaligned = reject("reject misaligned MoE SwiGLU input",
                                   kRows, false, misaligned_up, "misaligned");
    const MoeSwiGLUBuffers misaligned_output{
        gate_view, up_view,
        output.view().slice(1u, kOneRowRoutedBytes)};
    const bool misaligned_output_rejected = reject(
        "reject misaligned MoE SwiGLU output", kRows, false,
        misaligned_output, "misaligned");
    const MoeSwiGLUBuffers short_output{
        gate_view, up_view,
        output.view().slice(0u, kOneRowRoutedBytes - sizeof(float))};
    const bool undersized_output = reject(
        "reject short MoE SwiGLU output", kRows, false, short_output,
        "output");
    const MoeSwiGLUBuffers gate_output_alias{
        gate_view, up_view, gate.view()};
    const bool gate_alias = reject("reject MoE SwiGLU gate/output alias",
                                   kRows, false, gate_output_alias, "overlap");
    const MoeSwiGLUBuffers partial_up_output_alias{
        gate_view, up_view,
        up.view().slice(sizeof(float), kOneRowRoutedBytes)};
    const bool up_alias = reject(
        "reject partial MoE SwiGLU up/output alias", kRows, false,
        partial_up_output_alias, "overlap");

    std::fill(output.payload().begin(), output.payload().end(), kPoison);
    auto late_failure_command = context.make_command_buffer(
        label + " caller-owned late failure", err);
    check(late_failure_command != nullptr,
          label + " late-failure command buffer", err);
    if (!late_failure_command) return false;
    const bool late_failure_rejected = !Encoder::encode(
        context, *late_failure_command, "MoE SwiGLU late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_moe_swiglu(context, encoder, kRows, false, valid,
                                   encode_error)) {
                return false;
            }
            return encode_moe_swiglu(context, encoder, 9u, false, valid,
                                     encode_error);
        },
        err);
    const bool late_failure_discarded =
        late_failure_command->unsafe_native_handle() == nullptr;
    const bool late_failure_untouched =
        std::all_of(output.payload().begin(), output.payload().end(),
                    [=](std::uint8_t value) { return value == kPoison; });
    check(late_failure_rejected && late_failure_discarded &&
              late_failure_untouched &&
              err.find("1..8 rows") != std::string::npos,
          label + " caller-owned late failure rolls back its earlier dispatch",
          err);

    bool concurrent_boundary_ok = true;
    const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        auto group_command = context.make_command_buffer(
            label + " open concurrent group", err);
        check(group_command != nullptr,
              label + " open-group command buffer", err);
        if (!group_command) return false;
        bool saw_open_group = false;
        bool rejected_open_group = false;
        const bool group_encoded = Encoder::encode(
            context, *group_command, "MoE SwiGLU open-group boundary",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                saw_open_group = encoder.concurrent_group_open();
                if (saw_open_group) {
                    std::string boundary_error;
                    rejected_open_group = !encode_moe_swiglu(
                        context, encoder, kRows, false, valid,
                        boundary_error) &&
                        boundary_error.find("ordered dispatch boundary") !=
                            std::string::npos;
                }
                if (!encoder.group_end(encode_error)) return false;
                if (!saw_open_group || !rejected_open_group) {
                    encode_error =
                        "MoE SwiGLU did not reject an open concurrent group";
                    return false;
                }
                return true;
            },
            err);
        group_command->discard();
        concurrent_boundary_ok = group_encoded && saw_open_group &&
                                 rejected_open_group &&
                                 group_command->unsafe_native_handle() ==
                                     nullptr;
        check(concurrent_boundary_ok,
              label + " rejects an open concurrent group", err);
    }

    return zero_rows && too_many_rows && undersized_gate &&
           undersized_shared_gate && misaligned && misaligned_output_rejected &&
           undersized_output && gate_alias && up_alias && late_failure_rejected &&
           late_failure_discarded && late_failure_untouched &&
           concurrent_boundary_ok;
}

bool test_moe_down_dispatch_validation(Context& context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t experts = 256u;
    constexpr std::uint32_t layer = kMoeGroupedLayer;
    constexpr std::size_t entries =
        static_cast<std::size_t>(rows) * kTopK;
    constexpr std::size_t address_bytes =
        static_cast<std::size_t>(layer + 1u) * experts *
        sizeof(MoeExpertAddress);
    constexpr std::size_t activation_bytes =
        entries * (kMoeDownInputWidth / 32u) * sizeof(Q81);
    constexpr std::size_t plan_bytes =
        static_cast<std::size_t>(layer + 1u) * sizeof(TieredMoePlan);
    constexpr std::size_t parts_bytes =
        entries * kMoeDownWidth * sizeof(float);
    const std::string label = "MoE grouped down dispatch validation";
    std::string err;
    twin::GuardedBuffer addresses;
    twin::GuardedBuffer activations;
    twin::GuardedBuffer plans;
    twin::GuardedBuffer parts;
    check(addresses.allocate(context, label + " addresses",
                             address_bytes + 32u, err),
          label + " address allocation", err);
    check(activations.allocate(context, label + " activations",
                               activation_bytes + 32u, err),
          label + " activation allocation", err);
    check(plans.allocate(context, label + " plans", plan_bytes + 32u, err),
          label + " plan allocation", err);
    check(parts.allocate(context, label + " parts",
                         parts_bytes + 32u, err),
          label + " parts allocation", err);
    if (!addresses.view() || !activations.view() || !plans.view() ||
        !parts.view()) {
        return false;
    }

    const ReadOnlyBufferRef address_view =
        context.buffers().read_only(
            addresses.view().slice(0u, address_bytes));
    const ReadOnlyBufferRef activation_view =
        context.buffers().read_only(
            activations.view().slice(0u, activation_bytes));
    const ReadOnlyBufferRef plan_view =
        context.buffers().read_only(plans.view().slice(0u, plan_bytes));
    const BufferRef parts_view = parts.view().slice(0u, parts_bytes);
    const MoeGroupedDownBuffers valid{
        address_view, activation_view, plan_view, parts_view};
    const std::vector<std::uint8_t> parts_before(
        parts.payload().begin(), parts.payload().end());

    const auto reject = [&](const std::string& case_label,
                            std::uint32_t case_rows,
                            std::uint32_t case_experts,
                            std::uint32_t case_layer,
                            MoeDownFormat format,
                            const MoeGroupedDownBuffers& buffers,
                            std::string_view expected_error) {
        auto command = context.make_command_buffer(case_label, err);
        const bool rejected = command && !encode_moe_down(
            context, *command, case_rows, case_experts, case_layer, format,
            buffers, err);
        const bool discarded =
            command && command->unsafe_native_handle() == nullptr;
        const bool unchanged = std::equal(
            parts_before.begin(), parts_before.end(), parts.payload().begin(),
            parts.payload().end());
        const bool error_matches =
            err.find(expected_error) != std::string::npos;
        check(rejected && discarded && unchanged && error_matches,
              case_label + " rejects before dispatch and discards", err);
        return rejected && discarded && unchanged && error_matches;
    };

    const bool zero_rows = reject(
        label + " zero rows", 0u, experts, layer, MoeDownFormat::q2_0,
        valid, "1..8 rows");
    const bool too_many_rows = reject(
        label + " too many rows", 9u, experts, layer, MoeDownFormat::q2_0,
        valid, "1..8 rows");
    const bool zero_experts = reject(
        label + " zero experts", rows, 0u, layer, MoeDownFormat::q2_0,
        valid, "1..512 experts");
    const bool too_many_experts = reject(
        label + " too many experts", rows, 513u, layer,
        MoeDownFormat::q2_0, valid, "1..512 experts");
    const bool invalid_layer = reject(
        label + " invalid layer", rows, experts, kTieredMaxLayers,
        MoeDownFormat::q2_0, valid, "valid layer");
    const bool invalid_format = reject(
        label + " invalid format", rows, experts, layer,
        static_cast<MoeDownFormat>(99), valid, "format is unsupported");

    const MoeGroupedDownBuffers short_addresses{
        context.buffers().read_only(
            addresses.view().slice(0u, address_bytes - 1u)),
        activation_view, plan_view, parts_view};
    const bool short_address = reject(
        label + " short address table", rows, experts, layer,
        MoeDownFormat::q2_0, short_addresses, "address table");
    const MoeGroupedDownBuffers misaligned_addresses{
        context.buffers().read_only(
            addresses.view().slice(1u, address_bytes)),
        activation_view, plan_view, parts_view};
    const bool misaligned_address = reject(
        label + " misaligned address table", rows, experts, layer,
        MoeDownFormat::q2_0, misaligned_addresses, "misaligned");
    const MoeGroupedDownBuffers short_activations{
        address_view,
        context.buffers().read_only(
            activations.view().slice(0u, activation_bytes - sizeof(Q81))),
        plan_view, parts_view};
    const bool short_activation = reject(
        label + " short activations", rows, experts, layer,
        MoeDownFormat::q2_0, short_activations, "activations");
    const MoeGroupedDownBuffers misaligned_activations{
        address_view,
        context.buffers().read_only(
            activations.view().slice(1u, activation_bytes)),
        plan_view, parts_view};
    const bool misaligned_activation = reject(
        label + " misaligned activations", rows, experts, layer,
        MoeDownFormat::q2_0, misaligned_activations, "misaligned");
    const MoeGroupedDownBuffers short_plans{
        address_view, activation_view,
        context.buffers().read_only(
            plans.view().slice(0u, plan_bytes - sizeof(TieredMoePlan))),
        parts_view};
    const bool short_plan = reject(
        label + " short plans", rows, experts, layer,
        MoeDownFormat::q2_0, short_plans, "plan table");
    const MoeGroupedDownBuffers misaligned_plans{
        address_view, activation_view,
        context.buffers().read_only(plans.view().slice(1u, plan_bytes)),
        parts_view};
    const bool misaligned_plan = reject(
        label + " misaligned plans", rows, experts, layer,
        MoeDownFormat::q2_0, misaligned_plans, "misaligned");
    const MoeGroupedDownBuffers short_parts{
        address_view, activation_view, plan_view,
        parts.view().slice(0u, parts_bytes - sizeof(float))};
    const bool short_output = reject(
        label + " short output", rows, experts, layer,
        MoeDownFormat::q2_0, short_parts, "parts output");
    const MoeGroupedDownBuffers misaligned_parts{
        address_view, activation_view, plan_view,
        parts.view().slice(1u, parts_bytes)};
    const bool misaligned_output = reject(
        label + " misaligned output", rows, experts, layer,
        MoeDownFormat::q2_0, misaligned_parts, "misaligned");

    const ReadOnlyBufferRef parts_as_input =
        context.buffers().read_only(parts_view);
    const MoeGroupedDownBuffers address_output_alias{
        parts_as_input, activation_view, plan_view, parts_view};
    const bool address_alias = reject(
        label + " address/output alias", rows, experts, layer,
        MoeDownFormat::q2_0, address_output_alias, "overlap");
    const MoeGroupedDownBuffers activation_output_alias{
        address_view, parts_as_input, plan_view, parts_view};
    const bool activation_alias = reject(
        label + " activation/output alias", rows, experts, layer,
        MoeDownFormat::q2_0, activation_output_alias, "overlap");
    const MoeGroupedDownBuffers plan_output_alias{
        address_view, activation_view, parts_as_input, parts_view};
    const bool plan_alias = reject(
        label + " plan/output alias", rows, experts, layer,
        MoeDownFormat::q2_0, plan_output_alias, "overlap");
    const MoeGroupedDownBuffers readonly_inputs_alias{
        parts_as_input, parts_as_input, parts_as_input, parts_view};
    const bool readonly_alias = reject(
        label + " read-only input alias", rows, experts, layer,
        MoeDownFormat::q2_0, readonly_inputs_alias, "overlap");

    auto late_failure_command =
        context.make_command_buffer(label + " late failure", err);
    check(late_failure_command != nullptr,
          label + " late-failure command buffer", err);
    if (!late_failure_command) return false;
    const bool late_failure_rejected = !Encoder::encode(
        context, *late_failure_command, "MoE down late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_moe_down(context, encoder, rows, experts, layer,
                                 MoeDownFormat::q2_0, valid, encode_error)) {
                return false;
            }
            return encode_moe_down(context, encoder, 9u, experts, layer,
                                   MoeDownFormat::q2_0, valid, encode_error);
        },
        err);
    const bool late_failure_discarded =
        late_failure_command->unsafe_native_handle() == nullptr;
    const bool late_failure_untouched = std::equal(
        parts_before.begin(), parts_before.end(), parts.payload().begin(),
        parts.payload().end());
    check(late_failure_rejected && late_failure_discarded &&
              late_failure_untouched &&
              err.find("1..8 rows") != std::string::npos,
          label + " caller-owned late failure rolls back its dispatch", err);

    const auto dispatch_invalid_gpu_plan = [&](
        const std::string& case_label, MoeDownFormat format,
        std::int32_t expert, std::int32_t group_end,
        std::int32_t invalid_destination) {
        TieredMoePlan plan{};
        plan.n_groups = 1u;
        plan.n_entries = static_cast<std::uint32_t>(entries);
        plan.skip = 1u;
        plan.grp_e[0] = expert;
        plan.grp_start[0] = 0;
        plan.grp_start[1] = group_end;
        for (std::uint32_t entry = 0u; entry < entries; ++entry) {
            plan.ent_dst[entry] = static_cast<std::int32_t>(entry);
        }
        if (invalid_destination >= 0) {
            plan.ent_dst[0] = invalid_destination;
        }
        std::fill(plans.payload().begin(), plans.payload().end(), 0u);
        std::memcpy(plans.payload().data() +
                        static_cast<std::size_t>(layer) * sizeof(plan),
                    &plan, sizeof(plan));
        std::fill(parts.payload().begin(), parts.payload().end(), 0x5au);
        const std::vector<std::uint8_t> before(parts.payload().begin(),
                                               parts.payload().end());
        plans.capture_before_dispatch();
        parts.capture_before_dispatch();
        const bool completed = dispatch_grouped_down(
            context, format, rows, experts, layer, addresses, activations,
            plans, parts, err);
        const bool unchanged = std::equal(
            before.begin(), before.end(), parts.payload().begin(),
            parts.payload().end());
        const bool guards_unchanged =
            !parts.first_unwritten_change({});
        check(completed && unchanged && guards_unchanged,
              case_label +
                  " GPU rejects malformed plan without output or guard writes",
              err);
        return completed && unchanged && guards_unchanged;
    };
    const bool invalid_q2_destination = dispatch_invalid_gpu_plan(
        label + " Q2_0 destination bound", MoeDownFormat::q2_0, 0,
        static_cast<std::int32_t>(entries),
        static_cast<std::int32_t>(entries));
    const bool invalid_iq4_destination = dispatch_invalid_gpu_plan(
        label + " IQ4_NL destination bound", MoeDownFormat::iq4_nl, 0,
        static_cast<std::int32_t>(entries),
        static_cast<std::int32_t>(entries));
    const bool invalid_iq4_expert = dispatch_invalid_gpu_plan(
        label + " IQ4_NL expert bound", MoeDownFormat::iq4_nl,
        static_cast<std::int32_t>(experts),
        static_cast<std::int32_t>(entries), -1);
    const bool invalid_iq4_group_range = dispatch_invalid_gpu_plan(
        label + " IQ4_NL group range", MoeDownFormat::iq4_nl, 0,
        static_cast<std::int32_t>(entries + 1u), -1);

    bool concurrent_boundary_ok = true;
    const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        auto command =
            context.make_command_buffer(label + " open group", err);
        check(command != nullptr, label + " open-group command buffer", err);
        if (!command) return false;
        bool saw_open_group = false;
        bool rejected_open_group = false;
        const bool encoded = Encoder::encode(
            context, *command, "MoE down open-group boundary",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                saw_open_group = encoder.concurrent_group_open();
                if (saw_open_group) {
                    std::string boundary_error;
                    rejected_open_group = !encode_moe_down(
                        context, encoder, rows, experts, layer,
                        MoeDownFormat::q2_0, valid, boundary_error) &&
                        boundary_error.find("ordered dispatch boundary") !=
                            std::string::npos;
                }
                if (!encoder.group_end(encode_error)) return false;
                if (!saw_open_group || !rejected_open_group) {
                    encode_error =
                        "MoE down did not reject an open concurrent group";
                    return false;
                }
                return true;
            },
            err);
        command->discard();
        concurrent_boundary_ok =
            encoded && saw_open_group && rejected_open_group &&
            command->unsafe_native_handle() == nullptr;
        check(concurrent_boundary_ok,
              label + " rejects an open concurrent group", err);
    }

    return zero_rows && too_many_rows && zero_experts && too_many_experts &&
           invalid_layer && invalid_format && short_address &&
           misaligned_address && short_activation && misaligned_activation &&
           short_plan && misaligned_plan && short_output &&
           misaligned_output && address_alias && activation_alias &&
           plan_alias && readonly_alias && late_failure_rejected &&
           late_failure_discarded && late_failure_untouched &&
           invalid_q2_destination && invalid_iq4_destination &&
           invalid_iq4_expert &&
           invalid_iq4_group_range &&
           concurrent_boundary_ok;
}

bool test_moe_combine_dispatch_validation(Context& context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t entries =
        static_cast<std::size_t>(rows) * kTopK;
    constexpr std::size_t parts_bytes =
        entries * kMoeDownWidth * sizeof(float);
    constexpr std::size_t weights_bytes = entries * sizeof(float);
    constexpr std::size_t shared_bytes =
        static_cast<std::size_t>(rows) * kMoeDownWidth * sizeof(float);
    constexpr std::size_t gate_bytes =
        static_cast<std::size_t>(rows) * sizeof(float);
    constexpr std::size_t output_bytes = shared_bytes;
    const std::string label = "MoE combine dispatch validation";
    std::string err;
    twin::GuardedBuffer parts;
    twin::GuardedBuffer weights;
    twin::GuardedBuffer shared;
    twin::GuardedBuffer gate;
    twin::GuardedBuffer output;
    check(parts.allocate(context, label + " parts", parts_bytes + 32u, err),
          label + " parts allocation", err);
    check(weights.allocate(context, label + " weights",
                           weights_bytes + 32u, err),
          label + " weights allocation", err);
    check(shared.allocate(context, label + " shared",
                          shared_bytes + 32u, err),
          label + " shared allocation", err);
    check(gate.allocate(context, label + " gate", gate_bytes + 32u, err),
          label + " gate allocation", err);
    check(output.allocate(context, label + " output",
                          output_bytes + 32u, err),
          label + " output allocation", err);
    if (!parts.view() || !weights.view() || !shared.view() ||
        !gate.view() || !output.view()) {
        return false;
    }

    const ReadOnlyBufferRef parts_view =
        context.buffers().read_only(parts.view().slice(0u, parts_bytes));
    const ReadOnlyBufferRef weights_view =
        context.buffers().read_only(weights.view().slice(0u, weights_bytes));
    const ReadOnlyBufferRef shared_view =
        context.buffers().read_only(shared.view().slice(0u, shared_bytes));
    const ReadOnlyBufferRef gate_view =
        context.buffers().read_only(gate.view().slice(0u, gate_bytes));
    const BufferRef output_view = output.view().slice(0u, output_bytes);
    const MoeCombineBuffers valid{
        parts_view, weights_view, shared_view, gate_view, output_view};
    const std::vector<std::uint8_t> output_before(
        output.payload().begin(), output.payload().end());

    const auto reject = [&](const std::string& case_label,
                            std::uint32_t case_rows,
                            const MoeCombineBuffers& buffers,
                            std::string_view expected_error) {
        auto command = context.make_command_buffer(case_label, err);
        const bool rejected = command && !encode_moe_combine(
            context, *command, case_rows, buffers, err);
        const bool discarded =
            command && command->unsafe_native_handle() == nullptr;
        const bool unchanged = std::equal(
            output_before.begin(), output_before.end(), output.payload().begin(),
            output.payload().end());
        const bool error_matches =
            err.find(expected_error) != std::string::npos;
        check(rejected && discarded && unchanged && error_matches,
              case_label + " rejects before dispatch and discards", err);
        return rejected && discarded && unchanged && error_matches;
    };

    const bool zero_rows =
        reject(label + " zero rows", 0u, valid, "1..8 rows");
    const bool too_many_rows =
        reject(label + " too many rows", 9u, valid, "1..8 rows");
    const bool short_parts = reject(
        label + " short parts", rows,
        {parts_view.slice(0u, parts_bytes - sizeof(float)), weights_view,
         shared_view, gate_view, output_view},
        "routed parts");
    const bool short_weights = reject(
        label + " short weights", rows,
        {parts_view,
         weights_view.slice(0u, weights_bytes - sizeof(float)),
         shared_view, gate_view, output_view},
        "route weights");
    const bool short_shared = reject(
        label + " short shared output", rows,
        {parts_view, weights_view,
         shared_view.slice(0u, shared_bytes - sizeof(float)),
         gate_view, output_view},
        "shared output");
    const bool short_gate = reject(
        label + " short gate logits", rows,
        {parts_view, weights_view, shared_view,
         gate_view.slice(0u, gate_bytes - sizeof(float)), output_view},
        "shared gate logits");
    const bool short_output = reject(
        label + " short output", rows,
        {parts_view, weights_view, shared_view, gate_view,
         output.view().slice(0u, output_bytes - sizeof(float))},
        "output");
    const bool misaligned_parts = reject(
        label + " misaligned parts", rows,
        {context.buffers().read_only(parts.view().slice(1u, parts_bytes)),
         weights_view, shared_view, gate_view, output_view},
        "misaligned");
    const bool misaligned_weights = reject(
        label + " misaligned weights", rows,
        {parts_view,
         context.buffers().read_only(weights.view().slice(1u, weights_bytes)),
         shared_view, gate_view, output_view},
        "misaligned");
    const bool misaligned_shared = reject(
        label + " misaligned shared output", rows,
        {parts_view, weights_view,
         context.buffers().read_only(shared.view().slice(1u, shared_bytes)),
         gate_view, output_view},
        "misaligned");
    const bool misaligned_gate = reject(
        label + " misaligned gate logits", rows,
        {parts_view, weights_view, shared_view,
         context.buffers().read_only(gate.view().slice(1u, gate_bytes)),
         output_view},
        "misaligned");
    const bool misaligned_output = reject(
        label + " misaligned output", rows,
        {parts_view, weights_view, shared_view, gate_view,
         output.view().slice(1u, output_bytes)},
        "misaligned");

    const ReadOnlyBufferRef output_as_input =
        context.buffers().read_only(output_view);
    const bool parts_alias = reject(
        label + " parts/output alias", rows,
        {context.buffers().read_only(
             parts.view().slice(0u, parts_bytes)),
         weights_view, shared_view, gate_view,
         parts.view().slice(0u, output_bytes)},
        "overlap");
    const bool weights_alias = reject(
        label + " weights/output alias", rows,
        {parts_view,
         context.buffers().read_only(
             output.view().slice(0u, weights_bytes)),
         shared_view, gate_view, output_view},
        "overlap");
    const bool shared_alias = reject(
        label + " shared/output alias", rows,
        {parts_view, weights_view, output_as_input, gate_view, output_view},
        "overlap");
    const bool gate_alias = reject(
        label + " gate/output alias", rows,
        {parts_view, weights_view, shared_view, output_as_input, output_view},
        "overlap");

    auto late_failure_command =
        context.make_command_buffer(label + " late failure", err);
    check(late_failure_command != nullptr,
          label + " late-failure command buffer", err);
    if (!late_failure_command) return false;
    const bool late_failure_rejected = !Encoder::encode(
        context, *late_failure_command, "MoE combine late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_moe_combine(context, encoder, rows, valid,
                                    encode_error)) {
                return false;
            }
            return encode_moe_combine(context, encoder, 0u, valid,
                                      encode_error);
        },
        err);
    const bool late_failure_discarded =
        late_failure_command->unsafe_native_handle() == nullptr;
    const bool late_failure_untouched = std::equal(
        output_before.begin(), output_before.end(), output.payload().begin(),
        output.payload().end());
    check(late_failure_rejected && late_failure_discarded &&
              late_failure_untouched &&
              err.find("1..8 rows") != std::string::npos,
          label + " caller-owned late failure rolls back its dispatch", err);

    bool concurrent_boundary_ok = true;
    const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        auto command =
            context.make_command_buffer(label + " open group", err);
        check(command != nullptr, label + " open-group command buffer", err);
        if (!command) return false;
        bool saw_open_group = false;
        bool rejected_open_group = false;
        const bool encoded = Encoder::encode(
            context, *command, "MoE combine open-group boundary",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                saw_open_group = encoder.concurrent_group_open();
                if (saw_open_group) {
                    std::string boundary_error;
                    rejected_open_group = !encode_moe_combine(
                        context, encoder, rows, valid, boundary_error) &&
                        boundary_error.find("ordered dispatch boundary") !=
                            std::string::npos;
                }
                if (!encoder.group_end(encode_error)) return false;
                if (!saw_open_group || !rejected_open_group) {
                    encode_error =
                        "MoE combine did not reject an open concurrent group";
                    return false;
                }
                return true;
            },
            err);
        command->discard();
        concurrent_boundary_ok =
            encoded && saw_open_group && rejected_open_group &&
            command->unsafe_native_handle() == nullptr;
        check(concurrent_boundary_ok,
              label + " rejects an open concurrent group", err);
    }

    return zero_rows && too_many_rows && short_parts && short_weights &&
           short_shared && short_gate && short_output && misaligned_parts &&
           misaligned_weights && misaligned_shared && misaligned_gate &&
           misaligned_output && parts_alias && weights_alias &&
           shared_alias && gate_alias && late_failure_rejected &&
           late_failure_discarded && late_failure_untouched &&
           concurrent_boundary_ok;
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

    const bool dispatched = dispatch_moe_combine(
        context, rows, parts_buffer, weights_buffer, shared_buffer,
        gate_buffer, output_buffer, err);
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

void moe_gr_post_reference(std::span<const float> residual,
                           std::span<const float> block_out,
                           std::span<const float> inject,
                           std::uint32_t rows,
                           std::vector<twin::tf>& output) {
    constexpr std::size_t kStreams = 4u;
    const std::size_t residual_width = kStreams * kMoeDownWidth;
    output.resize(static_cast<std::size_t>(rows) * residual_width);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::size_t index = 0u; index < residual_width; ++index) {
            const std::size_t stream = index / kMoeDownWidth;
            const std::size_t column = index % kMoeDownWidth;
            const twin::tf scaled_inject = twin::fma(
                twin::tf{0.25f},
                twin::tf{inject[static_cast<std::size_t>(row) *
                                    kStreams +
                                stream]},
                twin::tf{});
            const twin::tf sigmoid =
                twin::tf{1.0f} /
                (twin::tf{1.0f} + twin::accurate_exp(-scaled_inject));
            const twin::tf weight =
                twin::fma(twin::tf{2.0f}, sigmoid, twin::tf{});
            const std::size_t residual_index =
                static_cast<std::size_t>(row) * residual_width + index;
            const std::size_t block_index =
                static_cast<std::size_t>(row) * kMoeDownWidth + column;
            output[residual_index] = twin::fma(
                twin::tf{block_out[block_index]}, weight,
                twin::tf{residual[residual_index]});
        }
    }
}

bool test_moe_routed_shared_ffn_write(
    Context& context, std::uint32_t rows, const twin::Options& options,
    const MoeRoutedFFNBuffers& routed,
    std::span<const Q81> input_activations,
    std::span<const twin::tf> expected_parts,
    const std::array<twin::GuardedBuffer*, 5u>& routed_storage) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t layer = kMoeGroupedLayer;
    constexpr twin::InputSet set = twin::InputSet::fixture;
    const std::uint32_t nc =
        rows <= 1u ? 1u : (rows <= 2u ? 2u : (rows <= 4u ? 4u : 8u));
    const std::string label =
        "Routed/shared MoE FFN+GR write rows=" + std::to_string(rows);
    std::string error;

    std::vector<block_q8_0> shared_gate_weights;
    std::vector<block_q8_0> shared_up_weights;
    std::vector<block_q8_0> shared_down_weights;
    std::vector<std::uint16_t> shared_gate_input_weights;
    make_shared_expert_weights(
        set, options.seed ^ (static_cast<std::uint64_t>(rows) << 40u),
        shared_gate_weights, shared_up_weights, shared_down_weights,
        shared_gate_input_weights);

    const std::size_t mixed_bytes =
        static_cast<std::size_t>(rows) * kMoeInputWidth * sizeof(float);
    std::vector<float> mixed(mixed_bytes / sizeof(float));
    for (std::size_t row = 0u; row < rows; ++row) {
        for (std::size_t block = 0u; block < kMoeInputWidth / 32u; ++block) {
            const Q81& activation =
                input_activations[row * (kMoeInputWidth / 32u) + block];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(activation.ds & 0xffffu);
            const float scale =
                strata::kernels::f32_from_f16(scale_bits);
            for (std::size_t lane = 0u; lane < 32u; ++lane) {
                mixed[row * kMoeInputWidth + block * 32u + lane] =
                    scale * static_cast<float>(activation.qs[lane]);
            }
        }
    }

    constexpr std::array<float, kTopK> base_weights{
        0.04f, 0.06f, 0.08f, 0.10f, 0.12f,
        0.14f, 0.16f, 0.10f, 0.08f, 0.12f};
    std::vector<float> route_weights(
        static_cast<std::size_t>(rows) * kTopK);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        std::copy(base_weights.begin(), base_weights.end(),
                  route_weights.begin() + row * kTopK);
    }
    const std::uint32_t active_rows = rows;
    const std::size_t gate_bytes =
        static_cast<std::size_t>(rows) * kMoeExpertWidth * sizeof(float);
    const std::size_t routed_gate_bytes =
        static_cast<std::size_t>(rows) * kTopK * kMoeExpertWidth *
        sizeof(float);
    const std::size_t down_q8_bytes =
        static_cast<std::size_t>(rows) * kMoeDownInputWidth / 32u *
        sizeof(Q81);
    const std::size_t routed_down_q8_bytes =
        static_cast<std::size_t>(rows) * kTopK *
        (kMoeDownInputWidth / 32u) * sizeof(Q81);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(rows) * kMoeDownWidth * sizeof(float);
    const std::size_t gate_logits_bytes =
        static_cast<std::size_t>(rows) * sizeof(float);
    const std::size_t route_weights_bytes =
        route_weights.size() * sizeof(float);
    const std::size_t gate_weight_bytes =
        shared_gate_weights.size() * sizeof(block_q8_0);
    const std::size_t up_weight_bytes =
        shared_up_weights.size() * sizeof(block_q8_0);
    const std::size_t down_weight_bytes =
        shared_down_weights.size() * sizeof(block_q8_0);
    const std::size_t gate_input_weight_bytes =
        shared_gate_input_weights.size() * sizeof(std::uint16_t);
    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * 4u * kMoeDownWidth * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(rows) * 4u * sizeof(float);
    const std::size_t padded_rows = nc - rows;
    const std::size_t gate_sink_bytes =
        std::max(sizeof(float), padded_rows * kMoeExpertWidth * sizeof(float));
    const std::size_t down_sink_bytes =
        std::max(sizeof(float), padded_rows * kMoeDownWidth * sizeof(float));
    const std::size_t shared_gate_sink_bytes =
        std::max(sizeof(float), padded_rows * sizeof(float));

    std::vector<float> expected_gate;
    std::vector<float> expected_up;
    std::vector<float> expected_shared;
    std::vector<float> expected_gate_logits;
    const auto pad_rows = [rows, nc](const auto& active_rows,
                                    std::size_t elements_per_row) {
        using Value = typename std::remove_cvref_t<
            decltype(active_rows)>::value_type;
        std::vector<Value> padded(
            static_cast<std::size_t>(nc) * elements_per_row);
        for (std::uint32_t row = 0u; row < nc; ++row) {
            const std::uint32_t source_row = row < rows ? row : 0u;
            std::copy_n(
                active_rows.begin() +
                    static_cast<std::size_t>(source_row) * elements_per_row,
                elements_per_row,
                padded.begin() +
                    static_cast<std::size_t>(row) * elements_per_row);
        }
        return padded;
    };
    const auto padded_input_activations =
        pad_rows(input_activations, kMoeInputWidth / 32u);
    const auto padded_mixed = pad_rows(mixed, kMoeInputWidth);
    bool oracle_ready = twin::q8_0_matvec_reference(
        shared_gate_weights, padded_input_activations, kMoeInputWidth,
        kMoeExpertWidth, nc, rows, expected_gate, error);
    oracle_ready &= twin::q8_0_matvec_reference(
        shared_up_weights, padded_input_activations, kMoeInputWidth,
        kMoeExpertWidth, nc, rows, expected_up, error);
    const std::size_t active_projection_values =
        static_cast<std::size_t>(rows) * kMoeExpertWidth;
    std::vector<twin::tf> expected_hidden;
    oracle_ready &= twin::moe_swiglu_reference(
        std::span<const float>(expected_gate).first(active_projection_values),
        std::span<const float>(expected_up).first(active_projection_values),
        expected_hidden, error);
    std::vector<float> hidden_values(expected_hidden.size());
    std::transform(expected_hidden.begin(), expected_hidden.end(),
                   hidden_values.begin(),
                   [](const twin::tf& value) { return value.value; });
    std::vector<Q81> expected_hidden_q8;
    oracle_ready &= twin::quantize_q8_1_reference(
        hidden_values, kMoeExpertWidth, rows, expected_hidden_q8, error);
    const auto padded_hidden_q8 =
        pad_rows(expected_hidden_q8, kMoeDownInputWidth / 32u);
    oracle_ready &= twin::q8_0_matvec_reference(
        shared_down_weights, padded_hidden_q8, kMoeDownInputWidth,
        kMoeDownWidth, nc, rows, expected_shared, error);
    oracle_ready &= twin::bf16_matvec_reference(
        shared_gate_input_weights, padded_mixed, kMoeInputWidth, 1u, nc,
        rows, 256u, expected_gate_logits, error);
    check(oracle_ready, label + " shared scalar oracles", error);
    if (!oracle_ready) return false;

    std::vector<float> parts_values(expected_parts.size());
    std::transform(expected_parts.begin(), expected_parts.end(),
                   parts_values.begin(),
                   [](const twin::tf& value) { return value.value; });
    std::vector<twin::tf> expected_combined;
    const bool combine_ready = twin::moe_combine_reference(
        parts_values, route_weights,
        std::span<const float>(expected_shared).first(
            static_cast<std::size_t>(rows) * kMoeDownWidth),
        std::span<const float>(expected_gate_logits).first(rows), rows,
        expected_combined, error);
    check(combine_ready, label + " combine scalar oracle", error);
    if (!combine_ready) return false;

    std::mt19937_64 residual_rng(
        options.seed ^ rows ^ 0x4752504f5354ull);
    const std::vector<float> initial_residual = twin::synthesize_floats(
        set, residual_bytes / sizeof(float), residual_rng, error);
    if (!error.empty()) {
        check(false, label + " residual synthesis", error);
        return false;
    }
    std::mt19937_64 inject_rng(options.seed ^ (rows << 8u) ^ 0x494e4aull);
    const std::vector<float> inject = twin::synthesize_floats(
        set, inject_bytes / sizeof(float), inject_rng, error);
    if (!error.empty()) {
        check(false, label + " injection synthesis", error);
        return false;
    }

    enum BufferIndex : std::size_t {
        gate_weight_data,
        up_weight_data,
        down_weight_data,
        shared_gate_weight_data,
        mixed_data,
        active_rows_data,
        route_weight_data,
        gate_output,
        up_output,
        hidden_output,
        down_activations_output,
        shared_output,
        shared_gate_logits_output,
        combined_output,
        gate_sink,
        up_sink,
        down_sink,
        shared_gate_sink,
        residual,
        injection,
        buffer_count,
    };
    const std::array<std::size_t, buffer_count> sizes{
        gate_weight_bytes,
        up_weight_bytes,
        down_weight_bytes,
        gate_input_weight_bytes,
        mixed_bytes,
        sizeof(active_rows),
        route_weights_bytes,
        gate_bytes,
        gate_bytes,
        gate_bytes,
        down_q8_bytes,
        shared_bytes,
        gate_logits_bytes,
        shared_bytes,
        gate_sink_bytes,
        gate_sink_bytes,
        down_sink_bytes,
        shared_gate_sink_bytes,
        residual_bytes,
        inject_bytes,
    };
    const std::array<const char*, buffer_count> names{
        "shared gate weights", "shared up weights", "shared down weights",
        "shared gate input weights", "mixed activation", "active rows",
        "route weights", "shared gate output", "shared up output",
        "shared hidden output", "shared down Q8_1", "shared down output",
        "shared gate logits", "combined output", "gate sink", "up sink",
        "down sink", "shared gate sink", "GR residual", "GR injection",
    };
    std::array<twin::GuardedBuffer, buffer_count> storage;
    for (std::size_t index = 0u; index < storage.size(); ++index) {
        if (!storage[index].allocate(context, label + " " + names[index],
                                     sizes[index], error)) {
            check(false, label + " buffer allocation", error);
            return false;
        }
    }
    const auto copy = [&](BufferIndex index, const void* source,
                          std::size_t bytes) {
        if (source == nullptr || bytes > storage[index].payload().size()) {
            return false;
        }
        std::memcpy(storage[index].payload().data(), source, bytes);
        return true;
    };
    const bool inputs_ready =
        copy(gate_weight_data, shared_gate_weights.data(),
             gate_weight_bytes) &&
        copy(up_weight_data, shared_up_weights.data(), up_weight_bytes) &&
        copy(down_weight_data, shared_down_weights.data(),
             down_weight_bytes) &&
        copy(shared_gate_weight_data, shared_gate_input_weights.data(),
             gate_input_weight_bytes) &&
        copy(mixed_data, mixed.data(), mixed_bytes) &&
        copy(active_rows_data, &active_rows, sizeof(active_rows)) &&
        copy(route_weight_data, route_weights.data(), route_weights_bytes) &&
        copy(residual, initial_residual.data(), residual_bytes) &&
        copy(injection, inject.data(), inject_bytes);
    check(inputs_ready, label + " input initialization");
    if (!inputs_ready) return false;

    for (const BufferIndex index :
         {gate_output, up_output, hidden_output, down_activations_output,
          shared_output, shared_gate_logits_output, combined_output,
          gate_sink, up_sink, down_sink, shared_gate_sink}) {
        std::fill(storage[index].payload().begin(),
                  storage[index].payload().end(), 0xa5u);
    }
    for (auto& buffer : storage) buffer.capture_before_dispatch();

    const MoeSharedFFNBuffers shared_buffers{
        routed.input_activations,
        context.buffers().read_only(storage[mixed_data].view()),
        context.buffers().read_only(storage[gate_weight_data].view()),
        context.buffers().read_only(storage[up_weight_data].view()),
        context.buffers().read_only(storage[down_weight_data].view()),
        context.buffers().read_only(
            storage[shared_gate_weight_data].view()),
        context.buffers().read_only(storage[active_rows_data].view()),
        context.buffers().read_only(routed.parts),
        context.buffers().read_only(storage[route_weight_data].view()),
        storage[gate_output].view(),
        storage[up_output].view(),
        storage[hidden_output].view(),
        storage[down_activations_output].view(),
        storage[shared_output].view(),
        storage[shared_gate_logits_output].view(),
        storage[combined_output].view(),
        storage[gate_sink].view(),
        storage[up_sink].view(),
        storage[down_sink].view(),
        storage[shared_gate_sink].view(),
    };
    const MoeSharedFFNWriteBuffers shared_write{
        shared_buffers, storage[residual].view(),
        storage[injection].view()};
    const MoeRoutedSharedFFNWriteBuffers pipeline{routed, shared_write};

    auto command = context.make_command_buffer(label, error);
    const bool caller_owned = rows == 1u;
    const bool encoded = command && (caller_owned
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_moe_routed_shared_ffn_write(
                      context, encoder, rows, experts, layer,
                      MoeExpertFormat::q2_0, MoeDownFormat::q2_0, pipeline,
                      encode_error);
              },
              error)
        : encode_moe_routed_shared_ffn_write(
              context, *command, rows, experts, layer,
              MoeExpertFormat::q2_0, MoeDownFormat::q2_0, pipeline, error));
    const bool dispatched =
        encoded && command->commit(error) &&
        command->wait_until_completed(error);
    check(dispatched, label + (caller_owned ? " caller-owned"
                                            : " standalone"),
          error);
    if (!dispatched) return false;

    const twin::Tolerance tolerance{
        twin::Comparison::units, 16.0, 0.0, 0u};
    const auto compare = [&](BufferIndex index,
                             std::span<const twin::tf> expected,
                             std::string_view description) {
        const auto* actual = reinterpret_cast<const float*>(
            storage[index].payload().data());
        const auto failure = twin::compare_floats(
            std::span<const float>(actual, expected.size()), expected,
            tolerance);
        if (failure) {
            twin::report_failure(label.c_str(), options, rows, *failure);
        }
        check(!failure, label + " " + std::string(description) + " U(16)");
        return !failure;
    };
    const auto to_tf = [](std::span<const float> values) {
        std::vector<twin::tf> result(values.size());
        std::transform(values.begin(), values.end(), result.begin(),
                       [](float value) { return twin::tf{value}; });
        return result;
    };
    const auto expected_gate_active =
        to_tf(std::span<const float>(expected_gate).first(
            active_projection_values));
    const auto expected_up_active =
        to_tf(std::span<const float>(expected_up).first(
            active_projection_values));
    const auto expected_shared_active =
        to_tf(std::span<const float>(expected_shared).first(
            static_cast<std::size_t>(rows) * kMoeDownWidth));
    const auto expected_gate_logits_active =
        to_tf(std::span<const float>(expected_gate_logits).first(rows));
    bool outputs_match =
        compare(gate_output, expected_gate_active, "shared gate") &&
        compare(up_output, expected_up_active, "shared up");
    outputs_match &=
        compare(hidden_output, expected_hidden, "shared SwiGLU");
    outputs_match &=
        compare(shared_output, expected_shared_active, "shared down");
    outputs_match &= compare(shared_gate_logits_output,
                             expected_gate_logits_active,
                             "shared gate logits");
    outputs_match &=
        compare(combined_output, expected_combined, "routed/shared combine");
    const auto* actual_parts =
        reinterpret_cast<const float*>(routed.parts.contents);
    const auto parts_failure = twin::compare_floats(
        std::span<const float>(actual_parts, expected_parts.size()),
        expected_parts, tolerance);
    check(!parts_failure, label + " routed producer/consumer handoff U(16)");
    outputs_match &= !parts_failure;

    const auto* actual_combined =
        reinterpret_cast<const float*>(
            storage[combined_output].payload().data());
    std::vector<twin::tf> expected_post;
    moe_gr_post_reference(
        initial_residual,
        std::span<const float>(actual_combined,
                               shared_bytes / sizeof(float)),
        inject, rows, expected_post);
    const auto* actual_residual =
        reinterpret_cast<const float*>(storage[residual].payload().data());
    const auto residual_failure = twin::compare_floats(
        std::span<const float>(actual_residual, expected_post.size()),
        expected_post, tolerance);
    check(!residual_failure, label + " GR residual write U(16)");
    outputs_match &= !residual_failure;

    const auto compare_sink = [&](BufferIndex index,
                                  std::span<const float> full_expected,
                                  std::size_t active_values,
                                  std::size_t write_bytes,
                                  std::string_view description) {
        const std::size_t padded_values = write_bytes / sizeof(float);
        if (padded_values == 0u) return true;
        const auto expected = to_tf(
            full_expected.subspan(active_values, padded_values));
        const auto* actual = reinterpret_cast<const float*>(
            storage[index].payload().data());
        const auto failure = twin::compare_floats(
            std::span<const float>(actual, padded_values), expected,
            tolerance);
        check(!failure,
              label + " " + std::string(description) + " padded values U(16)");
        return !failure;
    };
    const std::size_t gate_sink_write_bytes =
        padded_rows * kMoeExpertWidth * sizeof(float);
    const std::size_t down_sink_write_bytes =
        padded_rows * kMoeDownWidth * sizeof(float);
    const std::size_t shared_gate_sink_write_bytes =
        padded_rows * sizeof(float);
    outputs_match &= compare_sink(
        gate_sink, expected_gate, active_projection_values,
        gate_sink_write_bytes, "gate sink");
    outputs_match &= compare_sink(
        up_sink, expected_up, active_projection_values,
        gate_sink_write_bytes, "up sink");
    outputs_match &= compare_sink(
        down_sink, expected_shared,
        static_cast<std::size_t>(rows) * kMoeDownWidth,
        down_sink_write_bytes, "down sink");
    outputs_match &= compare_sink(
        shared_gate_sink, expected_gate_logits, rows,
        shared_gate_sink_write_bytes, "shared gate sink");

    const std::array<twin::ByteRange, 1> routed_parts_written{
        {{0u, expected_parts.size_bytes()}}};
    const std::array<twin::ByteRange, 1> gate_written{
        {{0u, routed_gate_bytes}}};
    const std::array<twin::ByteRange, 1> down_q8_written{
        {{0u, routed_down_q8_bytes}}};
    const std::array<twin::ByteRange, 1> shared_written{
        {{0u, shared_bytes}}};
    const std::array<twin::ByteRange, 1> gate_logits_written{
        {{0u, gate_logits_bytes}}};
    const std::array<twin::ByteRange, 1> residual_written{
        {{0u, residual_bytes}}};
    const std::array routed_written{
        gate_written, gate_written, gate_written, down_q8_written,
        routed_parts_written};
    for (std::size_t index = 0u; index < routed_storage.size(); ++index) {
        const bool unchanged =
            !routed_storage[index]->first_unwritten_change(
                routed_written[index]);
        check(unchanged,
              label + " routed stage " + std::to_string(index) +
                  " guards and write extent unchanged");
        outputs_match &= unchanged;
    }
    for (const BufferIndex index :
         {gate_weight_data, up_weight_data, down_weight_data,
          shared_gate_weight_data, mixed_data, active_rows_data,
          route_weight_data, injection}) {
        const bool unchanged = !storage[index].first_unwritten_change({});
        check(unchanged,
              label + " " + names[index] + " and guards unchanged");
        outputs_match &= unchanged;
    }
    const std::array<std::pair<BufferIndex,
                               std::span<const twin::ByteRange>>,
                     8u>
        shared_written_ranges{{
            {gate_output, gate_written},
            {up_output, gate_written},
            {hidden_output, gate_written},
            {down_activations_output, down_q8_written},
            {shared_output, shared_written},
            {shared_gate_logits_output, gate_logits_written},
            {combined_output, shared_written},
            {residual, residual_written},
        }};
    for (const auto& [index, written] : shared_written_ranges) {
        const bool unchanged = !storage[index].first_unwritten_change(written);
        check(unchanged,
              label + " " + names[index] +
                  " guard and write extent unchanged");
        outputs_match &= unchanged;
    }
    for (const BufferIndex index :
         {gate_sink, up_sink, down_sink, shared_gate_sink}) {
        std::size_t written_bytes = 0u;
        switch (index) {
            case gate_sink:
            case up_sink: written_bytes = gate_sink_write_bytes; break;
            case down_sink: written_bytes = down_sink_write_bytes; break;
            case shared_gate_sink:
                written_bytes = shared_gate_sink_write_bytes;
                break;
            default: break;
        }
        const std::array<twin::ByteRange, 1> written{{{0u, written_bytes}}};
        const bool unchanged =
            !storage[index].first_unwritten_change(written);
        check(unchanged,
              label + " " + names[index] +
                  " write extent and guards unchanged");
        outputs_match &= unchanged;
    }

    std::memcpy(storage[residual].payload().data(), initial_residual.data(),
                residual_bytes);
    std::memcpy(storage[injection].payload().data(), inject.data(),
                inject_bytes);
    for (const BufferIndex index :
         {gate_output, up_output, hidden_output, down_activations_output,
          shared_output, shared_gate_logits_output, combined_output,
          gate_sink, up_sink, down_sink, shared_gate_sink}) {
        std::fill(storage[index].payload().begin(),
                  storage[index].payload().end(), 0xa5u);
    }
    for (auto& buffer : storage) buffer.capture_before_dispatch();

    const MoeRoutedSharedFFNBuffers pending_pipeline{routed, shared_buffers};
    auto pending_command =
        context.make_command_buffer(label + " pending GR write", error);
    const bool pending_caller_owned = rows == 1u;
    const bool pending_encoded =
        pending_command &&
        (pending_caller_owned
             ? Encoder::encode(
                   context, *pending_command, label + " pending GR write",
                   [&](Encoder& encoder, std::string& encode_error) {
                       return encode_moe_routed_shared_ffn(
                           context, encoder, rows, experts, layer,
                           MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                           pending_pipeline, encode_error);
                   },
                   error)
             : encode_moe_routed_shared_ffn(
                   context, *pending_command, rows, experts, layer,
                   MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                   pending_pipeline, error));
    const bool pending_dispatched =
        pending_encoded && pending_command->commit(error) &&
        pending_command->wait_until_completed(error);
    check(pending_dispatched,
          label + (pending_caller_owned
                       ? " pending GR write caller-owned"
                       : " pending GR write standalone"),
          error);
    if (!pending_dispatched) return false;

    outputs_match &=
        compare(combined_output, expected_combined,
                "pending routed/shared combine");
    const bool residual_pending =
        std::memcmp(storage[residual].payload().data(),
                    initial_residual.data(), residual_bytes) == 0;
    const bool injection_pending =
        std::memcmp(storage[injection].payload().data(), inject.data(),
                    inject_bytes) == 0;
    const bool pending_residual_guard =
        !storage[residual].first_unwritten_change({});
    const bool pending_injection_guard =
        !storage[injection].first_unwritten_change({});
    const bool pending_output_guard =
        !storage[combined_output].first_unwritten_change(shared_written);
    check(residual_pending && injection_pending && pending_residual_guard &&
              pending_injection_guard && pending_output_guard,
          label + " combined output stays pending without changing GR residual or injection");
    outputs_match &= residual_pending && injection_pending &&
                     pending_residual_guard && pending_injection_guard &&
                     pending_output_guard;

    storage[residual].capture_before_dispatch();
    storage[injection].capture_before_dispatch();
    storage[combined_output].capture_before_dispatch();
    const GRPostBuffers pending_write_buffers{
        storage[residual].view(), storage[combined_output].view(),
        storage[injection].view(), storage[residual].view()};
    auto pending_write_command =
        context.make_command_buffer(label + " consume pending GR write", error);
    const bool pending_write_encoded =
        pending_write_command &&
        encode_gr_post(context, *pending_write_command, rows,
                       pending_write_buffers, error);
    const bool pending_write_dispatched =
        pending_write_encoded && pending_write_command->commit(error) &&
        pending_write_command->wait_until_completed(error);
    check(pending_write_dispatched,
          label + " consume pending combined output through GR post", error);
    if (!pending_write_dispatched) return false;
    const auto* pending_residual =
        reinterpret_cast<const float*>(
            storage[residual].payload().data());
    const auto pending_write_failure = twin::compare_floats(
        std::span<const float>(pending_residual,
                               expected_post.size()),
        expected_post, tolerance);
    check(!pending_write_failure,
          label + " pending combined output is applied once at GR post U(16)");
    outputs_match &= !pending_write_failure;
    const bool pending_write_guards =
        !storage[residual].first_unwritten_change(residual_written) &&
        !storage[injection].first_unwritten_change({}) &&
        !storage[combined_output].first_unwritten_change({});
    check(pending_write_guards,
          label + " GR post writes only the residual and preserves pending inputs");
    outputs_match &= pending_write_guards;

    std::vector<std::vector<std::uint8_t>> route_before;
    route_before.reserve(routed_storage.size());
    for (const twin::GuardedBuffer* buffer : routed_storage) {
        route_before.emplace_back(buffer->payload().begin(),
                                  buffer->payload().end());
    }
    std::array<std::vector<std::uint8_t>, buffer_count> shared_before{};
    for (std::size_t index = 0u; index < storage.size(); ++index) {
        shared_before[index].assign(storage[index].payload().begin(),
                                    storage[index].payload().end());
    }
    const auto unchanged = [&] {
        for (std::size_t index = 0u; index < routed_storage.size(); ++index) {
            if (!std::equal(route_before[index].begin(),
                            route_before[index].end(),
                            routed_storage[index]->payload().begin(),
                            routed_storage[index]->payload().end())) {
                return false;
            }
        }
        for (std::size_t index = 0u; index < storage.size(); ++index) {
            if (!std::equal(shared_before[index].begin(),
                            shared_before[index].end(),
                            storage[index].payload().begin(),
                            storage[index].payload().end())) {
                return false;
            }
        }
        return true;
    };
    const auto reject = [&](std::string_view case_name,
                            const MoeRoutedSharedFFNWriteBuffers& bad,
                            std::string_view expected_error) {
        auto reject_command = context.make_command_buffer(
            label + " " + std::string(case_name), error);
        const bool rejected = reject_command &&
            !encode_moe_routed_shared_ffn_write(
                context, *reject_command, rows, experts, layer,
                MoeExpertFormat::q2_0, MoeDownFormat::q2_0, bad, error);
        const bool discarded = reject_command &&
            reject_command->unsafe_native_handle() == nullptr;
        const bool error_matches =
            error.find(expected_error) != std::string::npos;
        check(rejected && discarded && error_matches && unchanged(),
              label + " rejects " + std::string(case_name), error);
        return rejected && discarded && error_matches && unchanged();
    };
    bool validation_ok = true;
    MoeRoutedSharedFFNWriteBuffers output_alias = pipeline;
    output_alias.shared.ffn.gate = routed.gate;
    validation_ok &= reject("cross-stage output alias", output_alias,
                            "stage outputs must be disjoint");
    MoeRoutedSharedFFNWriteBuffers input_alias = pipeline;
    input_alias.shared.ffn.mixed =
        context.buffers().read_only(routed.hidden);
    validation_ok &= reject("cross-stage output/input alias", input_alias,
                            "cross-stage output/input");
    MoeRoutedSharedFFNWriteBuffers handoff_alias = pipeline;
    handoff_alias.shared.ffn.routed_parts =
        context.buffers().read_only(storage[shared_output].view());
    validation_ok &= reject("wrong routed-parts handoff", handoff_alias,
                            "parts view must use the routed producer");

    if (rows == 1u) {
        auto overlap_storage = context.buffers().allocate(
            label + " residual tail overlap storage",
            routed_gate_bytes + residual_bytes, error);
        const bool overlap_storage_ready = static_cast<bool>(overlap_storage);
        check(overlap_storage_ready,
              label + " residual tail overlap storage allocation", error);
        validation_ok &= overlap_storage_ready;
        if (overlap_storage_ready) {
            const BufferRef oversized_gate =
                overlap_storage.slice(0u, routed_gate_bytes + residual_bytes);
            MoeRoutedSharedFFNWriteBuffers active_alias = pipeline;
            active_alias.routed.gate = oversized_gate;
            active_alias.shared.residual =
                overlap_storage.slice(0u, residual_bytes);
            const auto overlap_before = std::vector<std::uint8_t>(
                static_cast<const std::uint8_t*>(overlap_storage.contents),
                static_cast<const std::uint8_t*>(overlap_storage.contents) +
                    overlap_storage.length);
            auto active_alias_command = context.make_command_buffer(
                label + " active residual alias", error);
            const bool active_alias_rejected =
                active_alias_command &&
                !encode_moe_routed_shared_ffn_write(
                    context, *active_alias_command, rows, experts, layer,
                    MoeExpertFormat::q2_0, MoeDownFormat::q2_0, active_alias,
                    error);
            const bool active_alias_discarded =
                active_alias_command &&
                active_alias_command->unsafe_native_handle() == nullptr;
            const bool active_alias_unchanged =
                std::equal(overlap_before.begin(), overlap_before.end(),
                           static_cast<const std::uint8_t*>(
                               overlap_storage.contents),
                           static_cast<const std::uint8_t*>(
                               overlap_storage.contents) +
                               overlap_storage.length);
            const bool active_alias_error =
                error.find("residual writes must be disjoint") !=
                std::string::npos;
            check(active_alias_rejected && active_alias_discarded &&
                      active_alias_unchanged && active_alias_error,
                  label + " rejects overlapping active routed/residual ranges",
                  error);
            validation_ok &= active_alias_rejected &&
                             active_alias_discarded &&
                             active_alias_unchanged && active_alias_error;

            MoeRoutedSharedFFNWriteBuffers tail_alias = pipeline;
            tail_alias.routed.gate = oversized_gate;
            tail_alias.shared.residual =
                overlap_storage.slice(routed_gate_bytes, residual_bytes);
            auto tail_alias_command = context.make_command_buffer(
                label + " disjoint active residual tail alias", error);
            bool tail_alias_preflight_accepted = false;
            const bool tail_alias_discarded =
                tail_alias_command &&
                !Encoder::encode(
                    context, *tail_alias_command,
                    label + " disjoint active residual tail alias",
                    [&](Encoder& encoder, std::string& encode_error) {
                        if (!encode_moe_routed_shared_ffn_write(
                                context, encoder, rows, experts, layer,
                                MoeExpertFormat::q2_0,
                                MoeDownFormat::q2_0, tail_alias,
                                encode_error)) {
                            return false;
                        }
                        tail_alias_preflight_accepted = true;
                        return false;
                    },
                    error) &&
                tail_alias_command->unsafe_native_handle() == nullptr;
            check(tail_alias_preflight_accepted && tail_alias_discarded &&
                      unchanged(),
                  label + " accepts disjoint live ranges with overlapping capacity",
                  error);
            validation_ok &= tail_alias_preflight_accepted &&
                             tail_alias_discarded && unchanged();
        }
    }

    std::array<std::vector<std::uint8_t>, buffer_count> rollback_before{};
    for (std::size_t index = 0u; index < storage.size(); ++index) {
        rollback_before[index].assign(storage[index].payload().begin(),
                                      storage[index].payload().end());
    }
    auto rollback_command =
        context.make_command_buffer(label + " late caller failure", error);
    const bool rollback = rollback_command && !Encoder::encode(
        context, *rollback_command, label + " late caller failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_moe_routed_shared_ffn_write(
                    context, encoder, rows, experts, layer,
                    MoeExpertFormat::q2_0, MoeDownFormat::q2_0, pipeline,
                    encode_error)) {
                return false;
            }
            return encode_moe_routed_shared_ffn_write(
                context, encoder, 0u, experts, layer,
                MoeExpertFormat::q2_0, MoeDownFormat::q2_0, pipeline,
                encode_error);
        },
        error);
    const bool rollback_discarded = rollback_command &&
        rollback_command->unsafe_native_handle() == nullptr;
    bool rollback_routed_unchanged = true;
    for (std::size_t index = 0u; index < routed_storage.size(); ++index) {
        rollback_routed_unchanged &=
            std::equal(route_before[index].begin(), route_before[index].end(),
                       routed_storage[index]->payload().begin(),
                       routed_storage[index]->payload().end());
    }
    bool rollback_unchanged = true;
    for (std::size_t index = 0u; index < storage.size(); ++index) {
        rollback_unchanged &=
            std::equal(rollback_before[index].begin(),
                       rollback_before[index].end(),
                       storage[index].payload().begin(),
                       storage[index].payload().end());
    }
    check(rollback && rollback_discarded && rollback_routed_unchanged &&
              rollback_unchanged &&
              error.find("1..8 rows") != std::string::npos,
          label + " caller-owned late failure discards the whole sequence",
          error);
    validation_ok &= rollback && rollback_discarded &&
                     rollback_routed_unchanged && rollback_unchanged;

    const char* const dispatch_mode =
        std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent") {
        auto group_command =
            context.make_command_buffer(label + " concurrent group", error);
        const bool group_rejected = group_command && !Encoder::encode(
            context, *group_command, label + " concurrent group",
            [&](Encoder& encoder, std::string& encode_error) {
                return encoder.group_begin(encode_error) &&
                       encode_moe_routed_shared_ffn_write(
                           context, encoder, rows, experts, layer,
                           MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                           pipeline, encode_error);
            },
            error);
        const bool group_discarded = group_command &&
            group_command->unsafe_native_handle() == nullptr;
        const bool group_unchanged = unchanged();
        check(group_rejected && group_discarded && group_unchanged &&
                  error.find("ordered dispatch boundaries") !=
                      std::string::npos,
              label + " rejects an open concurrent group", error);
        validation_ok &= group_rejected && group_discarded &&
                         group_unchanged;
    }
    return outputs_match && validation_ok;
}

bool test_composed_shared_ffn(
    Context& context, std::uint32_t rows, std::uint32_t nc,
    twin::InputSet set, const twin::Options& options,
    std::span<const float> mixed,
    std::span<const Q81> mixed_q8_1,
    std::span<const block_q8_0> gate_weights,
    std::span<const block_q8_0> up_weights,
    std::span<const block_q8_0> down_weights,
    std::span<const std::uint16_t> gate_input_weights,
    std::span<const float> routed_parts,
    std::span<const float> route_weights,
    const twin::MoeSharedExpertOutput& expected_shared) {
    enum BufferIndex : std::size_t {
        input_q8_1,
        mixed_f32,
        gate_weight_data,
        up_weight_data,
        down_weight_data,
        shared_gate_weight_data,
        active_row_data,
        routed_part_data,
        route_weight_data,
        gate_output,
        up_output,
        hidden_output,
        down_q8_1_output,
        shared_output,
        shared_gate_logits,
        combined_output,
        gate_sink,
        up_sink,
        down_sink,
        shared_gate_sink,
        residual_input,
        injection_input,
        buffer_count,
    };
    const std::string label =
        "Composed shared MoE FFN rows=" + std::to_string(rows) +
        " nc=" + std::to_string(nc) +
        " set=" + twin::input_set_name(set);
    const std::size_t active_input_q8_bytes =
        static_cast<std::size_t>(rows) *
        kMoeInputWidth / 32u * sizeof(Q81);
    const std::size_t mixed_bytes =
        static_cast<std::size_t>(rows) * kMoeInputWidth * sizeof(float);
    const std::size_t gate_weights_bytes = gate_weights.size_bytes();
    const std::size_t up_weights_bytes = up_weights.size_bytes();
    const std::size_t down_weights_bytes = down_weights.size_bytes();
    const std::size_t shared_gate_weights_bytes =
        gate_input_weights.size_bytes();
    const std::size_t routed_parts_bytes = routed_parts.size_bytes();
    const std::size_t route_weights_bytes = route_weights.size_bytes();
    const std::size_t gate_bytes =
        static_cast<std::size_t>(rows) * kMoeExpertWidth * sizeof(float);
    const std::size_t up_bytes = gate_bytes;
    const std::size_t hidden_bytes = gate_bytes;
    const std::size_t down_q8_1_bytes =
        static_cast<std::size_t>(rows) * 20u * sizeof(Q81);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(rows) * kMoeDownWidth * sizeof(float);
    const std::size_t gate_logits_bytes =
        static_cast<std::size_t>(rows) * sizeof(float);
    const std::size_t combined_bytes = shared_bytes;
    const std::size_t gate_sink_write_bytes =
        static_cast<std::size_t>(nc - rows) * kMoeExpertWidth *
        sizeof(float);
    const std::size_t down_sink_write_bytes =
        static_cast<std::size_t>(nc - rows) * kMoeDownWidth *
        sizeof(float);
    const std::size_t shared_gate_sink_write_bytes =
        static_cast<std::size_t>(nc - rows) * sizeof(float);
    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * 4u * kMoeDownWidth *
        sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * 4u * sizeof(float);
    constexpr std::size_t kSinkExtraBytes = 64u;
    const std::array<std::size_t, buffer_count> sizes{
        active_input_q8_bytes,
        mixed_bytes,
        gate_weights_bytes,
        up_weights_bytes,
        down_weights_bytes,
        shared_gate_weights_bytes,
        sizeof(rows),
        routed_parts_bytes,
        route_weights_bytes,
        gate_bytes,
        up_bytes,
        hidden_bytes,
        down_q8_1_bytes,
        shared_bytes,
        gate_logits_bytes,
        combined_bytes,
        std::max(sizeof(float), gate_sink_write_bytes) + kSinkExtraBytes,
        std::max(sizeof(float), gate_sink_write_bytes) + kSinkExtraBytes,
        std::max(sizeof(float), down_sink_write_bytes) + kSinkExtraBytes,
        std::max(sizeof(float), shared_gate_sink_write_bytes) +
            kSinkExtraBytes,
        residual_bytes,
        injection_bytes,
    };
    const std::array<const char*, buffer_count> names{
        "Q8_1 input", "mixed input", "gate weights", "up weights",
        "down weights", "shared gate weights", "active rows",
        "routed parts", "route weights", "gate output", "up output",
        "hidden output", "down Q8_1 output", "shared output",
        "shared gate logits", "combined output", "gate sink", "up sink",
        "down sink", "shared gate sink",
        "GR residual", "GR injection",
    };
    std::array<twin::GuardedBuffer, buffer_count> storage;
    std::string error;
    for (std::size_t i = 0u; i < storage.size(); ++i) {
        if (!storage[i].allocate(context, label + " " + names[i], sizes[i],
                                 error)) {
            check(false, label + " buffer allocation", error);
            return false;
        }
    }
    const auto copy_bytes = [&](BufferIndex index, const void* source,
                                std::size_t bytes) {
        if (source == nullptr || bytes > storage[index].payload().size()) {
            return false;
        }
        std::memcpy(storage[index].payload().data(), source, bytes);
        return true;
    };
    const std::uint32_t live_rows = rows;
    const bool inputs_copied =
        copy_bytes(input_q8_1, mixed_q8_1.data(), active_input_q8_bytes) &&
        copy_bytes(mixed_f32, mixed.data(), mixed_bytes) &&
        copy_bytes(gate_weight_data, gate_weights.data(), gate_weights_bytes) &&
        copy_bytes(up_weight_data, up_weights.data(), up_weights_bytes) &&
        copy_bytes(down_weight_data, down_weights.data(), down_weights_bytes) &&
        copy_bytes(shared_gate_weight_data, gate_input_weights.data(),
                   shared_gate_weights_bytes) &&
        copy_bytes(active_row_data, &live_rows, sizeof(live_rows)) &&
        copy_bytes(routed_part_data, routed_parts.data(), routed_parts_bytes) &&
        copy_bytes(route_weight_data, route_weights.data(),
                   route_weights_bytes);
    check(inputs_copied, label + " input initialization");
    if (!inputs_copied) return false;
    std::mt19937_64 gr_rng(
        options.seed ^ (static_cast<std::uint64_t>(rows) << 32u) ^
        (static_cast<std::uint64_t>(set) << 48u) ^ 0x4752504f5354ull);
    const std::vector<float> initial_residual =
        twin::synthesize_floats(set, residual_bytes / sizeof(float), gr_rng,
                                error);
    if (!error.empty()) {
        check(false, label + " GR residual synthesis", error);
        return false;
    }
    const std::vector<float> inject =
        twin::synthesize_floats(set, injection_bytes / sizeof(float), gr_rng,
                                error);
    if (!error.empty()) {
        check(false, label + " GR injection synthesis", error);
        return false;
    }
    const bool write_inputs_copied =
        copy_bytes(residual_input, initial_residual.data(), residual_bytes) &&
        copy_bytes(injection_input, inject.data(), injection_bytes);
    check(write_inputs_copied, label + " GR write input initialization");
    if (!write_inputs_copied) return false;

    const auto to_tf = [](std::span<const float> values) {
        std::vector<twin::tf> result(values.size());
        std::transform(values.begin(), values.end(), result.begin(),
                       [](float value) { return twin::tf{value}; });
        return result;
    };
    std::vector<twin::tf> expected_combined;
    const bool combine_oracle_ready = twin::moe_combine_reference(
        routed_parts, route_weights, expected_shared.shared,
        expected_shared.gate_logits, rows, expected_combined, error);
    check(combine_oracle_ready, label + " scalar combine oracle", error);
    if (!combine_oracle_ready) return false;
    const auto expected_gate = to_tf(expected_shared.gate);
    const auto expected_up = to_tf(expected_shared.up);
    const auto expected_hidden = to_tf(expected_shared.hidden);
    const auto expected_shared_output = to_tf(expected_shared.shared);
    const auto expected_gate_logits = to_tf(expected_shared.gate_logits);
    check(storage[gate_output].poison_floats_against(
              expected_gate, twin::Comparison::units, error),
          label + " gate poison", error);
    check(storage[up_output].poison_floats_against(
              expected_up, twin::Comparison::units, error),
          label + " up poison", error);
    check(storage[hidden_output].poison_floats_against(
              expected_hidden, twin::Comparison::units, error),
          label + " hidden poison", error);
    const auto expected_hidden_q8 =
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(
                expected_shared.hidden_q8_1.data()),
            down_q8_1_bytes);
    check(storage[down_q8_1_output].poison_bytes_against(
              expected_hidden_q8, error),
          label + " down Q8_1 poison", error);
    check(storage[shared_output].poison_floats_against(
              expected_shared_output, twin::Comparison::units, error),
          label + " shared output poison", error);
    check(storage[shared_gate_logits].poison_floats_against(
              expected_gate_logits, twin::Comparison::units, error),
          label + " gate-logit poison", error);
    check(storage[combined_output].poison_floats_against(
              expected_combined, twin::Comparison::units, error),
          label + " combined-output poison", error);
    for (const BufferIndex sink :
         {gate_sink, up_sink, down_sink, shared_gate_sink}) {
        std::fill(storage[sink].payload().begin(),
                  storage[sink].payload().end(), 0xa5u);
    }
    for (auto& buffer : storage) buffer.capture_before_dispatch();

    const MoeSharedFFNBuffers buffers{
        context.buffers().read_only(storage[input_q8_1].view()),
        context.buffers().read_only(storage[mixed_f32].view()),
        context.buffers().read_only(storage[gate_weight_data].view()),
        context.buffers().read_only(storage[up_weight_data].view()),
        context.buffers().read_only(storage[down_weight_data].view()),
        context.buffers().read_only(storage[shared_gate_weight_data].view()),
        context.buffers().read_only(storage[active_row_data].view()),
        context.buffers().read_only(storage[routed_part_data].view()),
        context.buffers().read_only(storage[route_weight_data].view()),
        storage[gate_output].view(),
        storage[up_output].view(),
        storage[hidden_output].view(),
        storage[down_q8_1_output].view(),
        storage[shared_output].view(),
        storage[shared_gate_logits].view(),
        storage[combined_output].view(),
        storage[gate_sink].view(),
        storage[up_sink].view(),
        storage[down_sink].view(),
        storage[shared_gate_sink].view(),
    };
    const MoeSharedFFNWriteBuffers write_buffers{
        buffers, storage[residual_input].view(),
        storage[injection_input].view()};

    bool validation_ok = true;
    if (rows == 1u && set == twin::InputSet::fixture) {
        std::array<std::vector<std::uint8_t>, buffer_count> before{};
        for (std::size_t i = 0u; i < storage.size(); ++i) {
            before[i].assign(storage[i].payload().begin(),
                             storage[i].payload().end());
        }
        const auto reject = [&](const std::string& case_label,
                                std::uint32_t case_rows,
                                const MoeSharedFFNBuffers& case_buffers,
                                std::string_view expected_error) {
            auto command = context.make_command_buffer(case_label, error);
            const bool rejected = command && !encode_moe_shared_ffn(
                context, *command, case_rows, case_buffers, error);
            const bool discarded =
                command && command->unsafe_native_handle() == nullptr;
            bool unchanged = true;
            for (std::size_t i = 0u; i < storage.size(); ++i) {
                unchanged &=
                    std::equal(before[i].begin(), before[i].end(),
                               storage[i].payload().begin(),
                               storage[i].payload().end());
            }
            const bool error_matches =
                error.find(expected_error) != std::string::npos;
            check(rejected && discarded && unchanged && error_matches,
                  case_label + " rejects before dispatch", error);
            validation_ok &= rejected && discarded && unchanged &&
                             error_matches;
        };
        reject(label + " zero rows", 0u, buffers, "1..8 rows");
        MoeSharedFFNBuffers aliased = buffers;
        aliased.gate = storage[routed_part_data].view();
        reject(label + " cross-stage input alias", rows, aliased,
               "overlap inputs");
        const std::uint32_t invalid_live_rows = rows + 1u;
        std::memcpy(storage[active_row_data].payload().data(),
                    &invalid_live_rows, sizeof(invalid_live_rows));
        before[active_row_data].assign(
            storage[active_row_data].payload().begin(),
            storage[active_row_data].payload().end());
        reject(label + " active-row mismatch", rows, buffers,
               "active-row count");
        std::memcpy(storage[active_row_data].payload().data(), &live_rows,
                    sizeof(live_rows));
        before[active_row_data].assign(
            storage[active_row_data].payload().begin(),
            storage[active_row_data].payload().end());

        std::array<std::vector<std::uint8_t>, buffer_count> rollback_before{};
        for (std::size_t i = 0u; i < storage.size(); ++i) {
            rollback_before[i].assign(storage[i].payload().begin(),
                                      storage[i].payload().end());
        }
        auto rollback_command =
            context.make_command_buffer(label + " caller-owned rollback",
                                        error);
        const bool rollback_command_ready = rollback_command != nullptr;
        check(rollback_command_ready,
              label + " caller-owned rollback command buffer", error);
        if (rollback_command_ready) {
            const bool rejected = !Encoder::encode(
                context, *rollback_command, label + " caller-owned rollback",
                [&](Encoder& encoder, std::string& encode_error) {
                    if (!encode_moe_shared_ffn(context, encoder, rows, buffers,
                                               encode_error)) {
                        return false;
                    }
                    return encode_moe_shared_ffn(context, encoder, 0u, buffers,
                                                 encode_error);
                },
                error);
            const bool discarded =
                rollback_command->unsafe_native_handle() == nullptr;
            bool unchanged = true;
            for (std::size_t i = 0u; i < storage.size(); ++i) {
                unchanged &=
                    std::equal(rollback_before[i].begin(),
                               rollback_before[i].end(),
                               storage[i].payload().begin(),
                               storage[i].payload().end());
            }
            check(rejected && discarded && unchanged &&
                      error.find("1..8 rows") != std::string::npos,
                  label + " caller-owned late failure rolls back all stages",
                  error);
            validation_ok &= rejected && discarded && unchanged &&
                             error.find("1..8 rows") != std::string::npos;
        } else {
            validation_ok = false;
        }

        const auto reject_write = [&](const std::string& case_label,
                                      const MoeSharedFFNWriteBuffers&
                                          case_buffers,
                                      std::string_view expected_error) {
            auto write_command =
                context.make_command_buffer(case_label, error);
            const bool rejected = write_command &&
                !encode_moe_shared_ffn_write(
                    context, *write_command, rows, case_buffers, error);
            const bool discarded =
                write_command &&
                write_command->unsafe_native_handle() == nullptr;
            bool unchanged = true;
            for (std::size_t i = 0u; i < storage.size(); ++i) {
                unchanged &=
                    std::equal(before[i].begin(), before[i].end(),
                               storage[i].payload().begin(),
                               storage[i].payload().end());
            }
            const bool error_matches =
                error.find(expected_error) != std::string::npos;
            check(rejected && discarded && unchanged && error_matches,
                  case_label + " rejects before dispatch", error);
            validation_ok &= rejected && discarded && unchanged &&
                             error_matches;
        };
        MoeSharedFFNWriteBuffers residual_alias = write_buffers;
        residual_alias.residual = storage[routed_part_data].view();
        reject_write(label + " GR residual/read-only alias", residual_alias,
                     "disjoint from shared FFN inputs");
        MoeSharedFFNWriteBuffers injection_alias = write_buffers;
        injection_alias.inject = storage[gate_output].view();
        reject_write(label + " GR injection/shared-scratch alias",
                     injection_alias, "disjoint from shared FFN outputs");
        MoeSharedFFNWriteBuffers residual_injection_alias = write_buffers;
        residual_injection_alias.inject =
            storage[residual_input].view().slice(0u, injection_bytes);
        reject_write(label + " GR residual/injection alias",
                     residual_injection_alias, "residual and injection");

        BufferRef adjacent_residual_injection = context.buffers().allocate(
            label + " adjacent GR residual/injection views",
            residual_bytes + injection_bytes, error);
        if (!adjacent_residual_injection) {
            check(false, label + " adjacent GR views allocation", error);
            validation_ok = false;
        } else {
            std::memcpy(adjacent_residual_injection.contents,
                        initial_residual.data(), residual_bytes);
            std::memcpy(
                static_cast<std::byte*>(
                    adjacent_residual_injection.contents) + residual_bytes,
                inject.data(), injection_bytes);
            MoeSharedFFNWriteBuffers adjacent_views{
                buffers,
                adjacent_residual_injection.slice(
                    0u, residual_bytes + injection_bytes),
                adjacent_residual_injection.slice(
                    residual_bytes, injection_bytes)};
            auto adjacent_command = context.make_command_buffer(
                label + " adjacent GR residual/injection ranges", error);
            const bool accepted = adjacent_command &&
                encode_moe_shared_ffn_write(
                    context, *adjacent_command, rows, adjacent_views, error);
            if (adjacent_command &&
                adjacent_command->unsafe_native_handle() != nullptr) {
                adjacent_command->discard();
            }
            const bool discarded =
                adjacent_command &&
                adjacent_command->unsafe_native_handle() == nullptr;
            check(accepted && discarded,
                  label + " accepts disjoint live ranges with overlapping capacity",
                  error);
            validation_ok &= accepted && discarded;
        }

        std::array<std::vector<std::uint8_t>, buffer_count>
            write_rollback_before{};
        for (std::size_t i = 0u; i < storage.size(); ++i) {
            write_rollback_before[i].assign(storage[i].payload().begin(),
                                            storage[i].payload().end());
        }
        auto write_rollback_command = context.make_command_buffer(
            label + " caller-owned write rollback", error);
        check(write_rollback_command != nullptr,
              label + " caller-owned write rollback command buffer", error);
        if (write_rollback_command) {
            const bool rejected = !Encoder::encode(
                context, *write_rollback_command,
                label + " caller-owned write rollback",
                [&](Encoder& encoder, std::string& encode_error) {
                    if (!encode_moe_shared_ffn_write(
                            context, encoder, rows, write_buffers,
                            encode_error)) {
                        return false;
                    }
                    return encode_moe_shared_ffn_write(
                        context, encoder, 0u, write_buffers, encode_error);
                },
                error);
            const bool discarded = write_rollback_command
                    ->unsafe_native_handle() == nullptr;
            bool unchanged = true;
            for (std::size_t i = 0u; i < storage.size(); ++i) {
                unchanged &=
                    std::equal(write_rollback_before[i].begin(),
                               write_rollback_before[i].end(),
                               storage[i].payload().begin(),
                               storage[i].payload().end());
            }
            check(rejected && discarded && unchanged &&
                      error.find("1..8 rows") != std::string::npos,
                  label + " caller-owned GR-write failure rolls back all stages",
                  error);
            validation_ok &= rejected && discarded && unchanged &&
                             error.find("1..8 rows") != std::string::npos;
        } else {
            validation_ok = false;
        }

        const char* const dispatch_mode =
            std::getenv("STRATA_METAL_DISPATCH");
        if (dispatch_mode != nullptr &&
            std::string_view(dispatch_mode) == "concurrent") {
            auto group_command = context.make_command_buffer(
                label + " open concurrent group", error);
            check(group_command != nullptr,
                  label + " open-group command buffer", error);
            if (group_command) {
                std::string boundary_error;
                const bool rejected = !Encoder::encode(
                    context, *group_command,
                    label + " open concurrent group",
                    [&](Encoder& encoder, std::string& encode_error) {
                        if (!encoder.group_begin(encode_error)) return false;
                        const bool compose_rejected =
                            !encode_moe_shared_ffn_write(
                                context, encoder, rows, write_buffers,
                                boundary_error) &&
                            boundary_error.find(
                                "ordered dispatch boundaries") !=
                                std::string::npos;
                        if (!encoder.group_end(encode_error)) return false;
                        if (!compose_rejected) {
                            encode_error =
                                "shared MoE FFN/GR write accepted an open group";
                            return false;
                        }
                        encode_error = boundary_error;
                        return false;
                    },
                    error);
                const bool discarded =
                    group_command->unsafe_native_handle() == nullptr;
                check(rejected && discarded &&
                          error.find("ordered dispatch boundaries") !=
                              std::string::npos,
                      label + " rejects an open concurrent group", error);
                validation_ok &= rejected && discarded &&
                                 error.find("ordered dispatch boundaries") !=
                                     std::string::npos;
            } else {
                validation_ok = false;
            }
        }
    }

    auto command = context.make_command_buffer(label, error);
    const bool caller_owned =
        rows == 1u && set == twin::InputSet::fixture;
    const bool encoded = command && (caller_owned
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_moe_shared_ffn(
                      context, encoder, rows, buffers, encode_error);
              },
              error)
        : encode_moe_shared_ffn(context, *command, rows, buffers, error));
    const bool dispatched =
        encoded && command->commit(error) &&
        command->wait_until_completed(error);
    check(dispatched,
          label + (caller_owned ? " caller-owned encoder"
                                : " standalone transactional encoder"),
          error);
    if (!dispatched) return false;

    std::vector<float> oracle_gate_full;
    std::vector<float> oracle_up_full;
    std::vector<float> oracle_down_full;
    std::vector<float> oracle_gate_logits_full;
    bool oracle_ready = twin::q8_0_matvec_reference(
        gate_weights, expected_shared.mixed_q8_1, kMoeInputWidth,
        kMoeExpertWidth, nc, rows, oracle_gate_full, error);
    oracle_ready &= twin::q8_0_matvec_reference(
        up_weights, expected_shared.mixed_q8_1, kMoeInputWidth,
        kMoeExpertWidth, nc, rows, oracle_up_full, error);
    oracle_ready &= twin::q8_0_matvec_reference(
        down_weights, expected_shared.hidden_q8_1, kMoeDownInputWidth,
        kMoeDownWidth, nc, rows, oracle_down_full, error);
    oracle_ready &= twin::bf16_matvec_reference(
        gate_input_weights, mixed, kMoeInputWidth, 1u, nc, rows, 256u,
        oracle_gate_logits_full, error);
    check(oracle_ready, label + " padded sink oracles", error);
    if (!oracle_ready) return false;

    const auto compare_output = [&](BufferIndex index,
                                    std::span<const twin::tf> expected,
                                    std::string_view output_label) {
        const auto* actual_data = reinterpret_cast<const float*>(
            storage[index].payload().data());
        const auto failure = twin::compare_floats(
            std::span<const float>(actual_data, expected.size()), expected,
            {twin::Comparison::units, 16.0, 0.0, 0u});
        if (failure) {
            twin::report_failure(label.c_str(), options, rows, *failure);
        }
        check(!failure, label + " " + std::string(output_label) + " U(16)");
        return !failure;
    };
    bool outputs_match = true;
    outputs_match &= compare_output(gate_output, expected_gate, "gate");
    outputs_match &= compare_output(up_output, expected_up, "up");
    outputs_match &=
        compare_output(hidden_output, expected_hidden, "hidden");
    const auto down_q8_failure = twin::compare_bytes(
        std::span<const std::uint8_t>(
            storage[down_q8_1_output].payload().data(),
            down_q8_1_bytes),
        expected_hidden_q8);
    check(!down_q8_failure, label + " down Q8_1 bytes match exactly");
    outputs_match &= !down_q8_failure;
    outputs_match &= compare_output(shared_output, expected_shared_output,
                                    "shared output");
    outputs_match &= compare_output(shared_gate_logits,
                                    expected_gate_logits, "gate logits");

    const auto compare_sink = [&](BufferIndex index,
                                  std::size_t expected_write_bytes,
                                  std::size_t active_count,
                                  std::span<const float> full_expected,
                                  std::string_view sink_label) {
        if (expected_write_bytes == 0u) return true;
        const std::size_t padded_count =
            expected_write_bytes / sizeof(float);
        std::vector<twin::tf> expected(
            full_expected.size() - active_count);
        std::transform(full_expected.begin() + active_count,
                       full_expected.end(), expected.begin(),
                       [](float value) { return twin::tf{value}; });
        const auto* actual_data = reinterpret_cast<const float*>(
            storage[index].payload().data());
        const auto failure = twin::compare_floats(
            std::span<const float>(actual_data, padded_count), expected,
            {twin::Comparison::units, 16.0, 0.0, 0u});
        check(!failure, label + " " + std::string(sink_label) +
                           " padded values U(16)");
        return !failure;
    };
    outputs_match &= compare_sink(
        gate_sink, gate_sink_write_bytes,
        static_cast<std::size_t>(rows) * kMoeExpertWidth,
        oracle_gate_full, "gate sink");
    outputs_match &= compare_sink(
        up_sink, gate_sink_write_bytes,
        static_cast<std::size_t>(rows) * kMoeExpertWidth,
        oracle_up_full, "up sink");
    outputs_match &= compare_sink(
        down_sink, down_sink_write_bytes,
        static_cast<std::size_t>(rows) * kMoeDownWidth,
        oracle_down_full, "down sink");
    outputs_match &= compare_sink(
        shared_gate_sink, shared_gate_sink_write_bytes, rows,
        oracle_gate_logits_full, "shared-gate sink");

    const std::size_t output_float_count = shared_bytes / sizeof(float);
    const auto* actual_shared_data =
        reinterpret_cast<const float*>(
            storage[shared_output].payload().data());
    const auto* actual_gate_data =
        reinterpret_cast<const float*>(
            storage[shared_gate_logits].payload().data());
    std::vector<twin::tf> recombined_expected;
    const bool recombine_ready = twin::moe_combine_reference(
        routed_parts, route_weights,
        std::span<const float>(actual_shared_data, output_float_count),
        std::span<const float>(actual_gate_data, rows), rows,
        recombined_expected, error);
    check(recombine_ready, label + " combine oracle from GPU shared values",
          error);
    if (!recombine_ready) return false;
    outputs_match &= compare_output(combined_output, recombined_expected,
                                    "combined output");

    const std::array<twin::ByteRange, 1> gate_written{{{0u, gate_bytes}}};
    const std::array<twin::ByteRange, 1> up_written{{{0u, up_bytes}}};
    const std::array<twin::ByteRange, 1> hidden_written{{{0u, hidden_bytes}}};
    const std::array<twin::ByteRange, 1> down_q8_written{
        {{0u, down_q8_1_bytes}}};
    const std::array<twin::ByteRange, 1> shared_written{
        {{0u, shared_bytes}}};
    const std::array<twin::ByteRange, 1> gate_logits_written{
        {{0u, gate_logits_bytes}}};
    const std::array<twin::ByteRange, 1> combined_written{
        {{0u, combined_bytes}}};
    const auto sink_unchanged = [&](BufferIndex index,
                                    std::size_t written_bytes,
                                    std::string_view sink_label) {
        const std::array<twin::ByteRange, 1> written{{{0u, written_bytes}}};
        check(!storage[index].first_unwritten_change(written),
              label + " " + std::string(sink_label) +
                  " exact write range and guards");
        return !storage[index].first_unwritten_change(written);
    };
    bool guards_ok = true;
    for (const BufferIndex input :
         {input_q8_1, mixed_f32, gate_weight_data, up_weight_data,
          down_weight_data, shared_gate_weight_data, active_row_data,
          routed_part_data, route_weight_data}) {
        check(!storage[input].first_unwritten_change({}),
              label + " " + names[input] + " read-only bytes and guards");
        guards_ok &= !storage[input].first_unwritten_change({});
    }
    guards_ok &= !storage[gate_output].first_unwritten_change(gate_written);
    guards_ok &= !storage[up_output].first_unwritten_change(up_written);
    guards_ok &=
        !storage[hidden_output].first_unwritten_change(hidden_written);
    guards_ok &=
        !storage[down_q8_1_output].first_unwritten_change(down_q8_written);
    guards_ok &=
        !storage[shared_output].first_unwritten_change(shared_written);
    guards_ok &=
        !storage[shared_gate_logits].first_unwritten_change(gate_logits_written);
    guards_ok &=
        !storage[combined_output].first_unwritten_change(combined_written);
    guards_ok &=
        !storage[residual_input].first_unwritten_change({});
    guards_ok &=
        !storage[injection_input].first_unwritten_change({});
    guards_ok &= sink_unchanged(gate_sink, gate_sink_write_bytes, "gate sink");
    guards_ok &= sink_unchanged(up_sink, gate_sink_write_bytes, "up sink");
    guards_ok &= sink_unchanged(down_sink, down_sink_write_bytes, "down sink");
    guards_ok &= sink_unchanged(shared_gate_sink,
                                shared_gate_sink_write_bytes,
                                "shared-gate sink");
    check(guards_ok, label + " inputs, output ranges and guards");
    if (!validation_ok) return false;

    bool write_match = true;
    if (rows == 1u || rows == 3u || rows == 8u) {
        for (auto& buffer : storage) buffer.capture_before_dispatch();
        auto write_command = context.make_command_buffer(
            label + " shared FFN and GR post", error);
        const bool write_caller_owned =
            rows == 1u && set == twin::InputSet::fixture;
        const bool write_encoded = write_command && (write_caller_owned
            ? Encoder::encode(
                  context, *write_command, label + " shared FFN and GR post",
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_moe_shared_ffn_write(
                          context, encoder, rows, write_buffers,
                          encode_error);
                  },
                  error)
            : encode_moe_shared_ffn_write(context, *write_command, rows,
                                          write_buffers, error));
        const bool write_dispatched =
            write_encoded && write_command->commit(error) &&
            write_command->wait_until_completed(error);
        check(write_dispatched,
              label + (write_caller_owned
                           ? " caller-owned shared FFN and GR write"
                           : " standalone shared FFN and GR write"),
              error);
        write_match &= write_dispatched;
        if (write_dispatched) {
            const auto* combined_data =
                reinterpret_cast<const float*>(
                    storage[combined_output].payload().data());
            std::vector<twin::tf> expected_post;
            moe_gr_post_reference(
                initial_residual,
                std::span<const float>(combined_data,
                                       combined_bytes / sizeof(float)),
                inject, rows, expected_post);
            const auto* actual_residual =
                reinterpret_cast<const float*>(
                    storage[residual_input].payload().data());
            const auto post_failure = twin::compare_floats(
                std::span<const float>(actual_residual,
                                       expected_post.size()),
                expected_post,
                {twin::Comparison::units, 16.0, 0.0, 0u});
            if (post_failure) {
                twin::report_failure(label.c_str(), options, rows,
                                     *post_failure);
            }
            check(!post_failure,
                  label + " composed GR residual write U(16)");
            write_match &= !post_failure;

            const std::array<twin::ByteRange, 1> residual_written{
                {{0u, residual_bytes}}};
            const bool residual_guards =
                !storage[residual_input].first_unwritten_change(
                    residual_written);
            const bool injection_unchanged =
                !storage[injection_input].first_unwritten_change({});
            check(residual_guards,
                  label + " residual exact write range and guards");
            check(injection_unchanged,
                  label + " injection remains read-only with guards");
            write_match &= residual_guards && injection_unchanged;
        }
    }
    return outputs_match && guards_ok && write_match;
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
    completed &= test_composed_shared_ffn(
        context, rows, nc, set, options, mixed, mixed_q8_1, gate_weights,
        up_weights, down_weights, gate_input_weights, routed_parts,
        route_weights, composite);
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

    const bool dispatched = dispatch_moe_combine(
        context, rows, parts_buffer, weights_buffer, shared_buffer,
        gate_buffer, output_buffer, err, rows == 8u);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::q2_0, rows, experts, kMoeGroupedLayer, address_buffer, input_buffer,
        plan_buffer, gate_buffer, up_buffer, err);
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

bool test_moe_routed_ffn_composition(Context& context, std::uint32_t rows,
                                    const twin::Options& options) {
    constexpr std::uint32_t experts = 512u;
    constexpr std::uint32_t layer = kMoeGroupedLayer;
    constexpr twin::InputSet set = twin::InputSet::fixture;
    const std::string label =
        "MoE routed FFN composition rows=" + std::to_string(rows);
    const std::vector<std::uint32_t> active_experts =
        make_moe_q2_experts(experts);
    std::vector<std::int32_t> expert_slots(experts, -1);
    for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
        expert_slots[active_experts[slot]] = static_cast<std::int32_t>(slot);
    }

    std::vector<std::int32_t> ids(
        static_cast<std::size_t>(rows) * kTopK);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t rank = 0u; rank < kTopK; ++rank) {
            ids[static_cast<std::size_t>(row) * kTopK + rank] =
                static_cast<std::int32_t>(
                    active_experts[(rank + row) % active_experts.size()]);
        }
    }
    const std::size_t table_entries =
        static_cast<std::size_t>(layer + 1u) * experts;
    std::vector<std::int32_t> resident(table_entries, 1);
    std::vector<float> initial_usage(table_entries);
    for (std::size_t index = 0u; index < initial_usage.size(); ++index) {
        initial_usage[index] =
            0.25f + static_cast<float>(index % 31u) * 0.0625f;
    }
    TieredMoePlan expected_plan{};
    std::vector<float> expected_usage;
    std::string err;
    check(twin::moe_plan_reference(
              ids, rows, experts, false, layer, resident, initial_usage,
              expected_plan, expected_usage, err),
          std::string(label) + " CPU plan oracle", err);
    if (expected_usage.size() != initial_usage.size()) return false;

    std::vector<block_q2_0> gate_weights;
    std::vector<block_q2_0> up_weights;
    std::vector<block_q2_0> down_weights;
    const std::uint64_t seed =
        options.seed ^ (static_cast<std::uint64_t>(experts) << 32u) ^
        static_cast<std::uint64_t>(set);
    make_moe_q2_weights(set, seed, gate_weights, up_weights);
    make_moe_down_q2_0_weights(set, seed ^ 0x9e3779b97f4a7c15ull,
                               down_weights);

    std::vector<Q81> input_activations;
    check(make_moe_q2_activations(rows, set, seed ^ 0x51f15eull,
                                  input_activations, err),
          std::string(label) + " input Q8_1 oracle", err);
    std::vector<twin::tf> expected_gate;
    std::vector<twin::tf> expected_up;
    check(twin::moe_gu_q2_0_reference(
              gate_weights, up_weights, input_activations, expert_slots,
              expected_plan, rows, expected_gate, expected_up, err),
          std::string(label) + " gate/up oracle", err);
    if (expected_gate.size() !=
            static_cast<std::size_t>(rows) * kTopK * kMoeExpertWidth ||
        expected_up.size() != expected_gate.size()) {
        check(false, std::string(label) + " gate/up oracle shape");
        return false;
    }

    std::vector<float> expected_gate_f32(expected_gate.size());
    std::vector<float> expected_up_f32(expected_up.size());
    for (std::size_t index = 0u; index < expected_gate.size(); ++index) {
        expected_gate_f32[index] = expected_gate[index].value;
        expected_up_f32[index] = expected_up[index].value;
    }
    std::vector<twin::tf> expected_hidden;
    check(twin::moe_swiglu_reference(expected_gate_f32, expected_up_f32,
                                     expected_hidden, err),
          std::string(label) + " SwiGLU oracle", err);
    if (expected_hidden.size() != expected_gate.size()) {
        check(false, std::string(label) + " SwiGLU oracle shape");
        return false;
    }
    std::vector<float> hidden_f32(expected_hidden.size());
    for (std::size_t index = 0u; index < expected_hidden.size(); ++index) {
        hidden_f32[index] = expected_hidden[index].value;
    }
    std::vector<Q81> expected_down_activations;
    check(twin::quantize_q8_1_reference(
              hidden_f32, kMoeExpertWidth, rows * kTopK,
              expected_down_activations, err),
          std::string(label) + " hidden Q8_1 oracle", err);
    std::vector<twin::tf> expected_parts;
    check(twin::moe_down_q2_0_reference(
              down_weights, expected_down_activations, expert_slots,
              expected_plan, rows, expected_parts, err),
          std::string(label) + " down oracle", err);
    if (expected_parts.size() !=
        static_cast<std::size_t>(rows) * kTopK * kMoeDownWidth) {
        check(false, std::string(label) + " down oracle shape");
        return false;
    }

    std::vector<MoeExpertAddress> addresses(table_entries);
    const std::size_t gate_up_bytes_per_expert =
        static_cast<std::size_t>(kMoeExpertWidth) *
        kMoeQ2BlocksPerRow * sizeof(block_q2_0);
    const std::size_t down_bytes_per_expert =
        static_cast<std::size_t>(kMoeDownWidth) *
        kMoeDownQ2BlocksPerRow * sizeof(block_q2_0);
    twin::GuardedBuffer gate_weight_buffer;
    twin::GuardedBuffer up_weight_buffer;
    twin::GuardedBuffer down_weight_buffer;
    twin::GuardedBuffer address_buffer;
    twin::GuardedBuffer ids_buffer;
    twin::GuardedBuffer resident_buffer;
    twin::GuardedBuffer usage_buffer;
    twin::GuardedBuffer plans_buffer;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer up_buffer;
    twin::GuardedBuffer hidden_buffer;
    twin::GuardedBuffer down_activations_buffer;
    twin::GuardedBuffer parts_buffer;
    const std::size_t f32_bytes =
        expected_gate.size() * sizeof(float);
    const std::size_t down_activations_bytes =
        expected_down_activations.size() * sizeof(Q81);
    const std::size_t parts_bytes = expected_parts.size() * sizeof(float);
    std::array<TieredMoePlan, layer + 1u> expected_plans{};
    expected_plans[layer] = expected_plan;
    const std::size_t plans_bytes =
        expected_plans.size() * sizeof(TieredMoePlan);
    const std::size_t input_bytes =
        input_activations.size() * sizeof(Q81);
    const auto allocated =
        gate_weight_buffer.allocate(context, "routed FFN gate weights",
                                    gate_weights.size() * sizeof(block_q2_0),
                                    err) &&
        up_weight_buffer.allocate(context, "routed FFN up weights",
                                  up_weights.size() * sizeof(block_q2_0),
                                  err) &&
        down_weight_buffer.allocate(context, "routed FFN down weights",
                                    down_weights.size() * sizeof(block_q2_0),
                                    err) &&
        address_buffer.allocate(context, "routed FFN expert addresses",
                                addresses.size() * sizeof(MoeExpertAddress),
                                err) &&
        ids_buffer.allocate(context, "routed FFN route IDs",
                            ids.size() * sizeof(std::int32_t), err) &&
        resident_buffer.allocate(context, "routed FFN resident table",
                                 resident.size() * sizeof(std::int32_t), err) &&
        usage_buffer.allocate(context, "routed FFN usage table",
                              initial_usage.size() * sizeof(float), err) &&
        plans_buffer.allocate(context, "routed FFN plan table", plans_bytes,
                              err) &&
        input_buffer.allocate(context, "routed FFN input activations",
                              input_bytes, err) &&
        gate_buffer.allocate(context, "routed FFN gate", f32_bytes, err) &&
        up_buffer.allocate(context, "routed FFN up", f32_bytes, err) &&
        hidden_buffer.allocate(context, "routed FFN hidden", f32_bytes, err) &&
        down_activations_buffer.allocate(
            context, "routed FFN down activations", down_activations_bytes,
            err) &&
        parts_buffer.allocate(context, "routed FFN parts", parts_bytes, err);
    check(allocated, std::string(label) + " allocate fixture buffers", err);
    if (!allocated) return false;

    std::memcpy(gate_weight_buffer.payload().data(), gate_weights.data(),
                gate_weights.size() * sizeof(block_q2_0));
    std::memcpy(up_weight_buffer.payload().data(), up_weights.data(),
                up_weights.size() * sizeof(block_q2_0));
    std::memcpy(down_weight_buffer.payload().data(), down_weights.data(),
                down_weights.size() * sizeof(block_q2_0));
    for (std::uint32_t slot = 0u; slot < active_experts.size(); ++slot) {
        const std::size_t address_index =
            static_cast<std::size_t>(layer) * experts + active_experts[slot];
        addresses[address_index].role[0] =
            gate_weight_buffer.view().gpu_address +
            slot * gate_up_bytes_per_expert;
        addresses[address_index].role[1] =
            up_weight_buffer.view().gpu_address +
            slot * gate_up_bytes_per_expert;
        addresses[address_index].role[2] =
            down_weight_buffer.view().gpu_address +
            slot * down_bytes_per_expert;
    }
    std::memcpy(address_buffer.payload().data(), addresses.data(),
                addresses.size() * sizeof(MoeExpertAddress));
    std::memcpy(ids_buffer.payload().data(), ids.data(),
                ids.size() * sizeof(std::int32_t));
    std::memcpy(resident_buffer.payload().data(), resident.data(),
                resident.size() * sizeof(std::int32_t));
    std::memcpy(usage_buffer.payload().data(), initial_usage.data(),
                initial_usage.size() * sizeof(float));
    std::vector<std::uint8_t> poisoned_plans(plans_bytes, 0u);
    const std::size_t selected_plan_offset =
        static_cast<std::size_t>(layer) * sizeof(TieredMoePlan);
    const auto* expected_plan_bytes =
        reinterpret_cast<const std::uint8_t*>(&expected_plan);
    for (std::size_t byte = 0u; byte < sizeof(expected_plan); ++byte) {
        poisoned_plans[selected_plan_offset + byte] =
            static_cast<std::uint8_t>(expected_plan_bytes[byte] ^ 0xffu);
    }
    std::memcpy(plans_buffer.payload().data(), poisoned_plans.data(),
                plans_bytes);
    std::memcpy(input_buffer.payload().data(), input_activations.data(),
                input_bytes);
    for (twin::GuardedBuffer* output :
         {&gate_buffer, &up_buffer, &hidden_buffer,
          &down_activations_buffer, &parts_buffer}) {
        std::fill(output->payload().begin(), output->payload().end(), 0x5au);
    }
    const bool residency_ready = context.buffers().commit_residency(err);
    check(residency_ready, std::string(label) + " expert residency", err);
    if (!residency_ready) return false;

    for (twin::GuardedBuffer* buffer :
         {&address_buffer, &ids_buffer, &resident_buffer, &usage_buffer,
          &plans_buffer, &input_buffer, &gate_buffer, &up_buffer,
          &hidden_buffer, &down_activations_buffer, &parts_buffer}) {
        buffer->capture_before_dispatch();
    }
    const ReadOnlyBufferRef addresses_view =
        context.buffers().read_only(address_buffer.view());
    const ReadOnlyBufferRef ids_view =
        context.buffers().read_only(ids_buffer.view());
    const ReadOnlyBufferRef resident_view =
        context.buffers().read_only(resident_buffer.view());
    const ReadOnlyBufferRef input_view =
        context.buffers().read_only(input_buffer.view());
    const ReadOnlyBufferRef plans_view =
        context.buffers().read_only(plans_buffer.view());
    if (!addresses_view || !ids_view || !resident_view || !input_view ||
        !plans_view) {
        check(false, std::string(label) + " read-only views", err);
        return false;
    }
    const MoeRoutedFFNBuffers ffn_buffers{
        addresses_view, input_view, plans_view,
        gate_buffer.view(), up_buffer.view(), hidden_buffer.view(),
        down_activations_buffer.view(), parts_buffer.view()};
    const auto outputs_unchanged = [&] {
        return gate_buffer.first_unwritten_change({}) == std::nullopt &&
               up_buffer.first_unwritten_change({}) == std::nullopt &&
               hidden_buffer.first_unwritten_change({}) == std::nullopt &&
               down_activations_buffer.first_unwritten_change({}) ==
                   std::nullopt &&
               parts_buffer.first_unwritten_change({}) == std::nullopt;
    };
    const auto reject = [&](std::string case_label,
                            const MoeRoutedFFNBuffers& bad_buffers,
                            std::string_view expected_error) {
        auto command = context.make_command_buffer(case_label, err);
        const bool rejected = command && !encode_moe_routed_ffn(
            context, *command, rows, experts, layer,
            MoeExpertFormat::q2_0, MoeDownFormat::q2_0, bad_buffers, err);
        const bool discarded =
            command && command->unsafe_native_handle() == nullptr;
        const bool error_matches =
            err.find(expected_error) != std::string::npos;
        const bool unchanged = outputs_unchanged();
        check(rejected && discarded && error_matches && unchanged,
              case_label + " rejects before encoding and discards", err);
        return rejected && discarded && error_matches && unchanged;
    };
    const bool short_parts = reject(
        std::string(label) + " short parts",
        {addresses_view, input_view, plans_view, gate_buffer.view(),
         up_buffer.view(), hidden_buffer.view(),
         down_activations_buffer.view(),
         parts_buffer.view().slice(0u, parts_bytes - sizeof(float))},
        "scratch output");
    const bool output_input_alias = reject(
        std::string(label) + " output/input alias",
        {addresses_view, context.buffers().read_only(gate_buffer.view()),
         plans_view, gate_buffer.view(), up_buffer.view(),
         hidden_buffer.view(), down_activations_buffer.view(),
         parts_buffer.view()},
        "disjoint from inputs");
    const bool scratch_output_alias = reject(
        std::string(label) + " scratch output alias",
        {addresses_view, input_view, plans_view, gate_buffer.view(),
         gate_buffer.view(), hidden_buffer.view(),
         down_activations_buffer.view(), parts_buffer.view()},
        "scratch outputs must be disjoint");
    const bool readonly_input_alias = reject(
        std::string(label) + " read-only input alias",
        {addresses_view, input_view,
         addresses_view.slice(0u, plans_bytes),
         gate_buffer.view(), up_buffer.view(), hidden_buffer.view(),
         down_activations_buffer.view(), parts_buffer.view()},
        "read-only inputs must be disjoint");
    auto invalid_rows_command =
        context.make_command_buffer("routed FFN invalid rows", err);
    const bool invalid_rows = invalid_rows_command &&
        !encode_moe_routed_ffn(
            context, *invalid_rows_command, 0u, experts, layer,
            MoeExpertFormat::q2_0, MoeDownFormat::q2_0, ffn_buffers, err);
    const bool invalid_rows_discarded = invalid_rows_command &&
        invalid_rows_command->unsafe_native_handle() == nullptr;
    check(invalid_rows && invalid_rows_discarded &&
              err.find("1..8 rows") != std::string::npos &&
              outputs_unchanged(),
          std::string(label) + " rejects invalid rows transactionally", err);

    const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    const bool concurrent_mode = dispatch_mode != nullptr &&
        std::string_view(dispatch_mode) == "concurrent";
    bool open_group_rejected = true;
    if (concurrent_mode) {
        auto group_command =
            context.make_command_buffer("routed FFN open group", err);
        const bool encoded = group_command && !Encoder::encode(
            context, *group_command, "routed FFN open group",
            [&](Encoder& encoder, std::string& body_error) {
                return encoder.group_begin(body_error) &&
                    encode_moe_routed_ffn(
                        context, encoder, rows, experts, layer,
                        MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                        ffn_buffers, body_error);
            },
            err);
        const bool discarded = group_command &&
            group_command->unsafe_native_handle() == nullptr;
        open_group_rejected = encoded && discarded &&
            err.find("ordered dispatch boundaries") != std::string::npos &&
            outputs_unchanged();
        check(open_group_rejected,
              std::string(label) + " rejects an open concurrent group", err);
    }

    auto late_failure_command =
        context.make_command_buffer("routed FFN late caller failure", err);
    const bool late_failure = late_failure_command && !Encoder::encode(
        context, *late_failure_command, "routed FFN late caller failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_moe_routed_ffn(
                    context, encoder, rows, experts, layer,
                    MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                    ffn_buffers, encode_error)) {
                return false;
            }
            return encode_moe_down(
                context, encoder, 9u, experts, layer, MoeDownFormat::q2_0,
                {addresses_view,
                 context.buffers().read_only(down_activations_buffer.view()),
                 plans_view, parts_buffer.view()},
                encode_error);
        },
        err);
    const bool late_discarded = late_failure_command &&
        late_failure_command->unsafe_native_handle() == nullptr;
    check(late_failure && late_discarded && outputs_unchanged() &&
              err.find("1..8 rows") != std::string::npos,
          std::string(label) +
              " caller-owned late failure discards all routed FFN stages",
          err);

    const bool encoded_and_completed = [&] {
        auto command = context.make_command_buffer(
            "routed FFN plan and composition", err);
        if (!command) return false;
        const bool encoded = Encoder::encode(
            context, *command, "routed FFN plan and composition",
            [&](Encoder& encoder, std::string& encode_error) {
                const MoePlanBuffers plan_buffers{
                    ids_view, resident_view, usage_buffer.view(),
                    plans_buffer.view()};
                return encode_moe_plan(context, encoder, rows, experts,
                                       false, layer, plan_buffers,
                                       encode_error) &&
                       encode_moe_routed_ffn(
                           context, encoder, rows, experts, layer,
                           MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                           ffn_buffers, encode_error);
            },
            err);
        return encoded && command->commit(err) &&
               command->wait_until_completed(err);
    }();
    check(encoded_and_completed,
          std::string(label) + " GPU plan plus routed stages", err);
    if (!encoded_and_completed) return false;

    const auto* actual_plans =
        reinterpret_cast<const std::uint8_t*>(plans_buffer.payload().data());
    const std::span<const std::uint8_t> expected_plans_span(
        reinterpret_cast<const std::uint8_t*>(expected_plans.data()),
        plans_bytes);
    const auto plan_failure =
        twin::compare_bytes(std::span<const std::uint8_t>(
                                actual_plans, plans_bytes),
                            expected_plans_span);
    check(!plan_failure, std::string(label) + " exact GPU plan bytes");
    const auto usage_failure = twin::compare_bytes(
        usage_buffer.payload(),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_usage.data()),
            expected_usage.size() * sizeof(float)));
    check(!usage_failure,
          std::string(label) + " full mode preserves usage bytes");

    const twin::Tolerance tolerance{twin::Comparison::units, 16.0, 0.0, 0u};
    const auto compare_output = [&](const twin::GuardedBuffer& buffer,
                                    std::span<const twin::tf> expected,
                                    std::string output_label) {
        const auto* actual =
            reinterpret_cast<const float*>(buffer.payload().data());
        const auto failure = twin::compare_floats(
            std::span<const float>(actual, expected.size()), expected,
            tolerance);
        if (failure) {
            twin::report_failure(output_label.c_str(), options, rows,
                                 *failure);
        }
        check(!failure, std::string(label) + " " + output_label + " U(16)");
        return !failure;
    };
    const bool gate_matches = compare_output(
        gate_buffer, expected_gate, "gate");
    const bool up_matches = compare_output(up_buffer, expected_up, "up");
    const bool hidden_matches = compare_output(
        hidden_buffer, expected_hidden, "SwiGLU");
    const bool parts_matches = compare_output(
        parts_buffer, expected_parts, "down parts");
    const auto* actual_down_activations =
        reinterpret_cast<const std::uint8_t*>(
            down_activations_buffer.payload().data());
    const auto down_activations_failure = twin::compare_bytes(
        std::span<const std::uint8_t>(
            actual_down_activations, down_activations_bytes),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(
                expected_down_activations.data()),
            down_activations_bytes));
    check(!down_activations_failure,
          std::string(label) + " exact hidden Q8_1 bytes");

    check(!address_buffer.first_unwritten_change({}),
          std::string(label) + " addresses and guards unchanged");
    check(!ids_buffer.first_unwritten_change({}),
          std::string(label) + " IDs and guards unchanged");
    check(!resident_buffer.first_unwritten_change({}),
          std::string(label) + " resident table and guards unchanged");
    check(!input_buffer.first_unwritten_change({}),
          std::string(label) + " input Q8_1 and guards unchanged");
    check(!usage_buffer.first_unwritten_change({}),
          std::string(label) + " usage and guards unchanged");
    const std::array<twin::ByteRange, 1> plan_written = {
        {{selected_plan_offset, sizeof(TieredMoePlan)}}};
    check(!plans_buffer.first_unwritten_change(plan_written),
          std::string(label) + " plan writes stay within selected layer");
    const std::array<twin::ByteRange, 1> gate_up_written = {{{0u, f32_bytes}}};
    const std::array<twin::ByteRange, 1> down_q_written = {
        {{0u, down_activations_bytes}}};
    const std::array<twin::ByteRange, 1> parts_written = {{{0u, parts_bytes}}};
    check(!gate_buffer.first_unwritten_change(gate_up_written),
          std::string(label) + " gate writes stay within the output");
    check(!up_buffer.first_unwritten_change(gate_up_written),
          std::string(label) + " up writes stay within the output");
    check(!hidden_buffer.first_unwritten_change(gate_up_written),
          std::string(label) + " SwiGLU writes stay within the output");
    check(!down_activations_buffer.first_unwritten_change(down_q_written),
          std::string(label) + " hidden Q8_1 writes stay within the output");
    check(!parts_buffer.first_unwritten_change(parts_written),
          std::string(label) + " down writes stay within original route order");
    const bool routed_only_ok =
        short_parts && output_input_alias && scratch_output_alias &&
        readonly_input_alias && invalid_rows &&
        invalid_rows_discarded && open_group_rejected && late_failure &&
        late_discarded && !plan_failure && !usage_failure &&
        gate_matches && up_matches && hidden_matches && parts_matches &&
        !down_activations_failure;
    const std::array<twin::GuardedBuffer*, 5u> routed_storage{
        &gate_buffer, &up_buffer, &hidden_buffer, &down_activations_buffer,
        &parts_buffer};
    const bool routed_shared_ok = test_moe_routed_shared_ffn_write(
        context, rows, options, ffn_buffers, input_activations,
        expected_parts, routed_storage);
    return routed_only_ok && routed_shared_ok;
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq1_m, rows, experts, kMoeGroupedLayer, address_buffer, input_buffer,
        plan_buffer, gate_buffer, up_buffer, err);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq2_xxs, rows, experts, kMoeGroupedLayer,
        address_buffer, input_buffer, plan_buffer, gate_buffer, up_buffer, err);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq2_xxs, rows, experts, kMoeGroupedLayer,
        address_buffer, input_buffer, plan_buffer, gate_buffer, up_buffer, err,
        true);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq2_xs, rows, experts, kMoeGroupedLayer, address_buffer, input_buffer,
        plan_buffer, gate_buffer, up_buffer, err);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq2_s, rows, experts, kMoeGroupedLayer, address_buffer, input_buffer,
        plan_buffer, gate_buffer, up_buffer, err);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq3_xxs, rows, experts, kMoeGroupedLayer, address_buffer, input_buffer,
        plan_buffer, gate_buffer, up_buffer, err);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq3_s, rows, experts, kMoeGroupedLayer, address_buffer, input_buffer,
        plan_buffer, gate_buffer, up_buffer, err);
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

    const bool dispatched = dispatch_grouped_experts(
        context, MoeExpertFormat::iq4_xs, rows, experts, kMoeGroupedLayer, address_buffer, input_buffer,
        plan_buffer, gate_buffer, up_buffer, err);
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

    const bool dispatched = dispatch_grouped_down(
        context, MoeDownFormat::q2_0, rows, experts, kMoeGroupedLayer,
        address_buffer, input_buffer, plan_buffer, output_buffer, err,
        one_expert_per_entry);
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

    const bool dispatched = dispatch_grouped_down(
        context, MoeDownFormat::iq4_nl, rows, experts, kMoeGroupedLayer,
        address_buffer, input_buffer, plan_buffer, output_buffer, err,
        one_expert_per_entry);
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

    const bool single_dispatched = dispatch_grouped_down(
        context, MoeDownFormat::iq4_nl, rows, experts, kMoeGroupedLayer,
        address_buffer, input_buffer, single_plan_buffer,
        single_output_buffer, err);
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

        const bool flipped_dispatched = dispatch_grouped_down(
            context, MoeDownFormat::iq4_nl, rows, experts,
            kMoeGroupedLayer, flipped_address_buffer, input_buffer,
            single_plan_buffer, flipped_output_buffer, err);
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
    if (!test_grouped_expert_dispatch_validation(*context)) ++g_failures;
    if (!test_swiglu_dispatch_validation(*context)) ++g_failures;
    if (!test_moe_down_dispatch_validation(*context)) ++g_failures;
    if (!test_moe_combine_dispatch_validation(*context)) ++g_failures;
    for (const std::uint32_t rows :
         {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        if (!test_moe_routed_ffn_composition(*context, rows, options)) {
            ++g_failures;
        }
    }

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
