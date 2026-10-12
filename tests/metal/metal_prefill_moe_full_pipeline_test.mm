// End-to-end synthetic full-mode Prefill MoE layer composition (04 §6.6).
#include "metallib_sha256.inc"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_moe_full_pipeline.hpp"
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

constexpr std::size_t kWidth = 2560u;
constexpr std::size_t kHidden = 640u;
constexpr std::size_t kTopK = 10u;
constexpr std::size_t kAddressStride = 3u * sizeof(std::uint64_t);
constexpr std::size_t kPlanStride = 6400u;
constexpr std::size_t kQ2BlocksPerWidth = kWidth / 64u;
constexpr std::size_t kQ2BlocksPerHidden = kHidden / 64u;
constexpr std::size_t kQ2GateUpBytes =
    kHidden * kQ2BlocksPerWidth * sizeof(block_q2_0);
constexpr std::size_t kQ2DownBytes =
    kWidth * kQ2BlocksPerHidden * sizeof(block_q2_0);
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint8_t kGuardPoison = 0xffu;
constexpr float kCombineTolerance = 2.0e-6f;
constexpr double kProjectionAbsoluteTolerance = 1.0e-3;
constexpr double kProjectionRelativeTolerance = 1.0e-5;

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

struct PrefillMoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(PrefillMoeExpertAddress) == kAddressStride);

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

class CaseBuffers {
public:
    std::uint32_t rows = 0u;
    std::uint32_t experts = 0u;

    twin::GuardedBuffer router_hi;
    twin::GuardedBuffer router_lo;
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

    twin::GuardedBuffer shared_gate_weights;
    twin::GuardedBuffer shared_gate_dequant;
    twin::GuardedBuffer shared_gate_f32;
    twin::GuardedBuffer shared_up_weights;
    twin::GuardedBuffer shared_up_dequant;
    twin::GuardedBuffer shared_up_f32;
    twin::GuardedBuffer shared_hidden;
    twin::GuardedBuffer shared_down_weights;
    twin::GuardedBuffer shared_down_dequant;
    twin::GuardedBuffer shared_output;
    twin::GuardedBuffer shared_gate_hi;
    twin::GuardedBuffer shared_gate_lo;
    twin::GuardedBuffer shared_gate_router_weights;
    twin::GuardedBuffer shared_gate_logits;

    twin::GuardedBuffer routed_gate_up_weights;
    twin::GuardedBuffer routed_down_weights;
    twin::GuardedBuffer addresses;
    twin::GuardedBuffer gate_up_address_snapshot;
    twin::GuardedBuffer gate_up_hidden;
    twin::GuardedBuffer gate_up_status;
    twin::GuardedBuffer down_address_snapshot;
    twin::GuardedBuffer routed_parts;
    twin::GuardedBuffer down_status;
    twin::GuardedBuffer combined_output;

    std::array<PrefillMoeWeightStorageView, 2u> weight_storage{};

    bool allocate(Context& context, const std::string& label,
                  std::uint32_t row_count, std::uint32_t expert_count,
                  std::string& error) {
        const std::size_t entries =
            static_cast<std::size_t>(row_count) * kTopK;
        const std::size_t route_tiles =
            (entries + 63u) / 64u + 512u;
        const std::size_t histogram_tiles =
            (entries + 255u) / 256u;
        rows = row_count;
        experts = expert_count;
        return
            router_hi.allocate(context, label + " router high",
                               row_count * kWidth * sizeof(std::uint16_t),
                               error) &&
            router_lo.allocate(context, label + " router low",
                               row_count * kWidth * sizeof(std::uint16_t),
                               error) &&
            router_weights.allocate(
                context, label + " router weights",
                expert_count * kWidth * sizeof(std::uint16_t), error) &&
            logits.allocate(context, label + " logits",
                            row_count * expert_count * sizeof(float), error) &&
            route_ids.allocate(context, label + " route IDs",
                               entries * sizeof(std::int32_t), error) &&
            route_weights.allocate(context, label + " route weights",
                                   entries * sizeof(float), error) &&
            resident.allocate(context, label + " full-mode residency",
                              sizeof(std::int32_t), error) &&
            mixed_h.allocate(context, label + " mixed F16",
                             row_count * kWidth * sizeof(std::uint16_t), error) &&
            histogram.allocate(context, label + " histogram",
                               histogram_tiles * 512u *
                                   sizeof(std::uint32_t),
                               error) &&
            plan.allocate(context, label + " PfPlan", kPlanStride, error) &&
            tiles.allocate(
                context, label + " tiles",
                route_tiles * 2u * sizeof(std::uint32_t), error) &&
            source_rows.allocate(context, label + " source rows",
                                 entries * sizeof(std::int32_t), error) &&
            destination_rows.allocate(
                context, label + " destinations",
                entries * sizeof(std::int32_t), error) &&
            gathered_rows.allocate(
                context, label + " gathered rows",
                entries * kWidth * sizeof(std::uint16_t) +
                    sizeof(std::uint16_t),
                error) &&
            shared_gate_weights.allocate(
                context, label + " shared gate weights",
                kHidden * kQ2BlocksPerWidth * sizeof(block_q2_0), error) &&
            shared_gate_dequant.allocate(
                context, label + " shared gate dequant",
                kWidth * kHidden * sizeof(std::uint16_t), error) &&
            shared_gate_f32.allocate(
                context, label + " shared gate output",
                row_count * kHidden * sizeof(float), error) &&
            shared_up_weights.allocate(
                context, label + " shared up weights",
                kHidden * kQ2BlocksPerWidth * sizeof(block_q2_0), error) &&
            shared_up_dequant.allocate(
                context, label + " shared up dequant",
                kWidth * kHidden * sizeof(std::uint16_t), error) &&
            shared_up_f32.allocate(
                context, label + " shared up output",
                row_count * kHidden * sizeof(float), error) &&
            shared_hidden.allocate(
                context, label + " shared SwiGLU",
                row_count * kHidden * sizeof(std::uint16_t), error) &&
            shared_down_weights.allocate(
                context, label + " shared down weights",
                kWidth * kQ2BlocksPerHidden * sizeof(block_q2_0), error) &&
            shared_down_dequant.allocate(
                context, label + " shared down dequant",
                kHidden * kWidth * sizeof(std::uint16_t), error) &&
            shared_output.allocate(
                context, label + " shared output",
                row_count * kWidth * sizeof(float), error) &&
            shared_gate_hi.allocate(
                context, label + " shared gate high",
                row_count * kWidth * sizeof(std::uint16_t), error) &&
            shared_gate_lo.allocate(
                context, label + " shared gate low",
                row_count * kWidth * sizeof(std::uint16_t), error) &&
            shared_gate_router_weights.allocate(
                context, label + " shared gate router weights",
                kWidth * sizeof(std::uint16_t), error) &&
            shared_gate_logits.allocate(
                context, label + " shared gate logits",
                row_count * sizeof(float), error) &&
            routed_gate_up_weights.allocate(
                context, label + " routed gate/up weights",
                2u * kQ2GateUpBytes, error) &&
            routed_down_weights.allocate(
                context, label + " routed down weights",
                kQ2DownBytes, error) &&
            addresses.allocate(
                context, label + " expert address table",
                expert_count * kAddressStride, error) &&
            gate_up_address_snapshot.allocate(
                context, label + " gate/up address snapshot",
                expert_count * kAddressStride, error) &&
            gate_up_hidden.allocate(
                context, label + " routed SwiGLU",
                entries * kHidden * sizeof(std::uint16_t), error) &&
            gate_up_status.allocate(
                context, label + " gate/up status",
                sizeof(std::uint32_t), error) &&
            down_address_snapshot.allocate(
                context, label + " down address snapshot",
                expert_count * kAddressStride, error) &&
            routed_parts.allocate(
                context, label + " routed parts",
                entries * kWidth * sizeof(float), error) &&
            down_status.allocate(
                context, label + " down status",
                sizeof(std::uint32_t), error) &&
            combined_output.allocate(
                context, label + " combined output",
                row_count * kWidth * sizeof(float), error);
    }

    PrefillMoeFullPipelineBuffers views(Context& context) {
        const auto readonly = [&](twin::GuardedBuffer& buffer) {
            return context.buffers().read_only(buffer.view());
        };
        weight_storage = {{
            {readonly(routed_gate_up_weights)},
            {readonly(routed_down_weights)},
        }};
        const auto router = PrefillGemmBF16X2Buffers{
            readonly(router_hi), readonly(router_lo), readonly(router_weights),
            logits.view()};
        const auto route = PrefillMoeRouteGatherBuffers{
            router,
            route_ids.view(),
            route_weights.view(),
            readonly(resident),
            readonly(mixed_h),
            histogram.view(),
            plan.view(),
            tiles.view(),
            source_rows.view(),
            destination_rows.view(),
            gathered_rows.view(),
        };
        const auto shared = PrefillMoeSharedFFNBuffers{
            readonly(mixed_h),
            readonly(shared_gate_weights),
            shared_gate_dequant.view(),
            shared_gate_f32.view(),
            readonly(shared_up_weights),
            shared_up_dequant.view(),
            shared_up_f32.view(),
            shared_hidden.view(),
            readonly(shared_down_weights),
            shared_down_dequant.view(),
            shared_output.view(),
            {readonly(shared_gate_hi), readonly(shared_gate_lo),
             readonly(shared_gate_router_weights), shared_gate_logits.view()},
        };
        const auto gate_up = PrefillMoeExpertTileBuffers{
            readonly(addresses),
            gate_up_address_snapshot.view(),
            weight_storage,
            {},
            readonly(gathered_rows),
            readonly(plan),
            readonly(tiles),
            gate_up_hidden.view(),
            gate_up_status.view(),
        };
        const auto down = PrefillMoeDownTileBuffers{
            readonly(addresses),
            down_address_snapshot.view(),
            weight_storage,
            {},
            readonly(gate_up_hidden),
            readonly(plan),
            readonly(tiles),
            readonly(destination_rows),
            routed_parts.view(),
            down_status.view(),
        };
        const auto combine = PrefillMoeCombineBuffers{
            readonly(routed_parts),
            readonly(route_weights),
            readonly(shared_output),
            readonly(shared_gate_logits),
            combined_output.view(),
        };
        return {route, shared, gate_up, down, combine};
    }

    std::vector<twin::GuardedBuffer*> all_buffers() {
        return {
            &router_hi, &router_lo, &router_weights, &logits, &route_ids,
            &route_weights, &resident, &mixed_h, &histogram, &plan, &tiles,
            &source_rows, &destination_rows, &gathered_rows,
            &shared_gate_weights, &shared_gate_dequant, &shared_gate_f32,
            &shared_up_weights, &shared_up_dequant, &shared_up_f32,
            &shared_hidden, &shared_down_weights, &shared_down_dequant,
            &shared_output, &shared_gate_hi, &shared_gate_lo,
            &shared_gate_router_weights, &shared_gate_logits,
            &routed_gate_up_weights, &routed_down_weights, &addresses,
            &gate_up_address_snapshot, &gate_up_hidden, &gate_up_status,
            &down_address_snapshot, &routed_parts, &down_status,
            &combined_output,
        };
    }
};

bool allocate_case(Context& context, CaseBuffers& buffers,
                   const std::string& label, std::uint32_t rows,
                   std::uint32_t experts, std::string& error) {
    const bool allocated =
        buffers.allocate(context, label, rows, experts, error);
    check(allocated, label + " allocate", error);
    return allocated;
}

void fill_q2_matrix(std::span<block_q2_0> blocks, std::size_t output_rows,
                    std::size_t input_width, std::uint32_t seed) {
    const std::size_t blocks_per_row = input_width / 64u;
    for (std::size_t row = 0u; row < output_rows; ++row) {
        for (std::size_t block_index = 0u;
             block_index < blocks_per_row; ++block_index) {
            block_q2_0& block = blocks[row * blocks_per_row + block_index];
            const std::uint32_t scale_code = static_cast<std::uint32_t>(
                (row * 3u + block_index * 5u + seed) % 5u);
            block.d = strata::kernels::f16_from_f32(
                0.0078125f * static_cast<float>(scale_code + 1u));
            for (std::size_t byte = 0u; byte < sizeof(block.qs); ++byte) {
                std::uint8_t packed = 0u;
                for (std::size_t lane = 0u; lane < 4u; ++lane) {
                    const std::uint32_t code = static_cast<std::uint32_t>(
                        (row * 7u + block_index * 11u + byte * 3u +
                         lane * 2u + seed) %
                        4u);
                    packed |= static_cast<std::uint8_t>(code << (lane * 2u));
                }
                block.qs[byte] = packed;
            }
        }
    }
}

float q2_value(std::span<const block_q2_0> blocks, std::size_t row,
               std::size_t k_width, std::size_t k) {
    const std::size_t blocks_per_row = k_width / 64u;
    const block_q2_0& block =
        blocks[row * blocks_per_row + k / 64u];
    const std::uint8_t packed = block.qs[(k % 64u) / 4u];
    const int code =
        static_cast<int>((packed >> (2u * (k % 4u))) & 3u) - 1;
    const float scaled =
        strata::kernels::f32_from_f16(block.d) * static_cast<float>(code);
    return strata::kernels::f32_from_f16(
        strata::kernels::f16_from_f32(scaled));
}

double q2_dot(std::span<const block_q2_0> blocks, std::size_t output_row,
              std::size_t k_width,
              std::span<const std::uint16_t> input) {
    double result = 0.0;
    for (std::size_t k = 0u; k < k_width; ++k) {
        result +=
            static_cast<double>(strata::kernels::f32_from_f16(input[k])) *
            static_cast<double>(q2_value(blocks, output_row, k_width, k));
    }
    return result;
}

std::uint16_t expected_half(float value) {
    if (std::fpclassify(value) == FP_SUBNORMAL) value = 0.0f;
    value = std::clamp(value, -65504.0f, 65504.0f);
    return strata::kernels::f16_from_f32(value);
}

std::uint16_t ordered_half(std::uint16_t bits) {
    return (bits & 0x8000u) != 0u
               ? static_cast<std::uint16_t>(~bits)
               : static_cast<std::uint16_t>(bits | 0x8000u);
}

std::uint16_t half_ulp_distance(std::uint16_t left,
                                std::uint16_t right) {
    const std::uint16_t a = ordered_half(left);
    const std::uint16_t b = ordered_half(right);
    return a > b ? static_cast<std::uint16_t>(a - b)
                 : static_cast<std::uint16_t>(b - a);
}

std::uint16_t expected_swiglu(double gate, double up) {
    const float gate_f32 = static_cast<float>(gate);
    const float up_f32 = static_cast<float>(up);
    const float silu = gate_f32 / (1.0f + std::exp(-gate_f32));
    return expected_half(silu * up_f32);
}

void initialize_case(CaseBuffers& b) {
    fill_q2_matrix(typed<block_q2_0>(b.shared_gate_weights),
                   kHidden, kWidth, 1u);
    fill_q2_matrix(typed<block_q2_0>(b.shared_up_weights),
                   kHidden, kWidth, 2u);
    fill_q2_matrix(typed<block_q2_0>(b.shared_down_weights),
                   kWidth, kHidden, 3u);
    auto routed_up = typed<block_q2_0>(b.routed_gate_up_weights);
    const std::size_t routed_up_block_offset =
        kQ2GateUpBytes / sizeof(block_q2_0);
    fill_q2_matrix(routed_up.first(routed_up_block_offset),
                   kHidden, kWidth, 4u);
    fill_q2_matrix(routed_up.subspan(routed_up_block_offset),
                   kHidden, kWidth, 5u);
    fill_q2_matrix(typed<block_q2_0>(b.routed_down_weights),
                   kWidth, kHidden, 6u);

    constexpr std::array<std::uint32_t, 6u> active_k{
        0u, 17u, 128u, 511u, 1024u, 2559u};
    constexpr std::array<float, 6u> high_values{
        0.5f, 0.75f, 0.125f, 1.0f, 0.375f, 0.25f};
    constexpr std::array<float, 6u> low_values{
        0.015625f, 0.0078125f, 0.0234375f,
        0.00390625f, 0.01171875f, 0.01953125f};
    constexpr std::array<float, 6u> weight_values{
        0.25f, 0.5f, 0.75f, 0.125f, 0.375f, 0.625f};
    auto x_hi = typed<std::uint16_t>(b.router_hi);
    auto x_lo = typed<std::uint16_t>(b.router_lo);
    auto router_weights = typed<std::uint16_t>(b.router_weights);
    std::fill(x_hi.begin(), x_hi.end(), 0u);
    std::fill(x_lo.begin(), x_lo.end(), 0u);
    std::fill(router_weights.begin(), router_weights.end(), 0u);
    for (std::uint32_t expert = 0u; expert < b.experts; ++expert) {
        const float score =
            expert < kTopK ? static_cast<float>(kTopK - expert) : -100.0f;
        for (std::size_t item = 0u; item < active_k.size(); ++item) {
            const std::size_t index =
                static_cast<std::size_t>(expert) * kWidth + active_k[item];
            router_weights[index] = strata::kernels::bf16_from_f32(
                score * weight_values[item]);
        }
    }
    auto mixed = typed<std::uint16_t>(b.mixed_h);
    auto shared_hi = typed<std::uint16_t>(b.shared_gate_hi);
    auto shared_lo = typed<std::uint16_t>(b.shared_gate_lo);
    auto shared_gate_weights =
        typed<std::uint16_t>(b.shared_gate_router_weights);
    for (std::uint32_t row = 0u; row < b.rows; ++row) {
        const float row_scale = 1.0f + static_cast<float>(row) * 0.125f;
        const std::size_t offset = static_cast<std::size_t>(row) * kWidth;
        for (std::size_t item = 0u; item < active_k.size(); ++item) {
            x_hi[offset + active_k[item]] = strata::kernels::bf16_from_f32(
                high_values[item] * row_scale);
            x_lo[offset + active_k[item]] = strata::kernels::bf16_from_f32(
                low_values[item] *
                (1.0f + static_cast<float>(row) * 0.25f));
        }
        for (std::size_t k = 0u; k < kWidth; ++k) {
            mixed[offset + k] = strata::kernels::f16_from_f32(
                0.125f + static_cast<float>(row) * 0.0625f +
                static_cast<float>(k % 13u) * 0.015625f);
            shared_hi[offset + k] = strata::kernels::bf16_from_f32(
                0.25f + static_cast<float>(row) * 0.0625f);
            shared_lo[offset + k] =
                strata::kernels::bf16_from_f32(0.015625f);
        }
    }
    for (std::size_t k = 0u; k < kWidth; ++k) {
        shared_gate_weights[k] =
            strata::kernels::bf16_from_f32(0.00390625f);
    }
    std::fill(b.resident.payload().begin(), b.resident.payload().end(), 0u);

    auto* addresses =
        reinterpret_cast<PrefillMoeExpertAddress*>(b.addresses.payload().data());
    const std::uint64_t base = b.routed_gate_up_weights.view().gpu_address;
    const std::uint64_t down = b.routed_down_weights.view().gpu_address;
    for (std::uint32_t expert = 0u; expert < b.experts; ++expert) {
        addresses[expert].role[0] = base;
        addresses[expert].role[1] = base + kQ2GateUpBytes;
        addresses[expert].role[2] = down;
    }

    for (twin::GuardedBuffer* output : {
             &b.logits, &b.route_ids, &b.route_weights, &b.histogram, &b.plan,
             &b.tiles, &b.source_rows, &b.destination_rows, &b.gathered_rows,
             &b.shared_gate_dequant, &b.shared_gate_f32,
             &b.shared_up_dequant, &b.shared_up_f32, &b.shared_hidden,
             &b.shared_down_dequant, &b.shared_output,
             &b.shared_gate_logits, &b.gate_up_address_snapshot,
             &b.gate_up_hidden, &b.gate_up_status,
             &b.down_address_snapshot, &b.routed_parts, &b.down_status,
             &b.combined_output}) {
        std::fill(output->payload().begin(), output->payload().end(), kPoison);
    }
}

PrefillMoeFullPipelineFormats formats() {
    return {
        {PrefillNativeWeightFormat::Q2_0,
         PrefillNativeWeightFormat::Q2_0,
         PrefillNativeWeightFormat::Q2_0},
        PrefillMoeGateUpFormat::Q2_0,
        PrefillMoeDownFormat::Q2_0,
    };
}

bool encode(Context& context, CommandBuffer& command,
            const PrefillMoeFullPipelineBuffers& buffers,
            std::uint32_t rows, std::uint32_t experts,
            PrefillBF16X2Mode projection_mode, bool caller_owned,
            std::string& error) {
    if (!caller_owned) {
        return encode_prefill_moe_full_mode_layer(
            context, command, rows, experts, 0u, projection_mode, formats(),
            buffers, error);
    }
    return Encoder::encode(
        context, command, "full Prefill MoE caller-owned test",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_moe_full_mode_layer(
                context, encoder, rows, experts, 0u, projection_mode,
                formats(), buffers, encode_error);
        },
        error);
}

std::vector<std::vector<std::uint8_t>> snapshot_outputs(CaseBuffers& b) {
    std::vector<std::vector<std::uint8_t>> snapshots;
    for (twin::GuardedBuffer* buffer : {
             &b.logits, &b.route_ids, &b.route_weights, &b.histogram, &b.plan,
             &b.tiles, &b.source_rows, &b.destination_rows, &b.gathered_rows,
             &b.shared_gate_dequant, &b.shared_gate_f32,
             &b.shared_up_dequant, &b.shared_up_f32, &b.shared_hidden,
             &b.shared_down_dequant, &b.shared_output,
             &b.shared_gate_logits, &b.gate_up_address_snapshot,
             &b.gate_up_hidden, &b.gate_up_status,
             &b.down_address_snapshot, &b.routed_parts, &b.down_status,
             &b.combined_output}) {
        const auto bytes = buffer->payload();
        snapshots.emplace_back(bytes.begin(), bytes.end());
    }
    return snapshots;
}

bool outputs_equal(CaseBuffers& b,
                   const std::vector<std::vector<std::uint8_t>>& snapshots) {
    std::size_t index = 0u;
    for (twin::GuardedBuffer* buffer : {
             &b.logits, &b.route_ids, &b.route_weights, &b.histogram, &b.plan,
             &b.tiles, &b.source_rows, &b.destination_rows, &b.gathered_rows,
             &b.shared_gate_dequant, &b.shared_gate_f32,
             &b.shared_up_dequant, &b.shared_up_f32, &b.shared_hidden,
             &b.shared_down_dequant, &b.shared_output,
             &b.shared_gate_logits, &b.gate_up_address_snapshot,
             &b.gate_up_hidden, &b.gate_up_status,
             &b.down_address_snapshot, &b.routed_parts, &b.down_status,
             &b.combined_output}) {
        const auto bytes = buffer->payload();
        if (index >= snapshots.size() ||
            !std::equal(snapshots[index].begin(), snapshots[index].end(),
                        bytes.begin())) {
            return false;
        }
        ++index;
    }
    return index == snapshots.size();
}

bool run_case(Context& context, std::uint32_t rows, std::uint32_t experts,
              PrefillBF16X2Mode projection_mode, bool caller_owned) {
    const std::string label =
        std::string("full-mode Prefill MoE T=") + std::to_string(rows) +
        " experts=" + std::to_string(experts) +
        (projection_mode == PrefillBF16X2Mode::except_gr ? " mode=2 " :
                                                           " mode=1 ") +
        (caller_owned ? "caller-owned" : "standalone");
    CaseBuffers b;
    std::string error;
    if (!allocate_case(context, b, label, rows, experts, error)) return false;
    initialize_case(b);
    const auto buffers = b.views(context);
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode(context, *command, buffers, rows, experts,
                                projection_mode, caller_owned, error);
    check(encoded, label + " encodes steps 1-13", error);
    const bool completed =
        encoded && command->commit(error) &&
        command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto ids = typed<std::int32_t>(b.route_ids);
    const auto weights = typed<float>(b.route_weights);
    const auto router_hi = typed<std::uint16_t>(b.router_hi);
    const auto router_lo = typed<std::uint16_t>(b.router_lo);
    const auto router_weight_bits =
        typed<std::uint16_t>(b.router_weights);
    const auto router_logits = typed<float>(b.logits);
    const auto shared_gate_hi =
        typed<std::uint16_t>(b.shared_gate_hi);
    const auto shared_gate_lo =
        typed<std::uint16_t>(b.shared_gate_lo);
    const auto shared_gate_weight_bits =
        typed<std::uint16_t>(b.shared_gate_router_weights);
    const auto shared_gate_logits = typed<float>(b.shared_gate_logits);
    std::vector<double> expected_logits(
        static_cast<std::size_t>(rows) * experts, 0.0);
    bool router_logits_match = true;
    bool route_weights_match = true;
    bool shared_gate_matches = true;
    bool route_matches = true;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t expert = 0u; expert < experts; ++expert) {
            double high_product = 0.0;
            double low_product = 0.0;
            for (std::size_t k = 0u; k < kWidth; ++k) {
                const std::size_t activation_index =
                    static_cast<std::size_t>(row) * kWidth + k;
                const std::size_t weight_index =
                    static_cast<std::size_t>(expert) * kWidth + k;
                const double weight = strata::kernels::f32_from_bf16(
                    router_weight_bits[weight_index]);
                high_product +=
                    static_cast<double>(strata::kernels::f32_from_bf16(
                        router_hi[activation_index])) *
                    weight;
                low_product +=
                    static_cast<double>(strata::kernels::f32_from_bf16(
                        router_lo[activation_index])) *
                    weight;
            }
            const std::size_t logit_index =
                static_cast<std::size_t>(row) * experts + expert;
            expected_logits[logit_index] = high_product + low_product;
            router_logits_match &=
                std::isfinite(router_logits[logit_index]) &&
                std::abs(static_cast<double>(router_logits[logit_index]) -
                         expected_logits[logit_index]) <= 2.0e-4;
        }

        float weight_sum = 0.0f;
        double exponential_sum = 0.0;
        std::array<double, kTopK> exponentials{};
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            exponentials[rank] = std::exp(
                expected_logits[static_cast<std::size_t>(row) * experts +
                                rank]);
            exponential_sum += exponentials[rank];
        }
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kTopK + rank;
            route_matches &=
                ids[index] == static_cast<std::int32_t>(rank);
            route_matches &=
                std::isfinite(weights[index]) && weights[index] > 0.0f;
            route_weights_match &=
                std::abs(static_cast<double>(weights[index]) -
                         exponentials[rank] / exponential_sum) <= 3.0e-6;
            weight_sum += weights[index];
        }
        route_matches &= std::abs(weight_sum - 1.0f) <= 3.0e-6f;

        double expected_shared_gate = 0.0;
        for (std::size_t k = 0u; k < kWidth; ++k) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kWidth + k;
            const double gate_weight = strata::kernels::f32_from_bf16(
                shared_gate_weight_bits[k]);
            expected_shared_gate +=
                static_cast<double>(strata::kernels::f32_from_bf16(
                    shared_gate_hi[index])) *
                    gate_weight +
                static_cast<double>(strata::kernels::f32_from_bf16(
                    shared_gate_lo[index])) *
                    gate_weight;
        }
        shared_gate_matches &=
            std::isfinite(shared_gate_logits[row]) &&
            std::abs(static_cast<double>(shared_gate_logits[row]) -
                     expected_shared_gate) <=
                1.0e-3 + 1.0e-5 * std::abs(expected_shared_gate);
    }
    check(router_logits_match,
          label + " BF16X2 router logits match independent FP64 oracle");
    check(route_weights_match,
          label + " divide-mode route weights match independent oracle");
    check(shared_gate_matches,
          label + " BF16X2 shared gate matches independent FP64 oracle");
    check(route_matches, label + " top-10 routes and normalized weights");

    const auto* plan =
        reinterpret_cast<const PfPlan*>(b.plan.payload().data());
    const bool plan_matches =
        plan->n_groups == kTopK &&
        plan->n_entries == rows * kTopK &&
        plan->n_tiles == kTopK && plan->bad == 0u;
    check(plan_matches, label + " grouped PfPlan has ten valid expert tiles");
    const bool expert_status =
        typed<std::uint32_t>(b.gate_up_status)[0] == 0u &&
        typed<std::uint32_t>(b.down_status)[0] == 0u;
    check(expert_status, label + " routed tile status words are clear");

    const auto mixed = typed<std::uint16_t>(b.mixed_h);
    const auto shared_gate_weights =
        typed<block_q2_0>(b.shared_gate_weights);
    const auto shared_up_weights = typed<block_q2_0>(b.shared_up_weights);
    const auto shared_down_weights =
        typed<block_q2_0>(b.shared_down_weights);
    const auto routed_gate_up_weights =
        typed<block_q2_0>(b.routed_gate_up_weights);
    const auto routed_down_weights = typed<block_q2_0>(b.routed_down_weights);
    const std::size_t routed_up_block_offset =
        kQ2GateUpBytes / sizeof(block_q2_0);
    const auto routed_gate_weights =
        routed_gate_up_weights.first(routed_up_block_offset);
    const auto routed_up_weights =
        routed_gate_up_weights.subspan(routed_up_block_offset);
    const auto shared_gate_actual = typed<float>(b.shared_gate_f32);
    const auto shared_up_actual = typed<float>(b.shared_up_f32);
    const auto shared_hidden_actual = typed<std::uint16_t>(b.shared_hidden);
    const auto shared_output_actual = typed<float>(b.shared_output);
    const auto routed_hidden_actual = typed<std::uint16_t>(b.gate_up_hidden);
    const auto gathered_actual = typed<std::uint16_t>(b.gathered_rows);
    const auto source_rows_actual = typed<std::int32_t>(b.source_rows);
    const auto destinations_actual =
        typed<std::int32_t>(b.destination_rows);
    const auto parts = typed<float>(b.routed_parts);
    bool shared_gate_up_matches = true;
    bool down_projection_matches = true;
    bool route_destination_matches = true;
    bool gathered_rows_match = true;
    bool hidden_oracle_matches = true;
    const std::size_t active_entries =
        static_cast<std::size_t>(rows) * kTopK;
    std::vector<std::vector<std::uint16_t>> routed_hidden_oracle(
        rows, std::vector<std::uint16_t>(kHidden));
    std::vector<std::vector<double>> routed_down_oracle(
        rows, std::vector<double>(kWidth));
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const auto input =
            mixed.subspan(static_cast<std::size_t>(row) * kWidth, kWidth);
        const auto shared_hidden = shared_hidden_actual.subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        for (std::size_t column = 0u; column < kHidden; ++column) {
            const double shared_gate_expected =
                q2_dot(shared_gate_weights, column, kWidth, input);
            const double shared_up_expected =
                q2_dot(shared_up_weights, column, kWidth, input);
            const double routed_gate_expected =
                q2_dot(routed_gate_weights, column, kWidth, input);
            const double routed_up_expected =
                q2_dot(routed_up_weights, column, kWidth, input);
            const std::uint16_t expected_shared = expected_swiglu(
                shared_gate_expected, shared_up_expected);
            const std::uint16_t expected_routed = expected_swiglu(
                routed_gate_expected, routed_up_expected);
            const std::size_t hidden_index =
                static_cast<std::size_t>(row) * kHidden + column;
            shared_gate_up_matches &=
                std::abs(static_cast<double>(
                             shared_gate_actual[hidden_index]) -
                         shared_gate_expected) <=
                    kProjectionAbsoluteTolerance +
                        kProjectionRelativeTolerance *
                            std::abs(shared_gate_expected) &&
                std::abs(static_cast<double>(
                             shared_up_actual[hidden_index]) -
                         shared_up_expected) <=
                    kProjectionAbsoluteTolerance +
                        kProjectionRelativeTolerance *
                            std::abs(shared_up_expected);
            hidden_oracle_matches &=
                half_ulp_distance(shared_hidden[column],
                                  expected_shared) <= 2u;
            routed_hidden_oracle[row][column] = expected_routed;
        }
        const auto routed_hidden =
            std::span<const std::uint16_t>(routed_hidden_oracle[row]);
        for (std::size_t column = 0u; column < kWidth; ++column) {
            const double expected_shared_down =
                q2_dot(shared_down_weights, column, kHidden, shared_hidden);
            const double expected_routed_down =
                q2_dot(routed_down_weights, column, kHidden, routed_hidden);
            const std::size_t output_index =
                static_cast<std::size_t>(row) * kWidth + column;
            down_projection_matches &=
                std::abs(static_cast<double>(
                             shared_output_actual[output_index]) -
                         expected_shared_down) <=
                    kProjectionAbsoluteTolerance +
                        kProjectionRelativeTolerance *
                            std::abs(expected_shared_down);
            routed_down_oracle[row][column] = expected_routed_down;
        }
    }
    std::size_t grouped_entries_checked = 0u;
    bool group_ranges_valid = plan->n_groups == kTopK;
    if (group_ranges_valid) {
        group_ranges_valid = plan->grp_start[0] == 0u &&
                             static_cast<std::size_t>(
                                 plan->grp_start[plan->n_groups]) ==
                                 active_entries;
    }
    for (std::size_t group = 0u;
         group < plan->n_groups && group < kTopK; ++group) {
        const std::size_t group_begin =
            static_cast<std::size_t>(plan->grp_start[group]);
        const std::size_t group_end =
            static_cast<std::size_t>(plan->grp_start[group + 1u]);
        if (group_begin > group_end || group_end > active_entries) {
            group_ranges_valid = false;
            continue;
        }
        for (std::size_t grouped_index = group_begin;
             grouped_index < group_end; ++grouped_index) {
            ++grouped_entries_checked;
            const std::int32_t source_row = source_rows_actual[grouped_index];
            const std::int32_t destination =
                destinations_actual[grouped_index];
            const bool valid_row =
                source_row >= 0 &&
                static_cast<std::size_t>(source_row) < rows;
            const bool valid_destination =
                destination >= 0 &&
                static_cast<std::size_t>(destination) < active_entries;
            route_destination_matches &=
                valid_row && valid_destination;
            if (!valid_row || !valid_destination) continue;
            const std::size_t row = static_cast<std::size_t>(source_row);
            const std::size_t route_index =
                static_cast<std::size_t>(destination);
            route_destination_matches &=
                ids[route_index] == plan->grp_e[group] &&
                route_index / kTopK == row;
            const auto gathered_row = gathered_actual.subspan(
                grouped_index * kWidth, kWidth);
            const auto mixed_row = mixed.subspan(row * kWidth, kWidth);
            gathered_rows_match &=
                std::equal(gathered_row.begin(), gathered_row.end(),
                           mixed_row.begin());
            const auto routed_hidden = routed_hidden_actual.subspan(
                grouped_index * kHidden, kHidden);
            for (std::size_t column = 0u; column < kHidden; ++column) {
                hidden_oracle_matches &=
                    half_ulp_distance(routed_hidden[column],
                                      routed_hidden_oracle[row][column]) <=
                    2u;
            }
            for (std::size_t column = 0u; column < kWidth; ++column) {
                const double expected = routed_down_oracle[row][column];
                down_projection_matches &=
                    std::abs(static_cast<double>(
                                 parts[route_index * kWidth + column]) -
                             expected) <=
                    kProjectionAbsoluteTolerance +
                        kProjectionRelativeTolerance * std::abs(expected);
            }
        }
    }
    group_ranges_valid &= grouped_entries_checked == active_entries;
    route_destination_matches &= group_ranges_valid;
    check(shared_gate_up_matches,
          label + " shared and routed gate/up match independent Q2_0 dots");
    check(down_projection_matches,
          label + " shared and routed down match independent Q2_0 dots");
    check(route_destination_matches,
          label + " grouped destinations map to original expert routes");
    check(gathered_rows_match,
          label + " every gathered row matches its source token");
    check(hidden_oracle_matches,
          label + " shared and routed F16 SwiGLU match scalar oracles");

    const auto shared = typed<float>(b.shared_output);
    const auto gate = typed<float>(b.shared_gate_logits);
    const auto actual = typed<float>(b.combined_output);
    bool combine_matches = true;
    bool finite_nonzero = false;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * kWidth;
        const float sigmoid =
            1.0f / (1.0f + std::exp(-gate[row]));
        for (std::size_t column = 0u; column < kWidth; ++column) {
            float routed_sum = 0.0f;
            for (std::size_t rank = 0u; rank < kTopK; ++rank) {
                const std::size_t route =
                    static_cast<std::size_t>(row) * kTopK + rank;
                routed_sum = std::fma(
                    weights[route],
                    parts[route * kWidth + column], routed_sum);
            }
            const float expected = std::fma(
                shared[row_offset + column], sigmoid, routed_sum);
            combine_matches &=
                std::isfinite(actual[row_offset + column]) &&
                std::abs(actual[row_offset + column] - expected) <=
                    kCombineTolerance *
                        std::max(1.0f, std::abs(expected));
            finite_nonzero |=
                std::abs(actual[row_offset + column]) > 1.0e-5f;
        }
    }
    check(combine_matches,
          label + " weighted routed/shared output matches scalar oracle");
    check(finite_nonzero,
          label + " produces finite, nonzero MoE output");

    const auto all = b.all_buffers();
    const bool guards_match = std::all_of(
        all.begin(), all.end(),
        [](const twin::GuardedBuffer* buffer) {
            return guards_intact(*buffer);
        });
    check(guards_match, label + " all allocation guards remain intact");
    const std::size_t gathered_active_bytes =
        static_cast<std::size_t>(rows) * kTopK * kWidth *
        sizeof(std::uint16_t);
    const auto gathered_payload = b.gathered_rows.payload();
    const bool gathered_tail_unchanged =
        std::all_of(
            gathered_payload.begin() +
                static_cast<std::ptrdiff_t>(gathered_active_bytes),
            gathered_payload.end(),
            [](std::uint8_t byte) { return byte == kPoison; });
    check(gathered_tail_unchanged,
          label + " gather leaves its extra poisoned tail untouched");

    const auto before_alias = snapshot_outputs(b);
    auto alias_command = context.make_command_buffer(
        label + " cross-stage output alias", error);
    if (!alias_command) {
        check(false, label + " alias command buffer", error);
        return false;
    }
    auto alias_buffers = buffers;
    alias_buffers.combine.output = b.logits.view();
    const bool alias_rejected =
        !encode_prefill_moe_full_mode_layer(
            context, *alias_command, rows, experts, 0u, projection_mode,
            formats(), alias_buffers, error);
    check(alias_rejected,
          label + " rejects combined output aliasing router logits");
    check(alias_command->unsafe_native_handle() == nullptr &&
              outputs_equal(b, before_alias),
          label + " alias rejection preserves all existing outputs");

    auto shifted_alias_command = context.make_command_buffer(
        label + " shifted dataflow alias", error);
    if (!shifted_alias_command) {
        check(false, label + " shifted alias command buffer", error);
        return false;
    }
    auto shifted_alias_buffers = buffers;
    shifted_alias_buffers.routed_gate_up.gathered_rows =
        context.buffers()
            .read_only(b.gathered_rows.view())
            .slice(sizeof(std::uint16_t), gathered_active_bytes);
    const bool shifted_alias_rejected =
        !encode_prefill_moe_full_mode_layer(
            context, *shifted_alias_command, rows, experts, 0u,
            projection_mode, formats(), shifted_alias_buffers, error) &&
        error.find("route.gathered overlaps active gate_up.gathered") !=
            std::string::npos;
    check(shifted_alias_rejected,
          label + " rejects shifted gathered-row dataflow aliases");
    check(shifted_alias_command->unsafe_native_handle() == nullptr &&
              outputs_equal(b, before_alias),
          label + " shifted alias rejection preserves every active output");

    auto bad_mode_command =
        context.make_command_buffer(label + " mode zero", error);
    if (!bad_mode_command) {
        check(false, label + " mode command buffer", error);
        return false;
    }
    const bool mode_rejected = !encode_prefill_moe_full_mode_layer(
        context, *bad_mode_command, rows, experts, 0u,
        PrefillBF16X2Mode::off, formats(), buffers, error);
    check(mode_rejected &&
              bad_mode_command->unsafe_native_handle() == nullptr,
          label + " rejects mode 0 before encoding");

    const auto rejects_formats = [&](const std::string& format_label,
                                     const PrefillMoeFullPipelineFormats&
                                         invalid_formats) {
        auto bad_format_command =
            context.make_command_buffer(label + " " + format_label, error);
        if (!bad_format_command) {
            check(false, label + " " + format_label + " command buffer",
                  error);
            return false;
        }
        const bool rejected = !encode_prefill_moe_full_mode_layer(
            context, *bad_format_command, rows, experts, 0u,
            projection_mode, invalid_formats, buffers, error);
        const bool discarded =
            bad_format_command->unsafe_native_handle() == nullptr;
        check(rejected && discarded,
              label + " rejects unsupported " + format_label +
                  " before encoding");
        check(discarded && outputs_equal(b, before_alias),
              label + " unsupported " + format_label +
                  " preserves every output");
        return rejected && discarded;
    };
    auto bad_gate_up_formats = formats();
    bad_gate_up_formats.routed_gate_up =
        static_cast<PrefillMoeGateUpFormat>(0xffu);
    const bool bad_gate_up_rejected =
        rejects_formats("gate/up format", bad_gate_up_formats);
    auto bad_down_formats = formats();
    bad_down_formats.routed_down =
        static_cast<PrefillMoeDownFormat>(0xffu);
    const bool bad_down_rejected =
        rejects_formats("down format", bad_down_formats);
    auto bad_shared_formats = formats();
    bad_shared_formats.shared.gate =
        static_cast<PrefillNativeWeightFormat>(0xffu);
    const bool bad_shared_rejected =
        rejects_formats("shared format", bad_shared_formats);

    auto* addresses =
        reinterpret_cast<PrefillMoeExpertAddress*>(b.addresses.payload().data());
    const std::uint64_t original_gate = addresses[0].role[0];
    addresses[0].role[0] = original_gate + 1u;
    const auto before_late_failure = snapshot_outputs(b);
    auto late_command =
        context.make_command_buffer(label + " late expert failure", error);
    if (!late_command) {
        check(false, label + " late failure command buffer", error);
        return false;
    }
    const bool late_rejected =
        !encode(context, *late_command, buffers, rows, experts,
                projection_mode, caller_owned, error);
    check(late_rejected &&
              error.find("misaligned") != std::string::npos,
          label + " rejects a malformed expert address after route/shared");
    const bool discarded =
        late_command->unsafe_native_handle() == nullptr;
    check(discarded,
          label + " late failure discards the whole command segment");
    check(discarded && outputs_equal(b, before_late_failure),
          label + " late rollback preserves every output and scratch byte");
    addresses[0].role[0] = original_gate;

    return route_matches && router_logits_match && route_weights_match &&
           shared_gate_matches && plan_matches && expert_status &&
           combine_matches && finite_nonzero && guards_match &&
           gathered_tail_unchanged && gathered_rows_match &&
           route_destination_matches && alias_rejected &&
           shifted_alias_rejected && mode_rejected && bad_gate_up_rejected &&
           bad_down_rejected && bad_shared_rejected && late_rejected &&
           discarded;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(
            stderr,
            "usage: metal_prefill_moe_full_pipeline_test strata.metallib\n");
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
    passed &= run_case(*context, 1u, 256u,
                       PrefillBF16X2Mode::all_projections, false);
    passed &= run_case(*context, 7u, 512u,
                       PrefillBF16X2Mode::except_gr, true);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d full Prefill MoE test(s) failed\n",
                     g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all full-mode Prefill MoE layer tests passed");
    return 0;
}
