// Ordered full-mode Prefill MoE layer composition.
#include "strata/metal/prefill_moe_full_pipeline.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_moe_expert_dispatch.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::size_t kWidth = 2560u;
constexpr std::size_t kHidden = 640u;
constexpr std::size_t kTopK = 10u;
constexpr std::size_t kAddressStride = 3u * sizeof(std::uint64_t);
constexpr std::size_t kPlanStride = kPrefillMoePlanStrideBytes;
constexpr std::size_t kPlanBytes = kPrefillMoePlanBytes;

enum class Stage : std::uint8_t {
    route,
    shared,
    gate_up,
    down,
    combine,
};

struct ActiveRange {
    Stage stage = Stage::route;
    const void* cpu = nullptr;
    std::uint64_t gpu = 0u;
    std::size_t bytes = 0u;
    bool writable = false;
    std::string_view name;
};

bool checked_product(std::initializer_list<std::size_t> factors,
                     std::size_t& result) {
    result = 1u;
    for (const std::size_t factor : factors) {
        if (factor != 0u &&
            result > std::numeric_limits<std::size_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

std::size_t round_up_div(std::size_t value, std::size_t divisor) {
    return value / divisor +
           static_cast<std::size_t>(value % divisor != 0u);
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0u || right == 0u || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool overlaps(const ActiveRange& left, const ActiveRange& right) {
    const bool cpu_overlap =
        left.cpu != nullptr && right.cpu != nullptr &&
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.cpu), left.bytes,
            reinterpret_cast<std::uintptr_t>(right.cpu), right.bytes);
    const bool gpu_overlap =
        left.gpu != 0u && right.gpu != 0u &&
        address_ranges_overlap(left.gpu, left.bytes, right.gpu,
                               right.bytes);
    return cpu_overlap || gpu_overlap;
}

template <class Address>
bool address_range_contains(Address outer, std::size_t outer_bytes,
                            Address inner, std::size_t inner_bytes) {
    return outer != 0u && inner == outer && inner_bytes <= outer_bytes;
}

bool contains(const ActiveRange& outer, const ActiveRange& inner) {
    bool compared_address_space = false;
    bool fits = true;
    if (outer.cpu != nullptr && inner.cpu != nullptr) {
        compared_address_space = true;
        fits &= address_range_contains(
            reinterpret_cast<std::uintptr_t>(outer.cpu), outer.bytes,
            reinterpret_cast<std::uintptr_t>(inner.cpu), inner.bytes);
    }
    if (outer.gpu != 0u && inner.gpu != 0u) {
        compared_address_space = true;
        fits &= address_range_contains(outer.gpu, outer.bytes, inner.gpu,
                                       inner.bytes);
    }
    return compared_address_space && fits;
}

bool expected_dataflow(const ActiveRange& writer,
                       const ActiveRange& reader) {
    if (!writer.writable || reader.writable ||
        static_cast<unsigned>(writer.stage) >=
            static_cast<unsigned>(reader.stage)) {
        return false;
    }
    if (writer.stage == Stage::route) {
        if (writer.name == "route.gathered") {
            return reader.stage == Stage::gate_up &&
                   reader.name == "gate_up.gathered";
        }
        if (writer.name == "route.plan") {
            return (reader.stage == Stage::gate_up &&
                    reader.name == "gate_up.plan") ||
                   (reader.stage == Stage::down &&
                    reader.name == "down.plan");
        }
        if (writer.name == "route.tiles") {
            return (reader.stage == Stage::gate_up &&
                    reader.name == "gate_up.tiles") ||
                   (reader.stage == Stage::down &&
                    reader.name == "down.tiles");
        }
        if (writer.name == "route.destinations") {
            return reader.stage == Stage::down &&
                   reader.name == "down.destinations";
        }
        if (writer.name == "route.weights") {
            return reader.stage == Stage::combine &&
                   reader.name == "combine.weights";
        }
    }
    if (writer.stage == Stage::shared) {
        if (writer.name == "shared.output") {
            return reader.stage == Stage::combine &&
                   reader.name == "combine.shared";
        }
        if (writer.name == "shared.gate") {
            return reader.stage == Stage::combine &&
                   reader.name == "combine.gate";
        }
    }
    if (writer.stage == Stage::gate_up &&
        writer.name == "gate_up.hidden") {
        return reader.stage == Stage::down &&
               reader.name == "down.hidden";
    }
    if (writer.stage == Stage::down && writer.name == "down.parts") {
        return reader.stage == Stage::combine &&
               reader.name == "combine.parts";
    }
    return false;
}

bool add_range(Context& context, ReadOnlyBufferRef view,
               std::size_t active_bytes, Stage stage, std::string_view name,
               std::vector<ActiveRange>& ranges, std::string& error) {
    if (!view || !context.buffers().owns(view) || active_bytes == 0u ||
        view.length() < active_bytes) {
        error = "full-mode Prefill MoE " + std::string(name) +
                " input is foreign or undersized";
        return false;
    }
    ranges.push_back({stage, view.contents(), view.gpu_address(),
                      active_bytes, false, name});
    return true;
}

bool add_range(Context& context, BufferRef view,
               std::size_t active_bytes, Stage stage, std::string_view name,
               std::vector<ActiveRange>& ranges, std::string& error) {
    if (!view || !context.buffers().owns(view) || active_bytes == 0u ||
        view.length < active_bytes) {
        error = "full-mode Prefill MoE " + std::string(name) +
                " output is foreign or undersized";
        return false;
    }
    ranges.push_back({stage, view.contents, view.gpu_address, active_bytes,
                      true, name});
    return true;
}

bool add_shared_ranges(
    Context& context, std::uint32_t rows,
    const PrefillMoeSharedFFNFormats& formats,
    const PrefillMoeSharedFFNBuffers& buffers,
    std::vector<ActiveRange>& ranges, std::string& error) {
    const PrefillGemmMPPParams gate_params{
        rows, static_cast<std::uint32_t>(kHidden),
        static_cast<std::uint32_t>(kWidth),
        static_cast<std::uint32_t>(kHidden), 0u, false};
    const PrefillGemmMPPParams down_params{
        rows, static_cast<std::uint32_t>(kWidth),
        static_cast<std::uint32_t>(kHidden),
        static_cast<std::uint32_t>(kWidth), 0u, false};
    PrefillNativeProjectionBufferSizes gate_sizes;
    PrefillNativeProjectionBufferSizes up_sizes;
    PrefillNativeProjectionBufferSizes down_sizes;
    if (!prefill_native_projection_buffer_sizes(
            gate_params, formats.gate, gate_sizes, error) ||
        !prefill_native_projection_buffer_sizes(
            gate_params, formats.up, up_sizes, error) ||
        !prefill_native_projection_buffer_sizes(
            down_params, formats.down, down_sizes, error)) {
        return false;
    }

    std::size_t mixed_bytes = 0u;
    std::size_t gate_up_output_bytes = 0u;
    std::size_t hidden_bytes = 0u;
    std::size_t shared_output_bytes = 0u;
    std::size_t gate_router_weight_bytes = 0u;
    std::size_t gate_logits_bytes = 0u;
    if (!checked_product({rows, kWidth, sizeof(std::uint16_t)},
                         mixed_bytes) ||
        !checked_product({rows, kHidden, sizeof(float)},
                         gate_up_output_bytes) ||
        !checked_product({rows, kHidden, sizeof(std::uint16_t)},
                         hidden_bytes) ||
        !checked_product({rows, kWidth, sizeof(float)},
                         shared_output_bytes) ||
        !checked_product({kWidth, sizeof(std::uint16_t)},
                         gate_router_weight_bytes) ||
        !checked_product({rows, sizeof(float)}, gate_logits_bytes)) {
        error = "full-mode Prefill MoE shared extent overflows size_t";
        return false;
    }

    return
        add_range(context, buffers.mixed_h, mixed_bytes, Stage::shared,
                  "shared.mixed", ranges, error) &&
        add_range(context, buffers.gate_weights, gate_sizes.weights_bytes,
                  Stage::shared, "shared.gate_weights", ranges, error) &&
        add_range(context, buffers.gate_dequantized_f16,
                  gate_sizes.dequantized_f16_bytes, Stage::shared,
                  "shared.gate_dequant", ranges, error) &&
        add_range(context, buffers.gate_f32, gate_up_output_bytes,
                  Stage::shared, "shared.gate_f32", ranges, error) &&
        add_range(context, buffers.up_weights, up_sizes.weights_bytes,
                  Stage::shared, "shared.up_weights", ranges, error) &&
        add_range(context, buffers.up_dequantized_f16,
                  up_sizes.dequantized_f16_bytes, Stage::shared,
                  "shared.up_dequant", ranges, error) &&
        add_range(context, buffers.up_f32, gate_up_output_bytes,
                  Stage::shared, "shared.up_f32", ranges, error) &&
        add_range(context, buffers.hidden_f16, hidden_bytes, Stage::shared,
                  "shared.hidden", ranges, error) &&
        add_range(context, buffers.down_weights, down_sizes.weights_bytes,
                  Stage::shared, "shared.down_weights", ranges, error) &&
        add_range(context, buffers.down_dequantized_f16,
                  down_sizes.dequantized_f16_bytes, Stage::shared,
                  "shared.down_dequant", ranges, error) &&
        add_range(context, buffers.shared_f32, shared_output_bytes,
                  Stage::shared, "shared.output", ranges, error) &&
        add_range(context, buffers.router_gate.x_hi, mixed_bytes,
                  Stage::shared, "shared.gate_hi", ranges, error) &&
        add_range(context, buffers.router_gate.x_lo, mixed_bytes,
                  Stage::shared, "shared.gate_lo", ranges, error) &&
        add_range(context, buffers.router_gate.w,
                  gate_router_weight_bytes, Stage::shared,
                  "shared.gate_weights_bf16", ranges, error) &&
        add_range(context, buffers.router_gate.y, gate_logits_bytes,
                  Stage::shared, "shared.gate", ranges, error);
}

bool collect_active_ranges(
    Context& context, std::uint32_t rows, std::uint32_t n_experts,
    std::uint32_t layer, const PrefillMoeFullPipelineFormats& formats,
    const PrefillMoeFullPipelineBuffers& buffers,
    std::vector<ActiveRange>& ranges, std::string& error) {
    const std::size_t entries = static_cast<std::size_t>(rows) * kTopK;
    std::size_t router_input_bytes = 0u;
    std::size_t router_weight_bytes = 0u;
    std::size_t router_logits_bytes = 0u;
    std::size_t route_bytes = 0u;
    std::size_t route_weight_bytes = 0u;
    std::size_t mixed_bytes = 0u;
    std::size_t gathered_bytes = 0u;
    std::size_t histogram_bytes = 0u;
    std::size_t route_tile_bytes = 0u;
    std::size_t expert_tile_bytes = 0u;
    std::size_t destination_bytes = 0u;
    std::size_t gate_up_hidden_bytes = 0u;
    std::size_t down_parts_bytes = 0u;
    std::size_t address_table_bytes = 0u;
    std::size_t address_row_bytes = 0u;
    const std::size_t histogram_tiles = round_up_div(entries, 256u);
    const std::size_t route_tile_capacity =
        round_up_div(entries, 64u) + kPrefillMoeHistogramExperts;
    const std::size_t expert_tile_capacity =
        round_up_div(entries, 64u) +
        std::min<std::size_t>(n_experts, entries);
    if (!checked_product({rows, kWidth, sizeof(std::uint16_t)},
                         router_input_bytes) ||
        !checked_product({n_experts, kWidth, sizeof(std::uint16_t)},
                         router_weight_bytes) ||
        !checked_product({rows, n_experts, sizeof(float)},
                         router_logits_bytes) ||
        !checked_product({entries, sizeof(std::int32_t)}, route_bytes) ||
        !checked_product({entries, sizeof(float)}, route_weight_bytes) ||
        !checked_product({rows, kWidth, sizeof(std::uint16_t)}, mixed_bytes) ||
        !checked_product({entries, kWidth, sizeof(std::uint16_t)},
                         gathered_bytes) ||
        !checked_product({histogram_tiles,
                          kPrefillMoeHistogramExperts,
                          sizeof(std::uint32_t)},
                         histogram_bytes) ||
        !checked_product({route_tile_capacity, 2u * sizeof(std::uint32_t)},
                         route_tile_bytes) ||
        !checked_product({expert_tile_capacity, 2u * sizeof(std::uint32_t)},
                         expert_tile_bytes) ||
        !checked_product({entries, sizeof(std::uint32_t)},
                         destination_bytes) ||
        !checked_product({entries, kHidden, sizeof(std::uint16_t)},
                         gate_up_hidden_bytes) ||
        !checked_product({entries, kWidth, sizeof(float)},
                         down_parts_bytes) ||
        !checked_product({static_cast<std::size_t>(layer) + 1u,
                          n_experts, kAddressStride},
                         address_table_bytes) ||
        !checked_product({n_experts, kAddressStride}, address_row_bytes)) {
        error = "full-mode Prefill MoE extent overflows size_t";
        return false;
    }

    const auto& route = buffers.route_gather;
    if (!add_range(context, route.router.x_hi, router_input_bytes,
                   Stage::route, "route.x_hi", ranges, error) ||
        !add_range(context, route.router.x_lo, router_input_bytes,
                   Stage::route, "route.x_lo", ranges, error) ||
        !add_range(context, route.router.w, router_weight_bytes,
                   Stage::route, "route.router_weights", ranges, error) ||
        !add_range(context, route.router.y, router_logits_bytes,
                   Stage::route, "route.logits", ranges, error) ||
        !add_range(context, route.route_ids, route_bytes, Stage::route,
                   "route.ids", ranges, error) ||
        !add_range(context, route.route_weights, route_weight_bytes,
                   Stage::route, "route.weights", ranges, error) ||
        !add_range(context, route.resident, sizeof(std::int32_t),
                   Stage::route, "route.resident", ranges, error) ||
        !add_range(context, route.mixed_h, mixed_bytes, Stage::route,
                   "route.mixed", ranges, error) ||
        !add_range(context, route.histogram, histogram_bytes, Stage::route,
                   "route.histogram", ranges, error) ||
        !add_range(context, route.plan, kPlanStride, Stage::route,
                   "route.plan", ranges, error) ||
        !add_range(context, route.tiles, route_tile_bytes, Stage::route,
                   "route.tiles", ranges, error) ||
        !add_range(context, route.source_rows, route_bytes, Stage::route,
                   "route.sources", ranges, error) ||
        !add_range(context, route.destination_rows, destination_bytes,
                   Stage::route, "route.destinations", ranges, error) ||
        !add_range(context, route.gathered_rows, gathered_bytes,
                   Stage::route, "route.gathered", ranges, error)) {
        return false;
    }

    if (!add_shared_ranges(context, rows, formats.shared, buffers.shared,
                           ranges, error)) {
        return false;
    }

    const auto& gate_up = buffers.routed_gate_up;
    const auto& down = buffers.routed_down;
    if (gate_up.tiered_leases || down.tiered_leases ||
        gate_up.weight_storage.empty() || down.weight_storage.empty()) {
        error = "full-mode Prefill MoE requires full-residency weight storage "
                "without tiered leases";
        return false;
    }
    if (!add_range(context, gate_up.addresses, address_table_bytes,
                   Stage::gate_up, "gate_up.addresses", ranges, error) ||
        !add_range(context, gate_up.gathered_rows, gathered_bytes,
                   Stage::gate_up, "gate_up.gathered", ranges, error) ||
        !add_range(context, gate_up.plan, kPlanBytes, Stage::gate_up,
                   "gate_up.plan", ranges, error) ||
        !add_range(context, gate_up.tiles, expert_tile_bytes,
                   Stage::gate_up, "gate_up.tiles", ranges, error) ||
        !add_range(context, gate_up.address_snapshot, address_row_bytes,
                   Stage::gate_up, "gate_up.address_snapshot", ranges,
                   error) ||
        !add_range(context, gate_up.hidden, gate_up_hidden_bytes,
                   Stage::gate_up, "gate_up.hidden", ranges, error) ||
        !add_range(context, gate_up.status, sizeof(std::uint32_t),
                   Stage::gate_up, "gate_up.status", ranges, error)) {
        return false;
    }
    for (const auto& storage : gate_up.weight_storage) {
        if (storage.buffer.is_tiered_slot_arena()) {
            error = "full-mode Prefill MoE cannot use tiered expert slots";
            return false;
        }
        if (!add_range(context, storage.buffer, storage.buffer.length(),
                       Stage::gate_up, "gate_up.weight_storage", ranges,
                       error)) {
            return false;
        }
    }

    if (!add_range(context, down.addresses, address_table_bytes,
                   Stage::down, "down.addresses", ranges, error) ||
        !add_range(context, down.hidden, gate_up_hidden_bytes, Stage::down,
                   "down.hidden", ranges, error) ||
        !add_range(context, down.plan, kPlanBytes, Stage::down, "down.plan",
                   ranges, error) ||
        !add_range(context, down.tiles, expert_tile_bytes, Stage::down,
                   "down.tiles", ranges, error) ||
        !add_range(context, down.destinations, destination_bytes,
                   Stage::down, "down.destinations", ranges, error) ||
        !add_range(context, down.address_snapshot, address_row_bytes,
                   Stage::down, "down.address_snapshot", ranges, error) ||
        !add_range(context, down.parts, down_parts_bytes, Stage::down,
                   "down.parts", ranges, error) ||
        !add_range(context, down.status, sizeof(std::uint32_t),
                   Stage::down, "down.status", ranges, error)) {
        return false;
    }
    for (const auto& storage : down.weight_storage) {
        if (storage.buffer.is_tiered_slot_arena()) {
            error = "full-mode Prefill MoE cannot use tiered expert slots";
            return false;
        }
        if (!add_range(context, storage.buffer, storage.buffer.length(),
                       Stage::down, "down.weight_storage", ranges, error)) {
            return false;
        }
    }

    const auto& combine = buffers.combine;
    return add_range(context, combine.parts, down_parts_bytes, Stage::combine,
                     "combine.parts", ranges, error) &&
           add_range(context, combine.weights, route_weight_bytes,
                     Stage::combine, "combine.weights", ranges, error) &&
           add_range(context, combine.shared, rows * kWidth * sizeof(float),
                     Stage::combine, "combine.shared", ranges, error) &&
           add_range(context, combine.gate, rows * sizeof(float),
                     Stage::combine, "combine.gate", ranges, error) &&
           add_range(context, combine.output, rows * kWidth * sizeof(float),
                     Stage::combine, "combine.output", ranges, error);
}

bool validate_cross_stage_aliases(
    const std::vector<ActiveRange>& ranges, std::string& error) {
    for (std::size_t left_index = 0u; left_index < ranges.size();
         ++left_index) {
        const auto& left = ranges[left_index];
        for (std::size_t right_index = left_index + 1u;
             right_index < ranges.size(); ++right_index) {
            const auto& right = ranges[right_index];
            if (left.stage == right.stage || !overlaps(left, right) ||
                (!left.writable && !right.writable)) {
                continue;
            }
            const ActiveRange& writer = left.writable ? left : right;
            const ActiveRange& reader = left.writable ? right : left;
            if (expected_dataflow(writer, reader) &&
                contains(writer, reader)) {
                continue;
            }
            error = "full-mode Prefill MoE " + std::string(writer.name) +
                    " overlaps active " + std::string(reader.name);
            return false;
        }
    }
    return true;
}

bool validate_full_pipeline(
    Context& context, std::uint32_t rows, std::uint32_t n_experts,
    std::uint32_t layer, PrefillBF16X2Mode projection_mode,
    const PrefillMoeFullPipelineFormats& formats,
    const PrefillMoeFullPipelineBuffers& buffers, std::string& error) {
    if (rows == 0u || rows > 8u ||
        (n_experts != 256u && n_experts != 512u) ||
        layer >= kPrefillMoePlanLayers) {
        error = "full-mode Prefill MoE requires T=1..8, 256 or 512 experts, "
                "and a valid layer";
        return false;
    }
    if (projection_mode != PrefillBF16X2Mode::all_projections &&
        projection_mode != PrefillBF16X2Mode::except_gr) {
        error = "full-mode Prefill MoE requires BF16X2 projection mode 1 or 2";
        return false;
    }
    switch (formats.routed_gate_up) {
        case PrefillMoeGateUpFormat::Q2_0:
        case PrefillMoeGateUpFormat::IQ4_XS:
        case PrefillMoeGateUpFormat::IQ2_XXS:
        case PrefillMoeGateUpFormat::IQ2_XS:
        case PrefillMoeGateUpFormat::IQ2_S:
        case PrefillMoeGateUpFormat::IQ3_XXS:
        case PrefillMoeGateUpFormat::IQ3_S:
        case PrefillMoeGateUpFormat::IQ1_M:
            break;
        default:
            error = "full-mode Prefill MoE routed gate/up format is unsupported";
            return false;
    }
    switch (formats.routed_down) {
        case PrefillMoeDownFormat::Q2_0:
        case PrefillMoeDownFormat::IQ4_NL:
            break;
        default:
            error = "full-mode Prefill MoE routed down format is unsupported";
            return false;
    }

    std::vector<ActiveRange> ranges;
    ranges.reserve(64u + buffers.routed_gate_up.weight_storage.size() +
                   buffers.routed_down.weight_storage.size());
    if (!collect_active_ranges(context, rows, n_experts, layer, formats,
                               buffers, ranges, error)) {
        return false;
    }
    if (!validate_cross_stage_aliases(ranges, error)) return false;
    return true;
}

bool encode_gate_up(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillMoeGateUpFormat format,
    const PrefillMoeExpertTileBuffers& buffers, std::string& error) {
    switch (format) {
        case PrefillMoeGateUpFormat::Q2_0:
            return encode_prefill_moe_gu_q2_0(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeGateUpFormat::IQ4_XS:
            return encode_prefill_moe_gu_iq4_xs(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeGateUpFormat::IQ2_XXS:
            return encode_prefill_moe_gu_iq2_xxs(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeGateUpFormat::IQ2_XS:
            return encode_prefill_moe_gu_iq2_xs(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeGateUpFormat::IQ2_S:
            return encode_prefill_moe_gu_iq2_s(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeGateUpFormat::IQ3_XXS:
            return encode_prefill_moe_gu_iq3_xxs(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeGateUpFormat::IQ3_S:
            return encode_prefill_moe_gu_iq3_s(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeGateUpFormat::IQ1_M:
            return encode_prefill_moe_gu_iq1_m(
                context, encoder, rows, n_experts, layer, buffers, error);
        default:
            error = "full-mode Prefill MoE routed gate/up format is unsupported";
            return false;
    }
}

bool encode_down(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillMoeDownFormat format,
    const PrefillMoeDownTileBuffers& buffers, std::string& error) {
    switch (format) {
        case PrefillMoeDownFormat::Q2_0:
            return encode_prefill_moe_down_q2_0(
                context, encoder, rows, n_experts, layer, buffers, error);
        case PrefillMoeDownFormat::IQ4_NL:
            return encode_prefill_moe_down_iq4_nl(
                context, encoder, rows, n_experts, layer, buffers, error);
        default:
            error = "full-mode Prefill MoE routed down format is unsupported";
            return false;
    }
}

bool encode_full_pipeline(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeFullPipelineFormats& formats,
    const PrefillMoeFullPipelineBuffers& buffers, std::string& error) {
    error.clear();
    if (!validate_full_pipeline(context, rows, n_experts, layer,
                                projection_mode, formats, buffers, error)) {
        return false;
    }
    if (!encode_prefill_moe_full_mode_route_gather(
            context, encoder, rows, n_experts, layer, projection_mode,
            buffers.route_gather, error) ||
        !encode_prefill_moe_shared_ffn(
            context, encoder, rows, projection_mode, formats.shared,
            buffers.shared, error) ||
        !encode_gate_up(context, encoder, rows, n_experts, layer,
                        formats.routed_gate_up, buffers.routed_gate_up,
                        error) ||
        !encode_down(context, encoder, rows, n_experts, layer,
                     formats.routed_down, buffers.routed_down, error) ||
        !encode_prefill_moe_combine(context, encoder, rows,
                                    buffers.combine, error)) {
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_prefill_moe_full_mode_layer(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeFullPipelineFormats& formats,
    const PrefillMoeFullPipelineBuffers& buffers, std::string& error) {
    return encode_full_pipeline(context, encoder, rows, n_experts, layer,
                                projection_mode, formats, buffers, error);
}

bool encode_prefill_moe_full_mode_layer(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeFullPipelineFormats& formats,
    const PrefillMoeFullPipelineBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Full-mode Prefill MoE layer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_full_pipeline(
                context, encoder, rows, n_experts, layer, projection_mode,
                formats, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
