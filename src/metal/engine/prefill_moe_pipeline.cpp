// Ordered full-mode Prefill MoE router and grouped-input composition.
#include "strata/metal/prefill_moe_pipeline.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kPrefillMoeHiddenWidth = 2560u;
constexpr std::uint32_t kPrefillMoeTopK = 10u;
constexpr std::uint32_t kPrefillMoeHistogramExperts = 512u;
constexpr std::size_t kPrefillMoePlanStrideBytes = 6400u;

struct ActiveRange {
    const void* cpu = nullptr;
    std::uint64_t gpu = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
    void* handle = nullptr;
    std::size_t offset = 0u;
};

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool ranges_overlap(const ActiveRange& left, const ActiveRange& right) {
    if (left.handle != nullptr && left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.bytes, right.offset,
                               right.bytes)) {
        return true;
    }
    const bool cpu_overlap =
        left.cpu != nullptr && right.cpu != nullptr &&
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.cpu), left.bytes,
            reinterpret_cast<std::uintptr_t>(right.cpu), right.bytes);
    const bool gpu_overlap =
        left.gpu != 0u && right.gpu != 0u &&
        address_ranges_overlap(left.gpu, left.bytes, right.gpu, right.bytes);
    return cpu_overlap || gpu_overlap;
}

ActiveRange read_range(ReadOnlyBufferRef view, std::size_t bytes,
                       std::string_view name) {
    return ActiveRange{view.contents(), view.gpu_address(), bytes, name};
}

ActiveRange write_range(BufferRef view, std::size_t bytes,
                        std::string_view name) {
    return ActiveRange{view.contents, view.gpu_address, bytes, name,
                       view.handle, view.offset};
}

bool validate_composed_ranges(
    const PrefillMoeRouteGatherBuffers& buffers, std::uint32_t rows,
    std::uint32_t n_experts, std::string& error) {
    const std::size_t entries =
        static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    const std::size_t route_bytes = entries * sizeof(std::int32_t);
    const std::size_t route_weight_bytes = entries * sizeof(float);
    const std::size_t router_bytes =
        static_cast<std::size_t>(rows) * n_experts * sizeof(float);
    const std::size_t router_input_bytes =
        static_cast<std::size_t>(rows) * kPrefillMoeHiddenWidth *
        sizeof(std::uint16_t);
    const std::size_t router_weight_bytes =
        static_cast<std::size_t>(n_experts) * kPrefillMoeHiddenWidth *
        sizeof(std::uint16_t);
    const std::size_t mixed_bytes =
        static_cast<std::size_t>(rows) * kPrefillMoeHiddenWidth *
        sizeof(std::uint16_t);
    const std::size_t gathered_bytes = entries * kPrefillMoeHiddenWidth *
                                       sizeof(std::uint16_t);
    const std::size_t histogram_tiles =
        (entries + 255u) / 256u;
    const std::size_t histogram_bytes =
        histogram_tiles * kPrefillMoeHistogramExperts *
        sizeof(std::uint32_t);
    const std::size_t tile_capacity = (entries + 63u) / 64u +
                                      kPrefillMoeHistogramExperts;
    const std::size_t tile_bytes = tile_capacity * 2u * sizeof(std::uint32_t);

    const std::array<ActiveRange, 5u> inputs{{
        read_range(buffers.router.x_hi, router_input_bytes, "router high input"),
        read_range(buffers.router.x_lo, router_input_bytes, "router low input"),
        read_range(buffers.router.w, router_weight_bytes, "router weights"),
        read_range(buffers.resident, sizeof(std::int32_t), "residency"),
        read_range(buffers.mixed_h, mixed_bytes, "mixed activation"),
    }};
    const std::array<ActiveRange, 9u> outputs{{
        write_range(buffers.router.y, router_bytes, "router logits"),
        write_range(buffers.route_ids, route_bytes, "route IDs"),
        write_range(buffers.route_weights, route_weight_bytes, "route weights"),
        write_range(buffers.histogram, histogram_bytes, "histogram"),
        write_range(buffers.plan, kPrefillMoePlanStrideBytes, "PfPlan slot"),
        write_range(buffers.tiles, tile_bytes, "tile list"),
        write_range(buffers.source_rows, route_bytes, "source rows"),
        write_range(buffers.destination_rows, route_bytes,
                    "destination rows"),
        write_range(buffers.gathered_rows, gathered_bytes, "gathered rows"),
    }};

    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            if (ranges_overlap(output, input)) {
                error = "composed Prefill MoE output " +
                        std::string(output.name) + " overlaps input " +
                        std::string(input.name);
                return false;
            }
        }
    }
    for (std::size_t i = 0u; i < outputs.size(); ++i) {
        for (std::size_t j = i + 1u; j < outputs.size(); ++j) {
            if (ranges_overlap(outputs[i], outputs[j])) {
                error = "composed Prefill MoE output " +
                        std::string(outputs[i].name) + " overlaps output " +
                        std::string(outputs[j].name);
                return false;
            }
        }
    }
    return true;
}

bool encode_full_mode_route_gather(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error) {
    error.clear();
    if (rows == 0u || rows > 8u ||
        (n_experts != 256u && n_experts != 512u) ||
        layer >= kPrefillMoePlanLayers) {
        error = "full-mode Prefill MoE route/gather requires T=1..8, "
                "256 or 512 experts, and a valid layer";
        return false;
    }
    if (projection_mode != PrefillBF16X2Mode::all_projections &&
        projection_mode != PrefillBF16X2Mode::except_gr) {
        error = "full-mode Prefill MoE router requires BF16X2 projection mode 1 or 2";
        return false;
    }
    if (!validate_composed_ranges(buffers, rows, n_experts, error)) {
        return false;
    }

    const PrefillGemmMPPParams router_params{
        rows, n_experts, kPrefillMoeHiddenWidth, n_experts, 0u, false};
    if (!encode_prefill_gemm_bf16x2(
            context, encoder, router_params, buffers.router, error)) {
        return false;
    }

    const MoeRouteBuffers route_buffers{
        context.buffers().read_only(buffers.router.y), buffers.route_ids,
        buffers.route_weights};
    if (!encode_moe_route(context, encoder, rows, n_experts, true,
                          route_buffers, error)) {
        return false;
    }

    const PrefillMoeGroupingBuffers grouping_buffers{
        context.buffers().read_only(buffers.route_ids),
        buffers.resident,
        buffers.mixed_h,
        buffers.histogram,
        buffers.plan,
        buffers.tiles,
        buffers.source_rows,
        buffers.destination_rows,
        buffers.gathered_rows};
    return encode_prefill_moe_grouping(
        context, encoder, rows, n_experts, false, layer, grouping_buffers,
        error);
}

}  // namespace

bool encode_prefill_moe_full_mode_route_gather(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error) {
    return encode_full_mode_route_gather(context, encoder, rows, n_experts,
                                         layer, projection_mode, buffers,
                                         error);
}

bool encode_prefill_moe_full_mode_route_gather(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE full-mode route/gather",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_full_mode_route_gather(
                context, encoder, rows, n_experts, layer, projection_mode,
                buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
