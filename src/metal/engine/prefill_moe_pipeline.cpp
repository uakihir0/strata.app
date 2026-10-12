// Ordered full-mode Prefill MoE router and grouped-input composition.
#include "strata/metal/prefill_moe_pipeline.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"

namespace strata::metal {
namespace {

constexpr std::uint32_t kPrefillMoeHiddenWidth = 2560u;

bool encode_full_mode_route_gather(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error) {
    error.clear();
    if (rows == 0u || rows > 8u ||
        (n_experts != 256u && n_experts != 512u) ||
        layer >= kPrefillMoePlanLayers) {
        error = "full-mode Prefill MoE route/gather requires T=1..8, "
                "256 or 512 experts, and a valid layer";
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
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error) {
    return encode_full_mode_route_gather(context, encoder, rows, n_experts,
                                         layer, buffers, error);
}

bool encode_prefill_moe_full_mode_route_gather(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE full-mode route/gather",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_full_mode_route_gather(
                context, encoder, rows, n_experts, layer, buffers,
                encode_error);
        },
        error);
}

}  // namespace strata::metal
