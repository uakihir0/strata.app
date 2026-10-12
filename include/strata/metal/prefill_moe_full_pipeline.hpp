// Full-mode Prefill MoE layer composition.
#pragma once

#include "strata/metal/prefill_moe_down_dispatch.hpp"
#include "strata/metal/prefill_moe_pipeline.hpp"
#include "strata/metal/prefill_moe_post.hpp"
#include "strata/metal/prefill_moe_shared_pipeline.hpp"

#include <cstdint>

namespace strata::metal {

/// Routed gate/up formats supported by the Prefill expert tile encoders.
enum class PrefillMoeGateUpFormat : std::uint8_t {
    Q2_0,
    IQ4_XS,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ1_M,
};

/// Routed down formats supported by the Prefill expert tile encoders.
enum class PrefillMoeDownFormat : std::uint8_t {
    Q2_0,
    IQ4_NL,
};

/// Routed expert tile formats. Shared expert formats are selected
/// independently with `PrefillNativeWeightFormat`.
struct PrefillMoeFullPipelineFormats {
    PrefillMoeSharedFFNFormats shared;
    PrefillMoeGateUpFormat routed_gate_up =
        PrefillMoeGateUpFormat::Q2_0;
    PrefillMoeDownFormat routed_down = PrefillMoeDownFormat::Q2_0;
};

/// Caller-owned buffers for one full-mode Prefill MoE layer.
///
/// The nested views carry router/grouping, shared-expert, routed expert and
/// weighted-combine data. Routed expert address tables and weight storage
/// must describe full-residency allocations; tiered leases are unsupported.
/// After completion, the caller must check both routed-stage status words and
/// the PfPlan error field before consuming `combine.output`.
struct PrefillMoeFullPipelineBuffers {
    PrefillMoeRouteGatherBuffers route_gather;
    PrefillMoeSharedFFNBuffers shared;
    PrefillMoeExpertTileBuffers routed_gate_up;
    PrefillMoeDownTileBuffers routed_down;
    PrefillMoeCombineBuffers combine;
};

/// Encode a complete synthetic full-mode Prefill MoE layer in one ordered
/// segment: BF16X2 router and top-10 grouping/gather, shared native FFN and
/// gate, routed gate/up SwiGLU and down tiles, then weighted combine.
///
/// Supports T=1..8, 256 or 512 experts, BF16X2 projection modes 1 and 2,
/// the eight documented routed gate/up formats, Q2_0 or IQ4_NL routed down,
/// and all supported shared native formats. This does not schedule tiered
/// residency or integrate with `Prefill::run`.
///
/// Every writable active view must be disjoint from other active views except
/// for the listed producer-to-consumer edges documented by the implementation.
/// Caller-owned use must propagate false from the enclosing
/// `Encoder::encode` body so the command segment is discarded.
bool encode_prefill_moe_full_mode_layer(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeFullPipelineFormats& formats,
    const PrefillMoeFullPipelineBuffers& buffers, std::string& error);

/// Transactional standalone form. Any validation or dispatch failure
/// discards the complete layer segment.
bool encode_prefill_moe_full_mode_layer(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeFullPipelineFormats& formats,
    const PrefillMoeFullPipelineBuffers& buffers, std::string& error);

}  // namespace strata::metal
