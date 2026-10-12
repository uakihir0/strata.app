// Full-mode Prefill MoE router and grouped-input composition.
#pragma once

#include "strata/metal/prefill_gemm_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

/// Caller-owned buffers for Prefill MoE steps 1–6 in full residency mode.
///
/// `router.y` receives F32 logits. `route_ids` and `route_weights` receive
/// token-major top-10 results. Grouping then writes the selected layer's
/// `plan`, `tiles`, `source_rows`, `destination_rows` and `gathered_rows`.
/// `resident` is still required by the grouping ABI in full mode; it need
/// only expose one int32 value because the full-mode grouping kernel does not
/// read it. `mixed_h` is row-major F16 `[rows, 2560]`.
struct PrefillMoeRouteGatherBuffers {
    PrefillGemmBF16X2Buffers router;
    BufferRef route_ids;
    BufferRef route_weights;
    ReadOnlyBufferRef resident;
    ReadOnlyBufferRef mixed_h;
    BufferRef histogram;
    BufferRef plan;
    BufferRef tiles;
    BufferRef source_rows;
    BufferRef destination_rows;
    BufferRef gathered_rows;
};

/// Encode Prefill MoE steps 1–6 as one ordered full-mode segment:
/// BF16X2 router projection, divide-mode top-10 routing, stable GPU grouping,
/// and F16 input gather. Supports T=1..8 and 256 or 512 experts.
///
/// This composition deliberately excludes tiered mode. Tiered execution
/// requires host resolution after the plan is produced and an event wait
/// before expert consumers; callers should encode grouping on its own and
/// coordinate that handshake with `ExpertStore`.
bool encode_prefill_moe_full_mode_route_gather(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error);

/// Transactional standalone form. A failure in route or grouping validation
/// discards every earlier router dispatch in this composition.
bool encode_prefill_moe_full_mode_route_gather(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeRouteGatherBuffers& buffers, std::string& error);

}  // namespace strata::metal
