// Validated Prefill MoE down-tile encoders.
#pragma once

#include "strata/metal/prefill_moe_expert_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

struct PrefillMoeDownTileBuffers {
    /// Row-major `[layer, expert, gate/up/down]` GPU-address table.
    ReadOnlyBufferRef addresses;
    /// Dedicated snapshot scratch for the selected address row. Follow the
    /// lifetime and pending-command rules documented by
    /// `PrefillMoeExpertTileBuffers::address_snapshot`.
    BufferRef address_snapshot;
    /// Context-owned allocations backing each non-null down-weight address.
    std::span<const PrefillMoeWeightStorageView> weight_storage;
    /// Prepared for the same CommandBuffer by `ExpertStore`.
    std::shared_ptr<ExpertLeaseSet> tiered_leases;
    /// Grouped F16 SwiGLU rows: `[rows * 10, 640]`.
    ReadOnlyBufferRef hidden;
    /// The selected layer's 6,400-byte PfPlan slot.
    ReadOnlyBufferRef plan;
    /// Expert tile list from `encode_prefill_moe_grouping`.
    ReadOnlyBufferRef tiles;
    /// Sorted-entry to original route index map from grouping.
    ReadOnlyBufferRef destinations;
    /// F32 output restored to token/rank order: `[rows * 10, 2560]`.
    BufferRef parts;
    /// Dedicated GPU-cleared status for this stage. Do not share it with the
    /// gate/up stage: plan validation clears this word before checking tiles.
    /// Bits otherwise match `PrefillMoeExpertTileBuffers::status`.
    BufferRef status;
};

/// Encode the Q2_0 F16-activation down tile after SwiGLU. The GPU tile list
/// selects expert groups; `destinations` restores route order in `parts`.
bool encode_prefill_moe_down_q2_0(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error);

/// Encode the F16-activation Prefill down tile with IQ4_NL weights.
bool encode_prefill_moe_down_iq4_nl(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error);

/// Transactional standalone wrapper for `encode_prefill_moe_down_q2_0`.
bool encode_prefill_moe_down_q2_0(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error);

/// Transactional standalone wrapper for `encode_prefill_moe_down_iq4_nl`.
bool encode_prefill_moe_down_iq4_nl(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error);

}  // namespace strata::metal
