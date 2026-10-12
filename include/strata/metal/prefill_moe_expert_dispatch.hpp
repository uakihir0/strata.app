// Validated Prefill MoE expert-tile encoders.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace strata::metal {

class Encoder;
class ExpertLeaseSet;

struct PrefillMoeWeightStorageView {
    ReadOnlyBufferRef buffer;
};

struct PrefillMoeExpertTileBuffers {
    /// Row-major `[layer, expert, gate/up/down]` GPU-address table.
    ReadOnlyBufferRef addresses;
    /// Dedicated caller-owned scratch for one selected address row. Encoding
    /// copies the validated row here before dispatch, so earlier GPU writes
    /// to the source table cannot change the addresses used by this consumer.
    /// Use a distinct range for every pending expert dispatch. The
    /// CommandBuffer rejects ranges used by prior GPU accesses and later GPU
    /// writes to reserved scratch; leave it unchanged until command completion.
    /// Reuse after that command completes.
    BufferRef address_snapshot;
    /// Context-owned allocations backing every non-null gate/up address.
    std::span<const PrefillMoeWeightStorageView> weight_storage;
    /// Prepared for the same CommandBuffer by `ExpertStore`; keeps tiered
    /// slots pinned until that command buffer completes.
    std::shared_ptr<ExpertLeaseSet> tiered_leases;
    /// Grouped F16 input rows: `[rows * 10, 2560]`.
    ReadOnlyBufferRef gathered_rows;
    /// The selected layer's 6,400-byte PfPlan slot.
    ReadOnlyBufferRef plan;
    /// Expert tile list from `encode_prefill_moe_grouping`.
    ReadOnlyBufferRef tiles;
    /// F16 SwiGLU output: `[rows * 10, 640]`.
    BufferRef hidden;
    /// GPU-cleared before expert execution. Bit 0 reports a missing active
    /// gate/up address, bit 1 an invalid grouping plan, and bit 2 a tiered
    /// address without a live layer lease. Check for zero after completion
    /// before consuming `hidden`.
    BufferRef status;
};

/// Encode the Q2_0 64x64 SIMD-group gate/up tile and fused SwiGLU epilogue.
/// The selected expert layer may use full-mode or tiered GPU addresses.
bool encode_prefill_moe_gu_q2_0(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeExpertTileBuffers& buffers, std::string& error);

/// Transactional standalone wrapper for `encode_prefill_moe_gu_q2_0`.
bool encode_prefill_moe_gu_q2_0(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeExpertTileBuffers& buffers, std::string& error);

}  // namespace strata::metal
