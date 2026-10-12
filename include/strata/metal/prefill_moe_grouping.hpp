// Validated GPU grouping for Prefill's routed expert entries.
#pragma once

#include "strata/metal/context.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

inline constexpr std::uint32_t kPrefillMoeTopK = 10u;
inline constexpr std::uint32_t kPrefillMoeHistogramExperts = 512u;
inline constexpr std::uint32_t kPrefillMoePlanLayers = 48u;
inline constexpr std::size_t kPrefillMoePlanStrideBytes = 6400u;
inline constexpr std::size_t kPrefillMoePlanBytes = 6180u;

/// Views used by one Prefill chunk's stable expert grouping.
///
/// `ids` contains T*10 route IDs in token-major order. `resident` is the
/// layer-major residency table with `n_experts` entries per layer. `plans`
/// is the selected layer's 6,400-byte PfPlan slot (the first 6,180 bytes
/// hold the Metal struct). The scratch capacities are:
/// - histogram: ceil(T*10/256) * 512 uint32 values;
/// - tiles: ceil(T*10/64) + 512 uint2 values;
/// - source_rows and destination_rows: T*10 int32 values each;
/// - gathered_rows: T*10*2560 F16 values.
struct PrefillMoeGroupingBuffers {
    ReadOnlyBufferRef ids;
    ReadOnlyBufferRef resident;
    ReadOnlyBufferRef mixed_h;
    BufferRef histogram;
    BufferRef plans;
    BufferRef tiles;
    BufferRef source_rows;
    BufferRef destination_rows;
    BufferRef gathered_rows;
};

/// Encode stable expert grouping and input-row gather for a Prefill chunk.
///
/// The caller must ensure all views remain alive through command completion.
/// `plan.bad` reports route IDs outside [0, n_experts); callers must check it
/// before submitting expert work. `grp_res` stores a resident flag
/// (`resident >= 0`), not a slot index. The caller-owned Encoder form requires
/// an ordered dispatch boundary and callers must propagate failure so the
/// whole segment is discarded.
bool encode_prefill_moe_grouping(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, bool tiered, std::uint32_t layer,
    const PrefillMoeGroupingBuffers& buffers, std::string& error);

/// Transactional standalone form. Any late validation failure discards the
/// earlier count, scan, scatter and gather dispatches.
bool encode_prefill_moe_grouping(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, bool tiered, std::uint32_t layer,
    const PrefillMoeGroupingBuffers& buffers, std::string& error);

}  // namespace strata::metal
