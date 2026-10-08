// Validated Metal MoE routing dispatches for Window and Prefill.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct MoeRouteBuffers {
    ReadOnlyBufferRef logits;
    BufferRef ids;
    BufferRef weights;
};

struct MoePlanBuffers {
    ReadOnlyBufferRef ids;
    ReadOnlyBufferRef resident;
    BufferRef usage;
    BufferRef plans;
};

enum class MoeExpertFormat {
    q2_0,
    iq1_m,
    iq2_xxs,
    iq2_xs,
    iq2_s,
    iq3_xxs,
    iq3_s,
    iq4_xs,
};

struct MoeGroupedExpertBuffers {
    /// Row-major `[layer, expert, gate/up/down]` GPU-address table.
    ReadOnlyBufferRef addresses;
    /// Q8_1 activations, with 80 blocks for each input row.
    ReadOnlyBufferRef activations;
    /// `TieredMoePlan` table containing the selected layer's validated plan.
    ReadOnlyBufferRef plans;
    /// Each output reserves `[rows * 10, 640]` floats.
    BufferRef gate;
    BufferRef up;
};

enum class MoeDownFormat {
    q2_0,
    iq4_nl,
};

struct MoeGroupedDownBuffers {
    /// Row-major `[layer, expert, gate/up/down]` GPU-address table.
    ReadOnlyBufferRef addresses;
    /// Q8_1 activations, with 20 blocks for each routed expert entry.
    ReadOnlyBufferRef activations;
    /// `TieredMoePlan` table containing the selected layer's validated plan.
    ReadOnlyBufferRef plans;
    /// Each output reserves `[rows * 10, 2560]` floats in original route order.
    BufferRef parts;
};

struct MoeSwiGLUBuffers {
    ReadOnlyBufferRef gate;
    ReadOnlyBufferRef up;
    BufferRef output;
};

struct MoeCombineBuffers {
    /// Routed down-projection parts in `[rows, 10, 2560]` order.
    ReadOnlyBufferRef parts;
    /// Top-10 route weights in `[rows, 10]` order.
    ReadOnlyBufferRef route_weights;
    /// Shared-expert down output in `[rows, 2560]` order.
    ReadOnlyBufferRef shared_output;
    /// Shared-expert gate logits, one F32 per row.
    ReadOnlyBufferRef shared_gate_logits;
    /// Combined model output in `[rows, 2560]` order.
    BufferRef output;
};

/// Encode the stable top-10 route selection for 256 or 512 experts.
/// `divide_weights` selects the prompt-path division mode; Window decode must
/// pass false to match the model's multiply-by-reciprocal path. The views must
/// remain alive and unchanged where read until command completion.
/// When using the caller-owned Encoder overload, propagate false from the
/// enclosing `Encoder::encode` body so earlier dispatches in the segment are
/// discarded on validation or encoding failure.
bool encode_moe_route(Context& context, Encoder& encoder,
                      std::uint32_t rows, std::uint32_t experts,
                      bool divide_weights, const MoeRouteBuffers& buffers,
                      std::string& error);
/// Standalone transactional form. A validation failure discards the segment.
bool encode_moe_route(Context& context, CommandBuffer& command_buffer,
                      std::uint32_t rows, std::uint32_t experts,
                      bool divide_weights, const MoeRouteBuffers& buffers,
                      std::string& error);

/// Group top-10 expert IDs in first-appearance order for one Window layer.
/// `experts` is the active table stride (1..512); every input ID must be in
/// `[0, experts)`. Production callers pass IDs from the matching 256- or
/// 512-expert route kernel. The resident and usage tables use `experts` as their active layer stride;
/// both remain bound in full mode to match the kernel ABI, though full mode
/// does not read or update them.
/// The plan view starts at layer zero and contains the selected layer.
/// Inputs and outputs must remain alive until command completion.
bool encode_moe_plan(Context& context, Encoder& encoder,
                     std::uint32_t rows, std::uint32_t experts,
                     bool tiered, std::uint32_t layer,
                     const MoePlanBuffers& buffers, std::string& error);
/// Standalone transactional form. A validation failure discards the segment.
bool encode_moe_plan(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t rows, std::uint32_t experts,
                     bool tiered, std::uint32_t layer,
                     const MoePlanBuffers& buffers, std::string& error);

/// Encode the grouped gate/up projection for a plan produced for the same
/// rows, experts and layer by the ordered MoE plan encoder. In tiered mode,
/// queue this consumer only after ExpertStore successfully resolves the plan.
/// Dispatches all 640 rows of each projection. The GPU checks plan row/group
/// bounds, selected expert IDs and token indices before reading or writing.
/// The plan partition and selected expert addresses must still come from
/// trusted engine outputs and remain valid until command completion; this
/// method does not read back or fully validate GPU-produced plan arrays.
/// The inputs and output buffers must remain alive until command completion.
bool encode_moe_experts(Context& context, Encoder& encoder,
                        std::uint32_t rows, std::uint32_t experts,
                        std::uint32_t layer, MoeExpertFormat format,
                        const MoeGroupedExpertBuffers& buffers,
                        std::string& error);
/// Standalone transactional form. A validation failure discards the segment.
bool encode_moe_experts(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, std::uint32_t experts,
                        std::uint32_t layer, MoeExpertFormat format,
                        const MoeGroupedExpertBuffers& buffers,
                        std::string& error);

/// Encode the grouped routed-expert down projection for a plan produced for
/// the same rows, experts and layer by the ordered MoE plan encoder. In tiered
/// mode, queue this consumer only after ExpertStore successfully resolves the
/// plan. Supports Q2_0 and IQ4_NL expert weights. GPU checks plan group,
/// destination and row bounds before reading or writing; plan contents and
/// selected expert addresses must still come from trusted engine outputs and
/// remain valid until command completion. The inputs and output must be
/// disjoint and remain alive until command completion.
bool encode_moe_down(Context& context, Encoder& encoder,
                     std::uint32_t rows, std::uint32_t experts,
                     std::uint32_t layer, MoeDownFormat format,
                     const MoeGroupedDownBuffers& buffers,
                     std::string& error);
/// Standalone transactional form. A validation failure discards the segment.
bool encode_moe_down(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t rows, std::uint32_t experts,
                     std::uint32_t layer, MoeDownFormat format,
                     const MoeGroupedDownBuffers& buffers,
                     std::string& error);

/// Encode SwiGLU for routed MoE entries (rows * 10) or shared-expert rows.
/// Gate and up are read-only F32 arrays of 640 values per entry. The output
/// must be disjoint from both inputs and remain alive until command completion.
/// Keep this ordered before the Q8_1 quantization of the resulting values.
bool encode_moe_swiglu(Context& context, Encoder& encoder,
                       std::uint32_t rows, bool shared_expert,
                       const MoeSwiGLUBuffers& buffers,
                       std::string& error);
/// Standalone transactional form. A validation failure discards the segment.
bool encode_moe_swiglu(Context& context, CommandBuffer& command_buffer,
                       std::uint32_t rows, bool shared_expert,
                       const MoeSwiGLUBuffers& buffers,
                       std::string& error);

/// Combine ten routed expert parts with the shared-expert output for 1..8
/// rows. Inputs are read-only, the output must be disjoint from every input,
/// and all views must remain alive until command completion.
bool encode_moe_combine(Context& context, Encoder& encoder,
                        std::uint32_t rows,
                        const MoeCombineBuffers& buffers,
                        std::string& error);
/// Standalone transactional form. A validation failure discards the segment.
bool encode_moe_combine(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows,
                        const MoeCombineBuffers& buffers,
                        std::string& error);

}  // namespace strata::metal
