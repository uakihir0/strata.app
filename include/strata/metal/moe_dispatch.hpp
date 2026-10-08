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

struct MoeRoutedFFNBuffers {
    /// Row-major `[layer, expert, gate/up/down]` GPU-address table.
    ReadOnlyBufferRef addresses;
    /// Q8_1 rows of the FFN input, with 80 blocks for each token.
    ReadOnlyBufferRef input_activations;
    /// `TieredMoePlan` table containing the matching layer plan.
    ReadOnlyBufferRef plans;
    /// Routed gate, up and SwiGLU scratch: `[rows * 10, 640]` F32.
    BufferRef gate;
    BufferRef up;
    BufferRef hidden;
    /// Q8_1 quantization of `hidden`: `[rows * 10, 20]` blocks.
    BufferRef down_activations;
    /// Routed down results in original route order: `[rows * 10, 2560]`.
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

struct MoeSharedFFNBuffers {
    /// Q8_1 rows of the mixed FFN input, `[rows, 80]`.
    ReadOnlyBufferRef input_activations;
    /// F32 mixed FFN input for the BF16 shared gate, `[rows, 2560]`.
    ReadOnlyBufferRef mixed;
    /// Shared gate/up Q8_0 weights, each `[640, 80]` blocks.
    ReadOnlyBufferRef gate_weights;
    ReadOnlyBufferRef up_weights;
    /// Shared down Q8_0 weights, `[2560, 20]` blocks.
    ReadOnlyBufferRef down_weights;
    /// Shared gate-input BF16 weights, `[1, 2560]`.
    ReadOnlyBufferRef shared_gate_weights;
    /// Host-visible uint32 row count equal to `rows`.
    ReadOnlyBufferRef active_rows;
    /// Routed down results `[rows, 10, 2560]` and weights `[rows, 10]`.
    ReadOnlyBufferRef routed_parts;
    ReadOnlyBufferRef route_weights;

    /// F32 gate, up, hidden and shared down outputs; gate logits are F32.
    BufferRef gate;
    BufferRef up;
    BufferRef hidden;
    BufferRef down_activations;
    BufferRef shared_output;
    BufferRef shared_gate_logits;
    BufferRef output;

    /// Mat-vec padded-column outputs. Each requires at least one float when
    /// `rows` equals the compiled column width.
    BufferRef gate_sink;
    BufferRef up_sink;
    BufferRef down_sink;
    BufferRef shared_gate_sink;
};

struct MoeSharedFFNWriteBuffers {
    /// The shared MoE FFN and routed combine inputs and scratch.
    MoeSharedFFNBuffers ffn;
    /// GR residual `[rows, 4, 2560]` and FFN injection `[rows, 4]`.
    BufferRef residual;
    BufferRef inject;
};

struct MoeRoutedSharedFFNWriteBuffers {
    /// Routed experts, whose `parts` feed the shared combine below.
    MoeRoutedFFNBuffers routed;
    /// Shared expert, combine and GR residual write.
    MoeSharedFFNWriteBuffers shared;
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

/// Encode the routed expert FFN in order: grouped gate/up, SwiGLU, Q8_1 and
/// grouped down projection. The supplied plan must have been produced for the
/// same rows, experts and layer. In tiered mode, call this only after successful
/// plan resolution; in full mode it may follow the plan dispatch in the same
/// ordered segment. The address, activation and plan inputs must be disjoint.
/// Scratch outputs must be mutually disjoint and disjoint from every input.
/// Plan contents and selected role addresses must come from trusted engine
/// outputs and remain valid until command completion; this encoder does not
/// read back or fully validate GPU-produced plan arrays.
bool encode_moe_routed_ffn(Context& context, Encoder& encoder,
                           std::uint32_t rows, std::uint32_t experts,
                           std::uint32_t layer,
                           MoeExpertFormat gate_up_format,
                           MoeDownFormat down_format,
                           const MoeRoutedFFNBuffers& buffers,
                           std::string& error);
/// Standalone transactional form. A failure discards every earlier stage.
bool encode_moe_routed_ffn(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, std::uint32_t experts,
                           std::uint32_t layer,
                           MoeExpertFormat gate_up_format,
                           MoeDownFormat down_format,
                           const MoeRoutedFFNBuffers& buffers,
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

/// Encode the shared-expert gate/up, SwiGLU, Q8_1, down projection and BF16
/// gate, then combine that output with already available routed parts and
/// route weights. The Q8_1 input must contain `rows` rows; the F32 `mixed`
/// view is consumed by the shared gate projection. Rows must be in 1..8 and
/// the compiled mat-vec width is the smallest supported width covering rows.
///
/// Read-only inputs may overlap. Every scratch/output/sink view must be
/// disjoint from every input and every other writable view. All views must
/// remain alive until command completion, and read-only inputs (including
/// `active_rows`) must remain unchanged. Call this only after routed inputs
/// are ready; in tiered mode, wait for successful expert resolution first.
/// The caller-owned form must propagate false from its Encoder body so earlier
/// dispatches in the segment are discarded.
bool encode_moe_shared_ffn(Context& context, Encoder& encoder,
                           std::uint32_t rows,
                           const MoeSharedFFNBuffers& buffers,
                           std::string& error);
/// Standalone transactional form. A failure discards every stage.
bool encode_moe_shared_ffn(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows,
                           const MoeSharedFFNBuffers& buffers,
                           std::string& error);

/// Encode the shared MoE FFN/combine, then apply its output to the GR residual
/// through `strata_gr_post` using an in-place residual write. The residual and
/// injection must be disjoint from every shared-FFN input, scratch, output
/// and sink. All views must remain alive until command completion; the
/// read-only FFN inputs, injection and initial residual contents must remain
/// unchanged until then. Do not access written scratch or output before
/// completion. The caller-owned form must propagate false from its Encoder
/// body so failures discard the entire segment, including earlier shared-FFN
/// dispatches.
bool encode_moe_shared_ffn_write(Context& context, Encoder& encoder,
                                 std::uint32_t rows,
                                 const MoeSharedFFNWriteBuffers& buffers,
                                 std::string& error);
/// Standalone transactional form. A failure discards every stage.
bool encode_moe_shared_ffn_write(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const MoeSharedFFNWriteBuffers& buffers, std::string& error);

/// Encode routed experts, the shared expert and combine, then apply the
/// combined output to the GR residual. The shared FFN's Q8_1 input and routed
/// parts must refer to the routed producer's input and parts views. All other
/// routed outputs must be disjoint from the active shared inputs, outputs,
/// residual and injection ranges. Cross-stage alias checks use the byte
/// extents consumed or written by each stage, not unused view tails. Tiered
/// callers must submit only after successful expert resolution. Keep all views
/// alive, keep read inputs, injection and the initial residual unchanged until
/// completion, and avoid accessing outputs until completion. Propagate false
/// from a caller-owned Encoder body so any late failure discards the entire
/// sequence.
bool encode_moe_routed_shared_ffn_write(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const MoeRoutedSharedFFNWriteBuffers& buffers, std::string& error);
/// Standalone transactional form. A failure discards every stage.
bool encode_moe_routed_shared_ffn_write(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const MoeRoutedSharedFFNWriteBuffers& buffers, std::string& error);

}  // namespace strata::metal
