// Ordered Prefill GDN post-projection transform chain (docs/mac/04 §6.4).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

/// Buffers for one Prefill GDN layer chunk after its projections are ready.
/// history and state are persistent in-place state. gate, beta, h, oc and y_h
/// are caller-owned scratch/output buffers.
struct PrefillGDNTransformBuffers {
    ReadOnlyBufferRef alpha_beta;
    ReadOnlyBufferRef dt_bias;
    ReadOnlyBufferRef a;
    BufferRef history;
    ReadOnlyBufferRef qkv;
    ReadOnlyBufferRef conv_weights;
    BufferRef gate;
    BufferRef beta;
    BufferRef h;
    BufferRef state;
    BufferRef oc;
    ReadOnlyBufferRef z;
    ReadOnlyBufferRef gamma;
    BufferRef y_h;
};

/// Encode GDN gates, convolution, history advance, q/k normalization,
/// recurrence, and output normalization in the required order.
///
/// All views must belong to context and cover the active T-row shapes. Writable
/// stage ranges must be pairwise disjoint and disjoint from every read-only
/// range; read-only inputs may overlap one another. history and state are
/// updated in place; h and oc are intentionally reused by later stages.
bool encode_prefill_gdn_transforms(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNTransformBuffers& buffers, std::string& error);

/// Transactional standalone form. A validation or encoding failure discards
/// the entire command buffer, including any earlier encoded GDN stages.
bool encode_prefill_gdn_transforms(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNTransformBuffers& buffers, std::string& error);

}  // namespace strata::metal
