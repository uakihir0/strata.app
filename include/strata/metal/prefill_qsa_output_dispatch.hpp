// Ordered Prefill QSA gate and native attention-output projection.
#pragma once

#include "strata/metal/prefill_native_projection.hpp"
#include "strata/metal/prefill_qsa_gate_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillQsaOutputBuffers {
    PrefillQsaGateAttnBuffers gate;
    ReadOnlyBufferRef native_weights;
    BufferRef dequantized_f16;
    BufferRef y_f32;
};

/// Gate F32 sparse attention with Qf, then project the F16 result through
/// native `attn_output` weights. The fixed projection is 6144 -> 2560 and
/// writes F32 rows. The caller owns all buffers and any optional FWHT that
/// precedes the gate.
///
/// The caller-owned Encoder form requires ordered dispatch and must propagate
/// false from its enclosing Encoder::encode body so a late projection failure
/// discards the already-encoded gate.
bool encode_prefill_qsa_output(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillNativeWeightFormat format,
    const PrefillQsaOutputBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

/// Transactional standalone wrapper. A failure after gate dispatch discards
/// the complete unsubmitted QSA output segment.
bool encode_prefill_qsa_output(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillNativeWeightFormat format,
    const PrefillQsaOutputBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend = nullptr);

}  // namespace strata::metal
