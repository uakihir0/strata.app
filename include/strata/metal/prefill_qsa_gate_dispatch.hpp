// Validated encoder for Prefill's QSA attention FWHT gate stage.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillQsaGateAttnBuffers {
    ReadOnlyBufferRef attn;
    ReadOnlyBufferRef q_full;
    BufferRef attn_h;
};

/// Gate the 24×256 F32 attention rows by the second half of Qf and store F16.
/// The caller must encode the attention FWHT earlier in the same ordered
/// segment when the model's QSA layer enables that transform.
bool encode_prefill_qsa_gate_attn(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillQsaGateAttnBuffers& buffers, std::string& error);

/// Transactional standalone wrapper. A failure discards earlier work in the
/// same unsubmitted command buffer.
bool encode_prefill_qsa_gate_attn(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillQsaGateAttnBuffers& buffers, std::string& error);

}  // namespace strata::metal
