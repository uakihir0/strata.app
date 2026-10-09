// GPU encoding for Prefill's post-projection GDN gates (docs/mac/04 §6.4).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGDNGatesBuffers {
    ReadOnlyBufferRef alpha_beta;
    ReadOnlyBufferRef dt_bias;
    ReadOnlyBufferRef a;
    BufferRef gate;
    BufferRef beta;
};

/// Transform the T×96 F32 alpha/beta projection into T×48 GDN gate and beta
/// rows. Input views are read-only and may overlap; outputs must be disjoint
/// from every input and each other.
bool encode_prefill_gdn_gates(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNGatesBuffers& buffers, std::string& error);

/// Standalone transactional wrapper for Prefill GDN gate generation.
bool encode_prefill_gdn_gates(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNGatesBuffers& buffers, std::string& error);

}  // namespace strata::metal
