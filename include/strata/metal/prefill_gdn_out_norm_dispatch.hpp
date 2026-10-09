// GPU encoding for Prefill GDN output normalization (docs/mac/04 §6.4).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGDNOutNormBuffers {
    ReadOnlyBufferRef oc;
    ReadOnlyBufferRef z;
    ReadOnlyBufferRef gamma;
    BufferRef y_h;
};

/// Normalize and gate raw GDN recurrent outputs into F16 y_h rows.
/// Read-only inputs may overlap; y_h must be disjoint from every input.
bool encode_prefill_gdn_out_norm(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNOutNormBuffers& buffers, std::string& error);

/// Standalone transactional wrapper for Prefill GDN output normalization.
bool encode_prefill_gdn_out_norm(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNOutNormBuffers& buffers, std::string& error);

}  // namespace strata::metal
