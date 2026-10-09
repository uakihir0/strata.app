// GPU encoding for the Prefill GDN recurrent state update (docs/mac/04 §6.4).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGDNRecBuffers {
    BufferRef state;
    ReadOnlyBufferRef h;
    ReadOnlyBufferRef gate;
    ReadOnlyBufferRef beta;
    BufferRef output;
};

/// Update the GDN state across rows and write raw recurrent outputs.
/// Read-only inputs may overlap one another; state and output must be
/// disjoint from every other bound view.
bool encode_prefill_gdn_rec(Context& context, Encoder& encoder,
                            std::uint32_t rows,
                            const PrefillGDNRecBuffers& buffers,
                            std::string& error);

/// Standalone transactional wrapper for Prefill GDN recurrence.
bool encode_prefill_gdn_rec(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows,
                            const PrefillGDNRecBuffers& buffers,
                            std::string& error);

}  // namespace strata::metal
