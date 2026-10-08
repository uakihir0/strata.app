// GPU encoding for the final GR head mixer in a Window.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class CommandBuffer;
class Encoder;

struct WindowHeadMixerBuffers {
    BufferRef residual;
    ReadOnlyBufferRef norm_gamma;
    ReadOnlyBufferRef down_weights;
    ReadOnlyBufferRef up_weights;
    ReadOnlyBufferRef active_rows;
    BufferRef normalized;
    BufferRef low_rank;
    BufferRef gate;
    BufferRef mixed;
    BufferRef matvec_sink;
};

/// Encode the ordered final Window mixer: weighted RMS norm, BF16 down
/// projection, in-place down SiLU, BF16 up projection, and pre-gated mix.
/// `active_rows` must contain `rows`; mat-vec compiled width is the smallest
/// supported width at least `rows`. All scratch outputs must be disjoint.
/// `residual` and `normalized` may be the exact same view for in-place norm.
/// Read-only weights and active-row input must remain unchanged through
/// command completion. Residual is read only unless it exactly aliases
/// normalized. The Encoder form may have appended earlier stages when it
/// returns false; propagate false from the enclosing Encoder::encode body so
/// that segment is discarded. Call outside an open concurrent-dispatch group
/// so dependent stages receive their required barriers.
bool encode_window_head_mixer(Context& context, Encoder& encoder,
                              std::uint32_t rows,
                              const WindowHeadMixerBuffers& buffers,
                              std::string& error);

/// Standalone command-buffer form. A failed stage discards the unsubmitted
/// command buffer with the entire mixer segment.
bool encode_window_head_mixer(Context& context,
                              CommandBuffer& command_buffer,
                              std::uint32_t rows,
                              const WindowHeadMixerBuffers& buffers,
                              std::string& error);

}  // namespace strata::metal
