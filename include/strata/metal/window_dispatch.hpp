// T=1 Window input-stage composition for the Metal engine.
#pragma once

#include "strata/metal/embed_dispatch.hpp"
#include "strata/metal/window_params.hpp"

namespace strata::metal {

class Encoder;

struct WindowEmbeddingBuffers {
    /// The complete ABI-compatible block built by WindowParamsRing.
    BufferRef params;
    /// Read-only token embedding tensor from full-mode model storage.
    ReadOnlyBufferRef table;
    /// T=1 embedding and four-stream residual scratch regions.
    BufferRef embedding;
    BufferRef streams;
};

/// Encode the T=1 input stem: gather the token from WindowParams, then
/// broadcast the embedding into the four initial GR streams. The views must
/// remain alive and the parameter block must remain unchanged until GPU
/// completion. This does not encode the Window's model layers or imply that a
/// complete decode is available. When using the caller-owned Encoder overload,
/// propagate a false return from the enclosing Encoder::encode body so any
/// earlier gather in that segment is discarded.
bool encode_window_embedding(Context& context, Encoder& encoder,
                             std::uint32_t vocabulary_size,
                             EmbedRowsFormat format,
                             const WindowEmbeddingBuffers& buffers,
                             std::string& error);

/// Transactional standalone wrapper for a fresh command buffer.
bool encode_window_embedding(Context& context, CommandBuffer& command_buffer,
                             std::uint32_t vocabulary_size,
                             EmbedRowsFormat format,
                             const WindowEmbeddingBuffers& buffers,
                             std::string& error);

}  // namespace strata::metal
