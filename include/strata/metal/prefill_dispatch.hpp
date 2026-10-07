// GPU encoding for the embedding stage of a Metal Prefill chunk (docs/mac/04 §6.7).
#pragma once

#include "strata/metal/embed_dispatch.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/prefill.hpp"

namespace strata::metal {

struct PrefillEmbeddingBuffers {
    EmbedRowsBuffers rows;
    BufferRef image;
    BufferRef streams;
};

/// Encode gather -> optional image copy -> GR broadcast in one compute-encoder
/// segment, preserving the untracked-buffer dependencies between dispatches.
///
/// `rows.tokens` must contain the exact token words used to build `chunk`, and
/// the caller must keep all input views alive and unchanged until completion.
/// The caller must propagate failure from its Encoder body so earlier work in
/// the segment is discarded. This sequence cannot run inside an open concurrent
/// dispatch group because GR broadcast reads rows written by earlier passes.
bool encode_prefill_embedding(Context& context, Encoder& encoder,
                              const PrefillChunkMetadata& chunk,
                              std::uint32_t vocabulary_size,
                              EmbedRowsFormat format,
                              const PrefillEmbeddingBuffers& buffers,
                              std::string& error);

/// Standalone convenience wrapper for a caller-supplied fresh command buffer.
/// It routes the same validated stages through one Encoder segment. A token
/// mismatch or any later-stage validation failure discards that command buffer.
bool encode_prefill_embedding(Context& context, CommandBuffer& command_buffer,
                              const PrefillChunkMetadata& chunk,
                              std::uint32_t vocabulary_size,
                              EmbedRowsFormat format,
                              const PrefillEmbeddingBuffers& buffers,
                              std::string& error);

}  // namespace strata::metal
