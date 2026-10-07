// GPU encoding for the Metal embedding kernels (docs/mac/04 §6.7).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

enum class EmbedRowsFormat : std::uint32_t {
    q3_k = 0,
    iq3_s = 1,
    iq4_xs = 2,
    bf16 = 3,
};

struct EmbedRowsBuffers {
    BufferRef table;
    BufferRef tokens;
    BufferRef embedding;
};

struct EmbedImageBuffers {
    BufferRef tokens;
    BufferRef image;
    BufferRef embedding;
};

/// Encode text-token embedding gathers for a positive, shader-index-safe row
/// count. Non-image token ids must be below `vocabulary_size`; image-marked
/// rows are left untouched. The output must not overlap either input, while
/// the read-only table and token views may overlap. The caller must keep the
/// input views alive, and token values unchanged, until command completion.
bool encode_embed_rows(Context& context, CommandBuffer& command_buffer,
                       std::uint32_t rows, std::uint32_t vocabulary_size,
                       EmbedRowsFormat format,
                       const EmbedRowsBuffers& buffers, std::string& error);
/// Add text-token embedding gathers to a caller-owned Encoder segment.
bool encode_embed_rows(Context& context, Encoder& encoder,
                       std::uint32_t rows, std::uint32_t vocabulary_size,
                       EmbedRowsFormat format,
                       const EmbedRowsBuffers& buffers, std::string& error);

/// Encode image-row copies for a prefill chunk with a positive, shader-index-safe
/// row count. The embedding output must not overlap either input; the two
/// read-only inputs may overlap.
bool encode_embed_image(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, const EmbedImageBuffers& buffers,
                        std::string& error);
/// Add image-row copies to a caller-owned Encoder segment.
bool encode_embed_image(Context& context, Encoder& encoder,
                        std::uint32_t rows, const EmbedImageBuffers& buffers,
                        std::string& error);

}  // namespace strata::metal
