// src/metal/engine/prefill_dispatch.mm - composable Prefill embedding stage.
#include "strata/metal/prefill_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "embedding_encode_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace strata::metal {
namespace {

bool same_params(const PrefillParams& left, const PrefillParams& right) {
    return left.T == right.T && left.p0 == right.p0 &&
           left.idx_f == right.idx_f && left.idx_e == right.idx_e;
}

bool same_step(const PrefillStepRow& left, const PrefillStepRow& right) {
    return left.pos == right.pos && left.n_kv == right.n_kv &&
           left.n_bid == right.n_bid && left.width == right.width;
}

bool metadata_matches_builder(const PrefillChunkMetadata& chunk,
                              std::string& error) {
    PrefillChunkMetadata expected;
    if (!build_prefill_chunk_metadata(chunk.params.p0, chunk.tokens, expected,
                                      error)) {
        error = "Prefill embedding metadata is invalid: " + error;
        return false;
    }
    if (!same_params(chunk.params, expected.params) ||
        chunk.batch_rows != expected.batch_rows ||
        chunk.has_image_rows != expected.has_image_rows ||
        chunk.steps.size() != expected.steps.size()) {
        error = "Prefill embedding metadata does not match its token rows";
        return false;
    }
    for (std::size_t row = 0u; row < chunk.steps.size(); ++row) {
        if (!same_step(chunk.steps[row], expected.steps[row])) {
            error = "Prefill embedding StepRow does not match its token row";
            return false;
        }
    }
    return true;
}

}  // namespace

namespace {

bool validate_prefill_embedding(Context& context,
                                const PrefillChunkMetadata& chunk,
                                std::uint32_t vocabulary_size,
                                EmbedRowsFormat format,
                                const PrefillEmbeddingBuffers& buffers,
                                std::string& error) {
    if (!metadata_matches_builder(chunk, error)) return false;

    detail::EmbedRowsDispatch rows_dispatch;
    if (!detail::prepare_embed_rows(context, chunk.params.T, vocabulary_size,
                                    format, buffers.rows, rows_dispatch,
                                    error)) {
        return false;
    }

    const std::size_t token_bytes =
        chunk.tokens.size() * sizeof(std::uint32_t);
    if (std::memcmp(buffers.rows.tokens.contents, chunk.tokens.data(),
                    token_bytes) != 0) {
        error = "Prefill embedding token buffer does not match chunk metadata";
        return false;
    }

    if (chunk.has_image_rows) {
        const EmbedImageBuffers image_buffers{
            buffers.rows.tokens,
            buffers.image,
            buffers.rows.embedding,
        };
        detail::EmbedImageDispatch image_dispatch;
        if (!detail::prepare_embed_image(context, chunk.params.T,
                                         image_buffers, image_dispatch, error)) {
            return false;
        }
    }

    const GRBroadcastBuffers broadcast_buffers{
        buffers.rows.embedding,
        buffers.streams,
    };
    detail::GRBroadcastDispatch broadcast_dispatch;
    return detail::prepare_gr_broadcast(context, chunk.params.T,
                                        broadcast_buffers,
                                        broadcast_dispatch, error);
}

}  // namespace

bool encode_prefill_embedding(Context& context, Encoder& encoder,
                              const PrefillChunkMetadata& chunk,
                              std::uint32_t vocabulary_size,
                              EmbedRowsFormat format,
                              const PrefillEmbeddingBuffers& buffers,
                              std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill embedding requires an ordered concurrent-group boundary";
        return false;
    }
    if (!validate_prefill_embedding(context, chunk, vocabulary_size, format,
                                    buffers, error)) {
        return false;
    }

    if (!encode_embed_rows(context, encoder, chunk.params.T, vocabulary_size,
                           format, buffers.rows, error)) {
        return false;
    }
    if (chunk.has_image_rows) {
        const EmbedImageBuffers image_buffers{
            buffers.rows.tokens,
            buffers.image,
            buffers.rows.embedding,
        };
        if (!encode_embed_image(context, encoder, chunk.params.T,
                                image_buffers, error)) {
            return false;
        }
    }
    const GRBroadcastBuffers broadcast_buffers{
        buffers.rows.embedding,
        buffers.streams,
    };
    return encode_gr_broadcast(context, encoder, chunk.params.T,
                               broadcast_buffers, error);
}

bool encode_prefill_embedding(Context& context, CommandBuffer& command_buffer,
                              const PrefillChunkMetadata& chunk,
                              std::uint32_t vocabulary_size,
                              EmbedRowsFormat format,
                              const PrefillEmbeddingBuffers& buffers,
                              std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill embedding stage",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_embedding(context, encoder, chunk,
                                            vocabulary_size, format, buffers,
                                            encode_error);
        },
        error);
}

}  // namespace strata::metal
