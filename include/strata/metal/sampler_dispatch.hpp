// GPU encoding for the Metal sampler kernels (docs/mac/04-metal-kernels.md §5.12).
#pragma once

#include "strata/metal/context.hpp"

namespace strata::metal {
class Encoder;
}

#include <cstdint>
#include <string>

namespace strata::metal {

/// All buffers passed to one encoder must be views owned by the same Context
/// used to create its CommandBuffer. Encoders verify each view's handle, CPU
/// pointer and GPU address against its Metal allocation. Bound views must not
/// overlap within a dispatch. Use the CommandBuffer wrapper for encoding and
/// submission; do not submit its unsafe native handle directly.
/// Buffers shared by the greedy or split sampler encoders.
struct SamplerBuffers {
    BufferRef logits;
    BufferRef history;
    BufferRef rows;
    BufferRef candidates;
    BufferRef selected;
    BufferRef output;
};

/// Encoders append work to a caller-owned command buffer. If an encoder returns
/// false after earlier sampler stages were encoded, discard that whole buffer.
/// Encode greedy argmax work into a caller-owned, not-yet-submitted command buffer.
bool encode_sampler_greedy(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, std::uint32_t n_vocab,
                           const SamplerBuffers& buffers, std::string& error);
/// Append greedy argmax work to a caller-owned Encoder segment. Return false
/// from the enclosing Encoder::encode body to discard the whole segment.
bool encode_sampler_greedy(Context& context, Encoder& encoder,
                           std::uint32_t rows, std::uint32_t n_vocab,
                           const SamplerBuffers& buffers, std::string& error);

/// Encode split top-k and merge. If penalties_already_applied is true, the
/// split kernel reads logits as-is; use this after the coupled penalty kernel.
bool encode_sampler_sampled(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows, std::uint32_t n_vocab,
                            const SamplerBuffers& buffers,
                            bool penalties_already_applied, std::string& error,
                            std::uint32_t kpart = 0u);
/// Append split top-k and merge to a caller-owned Encoder segment. The two
/// dependent passes cannot run inside one open concurrent dispatch group.
/// Return false from the enclosing Encoder::encode body to discard the segment.
bool encode_sampler_sampled(Context& context, Encoder& encoder,
                            std::uint32_t rows, std::uint32_t n_vocab,
                            const SamplerBuffers& buffers,
                            bool penalties_already_applied, std::string& error,
                            std::uint32_t kpart = 0u);

/// Encode one split top-k pass or merge pass when the caller needs a host
/// synchronization point between them.
bool encode_sampler_split_part(Context& context, CommandBuffer& command_buffer,
                               std::uint32_t rows, std::uint32_t n_vocab,
                               const SamplerBuffers& buffers,
                               bool penalties_already_applied, std::uint32_t kpart,
                               std::string& error);
/// Append one split top-k pass to a caller-owned Encoder segment. Close any
/// concurrent group before a dependent merge pass.
bool encode_sampler_split_part(Context& context, Encoder& encoder,
                               std::uint32_t rows, std::uint32_t n_vocab,
                               const SamplerBuffers& buffers,
                               bool penalties_already_applied, std::uint32_t kpart,
                               std::string& error);
bool encode_sampler_split_merge(Context& context, CommandBuffer& command_buffer,
                                std::uint32_t rows, std::uint32_t n_vocab,
                                BufferRef rows_buffer, BufferRef candidates,
                                BufferRef selected, std::uint32_t kpart,
                                std::string& error);
/// Append the merge pass to a caller-owned Encoder segment. The caller must
/// order the split-part producer before this consumer; an open concurrent
/// dispatch group is rejected because its automatic buffer barrier is disabled.
bool encode_sampler_split_merge(Context& context, Encoder& encoder,
                                std::uint32_t rows, std::uint32_t n_vocab,
                                BufferRef rows_buffer, BufferRef candidates,
                                BufferRef selected, std::uint32_t kpart,
                                std::string& error);

/// Encode the in-place coupled-draft penalty pass. Follow it with
/// encode_sampler_sampled(..., penalties_already_applied=true).
bool encode_sampler_coupled_penalty(Context& context, CommandBuffer& command_buffer,
                                    std::uint32_t n_vocab, std::uint32_t sampled_vocab,
                                    bool has_id_to_subset_map, BufferRef logits,
                                    BufferRef id_to_subset, BufferRef history_ring,
                                    BufferRef row, std::string& error);
/// Append the in-place coupled-draft penalty pass to a caller-owned Encoder
/// segment. End a concurrent group before the dependent sampled pass.
bool encode_sampler_coupled_penalty(Context& context, Encoder& encoder,
                                    std::uint32_t n_vocab, std::uint32_t sampled_vocab,
                                    bool has_id_to_subset_map, BufferRef logits,
                                    BufferRef id_to_subset, BufferRef history_ring,
                                    BufferRef row, std::string& error);

}  // namespace strata::metal
