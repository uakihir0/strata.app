// GPU encoding for the Metal GDN kernels (docs/mac/04 §5.5).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

enum class GDNStateMode : std::uint32_t {
    window = 0,
    commit = 1,
    write_through = 2,
};

struct GDNConvL2Buffers {
    BufferRef history;
    BufferRef qkv;
    BufferRef conv_weights;
    BufferRef output;
};

struct GDNABBuffers {
    BufferRef input;
    BufferRef alpha;
    BufferRef beta_weights;
    BufferRef dt_bias;
    BufferRef a;
    BufferRef gate;
    BufferRef beta;
};

struct GDNStepNormBuffers {
    BufferRef state;
    BufferRef h;
    BufferRef gate;
    BufferRef beta;
    BufferRef z;
    BufferRef gamma;
    BufferRef output;
};

struct GDNConvCommitBuffers {
    BufferRef history;
    BufferRef qkv;
};

/// Encode the T-row convolution and q/k L2 normalization for T in 1..8.
/// Input views are read-only and may overlap one another. `output` must be
/// disjoint from all three inputs; every view and the command buffer must
/// belong to `context`.
bool encode_gdn_conv_l2(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, const GDNConvL2Buffers& buffers,
                        std::string& error);
/// Add convolution and q/k L2 normalization to a caller-owned Encoder
/// segment, with the same view checks as the standalone overload. The three
/// read-only inputs may overlap; output must be disjoint from each. Propagate
/// false from the enclosing Encoder::encode body so the complete segment is
/// discarded.
bool encode_gdn_conv_l2(Context& context, Encoder& encoder,
                        std::uint32_t rows, const GDNConvL2Buffers& buffers,
                        std::string& error);

/// Encode the alpha/beta projection and gate transforms for T in 1..8.
/// BF16 projection weights require 16-byte aligned offsets. Outputs must be
/// disjoint from one another and every input.
bool encode_gdn_ab(Context& context, CommandBuffer& command_buffer,
                   std::uint32_t rows, const GDNABBuffers& buffers,
                   std::string& error);
/// Add the alpha/beta projections and gate transforms to a caller-owned
/// Encoder segment. The seven buffer views must satisfy the same extent,
/// alignment, ownership and output-disjointness checks as the standalone
/// overload. Propagate false from the enclosing Encoder::encode body.
bool encode_gdn_ab(Context& context, Encoder& encoder, std::uint32_t rows,
                   const GDNABBuffers& buffers, std::string& error);

/// Encode one recurrence pass. `window` emits every row without writing the
/// state; `commit` replays rows 0..rows-1 into the state without output;
/// `write_through` emits every row and writes the final state. Commit accepts
/// rows=0 as a no-op; the other modes require rows in 1..8.
bool encode_gdn_step_norm(Context& context, CommandBuffer& command_buffer,
                          std::uint32_t rows, GDNStateMode mode,
                          const GDNStepNormBuffers& buffers,
                          std::string& error);
/// Add one recurrence pass to a caller-owned Encoder segment. Commit with
/// rows=0 remains a no-op; for other calls, preserve the standalone buffer,
/// alias and state-mode rules. Propagate false from the enclosing
/// Encoder::encode body so the complete segment is discarded.
bool encode_gdn_step_norm(Context& context, Encoder& encoder,
                          std::uint32_t rows, GDNStateMode mode,
                          const GDNStepNormBuffers& buffers,
                          std::string& error);

/// Advance the three-row convolution history by `rows` rows (0..8). Zero is
/// a no-op. The history is updated only after all required old taps are read.
bool encode_gdn_conv_commit(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows,
                            const GDNConvCommitBuffers& buffers,
                            std::string& error);
/// Add convolution-history advancement to a caller-owned Encoder segment.
/// Zero rows remains a no-op; otherwise the views must meet the standalone
/// extent, alignment, ownership and non-overlap checks. Propagate false from
/// the enclosing Encoder::encode body so the complete segment is discarded.
bool encode_gdn_conv_commit(Context& context, Encoder& encoder,
                            std::uint32_t rows,
                            const GDNConvCommitBuffers& buffers,
                            std::string& error);

}  // namespace strata::metal
