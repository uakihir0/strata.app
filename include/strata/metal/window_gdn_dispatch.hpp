// Ordered speculative Window GDN block encoding (docs/mac/04 §5.5).
#pragma once

#include "strata/metal/gdn_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct WindowGDNBlockBuffers {
    BufferRef mixed;
    BufferRef history;
    BufferRef state;
    MatVecWeightFormat qkv_format = MatVecWeightFormat::q8_0;
    BufferRef qkv_weights;
    BufferRef conv_weights;
    BufferRef alpha;
    BufferRef beta_weights;
    BufferRef dt_bias;
    BufferRef a;
    MatVecWeightFormat gate_format = MatVecWeightFormat::q8_0;
    BufferRef gate_weights;
    BufferRef gamma;
    MatVecWeightFormat output_format = MatVecWeightFormat::q8_0;
    BufferRef output_weights;
    BufferRef xq;
    BufferRef qkv;
    BufferRef h;
    BufferRef gate;
    BufferRef beta;
    BufferRef z;
    BufferRef y;
    BufferRef yq;
    BufferRef block_out;
    BufferRef sink;
    BufferRef active_rows;
};

/// Inputs and scratch for committing one GDN layer's accepted Window prefix.
/// `output_scratch` is a required, unused output binding for the recurrent
/// commit kernel; all views must be Context-owned and pairwise disjoint.
struct WindowGDNCommitBuffers {
    BufferRef history;
    BufferRef qkv;
    BufferRef state;
    BufferRef h;
    BufferRef gate;
    BufferRef beta;
    BufferRef z;
    BufferRef gamma;
    BufferRef output_scratch;
};

/// Encode the accepted prefix in documented order: advance convolution history,
/// then replay the GDN recurrence into state. Rows may be 0..8; zero is a no-op.
/// The caller-owned form requires an ordered segment and must propagate false
/// from its enclosing Encoder::encode body.
bool encode_window_gdn_commit(Context& context, Encoder& encoder,
                              std::uint32_t rows,
                              const WindowGDNCommitBuffers& buffers,
                              std::string& error);
/// Standalone transactional form. Any late validation failure discards both
/// the history update and the state replay.
bool encode_window_gdn_commit(Context& context, CommandBuffer& command_buffer,
                              std::uint32_t rows,
                              const WindowGDNCommitBuffers& buffers,
                              std::string& error);

/// Encode the speculative GDN block for T in 1..8 in documented order:
/// mixed→Q8_1→qkv→conv/L2, then alpha/beta, gate projection, window
/// recurrence/norm, Q8_1, and output projection. Q8_1 matvec stages require
/// quantized (non-BF16) weight formats. Every supplied view must be Context
/// owned and pairwise disjoint; callers must keep read-only views unchanged
/// through command completion. The GDN state and convolution history are
/// read-only in window mode and are committed separately after acceptance.
bool encode_window_gdn_block(Context& context, Encoder& encoder,
                             std::uint32_t rows,
                             const WindowGDNBlockBuffers& buffers,
                             std::string& error);
/// Add the complete block to a standalone transactional command buffer.
bool encode_window_gdn_block(Context& context, CommandBuffer& command_buffer,
                             std::uint32_t rows,
                             const WindowGDNBlockBuffers& buffers,
                             std::string& error);

}  // namespace strata::metal
