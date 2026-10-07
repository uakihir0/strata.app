// GPU encoding for the Metal PLE kernels (docs/mac/04 §5.3).
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/mv_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PLEBroadcastBuffers {
    BufferRef value;
    BufferRef gate;
    BufferRef gated;
};

struct PLEGateBuffers {
    BufferRef key;
    BufferRef query;
    BufferRef gate;
};

struct PLEConvResidualBuffers {
    BufferRef ple_norm;
    BufferRef history;
    BufferRef conv1d;
    BufferRef gated;
    BufferRef residual;
};

struct PLEHistoryBuffers {
    BufferRef history;
    BufferRef ple_norm;
};

struct PLEPostProjectionBuffers {
    BufferRef key;
    BufferRef residual;
    BufferRef query;
    BufferRef ple_norm;
    BufferRef value;
    BufferRef gate;
    BufferRef gated;
    BufferRef history;
    BufferRef conv1d;
    BufferRef key_gamma;
    BufferRef query_gamma;
    BufferRef conv_gamma;
};

/// Buffers for the complete decode PLE block. `input_f32` contains rows of
/// 2560 values; `post.key` and `post.value` receive the two projections.
/// `key_q8_1` is required for Q2_0/Q8_0 keys and unused for BF16 keys.
/// `matvec_sink` is shared by the serial key and value projections.
struct PLEBlockBuffers {
    BufferRef input_f32;
    BufferRef key_q8_1;
    BufferRef key_weights;
    BufferRef value_weights;
    BufferRef matvec_sink;
    BufferRef active_rows;
    PLEPostProjectionBuffers post;
};

enum class PLEHistoryUpdate : std::uint32_t {
    defer,
    advance_rows,
};

/// Encode the full decode PLE sequence for 1..8 token rows in one serial
/// Encoder segment: optional Q8_1 quantization and key projection, the
/// F32-activation BF16 value projection, then the post-projection sequence.
/// Only Q2_0, Q8_0 and BF16 key formats are supported. The caller must
/// propagate false from its enclosing `Encoder::encode` body so partial
/// dispatches are discarded. Projection inputs, weights, outputs and scratch
/// must obey the documented non-aliasing contract.
bool encode_ple_block(
    Context& context, Encoder& encoder, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const PLEBlockBuffers& buffers,
    std::string& error);
/// Standalone form. Any validation/encoding failure discards the whole
/// command buffer segment.
bool encode_ple_block(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const PLEBlockBuffers& buffers,
    std::string& error);

/// Encode the PLE post-projection sequence for rows in 1..8:
/// weighted key/query norms, gate, value broadcast, convolution norm,
/// and convolution-residual update. History advances by `rows` only when
/// `history_update` is `advance_rows`; `defer` leaves it unchanged so a
/// speculative Window can update only its accepted prefix later at commit.
/// `key` is normalized in place and `residual` is preserved until its final
/// in-place update. `query` and `ple_norm` must be disjoint or exact aliases;
/// the latter matches the documented scratch lifetime because query is dead
/// after the gate. Other buffer roles must be disjoint. A false return may
/// follow earlier dispatches being added to `encoder`; propagate it from the
/// enclosing `Encoder::encode` body so that the segment is discarded.
bool encode_ple_post_projection(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PLEHistoryUpdate history_update,
    const PLEPostProjectionBuffers& buffers, std::string& error);
/// Standalone form. A validation/encoding failure discards the command buffer.
bool encode_ple_post_projection(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PLEHistoryUpdate history_update,
    const PLEPostProjectionBuffers& buffers, std::string& error);

/// Encode the PLE gate reduction for rows in 1..8.
/// Read-only key and query views may overlap; the gate output must be disjoint.
/// The Encoder overload adds this dispatch to a caller-owned segment; the
/// Window path must create its documented serial segment.
bool encode_ple_gate(Context& context, Encoder& encoder, std::uint32_t rows,
                     const PLEGateBuffers& buffers, std::string& error);
bool encode_ple_gate(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t rows, const PLEGateBuffers& buffers,
                     std::string& error);

/// Encode batched PLE convolution and residual for rows in 1..8.
/// `residual` is updated in place and must be disjoint from all read-only views.
/// The Encoder overload adds this dispatch to a caller-owned segment; the
/// Window path must create its documented serial segment.
bool encode_ple_conv_residual(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PLEConvResidualBuffers& buffers, std::string& error);
bool encode_ple_conv_residual(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PLEConvResidualBuffers& buffers, std::string& error);

/// Encode the PLE value broadcast for rows in 1..8.
/// Read-only value and gate views may overlap; the output must be disjoint.
/// The Encoder overload adds this dispatch to a caller-owned segment; the
/// Window path must create its documented serial segment.
bool encode_ple_broadcast(Context& context, Encoder& encoder,
                          std::uint32_t rows,
                          const PLEBroadcastBuffers& buffers,
                          std::string& error);
bool encode_ple_broadcast(Context& context, CommandBuffer& command_buffer,
                          std::uint32_t rows,
                          const PLEBroadcastBuffers& buffers,
                          std::string& error);

/// Encode PLE history advancement for `n` input rows, updating history in place.
/// `ple_norm` must contain `n` rows; it must be disjoint from `history`.
/// The Encoder overload adds this dispatch to a caller-owned segment; the
/// Window path must create its documented serial segment.
bool encode_ple_history(Context& context, Encoder& encoder, std::uint32_t n,
                        const PLEHistoryBuffers& buffers,
                        std::string& error);
bool encode_ple_history(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t n,
                        const PLEHistoryBuffers& buffers,
                        std::string& error);

}  // namespace strata::metal
