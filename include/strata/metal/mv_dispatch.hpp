// include/strata/metal/mv_dispatch.hpp - validated Q8_1 and dense mat-vec encoders.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

enum class MatVecWeightFormat : std::uint8_t {
    q8_0,
    q4_0,
    q5_0,
    q2_0,
    iq4_nl,
    iq4_xs,
    iq3_s,
    q4_k,
    q5_k,
    q3_k,
    q6_k,
    bf16,
};

struct Q8_1Buffers {
    BufferRef input_f32;
    BufferRef output_q8_1;
};

/// Quantize row-major float activations to Q8_1, matching `strata_q_q8_1`.
/// Width must be a supported multiple of 32. Input values must remain
/// unchanged until the command buffer completes.
bool encode_q8_1(Context& context, CommandBuffer& command_buffer,
                 std::uint32_t rows, std::uint32_t width,
                 const Q8_1Buffers& buffers, std::string& error);
/// Add the same validated quantization dispatch to an existing compute
/// segment. Use one segment when a later dispatch consumes these Q8_1 rows.
bool encode_q8_1(Context& context, Encoder& encoder,
                 std::uint32_t rows, std::uint32_t width,
                 const Q8_1Buffers& buffers, std::string& error);

struct MatVecBuffers {
    // Q8_1 rows for quantized formats; row-major F32 activations for BF16.
    BufferRef activations;
    BufferRef weights;
    // Active output columns are packed row-major, each with `n_out` values.
    BufferRef output;
    // Receives padded compiled columns; at least one float is required when
    // there are no padded columns.
    BufferRef sink;
    // Host-visible uint32 value equal to active_columns.
    BufferRef active_rows;
};

struct ReadOnlyBF16MatVecBuffers {
    // F32 activations and packed BF16 weights are MSL const inputs.
    ReadOnlyBufferRef activations;
    ReadOnlyBufferRef weights;
    // Host-visible uint32 value equal to active_columns.
    ReadOnlyBufferRef active_rows;
    // Active output columns are packed row-major; padded columns go to sink.
    BufferRef output;
    BufferRef sink;
};

struct ReadOnlyQ8_0MatVecBuffers {
    // Q8_1 activation rows and packed Q8_0 weights are MSL const inputs.
    ReadOnlyBufferRef activations;
    ReadOnlyBufferRef weights;
    // Host-visible uint32 value equal to active_columns.
    ReadOnlyBufferRef active_rows;
    // Active output columns are packed row-major; padded columns go to sink.
    BufferRef output;
    BufferRef sink;
};

struct IndexedHeadMatVecBuffers {
    MatVecBuffers matvec;
    // Host-visible int32 row indices: output row -> full-head weight row.
    BufferRef row_map;
};

/// Encode one dense projection with a compiled NC width of 1, 2, 4 or 8.
/// `active_columns` may be smaller than `compiled_columns`; the unused outputs
/// are written to `sink`. Non-indexed dense/head variants are supported.
///
/// All views must belong to `context`. Read-only inputs may alias each other;
/// output and sink must not overlap any input or each other. Callers must keep
/// inputs and `active_rows` unchanged until command completion.
bool encode_matvec(Context& context, CommandBuffer& command_buffer,
                   MatVecWeightFormat format, std::uint32_t n_in,
                   std::uint32_t n_out, std::uint32_t active_columns,
                   std::uint32_t compiled_columns,
                   const MatVecBuffers& buffers, std::string& error);
/// Add the same validated dense projection to an existing compute segment.
bool encode_matvec(Context& context, Encoder& encoder,
                   MatVecWeightFormat format, std::uint32_t n_in,
                   std::uint32_t n_out, std::uint32_t active_columns,
                   std::uint32_t compiled_columns,
                   const MatVecBuffers& buffers, std::string& error);

/// Encode a BF16 projection whose activations, weights and active-row count
/// remain read-only through the dispatch. Supported widths match the BF16
/// kernel catalogue (320, 2560 or 10240). Read-only inputs may alias each
/// other; output and sink must not overlap any input or each other. The
/// caller-owned form can fail after earlier work was encoded, so its enclosing
/// Encoder::encode body must propagate false. The CommandBuffer form discards
/// the unsubmitted command buffer on failure.
bool encode_bf16_matvec(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t n_in, std::uint32_t n_out,
                        std::uint32_t active_columns,
                        std::uint32_t compiled_columns,
                        const ReadOnlyBF16MatVecBuffers& buffers,
                        std::string& error);
bool encode_bf16_matvec(Context& context, Encoder& encoder,
                        std::uint32_t n_in, std::uint32_t n_out,
                        std::uint32_t active_columns,
                        std::uint32_t compiled_columns,
                        const ReadOnlyBF16MatVecBuffers& buffers,
                        std::string& error);

/// Encode the Q8_0-by-Q8_1 projection with immutable model and activation
/// inputs. `n_in` must be divisible by 32. Supports compiled widths 1, 2, 4
/// or 8 and the safe Q8_0 kernel catalogue. Read-only inputs may alias each
/// other; output and sink must not overlap any input or each other.
/// All views must remain alive until command completion; keep every
/// read-only input, including `active_rows`, unchanged for that interval.
/// Propagate false from a caller-owned Encoder body so earlier dispatches in
/// its unsubmitted segment are discarded.
bool encode_q8_0_matvec(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t n_in, std::uint32_t n_out,
                        std::uint32_t active_columns,
                        std::uint32_t compiled_columns,
                        const ReadOnlyQ8_0MatVecBuffers& buffers,
                        std::string& error);
bool encode_q8_0_matvec(Context& context, Encoder& encoder,
                        std::uint32_t n_in, std::uint32_t n_out,
                        std::uint32_t active_columns,
                        std::uint32_t compiled_columns,
                        const ReadOnlyQ8_0MatVecBuffers& buffers,
                        std::string& error);

/// Encode the indexed draft-vocabulary head for Q5_K, Q6_K or IQ4_XS.
/// `weight_rows` is the full head row count; `output_rows` is the row-map
/// length and must not exceed it. The map is validated as signed int32 indices
/// into the full head. This path uses one active and compiled column.
///
/// All views must belong to `context`. Read-only inputs may alias each other;
/// output and sink must not overlap any input or each other. Callers must keep
/// inputs, `row_map` and `active_rows` unchanged until command completion.
bool encode_indexed_head_matvec(
    Context& context, CommandBuffer& command_buffer,
    MatVecWeightFormat format, std::uint32_t n_in,
    std::uint32_t weight_rows, std::uint32_t output_rows,
    const IndexedHeadMatVecBuffers& buffers, std::string& error);

}  // namespace strata::metal
