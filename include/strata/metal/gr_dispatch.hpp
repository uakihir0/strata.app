// GPU encoding for the Metal GR control-vector kernel (docs/mac/04 §5.4).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

enum class GRControlMode : std::uint32_t {
    project = 0,
    add = 1,
};

struct GRControlVectorBuffers {
    BufferRef residual;
    BufferRef direction;
    BufferRef layer_scale;
    BufferRef enabled;
    BufferRef block_out;
    BufferRef inject_prev;
};

struct GRNormBuffers {
    BufferRef residual;
    BufferRef block_out;
    BufferRef norm;
    BufferRef inject_prev;
    BufferRef xn;
    BufferRef rs_out;
};

struct GRDownBuffers {
    BufferRef xn;
    BufferRef hc_down;
    BufferRef hc_inject;
    BufferRef lo;
    BufferRef inj_out;
};

struct GRUpBuffers {
    BufferRef residual;
    BufferRef block_out;
    BufferRef inject_prev;
    BufferRef norm;
    BufferRef rs;
    BufferRef lo;
    BufferRef hc_up;
    BufferRef mixed;
};

/// Buffers for the ordered GR read pipeline: normalize, project down and
/// inject, then project up and mix. All twelve views must be pairwise
/// disjoint, including read-only roles; stage inputs and outputs are wired
/// from these shared fields.
struct GRReadBuffers {
    BufferRef residual;
    BufferRef block_out;
    BufferRef inject_prev;
    BufferRef norm;
    BufferRef normalized;
    BufferRef rs;
    BufferRef hc_down;
    BufferRef hc_inject;
    BufferRef lo;
    BufferRef inject_out;
    BufferRef hc_up;
    BufferRef mixed;
};

struct GRPostBuffers {
    BufferRef residual;
    BufferRef block_out;
    BufferRef inject;
    BufferRef output;
};

struct GRBroadcastBuffers {
    BufferRef embedding;
    BufferRef streams;
};

/// Encoder form with a const embedding input and a writable stream output.
struct GRBroadcastReadOnlyBuffers {
    ReadOnlyBufferRef embedding;
    BufferRef streams;
};

/// Broadcast each 2,560-value embedding row into four GR streams for
/// shader-index-safe row counts. The input and output views must be disjoint
/// and owned by `context`; the command buffer must be unsubmitted.
bool encode_gr_broadcast(Context& context, CommandBuffer& command_buffer,
                         std::uint32_t rows,
                         const GRBroadcastBuffers& buffers,
                         std::string& error);
/// Add the stream broadcast to a caller-owned Encoder segment.
bool encode_gr_broadcast(Context& context, Encoder& encoder,
                         std::uint32_t rows,
                         const GRBroadcastBuffers& buffers,
                         std::string& error);
/// Add stream broadcast from a read-only embedding view.
bool encode_gr_broadcast(Context& context, Encoder& encoder,
                         std::uint32_t rows,
                         const GRBroadcastReadOnlyBuffers& buffers,
                         std::string& error);

/// Encode the native GR post write for rows in 1..8. Buffer views and the
/// command buffer must belong to `context` and remain in bounds. Exact
/// residual/output view alias is allowed; every other overlap is rejected.
bool encode_gr_post(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, const GRPostBuffers& buffers,
                    std::string& error);
/// Add the native GR post write to a caller-owned Encoder segment.
bool encode_gr_post(Context& context, Encoder& encoder,
                    std::uint32_t rows, const GRPostBuffers& buffers,
                    std::string& error);

/// Encode the in-place GR down-projection scale and SiLU for rows in 1..8.
/// The encoder derives the required float count from `rows`; the low-rank
/// view and command buffer must belong to `context`.
bool encode_gr_down_silu(Context& context, CommandBuffer& command_buffer,
                         std::uint32_t rows, BufferRef low_rank,
                         std::string& error);
/// Add the in-place GR down-projection scale and SiLU to a caller-owned
/// Encoder segment.
bool encode_gr_down_silu(Context& context, Encoder& encoder,
                         std::uint32_t rows, BufferRef low_rank,
                         std::string& error);

struct GRPreGatedBuffers {
    BufferRef xn;
    BufferRef gate;
    BufferRef mixed;
};

struct WeightedRMSNormBuffers {
    BufferRef input;
    BufferRef gamma;
    BufferRef output;
};

/// Read-only model gamma input for weighted RMS normalization. Keep `gamma`
/// unchanged until command-buffer completion. For disjoint input and output,
/// both input and gamma are read-only bindings and may overlap. Exact in-place
/// input/output remains supported when gamma is disjoint from the output.
struct ReadOnlyWeightedRMSNormBuffers {
    BufferRef input;
    ReadOnlyBufferRef gamma;
    BufferRef output;
};

/// Encode `strata_rms_norm_w` for documented widths and gamma periods.
/// Exact input/output view alias is allowed; all other output overlap is
/// rejected. The Encoder overload adds this dispatch to a caller-owned segment.
bool encode_weighted_rms_norm(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_cols, std::uint32_t gamma_period, float epsilon,
    const WeightedRMSNormBuffers& buffers, std::string& error);
bool encode_weighted_rms_norm(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_cols, std::uint32_t gamma_period, float epsilon,
    const WeightedRMSNormBuffers& buffers, std::string& error);
/// Read-only-gamma overload for immutable model weights.
/// If input and output are disjoint, they are bound read-only and gamma may
/// overlap input. With exact input/output aliasing, gamma must be disjoint.
bool encode_weighted_rms_norm(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_cols, std::uint32_t gamma_period, float epsilon,
    const ReadOnlyWeightedRMSNormBuffers& buffers, std::string& error);
bool encode_weighted_rms_norm(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_cols, std::uint32_t gamma_period, float epsilon,
    const ReadOnlyWeightedRMSNormBuffers& buffers, std::string& error);

/// Encode the GR head mixer's pre-gated reduction for rows in 1..8.
/// Read-only `xn` and `gate` views may overlap; `mixed` must be disjoint.
bool encode_gr_pre_gated(Context& context, CommandBuffer& command_buffer,
                         std::uint32_t rows,
                         const GRPreGatedBuffers& buffers,
                         std::string& error);
/// Add the GR head-mixer projection to a caller-owned Encoder segment.
bool encode_gr_pre_gated(Context& context, Encoder& encoder,
                         std::uint32_t rows,
                         const GRPreGatedBuffers& buffers,
                         std::string& error);

/// Encode the fused GR up projection and mixer for rows in 1..8.
/// `hc_up` uses packed uint4 reads and requires a 16-byte aligned offset;
/// the other views require float alignment. All views and the command buffer
/// must belong to `context`, and the views must be disjoint.
bool encode_gr_up(Context& context, CommandBuffer& command_buffer,
                  std::uint32_t rows, bool apply,
                  const GRUpBuffers& buffers, std::string& error);
/// Add the fused GR up projection and mixer to a caller-owned Encoder
/// segment. The eight views must be disjoint and correctly aligned. Propagate
/// a false result from the enclosing Encoder::encode body so the whole segment
/// is discarded.
bool encode_gr_up(Context& context, Encoder& encoder, std::uint32_t rows,
                  bool apply, const GRUpBuffers& buffers,
                  std::string& error);

/// Encode the tiled GR down and injection projections for rows in 1..8.
/// BF16 weight views require 16-byte aligned offsets; the other views require
/// float alignment. All five views and the command buffer must belong to
/// `context`, and the views must be disjoint.
bool encode_gr_down(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, const GRDownBuffers& buffers,
                    std::string& error);
/// Add the tiled GR down and injection projections to a caller-owned Encoder
/// segment, with the same buffer and dynamic threadgroup-memory checks.
/// Propagate a false result from the enclosing Encoder::encode body so the
/// whole segment is discarded.
bool encode_gr_down(Context& context, Encoder& encoder,
                    std::uint32_t rows, const GRDownBuffers& buffers,
                    std::string& error);

/// Encode the fused GR read normalization for rows in 1..8. The six buffer
/// views and command buffer must belong to `context`, be correctly
/// sized/aligned and not overlap.
bool encode_gr_norm(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, bool apply,
                    const GRNormBuffers& buffers, std::string& error);
/// Add fused GR read normalization to a caller-owned Encoder segment.
bool encode_gr_norm(Context& context, Encoder& encoder,
                    std::uint32_t rows, bool apply,
                    const GRNormBuffers& buffers, std::string& error);

/// Compose normalization, down/injection projection, and up/mix for one GR
/// read in order. In concurrent dispatch mode call outside an open group so
/// the Encoder inserts barriers between dependent dispatches. The standalone
/// wrapper discards the full segment if any stage fails.
bool encode_gr_read(Context& context, Encoder& encoder, std::uint32_t rows,
                    bool apply, const GRReadBuffers& buffers,
                    std::string& error);
bool encode_gr_read(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, bool apply,
                    const GRReadBuffers& buffers, std::string& error);

/// Encode one fused write/project-or-add pass into a caller-owned command
/// buffer. Buffers and the command buffer must belong to `context`, be
/// correctly sized/aligned and not overlap. The residual is modified in place.
bool encode_gr_cvec(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, std::uint32_t layer,
                    GRControlMode mode, bool write,
                    const GRControlVectorBuffers& buffers,
                    std::string& error);
/// Add fused GR control-vector work to a caller-owned Encoder segment.
bool encode_gr_cvec(Context& context, Encoder& encoder,
                    std::uint32_t rows, std::uint32_t layer,
                    GRControlMode mode, bool write,
                    const GRControlVectorBuffers& buffers,
                    std::string& error);

}  // namespace strata::metal
