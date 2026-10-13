// Ordered speculative Window GDN block encoding (docs/mac/04 §5.5).
#pragma once

#include "strata/metal/gdn_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/ple_dispatch.hpp"
#include "strata/metal/qsa_dispatch.hpp"
#include "strata/metal/session_state.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace strata::metal {

class Encoder;
class KvStore;

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

/// Parameter block and per-layer buffers for one write-through GDN block.
/// `params` must describe exactly `rows` retained rows (`T == n_keep == rows`).
struct WindowGDNWriteThroughBuffers {
    BufferRef params;
    WindowGDNBlockBuffers gdn;
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

/// Inputs for committing all supported GDN layers in their state-layout order.
/// Each replay view is laid out [36][8][width] and stores the per-layer qkv,
/// h, gate and beta snapshots produced by the speculative Window. `states`
/// and `gamma` have one entry per GDN layer in `SessionStateStorage` ordinal
/// order. `z` and `output_scratch` are the reusable active-row scratch views;
/// commit mode does not read or write them.
struct WindowGDNCommitBatchBuffers {
    BufferRef qkv_replay;
    BufferRef h_replay;
    BufferRef gate_replay;
    BufferRef beta_replay;
    BufferRef z;
    BufferRef output_scratch;
    std::span<const GdnSessionStateViews> states;
    std::span<const BufferRef> gamma;
};

/// Inputs for the GDN plus PLE portion of one accepted-prefix state commit.
/// `rows` is the accepted count (`WindowParams.n_keep`); the PLE history is
/// advanced from the retained prefix of normalized PLE rows after all GDN
/// commits have been encoded in the same ordered segment.
struct WindowGdnPleCommitBuffers {
    WindowGDNCommitBatchBuffers gdn;
    PLEHistoryBuffers ple;
};

/// Inputs for an accepted-prefix state commit across GDN, PLE and QSA.
/// QSA indexer views must contain one entry per QSA layer.
struct WindowAcceptedPrefixStateCommitBuffers {
    WindowGdnPleCommitBuffers gdn_ple;
    std::span<const QsaIndexerAppendBuffers> qsa_indexers;
    std::uint32_t qsa_cell_capacity = 0u;
    float qsa_rope_zero_cos = 1.0f;
    float qsa_rope_zero_sin = 0.0f;
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

/// Encode accepted-prefix GDN history and recurrent-state commits for all 36
/// supported GDN layers, in session-state ordinal order. Replay regions use
/// the fixed T=8 WindowScratch capacity even when `rows` is smaller. The
/// caller-owned form must be inside an ordered segment and propagate false
/// from its enclosing Encoder::encode body to discard the whole batch.
bool encode_window_gdn_commit_batch(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowGDNCommitBatchBuffers& buffers, std::string& error);
/// Standalone transactional form; a late validation failure discards every
/// earlier layer's history and recurrent-state commit.
bool encode_window_gdn_commit_batch(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowGDNCommitBatchBuffers& buffers, std::string& error);

/// Encode the accepted-prefix GDN and PLE history updates in one ordered
/// segment. The session owner must register this command buffer with
/// `SessionStateStorage::track_use` and advance committed PLE token IDs only
/// after successful GPU completion. The caller-owned form must propagate
/// false from its enclosing Encoder::encode body. QSA/indexer rollback is
/// separate.
bool encode_window_gdn_ple_commit(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowGdnPleCommitBuffers& buffers, std::string& error);
/// Standalone transactional form; either both state-update sequences are
/// encoded or the command buffer is discarded.
bool encode_window_gdn_ple_commit(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowGdnPleCommitBuffers& buffers, std::string& error);

/// Encode one accepted prefix's GDN convolution-history/recurrent-state,
/// PLE-history and all-layer QSA-indexer updates in order. Cross-component
/// writable/readable views must not overlap. The caller-owned form must
/// propagate false from its enclosing Encoder::encode body so all earlier
/// dispatches are discarded. After successful GPU completion, the session
/// owner advances committed PLE token IDs.
bool encode_window_accepted_prefix_state_commit(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowAcceptedPrefixStateCommitBuffers& buffers,
    std::string& error);
/// Standalone transactional form; failure discards the complete state commit.
bool encode_window_accepted_prefix_state_commit(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowAcceptedPrefixStateCommitBuffers& buffers,
    std::string& error);

/// Execute a complete accepted-prefix GPU state commit and advance the
/// session-owned PLE token suffix only after the command buffer completes
/// successfully. `window_tokens` contains the candidate Window rows;
/// `accepted_count` selects the prefix replayed by the supplied buffers.
/// The token IDs, GDN/PLE replay views, QSA replay rows, and QSA append steps
/// must all describe the same speculative Window. The helper cannot infer
/// that provenance from raw buffer views. QSA persistent views must belong to
/// `qsa_store`. A zero-length prefix is rejected because the current QSA
/// snapshots do not preserve every value a speculative append can overwrite;
/// the caller must discard and recreate the sequence state after rejection.
/// If a submitted GPU commit reports an execution failure, discard and
/// recreate both session and KV state before retrying because writes may have
/// partially completed.
/// Callers must serialize this operation with Window preparation and session
/// reset for the same sequence.
bool commit_window_accepted_prefix_state(
    Context& context, SessionStateStorage& session, const KvStore& qsa_store,
    std::span<const std::uint32_t> window_tokens,
    std::size_t accepted_count, const Geometry& geometry,
    const WindowAcceptedPrefixStateCommitBuffers& buffers,
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

/// Encode one GDN layer of a write-through Window block. Every row is kept:
/// the recurrent step writes state in write-through mode and convolution
/// history advances after its last reader. The parameter block is checked
/// for `T == n_keep == rows`. This function does not roll back. Keep GDN
/// buffers alive through completion and propagate false from a caller-owned
/// Encoder body.
bool encode_window_gdn_write_through_block(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowGDNWriteThroughBuffers& buffers, std::string& error);
/// Standalone transactional form for an unsubmitted command buffer.
bool encode_window_gdn_write_through_block(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowGDNWriteThroughBuffers& buffers, std::string& error);

}  // namespace strata::metal
