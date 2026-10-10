// GPU encoding for the Metal QSA and indexer kernels.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/prefill.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace strata::metal {

class Encoder;

enum class QsaNormRopeMode : std::uint32_t {
    key = 0,
    query = 1,
    indexer_query = 2,
};

struct QsaNormRopeBuffers {
    BufferRef input;
    BufferRef gamma;
    BufferRef steps;
    BufferRef rope;
    BufferRef output;
};

struct QsaKvAppendF16Buffers {
    BufferRef keys;
    BufferRef values;
    BufferRef steps;
    BufferRef key_pool;
    BufferRef value_pool;
};

struct QsaKvAppendI8Buffers {
    BufferRef keys;
    BufferRef values;
    BufferRef steps;
    BufferRef key_codes;
    BufferRef key_scales;
    BufferRef value_codes;
    BufferRef value_scales;
};

struct QsaKvAppendQ4Buffers {
    BufferRef keys;
    BufferRef values;
    BufferRef steps;
    BufferRef key_pool;
    BufferRef value_pool;
};

struct QsaKvAppendK8V4Buffers {
    BufferRef keys;
    BufferRef values;
    BufferRef steps;
    BufferRef key_codes;
    BufferRef key_scales;
    BufferRef value_pool;
};

struct QsaIndexerSnapshotBuffers {
    BufferRef tails;
    BufferRef block_positions;
    BufferRef tail_snapshot;
    BufferRef block_pos_snapshot;
};

enum class QsaIndexerAppendMode : std::uint32_t {
    window = 0,
    commit = 1,
};

struct QsaIndexerAppendBuffers {
    BufferRef raw_keys;
    BufferRef steps;
    BufferRef gamma;
    BufferRef rope;
    BufferRef tail;
    BufferRef dead;
    BufferRef pooled;
    BufferRef block_position;
    BufferRef tail_snapshot;
    BufferRef block_pos_snapshot;
};

/// Per-layer views used by the batched Prefill indexer append path.
struct QsaPrefillIndexerAppendBuffers {
    BufferRef raw_keys;
    BufferRef gamma;
    BufferRef rope;
    BufferRef tail;
    BufferRef dead;
    BufferRef pooled;
    BufferRef block_position;
};

struct QsaSelectionScoresBuffers {
    BufferRef pooled;
    BufferRef dead;
    BufferRef indexer_queries;
    BufferRef steps;
    BufferRef scores;
};

struct QsaSelectionTopkBuffers {
    BufferRef scores;
    BufferRef steps;
    BufferRef ids;
};

struct QsaGateBuffers {
    BufferRef attn;
    BufferRef q_full;
    BufferRef output;
};

enum class QsaAttentionMode : std::uint32_t {
    f16 = 0,
    i8 = 1,
    q4 = 2,
    k8v4 = 3,
};

struct QsaAttentionScratch {
    BufferRef partial_acc;
    BufferRef partial_m;
    BufferRef partial_l;
};

struct QsaAttentionChunkBuffers {
    BufferRef query;
    BufferRef steps;
    BufferRef ids;
    BufferRef key_data;
    BufferRef key_scales;
    BufferRef value_data;
    BufferRef value_scales;
    QsaAttentionScratch scratch;
};

struct QsaAttentionMergeBuffers {
    QsaAttentionScratch scratch;
    BufferRef output;
};

struct QsaSelectionAttentionBuffers {
    QsaSelectionScoresBuffers selection_scores;
    QsaSelectionTopkBuffers selection_topk;
    QsaAttentionChunkBuffers attention_chunk;
    QsaAttentionMergeBuffers attention_merge;
};

/// Encode one QSA norm/RoPE row for each head and token. `token_rows` is T
/// in 1..8. Key and indexer-query modes permit exact in-place operation;
/// query mode reads 512-wide q_full rows and writes compact 256-wide qcur
/// rows, so input and output must be disjoint. Step positions must index the
/// supplied float2 RoPE table. Every view and the unsubmitted command buffer
/// must belong to `context`.
bool encode_qsa_norm_rope(Context& context, CommandBuffer& command_buffer,
                          std::uint32_t token_rows, QsaNormRopeMode mode,
                          const QsaNormRopeBuffers& buffers,
                          std::string& error);
/// Add QSA normalization and RoPE to a caller-owned Encoder segment. Preserve
/// the standalone view, position and mode checks, including exact in-place
/// K/indexer-query operation and disjoint query input/output. Propagate false
/// from the enclosing Encoder::encode body so the complete segment is dropped.
bool encode_qsa_norm_rope(Context& context, Encoder& encoder,
                          std::uint32_t token_rows, QsaNormRopeMode mode,
                          const QsaNormRopeBuffers& buffers,
                          std::string& error);

/// Encode the same QSA norm/RoPE stage for Prefill chunks, whose row count can
/// exceed the 1..8 Window decode limit. `token_rows` must fit
/// `kPrefillMaxShaderRows`; the Prefill-specific API keeps the decode contract
/// unchanged. Input/output aliasing and StepRow position checks match the
/// Window form.
bool encode_prefill_qsa_norm_rope(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, QsaNormRopeMode mode,
    const QsaNormRopeBuffers& buffers, std::string& error);
/// Caller-owned Encoder form for Prefill chunks. Propagate false from the
/// enclosing Encoder body so earlier dispatches in the segment are discarded.
bool encode_prefill_qsa_norm_rope(
    Context& context, Encoder& encoder, std::uint32_t token_rows,
    QsaNormRopeMode mode, const QsaNormRopeBuffers& buffers,
    std::string& error);

/// Append the T rows of two 256-wide K/V heads to the interleaved fp16
/// cache. `cell_capacity` must be a positive multiple of four, and every
/// `StepRow::pos` must be within that capacity and unique within the batch.
/// Current K/V and step views are read-only; both pools must be disjoint from
/// one another and the inputs. Every view and the unsubmitted command buffer
/// must belong to `context`.
bool encode_qsa_kv_append_f16(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendF16Buffers& buffers, std::string& error);
/// Add fp16 KV append to a caller-owned Encoder segment. Preserve
/// the standalone validation and binding contract; propagate false
/// from the enclosing Encoder::encode body to discard the segment.
bool encode_qsa_kv_append_f16(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendF16Buffers& buffers, std::string& error);
/// Prefill-sized fp16 KV append. Supports chunks through
/// `kPrefillMaxShaderRows` while retaining the Window overload's 1..8 limit.
/// The standalone form validates before encoding; the caller-owned form must
/// propagate false so the enclosing Encoder segment is discarded.
bool encode_prefill_qsa_kv_append_f16(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendF16Buffers& buffers, std::string& error);
bool encode_prefill_qsa_kv_append_f16(
    Context& context, Encoder& encoder, std::uint32_t token_rows,
    std::uint32_t cell_capacity, const QsaKvAppendF16Buffers& buffers,
    std::string& error);

/// Append the T rows of two 256-wide K/V heads to the interleaved int8
/// cache. Scales are fp16 bit patterns, four per cache row (one per 64
/// values). Cell positions must be unique within the batch. All output arrays
/// must be disjoint from one another and the inputs; disjoint output views
/// may share one backing allocation. K/V values must be finite. The encoder
/// does not scan their contents because they may be produced by earlier GPU
/// work in the same command buffer. Every view and the unsubmitted command
/// buffer must belong to `context`.
bool encode_qsa_kv_append_i8(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendI8Buffers& buffers, std::string& error);
/// Add int8 KV append to a caller-owned Encoder segment. Preserve
/// the standalone validation and binding contract; propagate false
/// from the enclosing Encoder::encode body to discard the segment.
bool encode_qsa_kv_append_i8(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendI8Buffers& buffers, std::string& error);
/// Prefill-sized int8 KV append, preserving the Window overload's 1..8 limit.
/// Cell positions must be unique, and K/V values must be finite. The encoder
/// does not scan values that may have been produced by earlier GPU work in the
/// same command buffer. The standalone form validates before encoding; the
/// caller-owned form must propagate false so its Encoder segment is discarded.
bool encode_prefill_qsa_kv_append_i8(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendI8Buffers& buffers, std::string& error);
bool encode_prefill_qsa_kv_append_i8(
    Context& context, Encoder& encoder, std::uint32_t token_rows,
    std::uint32_t cell_capacity, const QsaKvAppendI8Buffers& buffers,
    std::string& error);

/// Append FWHT256-rotated K/V rows to the packed q4_0 cache. Each cache row
/// contains eight 18-byte blocks. K/V values must be finite and each
/// nonzero block's fp32 reciprocal scale and stored fp16 scale must remain
/// finite. Cell positions must be unique; the output pools must be disjoint
/// from each other and the inputs. In the same command buffer, the caller
/// must encode FWHT256 on both K and V before this operation.
bool encode_qsa_kv_append_q4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendQ4Buffers& buffers, std::string& error);
/// Add q4_0 KV append to a caller-owned Encoder segment. The caller must
/// encode FWHT256 on both K and V earlier in the same segment, then close any
/// concurrent dispatch group before this append. Encoder dispatches inside one
/// open concurrent group have no inter-dispatch buffer barrier. Preserve the
/// standalone validation and binding contract; propagate false from the
/// enclosing Encoder::encode body to discard the segment.
bool encode_qsa_kv_append_q4(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendQ4Buffers& buffers, std::string& error);
/// Prefill-sized q4_0 KV append, preserving the Window overload's 1..8 limit.
/// K/V values must be finite; each nonzero block's fp32 reciprocal scale and
/// stored fp16 scale must remain finite. The caller must encode FWHT256 on
/// both K and V earlier in the same ordered command buffer/Encoder segment,
/// outside an open concurrent dispatch group. The standalone form validates
/// before encoding; the caller-owned form must propagate false.
bool encode_prefill_qsa_kv_append_q4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendQ4Buffers& buffers, std::string& error);
bool encode_prefill_qsa_kv_append_q4(
    Context& context, Encoder& encoder, std::uint32_t token_rows,
    std::uint32_t cell_capacity, const QsaKvAppendQ4Buffers& buffers,
    std::string& error);

/// Append unrotated K and FWHT256-rotated V rows to the hybrid k8v4 cache.
/// K uses signed int8 codes and four fp16 scales per row; V uses eight packed
/// 18-byte q4_0 blocks per row. Cell positions must be unique. The three
/// output arrays must be disjoint from one another and the inputs; disjoint
/// views may share one backing allocation. K/V values must be finite, and
/// their stored fp16 scales must remain finite. Each nonzero transformed V
/// block's fp32 reciprocal q4 scale must also remain finite. K, V and steps
/// must be mutually disjoint because V is transformed in place. In the same
/// command buffer, the caller must encode FWHT256 on V only before this
/// operation. Every view and the unsubmitted command buffer must belong to
/// `context`.
bool encode_qsa_kv_append_k8v4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error);
/// Add hybrid k8v4 KV append to a caller-owned Encoder segment. The caller
/// must encode FWHT256 on V earlier in the same segment, then close any
/// concurrent dispatch group before this append. Encoder dispatches inside one
/// open concurrent group have no inter-dispatch buffer barrier. Preserve the
/// standalone validation and binding contract; propagate false from the
/// enclosing Encoder::encode body to discard the segment.
bool encode_qsa_kv_append_k8v4(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error);
/// Prefill-sized hybrid KV append, preserving the Window overload's 1..8
/// limit. K/V values and their stored fp16 scales must remain finite; each
/// nonzero transformed V block's fp32 reciprocal q4 scale must also remain
/// finite. K, V and steps must be disjoint because V is transformed in place.
/// The caller must encode FWHT256 on V earlier in the same ordered command
/// buffer/Encoder segment, outside an open concurrent dispatch group. The
/// standalone form validates its inputs before encoding; the caller-owned
/// form must propagate false.
bool encode_prefill_qsa_kv_append_k8v4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error);
bool encode_prefill_qsa_kv_append_k8v4(
    Context& context, Encoder& encoder, std::uint32_t token_rows,
    std::uint32_t cell_capacity, const QsaKvAppendK8V4Buffers& buffers,
    std::string& error);

/// Encode an in-place, orthonormal FWHT over `rows` independent 256-float
/// rows. One simdgroup handles each row; `values` must contain all rows and
/// belong to `context`, as must the unsubmitted command buffer.
bool encode_qsa_fwht256(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, BufferRef values,
                        std::string& error);
/// Add the same in-place FWHT to a caller-owned Encoder segment. Preserve the
/// standalone positive-row, overflow, buffer and pipeline checks. Propagate
/// false from the enclosing Encoder::encode body to discard the segment.
bool encode_qsa_fwht256(Context& context, Encoder& encoder,
                        std::uint32_t rows, BufferRef values,
                        std::string& error);

/// Snapshot all 12 QSA layers' indexer tails (384 floats per layer) and
/// block positions (one int32 per layer). Source regions use the supplied
/// per-layer strides in float and int32 elements; destination snapshots are
/// tightly packed in QSA-layer order. The source and destination regions must
/// be disjoint. Encode this before any indexer append in a rollback-capable
/// window. Every view and the unsubmitted command buffer must belong to
/// `context`.
bool encode_qsa_indexer_snapshot(
    Context& context, CommandBuffer& command_buffer,
    std::uint64_t tail_layer_stride_floats,
    std::uint64_t block_pos_layer_stride_ints,
    const QsaIndexerSnapshotBuffers& buffers, std::string& error);
/// Add the same raw-bit snapshot to a caller-owned Encoder segment. Preserve
/// the standalone stride, extent, ownership and alias checks. Encode this
/// before any indexer append in a rollback-capable window, and propagate false
/// from the enclosing Encoder::encode body so earlier dispatches are discarded.
bool encode_qsa_indexer_snapshot(
    Context& context, Encoder& encoder,
    std::uint64_t tail_layer_stride_floats,
    std::uint64_t block_pos_layer_stride_ints,
    const QsaIndexerSnapshotBuffers& buffers, std::string& error);

/// Append 1..8 sequential raw indexer keys to one QSA layer. `cell_capacity`
/// must be a positive multiple of four and addressable by StepRow;
/// `qsa_layer_index` must be in 0..11. Window mode consumes contiguous
/// StepRow::pos values. Commit mode restores that layer's tail and block
/// position from the packed 12-layer snapshot, then replays the first
/// `token_rows` inputs beginning at steps[0].pos. Cell positions index both
/// the KV/indexer state and folded RoPE table. Completed block b reads RoPE
/// row 4*b; position zero uses the explicit angle-zero pair, including YaRN
/// magnitude. The pooled allocation must contain `cell_capacity / 4 + 2`
/// rows. Raw inputs must be finite and must round to finite fp16 values.
/// Window mode ignores snapshot buffers; commit mode requires packed
/// snapshots from `encode_qsa_indexer_snapshot`. Every writable output must
/// be disjoint from every other output and all inputs. Every view and the
/// unsubmitted command buffer must belong to `context`.
bool encode_qsa_indexer_append(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    QsaIndexerAppendMode mode, std::uint32_t qsa_layer_index,
    float rope_zero_cos, float rope_zero_sin,
    const QsaIndexerAppendBuffers& buffers, std::string& error);
/// Add the same validated append to a caller-owned Encoder segment. A commit
/// may follow a shared snapshot in that segment; propagate false from the
/// enclosing Encoder::encode body so all earlier dispatches are discarded.
/// Do not place dependent snapshot/append dispatches in one concurrent group.
bool encode_qsa_indexer_append(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    QsaIndexerAppendMode mode, std::uint32_t qsa_layer_index,
    float rope_zero_cos, float rope_zero_sin,
    const QsaIndexerAppendBuffers& buffers, std::string& error);

/// Append one Prefill chunk to one QSA indexer layer using the documented
/// first-row, full-block and tail kernels. `params` must be consistent with
/// `build_prefill_chunk_metadata`; the cache capacity must be a positive
/// multiple of four that contains `[p0, p0 + T)`. The pooled view must contain
/// `cell_capacity / 4 + 2` rows. Raw keys must be finite and round to finite
/// fp16 values. Writable views must be mutually disjoint and disjoint from
/// every read-only view. The three dispatches are ordered and must not be
/// placed inside a concurrent dispatch group.
bool encode_prefill_qsa_indexer_append(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& params, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    const QsaPrefillIndexerAppendBuffers& buffers, std::string& error);
/// Caller-owned ordered Encoder form. Propagate false from the enclosing
/// Encoder::encode body so any earlier dispatches in that segment are dropped.
bool encode_prefill_qsa_indexer_append(
    Context& context, Encoder& encoder,
    const PrefillParams& params, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    const QsaPrefillIndexerAppendBuffers& buffers, std::string& error);

/// Restore each QSA indexer from the shared packed snapshot, then replay the
/// accepted prefix in layer order. `layers` must contain exactly 12 entries;
/// each entry supplies that layer's raw keys, gamma and mutable indexer views,
/// while steps, RoPE and both snapshot views must be identical across entries.
/// All layer views are preflighted, including cross-layer output/input aliases
/// over the active read/write extents, before the first append is encoded.
/// Unused input suffixes may share storage with a disjoint output range. The
/// caller-owned form requires an ordered Encoder segment and must propagate
/// false from its enclosing body.
bool encode_qsa_indexer_commit_batch(
    Context& context, Encoder& encoder, std::uint32_t accepted_rows,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    std::span<const QsaIndexerAppendBuffers> layers, std::string& error);
/// Standalone transactional form; failure discards the complete batch.
bool encode_qsa_indexer_commit_batch(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t accepted_rows, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    std::span<const QsaIndexerAppendBuffers> layers, std::string& error);

/// Score all indexer blocks needed by 1..8 query rows. `cell_capacity` must
/// be a positive multiple of four addressable by StepRow. `pooled` contains
/// `cell_capacity / 4 + 2` rows of 128 floats, `dead` one 128-float key,
/// `indexer_queries` four 128-float heads per query, and `scores` one
/// `cell_capacity / 4 + 2` score row per query. Each StepRow must describe
/// its own position (`n_kv = pos + 1`, `n_bid = n_kv / 4`, and
/// `width = min(n_kv, 2051)`). Scores beyond each row's `n_bid` are left
/// untouched. Inputs may overlap; the score output must be disjoint from
/// every input. Every view and the unsubmitted command buffer or Encoder
/// segment must belong to `context`.
bool encode_qsa_selection_scores(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionScoresBuffers& buffers, std::string& error);
/// Add the same validated score dispatch to a caller-owned Encoder segment.
/// Propagate false from the enclosing `Encoder::encode` body so earlier
/// dispatches in the segment are discarded on failure.
bool encode_qsa_selection_scores(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionScoresBuffers& buffers, std::string& error);

/// Select each row's `min(n_kv, 2051)` cells by its block scores and write
/// their ids in ascending order. Equal order keys prefer lower cell ids;
/// negative zero is positive zero and NaN has order key zero. `cell_capacity`
/// and StepRows use the same constraints as `encode_qsa_selection_scores`.
/// `scores` contains `cell_capacity / 4 + 2` floats per row and `ids`
/// contains 2051 int32 ids per row; cells after each row's width are left
/// untouched. The encoder uses the register form through 33,792 max-blocks
/// and the memory form above that limit. The id output must be disjoint from
/// scores and steps. Every view and the unsubmitted command buffer or Encoder
/// segment must belong to `context`.
bool encode_qsa_selection_topk(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionTopkBuffers& buffers, std::string& error);
/// Add the same validated top-k dispatch to a caller-owned Encoder segment.
/// Propagate false from the enclosing `Encoder::encode` body so earlier
/// dispatches in the segment, including selection scores, are discarded on
/// failure.
bool encode_qsa_selection_topk(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionTopkBuffers& buffers, std::string& error);

/// Multiply each QSA attention value by the sigmoid gate in the second half
/// of its corresponding `q_full` head. `attn` and `output` contain
/// `token_rows * 24 * 256` floats; `q_full` contains
/// `token_rows * 24 * 512` floats. The output must be disjoint from both
/// inputs; the two read-only inputs may overlap. Every view and the
/// unsubmitted command buffer must belong to `context`.
bool encode_qsa_gate(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t token_rows,
                     const QsaGateBuffers& buffers, std::string& error);
/// Add the same validated gate dispatch to a caller-owned Encoder segment.
/// Propagate false from the enclosing `Encoder::encode` body so earlier
/// dispatches in that segment are discarded if a later operation fails.
bool encode_qsa_gate(Context& context, Encoder& encoder,
                     std::uint32_t token_rows,
                     const QsaGateBuffers& buffers, std::string& error);

/// Encode the sparse-attention chunk pass for 1..8 rows. `cell_capacity` is
/// the cache capacity in cells; `id_row_stride` is the per-row ID capacity
/// and determines ceil(id_row_stride / 64) chunks. The ID stride may exceed
/// the cache capacity; `StepRow::width` bounds how many IDs are read. Each
/// StepRow must have a valid position/count, a positive width no larger than
/// its ID row, and every ID in its live prefix must be in
/// [0, min(cell_capacity, StepRow::n_kv)).
/// The kernel masks negative, out-of-capacity and future-cell IDs defensively,
/// but callers must provide valid IDs for complete attention results. KV views
/// use the selected mode's layout from docs/mac/04-metal-kernels.md §4.5.
/// Scratch extents are derived from T and the chunk count. Read-only views may
/// overlap; all scratch outputs must be disjoint from inputs and one another.
bool encode_qsa_attention_chunk(
    Context& context, CommandBuffer& command_buffer, QsaAttentionMode mode,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride,
    const QsaAttentionChunkBuffers& buffers, std::string& error);
/// Add the chunk pass to a caller-owned Encoder segment. Encode the merge
/// afterward in the same ordered segment, and propagate false from the
/// enclosing `Encoder::encode` body so a late failure discards both passes.
bool encode_qsa_attention_chunk(
    Context& context, Encoder& encoder, QsaAttentionMode mode,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride,
    const QsaAttentionChunkBuffers& buffers, std::string& error);

/// Merge the chunk partials in increasing chunk order into T×24×256 floats.
/// `id_row_stride` must match the chunk pass; output must be disjoint from
/// every scratch input. Every view and the unsubmitted command buffer must
/// belong to `context`.
bool encode_qsa_attention_merge(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t id_row_stride,
    const QsaAttentionMergeBuffers& buffers, std::string& error);
/// Add the fixed-order merge pass to a caller-owned Encoder segment after its
/// chunk producer. The caller must preserve that dependency in concurrent
/// dispatch mode.
bool encode_qsa_attention_merge(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t id_row_stride,
    const QsaAttentionMergeBuffers& buffers, std::string& error);

/// Compose selection scoring, top-k, sparse-attention chunks and merge in
/// one caller-owned Encoder segment. Cell capacity must be a positive multiple
/// of four; ID rows always use the fixed 2051-element stride, including when
/// fewer cache cells are configured. Top-k scores and StepRows, its ID rows
/// and the chunk pass must share the same views, and chunk and merge must share
/// all scratch views.
/// Other stage outputs must not overlap unrelated stage inputs or outputs.
/// This ordered chain must be called outside a concurrent dispatch group so
/// Encoder inserts a buffer barrier between dependent passes. Propagate false
/// from the enclosing Encoder::encode body so a failure discards the flow.
bool encode_qsa_selection_attention(
    Context& context, Encoder& encoder, QsaAttentionMode mode,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride,
    const QsaSelectionAttentionBuffers& buffers, std::string& error);

}  // namespace strata::metal
