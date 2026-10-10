// GPU encoding for one Prefill GR read half (docs/mac/04 §6.3).
#pragma once

#include "strata/metal/prefill_gr_norm_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGRReadBuffers {
    BufferRef residual;
    BufferRef norm;
    BufferRef rs;
    BufferRef xn16;
    /// Required only in all_projections mode; must be empty otherwise.
    BufferRef xn16_lo;

    ReadOnlyBufferRef down_weights;
    BufferRef low_rank;
    BufferRef low_rank_bf16;
    /// Required only in all_projections mode; must be empty otherwise.
    BufferRef low_rank_bf16_lo;

    ReadOnlyBufferRef up_weights;
    BufferRef gated;
    ReadOnlyBufferRef inject_weights;
    BufferRef inject;
};

/// Encode GR normalization, down projection, SiLU conversion, up projection,
/// and injection projection for the requested rows. Buffers follow the arena
/// shapes in docs/mac/04 §6.3. Modes 0 and 2 omit BF16 remainders; mode 1 uses
/// BF16X2 for all three projections and writes both activation remainders.
///
/// The caller-owned form rejects open concurrent groups. Propagate false from
/// the enclosing Encoder body so any already-encoded prefix is discarded.
/// Writable buffers must be disjoint from every input and from each other.
bool encode_prefill_gr_read(Context& context, Encoder& encoder,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRReadBuffers& buffers,
                            std::string& error);

/// Encode the projection stages using rs/xn16 (and mode-1 xn16_lo) already
/// produced for the same active rows and BF16X2 mode by an earlier GR
/// write/norm transition. Order the producer before this consumer in the same
/// Encoder, or establish a supported cross-command-buffer dependency before
/// submission. These views are read-only inputs and must not overlap
/// projection outputs.
/// residual and norm are unused by this projection-only form.
bool encode_prefill_gr_read_pre_normalized(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadBuffers& buffers,
    std::string& error);

/// Transactional wrapper for a caller-supplied fresh command buffer.
bool encode_prefill_gr_read(Context& context,
                            CommandBuffer& command_buffer,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRReadBuffers& buffers,
                            std::string& error);

/// Transactional wrapper for the pre-normalized projection path.
bool encode_prefill_gr_read_pre_normalized(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadBuffers& buffers,
    std::string& error);

}  // namespace strata::metal
