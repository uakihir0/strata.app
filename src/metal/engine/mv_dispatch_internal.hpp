// Internal fast path for caller-owned, previously validated indexed head rows.
#pragma once

#include "strata/metal/mv_dispatch.hpp"

namespace strata::metal {

/// Encode an indexed head projection without rechecking every row-map entry.
/// Only call this when the immutable map was validated against weight_rows
/// before it was bound and will stay unchanged through GPU completion.
bool encode_indexed_head_matvec_prevalidated(
    Context& context, Encoder& encoder, MatVecWeightFormat format,
    std::uint32_t n_in, std::uint32_t weight_rows,
    std::uint32_t output_rows,
    const ReadOnlyIndexedHeadMatVecBuffers& buffers, std::string& error);

}  // namespace strata::metal
