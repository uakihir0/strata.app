// Validated Metal MoE routing dispatches for Window and Prefill.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct MoeRouteBuffers {
    ReadOnlyBufferRef logits;
    BufferRef ids;
    BufferRef weights;
};

/// Encode the stable top-10 route selection for 256 or 512 experts.
/// `divide_weights` selects the prompt-path division mode; Window decode must
/// pass false to match the model's multiply-by-reciprocal path. The views must
/// remain alive and unchanged where read until command completion.
/// When using the caller-owned Encoder overload, propagate false from the
/// enclosing `Encoder::encode` body so earlier dispatches in the segment are
/// discarded on validation or encoding failure.
bool encode_moe_route(Context& context, Encoder& encoder,
                      std::uint32_t rows, std::uint32_t experts,
                      bool divide_weights, const MoeRouteBuffers& buffers,
                      std::string& error);
/// Standalone transactional form. A validation failure discards the segment.
bool encode_moe_route(Context& context, CommandBuffer& command_buffer,
                      std::uint32_t rows, std::uint32_t experts,
                      bool divide_weights, const MoeRouteBuffers& buffers,
                      std::string& error);

}  // namespace strata::metal
