// GPU encoding for Prefill GDN q/k head normalization (docs/mac/04 §6.4).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

/// Normalize the 32 q/k heads of a T×10240 F32 convolution output in place.
/// The remaining value heads are left unchanged.
bool encode_prefill_gdn_l2(Context& context, Encoder& encoder,
                           std::uint32_t rows, BufferRef h,
                           std::string& error);

/// Standalone transactional wrapper for Prefill GDN q/k normalization.
bool encode_prefill_gdn_l2(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, BufferRef h,
                           std::string& error);

}  // namespace strata::metal
