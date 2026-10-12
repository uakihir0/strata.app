// GPU encoding for Prefill MoE SwiGLU and weighted combine post-ops.
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillMoeSwiGLUBuffers {
    ReadOnlyBufferRef gate;
    ReadOnlyBufferRef up;
    BufferRef output;
};

struct PrefillMoeCombineBuffers {
    ReadOnlyBufferRef parts;
    ReadOnlyBufferRef weights;
    ReadOnlyBufferRef shared;
    ReadOnlyBufferRef gate;
    BufferRef output;
};

/// Encode the shared-expert F32 SwiGLU followed by saturated F16 conversion.
/// The active tensors contain `rows * 640` values. Read-only inputs may alias;
/// the output must be disjoint from every active input range.
bool encode_prefill_moe_swiglu(Context& context, Encoder& encoder,
                               std::uint32_t rows,
                               const PrefillMoeSwiGLUBuffers& buffers,
                               std::string& error);

/// Standalone transactional wrapper for `encode_prefill_moe_swiglu`.
bool encode_prefill_moe_swiglu(Context& context,
                               CommandBuffer& command_buffer,
                               std::uint32_t rows,
                               const PrefillMoeSwiGLUBuffers& buffers,
                               std::string& error);

/// Encode the ordered rank-weighted routed outputs plus the shared expert.
/// `parts` is [rows][10][2560], `weights` is [rows][10], `shared` and
/// `output` are [rows][2560], and `gate` is [rows].
bool encode_prefill_moe_combine(Context& context, Encoder& encoder,
                                std::uint32_t rows,
                                const PrefillMoeCombineBuffers& buffers,
                                std::string& error);

/// Standalone transactional wrapper for `encode_prefill_moe_combine`.
bool encode_prefill_moe_combine(Context& context,
                                CommandBuffer& command_buffer,
                                std::uint32_t rows,
                                const PrefillMoeCombineBuffers& buffers,
                                std::string& error);

}  // namespace strata::metal
