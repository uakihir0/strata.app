// GPU encoding for Prefill GDN convolution stages (docs/mac/04 §6.4).
#pragma once

#include "strata/metal/context.hpp"

#include <cstdint>
#include <string>

namespace strata::metal {

class Encoder;

struct PrefillGDNConvTiledBuffers {
    ReadOnlyBufferRef history;
    ReadOnlyBufferRef qkv;
    ReadOnlyBufferRef conv_weights;
    BufferRef output;
};

struct PrefillGDNConvHistoryBuffers {
    BufferRef history;
    ReadOnlyBufferRef qkv;
};

/// Convolve each Prefill QKV row using the prior three-row history. The
/// output is the unnormalized SiLU convolution; history is read-only and is
/// advanced by a separate, ordered history encoder.
bool encode_prefill_gdn_conv_tiled(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNConvTiledBuffers& buffers, std::string& error);

/// Standalone transactional wrapper for Prefill GDN tiled convolution.
bool encode_prefill_gdn_conv_tiled(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNConvTiledBuffers& buffers, std::string& error);

/// Advance the three-row Prefill convolution history after convolution has
/// consumed its old values. History and QKV must be disjoint and belong to
/// this Context.
bool encode_prefill_gdn_conv_history(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNConvHistoryBuffers& buffers, std::string& error);

/// Standalone transactional wrapper for Prefill GDN convolution history.
bool encode_prefill_gdn_conv_history(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNConvHistoryBuffers& buffers, std::string& error);

}  // namespace strata::metal
