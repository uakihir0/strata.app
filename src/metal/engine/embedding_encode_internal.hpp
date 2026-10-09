// Private prepare/encode split shared by embedding and Prefill dispatch.
#pragma once

#import <Metal/Metal.h>

#include "strata/metal/embed_dispatch.hpp"
#include "strata/metal/gr_dispatch.hpp"

namespace strata::metal::detail {

struct EmbedRowsDispatch {
    std::uint32_t rows = 0u;
    std::uint32_t vocabulary_size = 0u;
    std::uint32_t tokens_are_gpu_written = 0u;
    EmbedRowsBuffers buffers{};
    const char* kernel_name = nullptr;
    const char* format_name = nullptr;
};

struct EmbedImageDispatch {
    std::uint32_t rows = 0u;
    EmbedImageBuffers buffers{};
};

struct GRBroadcastDispatch {
    std::uint32_t rows = 0u;
    GRBroadcastBuffers buffers{};
};

bool prepare_embed_rows(Context& context, std::uint32_t rows,
                        std::uint32_t vocabulary_size, EmbedRowsFormat format,
                        const EmbedRowsBuffers& buffers,
                        EmbedRowsDispatch& dispatch, std::string& error);

void encode_embed_rows(Context& context,
                       id<MTLComputeCommandEncoder> encoder,
                       const EmbedRowsDispatch& dispatch);

bool prepare_embed_image(Context& context, std::uint32_t rows,
                         const EmbedImageBuffers& buffers,
                         EmbedImageDispatch& dispatch, std::string& error);

void encode_embed_image(Context& context,
                        id<MTLComputeCommandEncoder> encoder,
                        const EmbedImageDispatch& dispatch);

bool prepare_gr_broadcast(Context& context, std::uint32_t rows,
                          const GRBroadcastBuffers& buffers,
                          GRBroadcastDispatch& dispatch,
                          std::string& error);

void encode_gr_broadcast(Context& context,
                         id<MTLComputeCommandEncoder> encoder,
                         const GRBroadcastDispatch& dispatch);

}  // namespace strata::metal::detail
