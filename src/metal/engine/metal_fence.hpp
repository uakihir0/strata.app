// Queue-wide hazard fence helpers for Metal encoders.
#pragma once

#import <Metal/Metal.h>

#include "strata/metal/context.hpp"

namespace strata::metal {

inline id<MTLFence> context_hazard_fence(Context& context) {
    return (__bridge id<MTLFence>)context.hazard_fence();
}

inline id<MTLComputeCommandEncoder> begin_compute_encoder(
    Context& context, id<MTLCommandBuffer> command_buffer) {
    id<MTLFence> fence = context_hazard_fence(context);
    if (fence == nil || command_buffer == nil) return nil;
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer
            computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (encoder != nil) [encoder waitForFence:fence];
    return encoder;
}

inline void end_compute_encoder(
    Context& context, id<MTLComputeCommandEncoder> encoder) {
    [encoder updateFence:context_hazard_fence(context)];
    [encoder endEncoding];
}

inline id<MTLBlitCommandEncoder> begin_blit_encoder(
    Context& context, id<MTLCommandBuffer> command_buffer) {
    id<MTLFence> fence = context_hazard_fence(context);
    if (fence == nil || command_buffer == nil) return nil;
    id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
    if (encoder != nil) [encoder waitForFence:fence];
    return encoder;
}

inline void end_blit_encoder(Context& context, id<MTLBlitCommandEncoder> encoder) {
    [encoder updateFence:context_hazard_fence(context)];
    [encoder endEncoding];
}

}  // namespace strata::metal
