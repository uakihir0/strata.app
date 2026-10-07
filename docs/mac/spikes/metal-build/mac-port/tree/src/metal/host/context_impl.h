// src/metal/host/context_impl.h - Context's Objective-C state, shared by the src/metal/host/*.mm files only.
#pragma once

#import <Metal/Metal.h>

#include "strata/metal/context.hpp"

#include <string>
#include <unordered_map>

namespace strata::metal {

struct Context::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    PipelineStats stats;
    std::string capture;   ///< the .gputrace path, empty when off
    bool capturing = false;
};

bool build_pipelines(Context::Impl& im, const std::string& metallib_sha256, std::string& err);

}  // namespace strata::metal
