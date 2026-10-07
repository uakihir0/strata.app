// src/metal/host/context_impl.h - Context's Objective-C state, shared by the src/metal/host/*.mm files only.
#pragma once

#import <Metal/Metal.h>

#include "strata/metal/context.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::metal {

struct KernelBindingSchema {
    std::vector<std::uint32_t> buffers;
    std::vector<std::uint32_t> threadgroup_memory;
    bool supported = true;
};

struct Context::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLFence> hazard_fence = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    std::unordered_map<std::string, KernelBindingSchema> kernel_bindings;
    PipelineStats stats;
    DeviceFacts facts;
    std::string capture;   ///< the .gputrace path, empty when off
    std::uint64_t max_buffer_length = 0;
    bool capturing = false;
};

bool build_pipelines(Context::Impl& im, const std::string& metallib_sha256, std::string& err);
std::mutex& command_buffer_submission_mutex();

}  // namespace strata::metal
