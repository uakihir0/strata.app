// src/metal/host/encoder.mm - common Metal encoder lifecycle and dispatch policy.
#import <Metal/Metal.h>

#include "strata/metal/encoder.hpp"
#include "strata/metal/kernel_env.hpp"
#include "../engine/metal_fence.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::size_t kMaxInlineBytesPerEncoder = 4096;

bool dispatch_mode(DispatchMode& mode, std::string& error) {
    return parse_dispatch_mode(std::getenv("STRATA_METAL_DISPATCH"), mode, error);
}

const KernelInfo* find_kernel(std::string_view name) {
    std::size_t count = 0;
    const KernelInfo* const manifest = kernel_manifest(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (manifest[index].name != nullptr &&
            std::string_view(manifest[index].name) == name) {
            return &manifest[index];
        }
    }
    return nullptr;
}

bool checked_thread_count(EncoderSize threads, std::size_t& count) {
    if (threads.x == 0 || threads.y == 0 || threads.z == 0 ||
        threads.x > std::numeric_limits<std::size_t>::max() / threads.y) {
        return false;
    }
    const std::size_t xy = threads.x * threads.y;
    if (xy > std::numeric_limits<std::size_t>::max() / threads.z) return false;
    count = xy * threads.z;
    return true;
}

bool validate_buffer(Context& context, const EncoderBufferBinding& binding,
                     std::string& error) {
    const BufferRef buffer = binding.buffer;
    if (binding.required_bytes == 0 || binding.alignment == 0 ||
        !buffer || buffer.length < binding.required_bytes ||
        buffer.offset % binding.alignment != 0 ||
        !context.buffers().owns(buffer)) {
        error = "encoder buffer binding is stale, foreign, undersized or misaligned";
        return false;
    }

    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || buffer.offset > native.length ||
        buffer.length > native.length - buffer.offset) {
        error = "encoder buffer view exceeds its Metal allocation";
        return false;
    }
    if (native.contents != nullptr) {
        const std::uintptr_t native_contents =
            reinterpret_cast<std::uintptr_t>(native.contents);
        if (native_contents >
                std::numeric_limits<std::uintptr_t>::max() - buffer.offset ||
            buffer.contents !=
                reinterpret_cast<void*>(native_contents + buffer.offset)) {
            error = "encoder buffer CPU backing metadata does not match its Metal view";
            return false;
        }
    } else if (buffer.contents != nullptr) {
        error = "encoder buffer exposes CPU contents absent from its Metal allocation";
        return false;
    }
    const std::uint64_t native_gpu_address =
        static_cast<std::uint64_t>(native.gpuAddress);
    std::uint64_t expected_gpu_address = 0;
    if (!detail::gpu_address_with_offset(native_gpu_address, buffer.offset,
                                         expected_gpu_address) ||
        buffer.gpu_address != expected_gpu_address) {
        error = "encoder buffer GPU address does not match its Metal view";
        return false;
    }
    return true;
}

template <typename Address>
bool address_ranges_overlap(Address first, std::size_t first_length,
                            Address second, std::size_t second_length) {
    if (first_length == 0 || second_length == 0) return false;
    const Address limit = std::numeric_limits<Address>::max();
    if (first_length > limit - first || second_length > limit - second) return true;
    const Address first_end = first + static_cast<Address>(first_length);
    const Address second_end = second + static_cast<Address>(second_length);
    return first < second_end && second < first_end;
}

bool buffer_views_overlap(const BufferRef& first, const BufferRef& second) {
    if (first.handle == second.handle &&
        address_ranges_overlap(first.offset, first.length,
                               second.offset, second.length)) {
        return true;
    }
    if (first.contents != nullptr && second.contents != nullptr &&
        address_ranges_overlap(reinterpret_cast<std::uintptr_t>(first.contents),
                               first.length,
                               reinterpret_cast<std::uintptr_t>(second.contents),
                               second.length)) {
        return true;
    }
    return first.gpu_address != 0 && second.gpu_address != 0 &&
           address_ranges_overlap(first.gpu_address, first.length,
                                  second.gpu_address, second.length);
}

bool claim_index(std::vector<std::uint32_t>& indices, std::uint32_t index) {
    if (std::find(indices.begin(), indices.end(), index) != indices.end()) return false;
    indices.push_back(index);
    return true;
}

}  // namespace

struct Encoder::Impl {
    Impl(Context& owner, id<MTLComputeCommandEncoder> native_encoder,
         DispatchMode dispatch_mode, CommandBuffer& owner_command_buffer,
         std::shared_ptr<const void> command_buffer_identity_token)
        : context(owner), encoder(native_encoder), mode(dispatch_mode),
          command_buffer(&owner_command_buffer),
          command_buffer_identity(std::move(command_buffer_identity_token)) {}

    ~Impl() { finish(); }

    void fail(std::string& error, std::string message) {
        failed = true;
        if (error.empty()) error = std::move(message);
    }

    void finish() noexcept {
        if (encoder == nil) return;
        [encoder updateFence:context_hazard_fence(context)];
        [encoder endEncoding];
        encoder = nil;
    }

    Context& context;
    __strong id<MTLComputeCommandEncoder> encoder = nil;
    DispatchMode mode = DispatchMode::serial;
    CommandBuffer* command_buffer = nullptr;
    std::shared_ptr<const void> command_buffer_identity;
    bool failed = false;
    bool group_open = false;
    bool has_encoded_dispatches = false;
    std::vector<std::uint32_t> active_buffer_indices;
    std::vector<std::uint32_t> active_threadgroup_memory_indices;
    std::size_t inline_bytes_encoded = 0;
};

Encoder::Encoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Encoder::~Encoder() = default;

bool Encoder::encode(Context& context, CommandBuffer& command_buffer,
                     std::string_view label, const Body& body,
                     std::string& error) {
    error.clear();
    if (!command_buffer.belongs_to(context)) {
        error = "Metal command buffer belongs to a different Context";
        return false;
    }

    return command_buffer.encode_unsubmitted(
        [&](void* command_handle, std::string& encode_error) {
            DispatchMode mode = DispatchMode::serial;
            if (!dispatch_mode(mode, encode_error)) return false;
            if (label.empty() || label.find('\0') != std::string_view::npos) {
                encode_error = "Metal compute encoder label is empty or contains a null byte";
                return false;
            }
            if (!body) {
                encode_error = "Metal encoder body is empty";
                return false;
            }
            id<MTLFence> fence = context_hazard_fence(context);
            id<MTLCommandBuffer> native_command =
                (__bridge id<MTLCommandBuffer>)command_handle;
            if (fence == nil || native_command == nil) {
                encode_error = "Metal Context fence or command buffer is unavailable";
                return false;
            }
            id<MTLComputeCommandEncoder> native_encoder =
                [native_command
                    computeCommandEncoderWithDispatchType:
                        mode == DispatchMode::serial ? MTLDispatchTypeSerial
                                                     : MTLDispatchTypeConcurrent];
            if (native_encoder == nil) {
                encode_error = "cannot create Metal compute encoder";
                return false;
            }
            NSString* native_label =
                [[NSString alloc] initWithBytes:label.data()
                                         length:label.size()
                                       encoding:NSUTF8StringEncoding];
            if (native_label == nil) {
                [native_encoder endEncoding];
                encode_error = "Metal compute encoder label is not valid UTF-8";
                return false;
            }
            native_encoder.label = native_label;
            [native_encoder waitForFence:fence];

            Encoder encoder(std::make_unique<Impl>(
                context, native_encoder, mode, command_buffer,
                command_buffer.identity_token()));
            bool body_succeeded = body(encoder, encode_error);
            if (encoder.impl_->group_open) {
                encoder.impl_->fail(encode_error,
                                    "Metal concurrent dispatch group was not closed");
            }
            if (!body_succeeded && encode_error.empty()) {
                encode_error = "Metal encoder body failed";
            }
            body_succeeded = body_succeeded && !encoder.impl_->failed;
            encoder.impl_->finish();
            return body_succeeded;
        },
        error, true);
}

bool Encoder::dispatch(
    std::string_view kernel, EncoderSize grid, EncoderSize threads_per_group,
    std::span<const EncoderBufferBinding> buffers,
    std::span<const EncoderBytesBinding> bytes,
    std::span<const EncoderThreadgroupMemory> threadgroup_memory,
    std::string& error) {
    return dispatch(kernel, grid, threads_per_group, buffers,
                    std::span<const EncoderReadOnlyBufferBinding>{}, bytes,
                    threadgroup_memory, error);
}

bool Encoder::dispatch(
    std::string_view kernel, EncoderSize grid, EncoderSize threads_per_group,
    std::span<const EncoderBufferBinding> buffers,
    std::span<const EncoderReadOnlyBufferBinding> read_only_buffers,
    std::span<const EncoderBytesBinding> bytes,
    std::span<const EncoderThreadgroupMemory> threadgroup_memory,
    std::string& error) {
    if (impl_ == nullptr || impl_->failed) {
        error = "Metal compute encoder is unavailable after an earlier failure";
        return false;
    }
    const auto fail = [&](std::string message) {
        impl_->fail(error, std::move(message));
        return false;
    };
    if (kernel.empty() || kernel.find('\0') != std::string_view::npos ||
        grid.x == 0 || grid.y == 0 || grid.z == 0) {
        return fail("Metal dispatch kernel name or grid is invalid");
    }
    const KernelInfo* const info = find_kernel(kernel);
    if (info == nullptr) {
        return fail("Metal dispatch kernel is not present in the manifest");
    }
    std::size_t thread_count = 0;
    if (info->threadgroup_x != threads_per_group.x ||
        info->threadgroup_y != threads_per_group.y ||
        info->threadgroup_z != threads_per_group.z ||
        !checked_thread_count(threads_per_group, thread_count) ||
        info->threads != thread_count) {
        return fail("Metal dispatch threadgroup does not match the kernel manifest");
    }

    const std::string kernel_name(kernel);
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)impl_->context.pipeline(kernel_name.c_str());
    if (pipeline == nil) return fail("Metal dispatch pipeline is unavailable");
    if (pipeline.maxTotalThreadsPerThreadgroup < thread_count) {
        return fail("Metal dispatch exceeds the pipeline threadgroup limit");
    }

    std::vector<std::uint32_t> argument_indices;
    argument_indices.reserve(buffers.size() + read_only_buffers.size() + bytes.size());
    for (const EncoderBufferBinding& binding : buffers) {
        if (!claim_index(argument_indices, binding.index)) {
            return fail("Metal dispatch has duplicate buffer argument indices");
        }
        if (!validate_buffer(impl_->context, binding, error)) {
            impl_->failed = true;
            return false;
        }
    }
    for (const EncoderReadOnlyBufferBinding& binding : read_only_buffers) {
        if (!claim_index(argument_indices, binding.index)) {
            return fail("Metal dispatch has duplicate read-only buffer argument indices");
        }
        const EncoderBufferBinding validation_binding{
            binding.buffer.buffer_, binding.index, binding.required_bytes,
            binding.alignment};
        if (!validate_buffer(impl_->context, validation_binding, error)) {
            impl_->failed = true;
            return false;
        }
        for (const EncoderBufferBinding& writable : buffers) {
            if (buffer_views_overlap(binding.buffer.buffer_, writable.buffer)) {
                return fail("Metal dispatch read-only input overlaps a writable buffer view");
            }
        }
    }
    for (const EncoderBytesBinding& binding : bytes) {
        if (!claim_index(argument_indices, binding.index)) {
            return fail("Metal dispatch has duplicate buffer and inline-byte indices");
        }
        if (binding.bytes == nullptr || binding.length == 0) {
            return fail("Metal dispatch inline-byte binding is empty");
        }
    }
    std::size_t dispatch_inline_bytes = 0;
    for (const EncoderBytesBinding& binding : bytes) {
        if (binding.length > kMaxInlineBytesPerEncoder ||
            dispatch_inline_bytes >
                kMaxInlineBytesPerEncoder - binding.length) {
            return fail("Metal dispatch inline-byte bindings exceed the 4 KiB encoder limit");
        }
        dispatch_inline_bytes += binding.length;
    }
    if (impl_->inline_bytes_encoded >
        kMaxInlineBytesPerEncoder - dispatch_inline_bytes) {
        return fail("Metal segment inline-byte bindings exceed the 4 KiB encoder limit");
    }

    std::size_t dynamic_memory_bytes = 0;
    for (std::size_t index = 0; index < threadgroup_memory.size(); ++index) {
        const EncoderThreadgroupMemory& binding = threadgroup_memory[index];
        if (binding.length == 0 || binding.length % 16u != 0u ||
            binding.length >
                std::numeric_limits<std::size_t>::max() - dynamic_memory_bytes) {
            return fail("Metal dispatch dynamic threadgroup memory size is invalid");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (threadgroup_memory[earlier].index == binding.index) {
                return fail("Metal dispatch has duplicate threadgroup-memory indices");
            }
        }
        dynamic_memory_bytes += binding.length;
    }
    std::vector<std::uint32_t> threadgroup_indices;
    threadgroup_indices.reserve(threadgroup_memory.size());
    for (const EncoderThreadgroupMemory& binding : threadgroup_memory) {
        threadgroup_indices.push_back(binding.index);
    }
    if (!impl_->context.validate_pipeline_bindings(
            kernel, argument_indices, threadgroup_indices, error)) {
        impl_->failed = true;
        return false;
    }
    const std::uint64_t max_threadgroup_memory =
        impl_->context.device_facts().max_threadgroup_memory_length;
    const std::uint64_t static_memory =
        static_cast<std::uint64_t>(pipeline.staticThreadgroupMemoryLength);
    if (dynamic_memory_bytes > max_threadgroup_memory ||
        static_memory > max_threadgroup_memory - dynamic_memory_bytes) {
        return fail("Metal dispatch exceeds the device threadgroup-memory limit");
    }

    std::vector<BufferRef> access_ranges;
    std::vector<BufferRef> write_ranges;
    access_ranges.reserve(buffers.size() + read_only_buffers.size());
    write_ranges.reserve(buffers.size());
    for (const EncoderBufferBinding& binding : buffers) {
        const BufferRef active =
            binding.buffer.slice(0u, binding.required_bytes);
        if (!active) return fail("Metal dispatch writable range is invalid");
        access_ranges.push_back(active);
        write_ranges.push_back(active);
    }
    for (const EncoderReadOnlyBufferBinding& binding : read_only_buffers) {
        const BufferRef active =
            binding.buffer.buffer_.slice(0u, binding.required_bytes);
        if (!active) return fail("Metal dispatch read-only range is invalid");
        access_ranges.push_back(active);
    }
    if (!impl_->command_buffer->track_access_ranges(access_ranges,
                                                    write_ranges, error)) {
        impl_->failed = true;
        return false;
    }

    for (const std::uint32_t index : impl_->active_buffer_indices) {
        if (std::find(argument_indices.begin(), argument_indices.end(), index) ==
            argument_indices.end()) {
            [impl_->encoder setBuffer:nil
                              offset:0
                             atIndex:static_cast<NSUInteger>(index)];
        }
    }
    for (const std::uint32_t index : impl_->active_threadgroup_memory_indices) {
        if (std::find(threadgroup_indices.begin(), threadgroup_indices.end(), index) ==
            threadgroup_indices.end()) {
            [impl_->encoder setThreadgroupMemoryLength:0
                                               atIndex:static_cast<NSUInteger>(index)];
        }
    }
    for (const EncoderBufferBinding& binding : buffers) {
        id<MTLBuffer> native = (__bridge id<MTLBuffer>)binding.buffer.handle;
        [impl_->encoder setBuffer:native
                           offset:static_cast<NSUInteger>(binding.buffer.offset)
                          atIndex:static_cast<NSUInteger>(binding.index)];
    }
    for (const EncoderReadOnlyBufferBinding& binding : read_only_buffers) {
        const BufferRef buffer = binding.buffer.buffer_;
        id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
        [impl_->encoder setBuffer:native
                           offset:static_cast<NSUInteger>(buffer.offset)
                          atIndex:static_cast<NSUInteger>(binding.index)];
    }
    for (const EncoderBytesBinding& binding : bytes) {
        [impl_->encoder setBytes:binding.bytes
                          length:binding.length
                         atIndex:static_cast<NSUInteger>(binding.index)];
    }
    for (const EncoderThreadgroupMemory& binding : threadgroup_memory) {
        [impl_->encoder setThreadgroupMemoryLength:binding.length
                                           atIndex:static_cast<NSUInteger>(binding.index)];
    }
    [impl_->encoder setComputePipelineState:pipeline];
    [impl_->encoder
        dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(grid.x),
                                         static_cast<NSUInteger>(grid.y),
                                         static_cast<NSUInteger>(grid.z))
        threadsPerThreadgroup:
            MTLSizeMake(static_cast<NSUInteger>(threads_per_group.x),
                        static_cast<NSUInteger>(threads_per_group.y),
                        static_cast<NSUInteger>(threads_per_group.z))];
    impl_->active_buffer_indices = std::move(argument_indices);
    impl_->active_threadgroup_memory_indices = std::move(threadgroup_indices);
    impl_->inline_bytes_encoded += dispatch_inline_bytes;
    impl_->has_encoded_dispatches = true;

    if (impl_->mode == DispatchMode::concurrent && !impl_->group_open) {
        [impl_->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    error.clear();
    return true;
}

bool Encoder::use_resources(
    std::span<const EncoderResourceBinding> resources,
    std::string& error) {
    error.clear();
    if (impl_ == nullptr || impl_->failed) {
        error = "Metal compute encoder is unavailable after an earlier failure";
        return false;
    }
    std::vector<void*> used_handles;
    used_handles.reserve(resources.size());
    std::vector<BufferRef> access_ranges;
    access_ranges.reserve(resources.size());
    for (const EncoderResourceBinding& resource : resources) {
        const EncoderBufferBinding validation_binding{
            resource.buffer.buffer_, 0u, resource.required_bytes,
            resource.alignment};
        if (!validate_buffer(impl_->context, validation_binding, error)) {
            impl_->failed = true;
            return false;
        }
        // `required_bytes` validates the minimum slice expected by the
        // caller, but indirect shader addresses may reach any byte in the
        // declared view. Track that complete range for future scratch claims.
        access_ranges.push_back(resource.buffer.buffer_);
    }
    if (!impl_->command_buffer->track_access_ranges(
            access_ranges, std::span<const BufferRef>{}, error)) {
        impl_->failed = true;
        return false;
    }
    for (const EncoderResourceBinding& resource : resources) {
        void* const handle = resource.buffer.buffer_.handle;
        if (std::find(used_handles.begin(), used_handles.end(), handle) !=
            used_handles.end()) {
            continue;
        }
        id<MTLBuffer> native = (__bridge id<MTLBuffer>)handle;
        [impl_->encoder useResource:native usage:MTLResourceUsageRead];
        used_handles.push_back(handle);
    }
    return true;
}

bool CommandBuffer::claim_staging_range(BufferRef range,
                                        std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(encoding_mutex_);
    if (!can_encode() || !range) {
        error = "cannot reserve staging scratch on an unavailable command buffer";
        return false;
    }
    for (const BufferRef& existing : staging_ranges_) {
        if (buffer_views_overlap(existing, range)) {
            error = "command buffer reuses overlapping staging scratch";
            return false;
        }
    }
    for (const BufferRef& existing : access_ranges_) {
        if (buffer_views_overlap(existing, range)) {
            error = "staging scratch overlaps an earlier GPU buffer access";
            return false;
        }
    }
    staging_ranges_.push_back(range);
    error.clear();
    return true;
}

bool CommandBuffer::track_access_ranges(
    std::span<const BufferRef> ranges, std::span<const BufferRef> writes,
    std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(encoding_mutex_);
    if (!can_encode()) {
        error = "cannot track GPU buffer access on an unavailable command buffer";
        return false;
    }
    for (const BufferRef& write : writes) {
        if (!write) {
            error = "cannot track an invalid GPU writable range";
            return false;
        }
        for (const BufferRef& staging : staging_ranges_) {
            if (buffer_views_overlap(write, staging)) {
                error = "GPU dispatch write overlaps reserved staging scratch";
                return false;
            }
        }
    }
    for (const BufferRef& range : ranges) {
        if (!range) {
            error = "cannot track an invalid GPU buffer range";
            return false;
        }
    }
    access_ranges_.insert(access_ranges_.end(), ranges.begin(), ranges.end());
    error.clear();
    return true;
}
bool Encoder::claim_unique_staging_range(BufferRef buffer,
                                         std::size_t required_bytes,
                                         std::size_t alignment,
                                         std::string& error) {
    error.clear();
    if (impl_ == nullptr || impl_->failed ||
        impl_->command_buffer == nullptr) {
        error = "Metal encoder is unavailable for staging scratch";
        if (impl_ != nullptr) impl_->failed = true;
        return false;
    }
    const EncoderBufferBinding binding{buffer, 0u, required_bytes, alignment};
    if (!validate_buffer(impl_->context, binding, error)) {
        impl_->failed = true;
        return false;
    }
    const BufferRef active_range = buffer.slice(0u, required_bytes);
    if (!active_range ||
        !impl_->command_buffer->claim_staging_range(active_range, error)) {
        if (error.empty()) {
            error = "Metal encoder cannot reserve its staging scratch range";
        }
        impl_->failed = true;
        return false;
    }
    return true;
}

const void* Encoder::command_buffer_identity() const noexcept {
    return impl_ == nullptr ? nullptr
                            : impl_->command_buffer_identity.get();
}

bool Encoder::concurrent_group_open() const noexcept {
    return impl_ != nullptr && impl_->mode == DispatchMode::concurrent &&
           impl_->group_open;
}

bool Encoder::has_encoded_dispatches() const noexcept {
    return impl_ != nullptr && impl_->has_encoded_dispatches;
}

bool Encoder::group_begin(std::string& error) {
    if (impl_ == nullptr || impl_->failed) {
        error = "Metal compute encoder is unavailable after an earlier failure";
        return false;
    }
    if (impl_->mode == DispatchMode::serial) {
        error.clear();
        return true;
    }
    if (impl_->group_open) {
        impl_->fail(error, "Metal concurrent dispatch groups cannot be nested");
        return false;
    }
    impl_->group_open = true;
    error.clear();
    return true;
}

bool Encoder::group_end(std::string& error) {
    if (impl_ == nullptr || impl_->failed) {
        error = "Metal compute encoder is unavailable after an earlier failure";
        return false;
    }
    if (impl_->mode == DispatchMode::serial) {
        error.clear();
        return true;
    }
    if (!impl_->group_open) {
        impl_->fail(error, "Metal concurrent dispatch group end has no matching begin");
        return false;
    }
    [impl_->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    impl_->group_open = false;
    error.clear();
    return true;
}

}  // namespace strata::metal
