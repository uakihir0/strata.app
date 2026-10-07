// src/metal/engine/sampler_dispatch.mm - production Metal sampler encoders.
#import <Metal/Metal.h>

#include "strata/metal/sampler_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "metal_fence.hpp"
#include "strata/metal/sampler_tail.hpp"
#include "../kernels/common/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kSamplerBlockSpan = 4096u;
constexpr std::uint32_t kSamplerCandidateStride = 64u;
constexpr std::uint32_t kSamplerMaxHistory = 4096u;
constexpr std::uint32_t kGreedyChunkVocab = 131072u;

struct BufferBinding {
    BufferRef buffer;
    NSUInteger index;
};

struct GridSize {
    NSUInteger x;
    NSUInteger y;
    NSUInteger z;
};

const KernelInfo* find_kernel_info(const char* name) {
    std::size_t count = 0;
    const KernelInfo* const manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == name) return &manifest[i];
    }
    return nullptr;
}

bool validate_buffer(BufferRef buffer, std::size_t required_bytes,
                     std::size_t alignment, std::string_view name,
                     std::string& error) {
    if (!buffer || buffer.length < required_bytes || buffer.offset % alignment != 0) {
        error = std::string(name) + " buffer is missing, undersized or misaligned";
        return false;
    }
    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || buffer.offset > native.length ||
        buffer.length > native.length - buffer.offset) {
        error = std::string(name) + " buffer view exceeds its Metal allocation";
        return false;
    }
    if (native.contents == nullptr) {
        error = std::string(name) + " buffer has no CPU-visible backing metadata";
        return false;
    }
    const std::uintptr_t native_contents =
        reinterpret_cast<std::uintptr_t>(native.contents);
    if (native_contents >
            std::numeric_limits<std::uintptr_t>::max() - buffer.offset ||
        buffer.contents !=
            reinterpret_cast<void*>(native_contents + buffer.offset)) {
        error = std::string(name) +
                " CPU backing metadata does not match its Metal view";
        return false;
    }
    const std::uint64_t native_gpu_address =
        static_cast<std::uint64_t>(native.gpuAddress);
    std::uint64_t expected_gpu_address = 0;
    if (!detail::gpu_address_with_offset(native_gpu_address, buffer.offset,
                                         expected_gpu_address) ||
        buffer.gpu_address != expected_gpu_address) {
        error = std::string(name) +
                " GPU backing metadata does not match its Metal view";
        return false;
    }
    return true;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_views_overlap(BufferRef left, BufferRef right) {
    const bool same_buffer_overlap =
        left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.length,
                               right.offset, right.length);
    const bool cpu_backing_overlap = address_ranges_overlap(
        reinterpret_cast<std::uintptr_t>(left.contents), left.length,
        reinterpret_cast<std::uintptr_t>(right.contents), right.length);
    const bool gpu_backing_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        address_ranges_overlap(left.gpu_address, left.length,
                               right.gpu_address, right.length);
    return same_buffer_overlap || cpu_backing_overlap || gpu_backing_overlap;
}

bool validate_rows_and_history(BufferRef rows, BufferRef history,
                               std::uint32_t row_count, std::string& error) {
    if (rows.contents == nullptr) {
        error = "sampler row parameters must be host-visible";
        return false;
    }
    const auto* const row_data = static_cast<const SampRow*>(rows.contents);
    const std::uint64_t history_entries = history.length / sizeof(std::int32_t);
    for (std::uint32_t row_index = 0; row_index < row_count; ++row_index) {
        const SampRow& row = row_data[row_index];
        const std::uint64_t end =
            static_cast<std::uint64_t>(row.hist_off) + row.hist_len;
        if (row.hist_len > kSamplerMaxHistory || end > history_entries) {
            error = "sampler row history exceeds its buffer or the 4096-token limit";
            return false;
        }
    }
    return true;
}

std::uint32_t sampled_k(const SampRow& row, std::uint32_t n_vocab) {
    const std::uint32_t requested =
        row.top_k > 0 && row.top_k < 64 ? static_cast<std::uint32_t>(row.top_k) : 64u;
    return std::min(n_vocab, requested);
}

bool validate_kpart_rows(BufferRef rows, std::uint32_t row_count,
                         std::uint32_t n_vocab, std::uint32_t kpart,
                         std::string& error) {
    if (kpart == 0) return true;
    if (rows.contents == nullptr) {
        error = "sampler rows must be host-visible to validate kpart";
        return false;
    }
    const auto* const row_data = static_cast<const SampRow*>(rows.contents);
    for (std::uint32_t row_index = 0; row_index < row_count; ++row_index) {
        if (kpart < sampled_k(row_data[row_index], n_vocab)) {
            error = "sampler kpart is smaller than the requested top-k";
            return false;
        }
    }
    return true;
}

bool validate_kernel(Context& context, const char* kernel_name,
                     const BufferBinding* bindings, std::size_t binding_count,
                     const void* parameters, std::size_t parameter_bytes,
                     std::size_t threadgroup_memory_bytes,
                     GridSize threads_per_group,
                     id<MTLComputePipelineState> __strong& pipeline,
                     std::string& error) {
    const KernelInfo* const info = find_kernel_info(kernel_name);
    if (info == nullptr ||
        info->threadgroup_x != threads_per_group.x ||
        info->threadgroup_y != threads_per_group.y ||
        info->threadgroup_z != threads_per_group.z ||
        info->threads != threads_per_group.x * threads_per_group.y *
                             threads_per_group.z) {
        error = std::string("sampler threadgroup does not match its manifest: ") + kernel_name;
        return false;
    }
    void* const pipeline_handle = context.pipeline(kernel_name);
    if (pipeline_handle == nullptr) {
        error = std::string("sampler pipeline is unavailable: ") + kernel_name;
        return false;
    }
    pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_handle;
    if (pipeline.maxTotalThreadsPerThreadgroup <
            threads_per_group.x * threads_per_group.y * threads_per_group.z ||
        (threadgroup_memory_bytes != 0 &&
         (threadgroup_memory_bytes % 16u != 0u ||
          threadgroup_memory_bytes > context.device_facts().max_threadgroup_memory_length))) {
        error = std::string("sampler dispatch exceeds device limits: ") + kernel_name;
        return false;
    }
    if (parameter_bytes != 0 && parameters == nullptr) {
        error = std::string("sampler parameter block is null: ") + kernel_name;
        return false;
    }
    for (std::size_t i = 0; i < binding_count; ++i) {
        if (!validate_buffer(bindings[i].buffer, 1u, 1u, "sampler binding", error)) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (buffer_views_overlap(bindings[i].buffer, bindings[j].buffer)) {
                error = "sampler buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    error.clear();
    return true;
}

bool encode_kernel(Context& context, CommandBuffer& command_buffer,
                   const char* kernel_name, const std::string& label,
                   const BufferBinding* bindings, std::size_t binding_count,
                   const void* parameters, std::size_t parameter_bytes,
                   NSUInteger parameter_index, std::size_t threadgroup_memory_bytes,
                   GridSize grid, GridSize threads_per_group, std::string& error,
                   id<MTLComputeCommandEncoder> shared_encoder = nil) {
    id<MTLComputePipelineState> pipeline = nil;
    if (!validate_kernel(context, kernel_name, bindings, binding_count,
                         parameters, parameter_bytes,
                         threadgroup_memory_bytes, threads_per_group,
                         pipeline, error)) {
        return false;
    }
    void* const command_handle = command_buffer.unsafe_native_handle();
    if (command_handle == nullptr) {
        error = "sampler command buffer is unavailable";
        return false;
    }
    id<MTLCommandBuffer> native_command =
        (__bridge id<MTLCommandBuffer>)command_handle;
    if (native_command == nil ||
        native_command.status != MTLCommandBufferStatusNotEnqueued) {
        error = "sampler work must be encoded before its command buffer is enqueued or committed";
        return false;
    }
    @autoreleasepool {
        const bool owns_encoder = shared_encoder == nil;
        id<MTLComputeCommandEncoder> encoder = shared_encoder;
        if (owns_encoder) encoder = begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = std::string("cannot create sampler encoder: ") + kernel_name;
            return false;
        }
        encoder.label = @(label.c_str());
        [encoder setComputePipelineState:pipeline];
        for (std::size_t i = 0; i < binding_count; ++i) {
            const BufferRef buffer = bindings[i].buffer;
            [encoder setBuffer:(__bridge id<MTLBuffer>)buffer.handle
                        offset:buffer.offset
                       atIndex:bindings[i].index];
        }
        if (parameter_bytes != 0) {
            [encoder setBytes:parameters length:parameter_bytes atIndex:parameter_index];
        }
        if (threadgroup_memory_bytes != 0) {
            [encoder setThreadgroupMemoryLength:threadgroup_memory_bytes atIndex:0];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
                  threadsPerThreadgroup:MTLSizeMake(
                      threads_per_group.x, threads_per_group.y, threads_per_group.z)];
        if (owns_encoder) end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_kernel(Context& context, Encoder& encoder,
                   const char* kernel_name,
                   const BufferBinding* bindings, std::size_t binding_count,
                   const void* parameters, std::size_t parameter_bytes,
                   NSUInteger parameter_index,
                   std::size_t threadgroup_memory_bytes,
                   GridSize grid, GridSize threads_per_group,
                   std::string& error) {
    [[maybe_unused]] id<MTLComputePipelineState> pipeline = nil;
    if (!validate_kernel(context, kernel_name, bindings, binding_count,
                         parameters, parameter_bytes,
                         threadgroup_memory_bytes, threads_per_group,
                         pipeline, error)) {
        return false;
    }

    std::vector<EncoderBufferBinding> encoder_bindings;
    encoder_bindings.reserve(binding_count);
    for (std::size_t i = 0; i < binding_count; ++i) {
        if (bindings[i].index > std::numeric_limits<std::uint32_t>::max()) {
            error = "sampler binding index exceeds the shared Encoder limit";
            return false;
        }
        encoder_bindings.push_back(EncoderBufferBinding{
            bindings[i].buffer, static_cast<std::uint32_t>(bindings[i].index),
            1u, 1u});
    }

    std::array<EncoderBytesBinding, 1> byte_bindings{};
    std::span<const EncoderBytesBinding> byte_span;
    if (parameter_bytes != 0u) {
        if (parameter_index > std::numeric_limits<std::uint32_t>::max()) {
            error = "sampler inline-byte index exceeds the shared Encoder limit";
            return false;
        }
        byte_bindings[0] = EncoderBytesBinding{
            parameters, parameter_bytes,
            static_cast<std::uint32_t>(parameter_index)};
        byte_span = byte_bindings;
    }

    std::array<EncoderThreadgroupMemory, 1> threadgroup_bindings{};
    std::span<const EncoderThreadgroupMemory> threadgroup_span;
    if (threadgroup_memory_bytes != 0u) {
        threadgroup_bindings[0] = EncoderThreadgroupMemory{
            0u, threadgroup_memory_bytes};
        threadgroup_span = threadgroup_bindings;
    }

    return encoder.dispatch(
        kernel_name, {grid.x, grid.y, grid.z},
        {threads_per_group.x, threads_per_group.y, threads_per_group.z},
        encoder_bindings, byte_span, threadgroup_span, error);
}

bool validate_common_buffers(std::uint32_t rows, std::uint32_t n_vocab,
                             const SamplerBuffers& buffers, std::string& error) {
    if (!sampler_shape_supported(rows, n_vocab)) {
        error = "sampler shape exceeds T=1..8 or the 262144-token vocabulary limit";
        return false;
    }
    if (!validate_buffer(buffers.logits,
                         static_cast<std::size_t>(rows) * n_vocab * sizeof(float),
                         alignof(float), "sampler logits", error) ||
        !validate_buffer(buffers.history, sizeof(std::int32_t), alignof(std::int32_t),
                         "sampler history", error) ||
        !validate_buffer(buffers.rows, static_cast<std::size_t>(rows) * sizeof(SampRow),
                         alignof(SampRow), "sampler rows", error) ||
        !validate_rows_and_history(buffers.rows, buffers.history, rows, error)) {
        return false;
    }
    return true;
}

bool validate_split_buffers(std::uint32_t rows, std::uint32_t n_vocab,
                            const SamplerBuffers& buffers, std::uint32_t kpart,
                            std::string& error) {
    if (!validate_common_buffers(rows, n_vocab, buffers, error)) return false;
    if (kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler kpart exceeds the vocabulary or 64-candidate limit";
        return false;
    }
    if (!validate_kpart_rows(buffers.rows, rows, n_vocab, kpart, error)) return false;
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const std::size_t candidate_count =
        static_cast<std::size_t>(rows) * n_blocks * kSamplerCandidateStride;
    const std::size_t selected_count =
        static_cast<std::size_t>(rows) * kSamplerCandidateStride;
    return validate_buffer(buffers.candidates, candidate_count * sizeof(SampCandidate),
                           alignof(SampCandidate), "sampler candidates", error) &&
           validate_buffer(buffers.selected, selected_count * sizeof(SampCandidate),
                           alignof(SampCandidate), "sampler selected", error);
}

bool validate_sampler_command(Context& context, CommandBuffer& command_buffer,
                              std::string& error) {
    if (!command_buffer.belongs_to(context)) {
        error = "sampler command buffer belongs to a different Context";
        return false;
    }
    return true;
}

bool validate_sampler_buffers(Context& context,
                              std::initializer_list<BufferRef> buffers,
                              std::string& error) {
    for (const BufferRef& buffer : buffers) {
        if (buffer.handle == nullptr) continue;
        if (!context.buffers().owns(buffer)) {
            error = "sampler buffer is not owned by this Context";
            return false;
        }
    }
    return true;
}

std::size_t greedy_scratch_bytes(std::uint32_t n_vocab) {
    const std::size_t chunk_vocab = std::min<std::size_t>(n_vocab, kGreedyChunkVocab);
    const std::size_t bitmap_words = (chunk_vocab + 31u) / 32u;
    constexpr std::size_t partial_words = 2u * 32u;
    const std::size_t words = bitmap_words + partial_words;
    const std::size_t bytes = words * sizeof(std::uint32_t);
    return (bytes + 15u) & ~std::size_t{15u};
}

std::size_t rounded_threadgroup_bytes(std::size_t bytes) {
    return (bytes + 15u) & ~std::size_t{15u};
}

}  // namespace

bool encode_sampler_greedy(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, std::uint32_t n_vocab,
                           const SamplerBuffers& buffers, std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_sampler_command(context, command_buffer, error)) {
        return false;
    }
    if (!sampler_shape_supported(rows, n_vocab)) {
        error = "sampler shape exceeds T=1..8 or the 262144-token vocabulary limit";
        return false;
    }
    if (!validate_sampler_buffers(
            context, {buffers.logits, buffers.history, buffers.rows,
                      buffers.output}, error)) {
        return false;
    }
    if (!validate_common_buffers(rows, n_vocab, buffers, error) ||
        !validate_buffer(buffers.output, static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                         alignof(std::int32_t), "sampler output", error)) {
        return false;
    }
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const SampArgs args{n_vocab, n_blocks, 0u, 1u};
    const std::array bindings{
        BufferBinding{buffers.logits, 0},
        BufferBinding{buffers.history, 1},
        BufferBinding{buffers.rows, 2},
        BufferBinding{buffers.output, 3},
    };
    return encode_kernel(context, command_buffer, "strata_samp_greedy",
                         "strata_samp_greedy", bindings.data(), bindings.size(),
                         &args, sizeof(args), 4, greedy_scratch_bytes(n_vocab),
                         {rows, 1, 1}, {1024, 1, 1}, error);
}

bool encode_sampler_sampled(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows, std::uint32_t n_vocab,
                            const SamplerBuffers& buffers,
                            bool penalties_already_applied, std::string& error,
                            std::uint32_t kpart) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_sampler_command(context, command_buffer, error)) {
        return false;
    }
    if (!sampler_shape_supported(rows, n_vocab) ||
        kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler shape or kpart exceeds supported bounds";
        return false;
    }
    if (!validate_sampler_buffers(
            context, {buffers.logits, buffers.history, buffers.rows,
                      buffers.candidates, buffers.selected}, error)) {
        return false;
    }
    if (!validate_split_buffers(rows, n_vocab, buffers, kpart, error)) {
        return false;
    }
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const SampArgs part_args{
        n_vocab, n_blocks, kpart, penalties_already_applied ? 0u : 1u};
    const std::array part_bindings{
        BufferBinding{buffers.logits, 0},
        BufferBinding{buffers.history, 1},
        BufferBinding{buffers.rows, 2},
        BufferBinding{buffers.candidates, 3},
    };

    const std::size_t candidate_count =
        static_cast<std::size_t>(rows) * n_blocks * kSamplerCandidateStride;
    const std::size_t selected_count =
        static_cast<std::size_t>(rows) * kSamplerCandidateStride;
    if (!validate_buffer(buffers.rows,
                         static_cast<std::size_t>(rows) * sizeof(SampRow),
                         alignof(SampRow), "sampler rows", error) ||
        !validate_buffer(buffers.candidates,
                         candidate_count * sizeof(SampCandidate),
                         alignof(SampCandidate), "sampler candidates", error) ||
        !validate_buffer(buffers.selected,
                         selected_count * sizeof(SampCandidate),
                         alignof(SampCandidate), "sampler selected", error) ||
        !validate_kpart_rows(buffers.rows, rows, n_vocab, kpart, error)) {
        return false;
    }
    const SampArgs merge_args{n_vocab, n_blocks, kpart, 0u};
    const std::array merge_bindings{
        BufferBinding{buffers.candidates, 0},
        BufferBinding{buffers.rows, 1},
        BufferBinding{buffers.selected, 2},
    };
    const std::size_t merge_scratch =
        static_cast<std::size_t>(n_blocks) * kSamplerCandidateStride *
        sizeof(SampCandidate);
    id<MTLComputePipelineState> part_pipeline = nil;
    id<MTLComputePipelineState> merge_pipeline = nil;
    if (!validate_kernel(context, "strata_samp_split_part",
                         part_bindings.data(), part_bindings.size(),
                         &part_args, sizeof(part_args), 0u, {128, 1, 1},
                         part_pipeline, error) ||
        !validate_kernel(context, "strata_samp_split_merge",
                         merge_bindings.data(), merge_bindings.size(),
                         &merge_args, sizeof(merge_args), merge_scratch,
                         {32, 1, 1}, merge_pipeline, error)) {
        return false;
    }

    return command_buffer.encode_unsubmitted(
        [&](void*, std::string& encode_error) {
            void* const command_handle = command_buffer.unsafe_native_handle();
            if (command_handle == nullptr) {
                encode_error = "sampler command buffer is unavailable";
                return false;
            }
            id<MTLCommandBuffer> native_command =
                (__bridge id<MTLCommandBuffer>)command_handle;
            id<MTLComputeCommandEncoder> encoder =
                begin_compute_encoder(context, native_command);
            if (encoder == nil) {
                encode_error = "cannot create sampler sampled encoder";
                return false;
            }
            encoder.label = @"sampler sampled split and merge";
            const bool part_ok = encode_kernel(
                context, command_buffer, "strata_samp_split_part",
                "strata_samp_split_part", part_bindings.data(),
                part_bindings.size(), &part_args, sizeof(part_args), 4, 0,
                {n_blocks, rows, 1}, {128, 1, 1}, encode_error, encoder);
            const bool merge_ok =
                part_ok &&
                encode_kernel(context, command_buffer,
                              "strata_samp_split_merge",
                              "strata_samp_split_merge",
                              merge_bindings.data(), merge_bindings.size(),
                              &merge_args, sizeof(merge_args), 3,
                              merge_scratch, {rows, 1, 1}, {32, 1, 1},
                              encode_error, encoder);
            end_compute_encoder(context, encoder);
            return part_ok && merge_ok;
        },
        error, true);
}

bool encode_sampler_split_part(Context& context, CommandBuffer& command_buffer,
                               std::uint32_t rows, std::uint32_t n_vocab,
                               const SamplerBuffers& buffers,
                               bool penalties_already_applied, std::uint32_t kpart,
                               std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_sampler_command(context, command_buffer, error)) {
        return false;
    }
    if (!sampler_shape_supported(rows, n_vocab) ||
        kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler shape or kpart exceeds supported bounds";
        return false;
    }
    if (!validate_sampler_buffers(
            context, {buffers.logits, buffers.history, buffers.rows,
                      buffers.candidates, buffers.selected}, error)) {
        return false;
    }
    if (!validate_split_buffers(rows, n_vocab, buffers, kpart, error)) return false;
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const SampArgs args{n_vocab, n_blocks, kpart,
                        penalties_already_applied ? 0u : 1u};
    const std::array part_bindings{
        BufferBinding{buffers.logits, 0},
        BufferBinding{buffers.history, 1},
        BufferBinding{buffers.rows, 2},
        BufferBinding{buffers.candidates, 3},
    };
    return encode_kernel(context, command_buffer, "strata_samp_split_part",
                         "strata_samp_split_part", part_bindings.data(),
                         part_bindings.size(), &args, sizeof(args), 4, 0,
                         {n_blocks, rows, 1}, {128, 1, 1}, error);
}

bool encode_sampler_split_merge(Context& context, CommandBuffer& command_buffer,
                                std::uint32_t rows, std::uint32_t n_vocab,
                                BufferRef rows_buffer, BufferRef candidates,
                                BufferRef selected, std::uint32_t kpart,
                                std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!sampler_shape_supported(rows, n_vocab) ||
        kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler merge shape or kpart exceeds supported bounds";
        return false;
    }
    if (!validate_sampler_command(context, command_buffer, error)) {
        return false;
    }
    if (!validate_sampler_buffers(
            context, {rows_buffer, candidates, selected}, error)) {
        return false;
    }
    if (!sampler_shape_supported(rows, n_vocab) ||
        kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler merge shape or kpart exceeds supported bounds";
        return false;
    }
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const std::size_t candidate_count =
        static_cast<std::size_t>(rows) * n_blocks * kSamplerCandidateStride;
    const std::size_t selected_count =
        static_cast<std::size_t>(rows) * kSamplerCandidateStride;
    if (!validate_buffer(rows_buffer, static_cast<std::size_t>(rows) * sizeof(SampRow),
                         alignof(SampRow), "sampler rows", error) ||
        !validate_buffer(candidates, candidate_count * sizeof(SampCandidate),
                         alignof(SampCandidate), "sampler candidates", error) ||
        !validate_buffer(selected, selected_count * sizeof(SampCandidate),
                         alignof(SampCandidate), "sampler selected", error)) {
        return false;
    }
    if (!validate_kpart_rows(rows_buffer, rows, n_vocab, kpart, error)) return false;
    const SampArgs args{n_vocab, n_blocks, kpart, 0u};
    const std::array bindings{
        BufferBinding{candidates, 0},
        BufferBinding{rows_buffer, 1},
        BufferBinding{selected, 2},
    };
    const std::size_t merge_scratch =
        static_cast<std::size_t>(n_blocks) * kSamplerCandidateStride *
        sizeof(SampCandidate);
    return encode_kernel(context, command_buffer, "strata_samp_split_merge",
                         "strata_samp_split_merge", bindings.data(),
                         bindings.size(), &args, sizeof(args), 3,
                         merge_scratch, {rows, 1, 1}, {32, 1, 1}, error);
}

bool encode_sampler_coupled_penalty(Context& context, CommandBuffer& command_buffer,
                                    std::uint32_t n_vocab, std::uint32_t sampled_vocab,
                                    bool has_id_to_subset_map, BufferRef logits,
                                    BufferRef id_to_subset, BufferRef history_ring,
                                    BufferRef row, std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!sampler_shape_supported(1u, n_vocab) ||
        !sampler_shape_supported(1u, sampled_vocab) ||
        sampled_vocab > n_vocab ||
        (!has_id_to_subset_map && n_vocab != sampled_vocab)) {
        error = "coupled sampler vocabulary must be a valid subset and needs an id map";
        return false;
    }
    if (!validate_sampler_command(context, command_buffer, error)) {
        return false;
    }
    if (!validate_sampler_buffers(
            context, {logits, id_to_subset, history_ring, row}, error)) {
        return false;
    }
    const std::size_t inverse_map_bytes =
        has_id_to_subset_map ? static_cast<std::size_t>(n_vocab) * sizeof(std::int32_t)
                             : sizeof(std::int32_t);
    if (!validate_buffer(logits, static_cast<std::size_t>(sampled_vocab) * sizeof(float),
                         alignof(float), "coupled sampler logits", error) ||
        !validate_buffer(id_to_subset, inverse_map_bytes, alignof(std::int32_t),
                         "coupled sampler inverse map", error) ||
        !validate_buffer(history_ring, sizeof(std::int32_t), alignof(std::int32_t),
                         "coupled sampler history", error) ||
        !validate_buffer(row, sizeof(SampRow), alignof(SampRow),
                         "coupled sampler row", error)) {
        return false;
    }
    if (row.contents == nullptr) {
        error = "coupled sampler row must be host-visible";
        return false;
    }
    const SampRow& row_data = *static_cast<const SampRow*>(row.contents);
    const std::uint64_t history_entries =
        history_ring.length / sizeof(std::int32_t);
    if (row_data.hist_len > kSamplerMaxHistory ||
        static_cast<std::uint64_t>(row_data.hist_off) + row_data.hist_len > history_entries) {
        error = "coupled sampler history exceeds its buffer or the 4096-token limit";
        return false;
    }

    const SampPenalizeArgs args{
        sampled_vocab, n_vocab, has_id_to_subset_map ? 1u : 0u};
    const std::array bindings{
        BufferBinding{logits, 0},
        BufferBinding{id_to_subset, 1},
        BufferBinding{history_ring, 2},
        BufferBinding{row, 3},
    };
    const std::size_t scratch = rounded_threadgroup_bytes(
        ((static_cast<std::size_t>(sampled_vocab) + 31u) / 32u) *
        sizeof(std::uint32_t));
    return encode_kernel(context, command_buffer, "strata_samp_coupled_penalize",
                         "strata_samp_coupled_penalize", bindings.data(),
                         bindings.size(), &args, sizeof(args), 4, scratch,
                         {1, 1, 1}, {1024, 1, 1}, error);
}



bool encode_sampler_greedy(Context& context, Encoder& encoder,
                           std::uint32_t rows, std::uint32_t n_vocab,
                           const SamplerBuffers& buffers, std::string& error) {
    if (!sampler_shape_supported(rows, n_vocab)) {
        error = "sampler shape exceeds T=1..8 or the 262144-token vocabulary limit";
        return false;
    }
    if (!validate_sampler_buffers(
            context, {buffers.logits, buffers.history, buffers.rows,
                      buffers.output}, error) ||
        !validate_common_buffers(rows, n_vocab, buffers, error) ||
        !validate_buffer(buffers.output,
                         static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                         alignof(std::int32_t), "sampler output", error)) {
        return false;
    }
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const SampArgs args{n_vocab, n_blocks, 0u, 1u};
    const std::array bindings{
        BufferBinding{buffers.logits, 0},
        BufferBinding{buffers.history, 1},
        BufferBinding{buffers.rows, 2},
        BufferBinding{buffers.output, 3},
    };
    return encode_kernel(context, encoder, "strata_samp_greedy",
                         bindings.data(), bindings.size(), &args, sizeof(args),
                         4, greedy_scratch_bytes(n_vocab), {rows, 1, 1},
                         {1024, 1, 1}, error);
}

bool encode_sampler_sampled(Context& context, Encoder& encoder,
                            std::uint32_t rows, std::uint32_t n_vocab,
                            const SamplerBuffers& buffers,
                            bool penalties_already_applied, std::string& error,
                            std::uint32_t kpart) {
    if (encoder.concurrent_group_open()) {
        error = "sampler split part and merge require an ordered concurrent-group boundary";
        return false;
    }
    if (!sampler_shape_supported(rows, n_vocab) ||
        kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler shape or kpart exceeds supported bounds";
        return false;
    }
    if (!validate_sampler_buffers(
            context, {buffers.logits, buffers.history, buffers.rows,
                      buffers.candidates, buffers.selected}, error) ||
        !validate_split_buffers(rows, n_vocab, buffers, kpart, error)) {
        return false;
    }
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const SampArgs part_args{
        n_vocab, n_blocks, kpart, penalties_already_applied ? 0u : 1u};
    const std::array part_bindings{
        BufferBinding{buffers.logits, 0},
        BufferBinding{buffers.history, 1},
        BufferBinding{buffers.rows, 2},
        BufferBinding{buffers.candidates, 3},
    };
    const SampArgs merge_args{n_vocab, n_blocks, kpart, 0u};
    const std::array merge_bindings{
        BufferBinding{buffers.candidates, 0},
        BufferBinding{buffers.rows, 1},
        BufferBinding{buffers.selected, 2},
    };
    const std::size_t merge_scratch =
        static_cast<std::size_t>(n_blocks) * kSamplerCandidateStride *
        sizeof(SampCandidate);
    id<MTLComputePipelineState> part_pipeline = nil;
    id<MTLComputePipelineState> merge_pipeline = nil;
    if (!validate_kernel(context, "strata_samp_split_part",
                         part_bindings.data(), part_bindings.size(),
                         &part_args, sizeof(part_args), 0u, {128, 1, 1},
                         part_pipeline, error) ||
        !validate_kernel(context, "strata_samp_split_merge",
                         merge_bindings.data(), merge_bindings.size(),
                         &merge_args, sizeof(merge_args), merge_scratch,
                         {32, 1, 1}, merge_pipeline, error)) {
        return false;
    }

    return encode_kernel(context, encoder, "strata_samp_split_part",
                         part_bindings.data(), part_bindings.size(),
                         &part_args, sizeof(part_args), 4, 0,
                         {n_blocks, rows, 1}, {128, 1, 1}, error) &&
           encode_kernel(context, encoder, "strata_samp_split_merge",
                         merge_bindings.data(), merge_bindings.size(),
                         &merge_args, sizeof(merge_args), 3, merge_scratch,
                         {rows, 1, 1}, {32, 1, 1}, error);
}

bool encode_sampler_split_part(Context& context, Encoder& encoder,
                               std::uint32_t rows, std::uint32_t n_vocab,
                               const SamplerBuffers& buffers,
                               bool penalties_already_applied,
                               std::uint32_t kpart, std::string& error) {
    if (!sampler_shape_supported(rows, n_vocab) ||
        kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler shape or kpart exceeds supported bounds";
        return false;
    }
    if (!validate_sampler_buffers(
            context, {buffers.logits, buffers.history, buffers.rows,
                      buffers.candidates, buffers.selected}, error) ||
        !validate_split_buffers(rows, n_vocab, buffers, kpart, error)) {
        return false;
    }
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const SampArgs args{
        n_vocab, n_blocks, kpart, penalties_already_applied ? 0u : 1u};
    const std::array bindings{
        BufferBinding{buffers.logits, 0},
        BufferBinding{buffers.history, 1},
        BufferBinding{buffers.rows, 2},
        BufferBinding{buffers.candidates, 3},
    };
    return encode_kernel(context, encoder, "strata_samp_split_part",
                         bindings.data(), bindings.size(), &args, sizeof(args),
                         4, 0, {n_blocks, rows, 1}, {128, 1, 1}, error);
}

bool encode_sampler_split_merge(Context& context, Encoder& encoder,
                                std::uint32_t rows, std::uint32_t n_vocab,
                                BufferRef rows_buffer, BufferRef candidates,
                                BufferRef selected, std::uint32_t kpart,
                                std::string& error) {
    if (encoder.concurrent_group_open()) {
        error = "sampler split merge requires an ordered concurrent-group boundary";
        return false;
    }
    if (!validate_sampler_buffers(context,
                                  {rows_buffer, candidates, selected}, error)) {
        return false;
    }
    if (!sampler_shape_supported(rows, n_vocab) ||
        kpart > std::min(n_vocab, kSamplerCandidateStride)) {
        error = "sampler merge shape or kpart exceeds supported bounds";
        return false;
    }
    const std::uint32_t n_blocks =
        (n_vocab + kSamplerBlockSpan - 1u) / kSamplerBlockSpan;
    const std::size_t candidate_count =
        static_cast<std::size_t>(rows) * n_blocks * kSamplerCandidateStride;
    const std::size_t selected_count =
        static_cast<std::size_t>(rows) * kSamplerCandidateStride;
    if (!validate_buffer(rows_buffer,
                         static_cast<std::size_t>(rows) * sizeof(SampRow),
                         alignof(SampRow), "sampler rows", error) ||
        !validate_buffer(candidates,
                         candidate_count * sizeof(SampCandidate),
                         alignof(SampCandidate), "sampler candidates", error) ||
        !validate_buffer(selected,
                         selected_count * sizeof(SampCandidate),
                         alignof(SampCandidate), "sampler selected", error) ||
        !validate_kpart_rows(rows_buffer, rows, n_vocab, kpart, error)) {
        return false;
    }
    const SampArgs args{n_vocab, n_blocks, kpart, 0u};
    const std::array bindings{
        BufferBinding{candidates, 0},
        BufferBinding{rows_buffer, 1},
        BufferBinding{selected, 2},
    };
    const std::size_t merge_scratch =
        static_cast<std::size_t>(n_blocks) * kSamplerCandidateStride *
        sizeof(SampCandidate);
    return encode_kernel(context, encoder, "strata_samp_split_merge",
                         bindings.data(), bindings.size(), &args, sizeof(args),
                         3, merge_scratch, {rows, 1, 1}, {32, 1, 1}, error);
}

bool encode_sampler_coupled_penalty(Context& context, Encoder& encoder,
                                    std::uint32_t n_vocab,
                                    std::uint32_t sampled_vocab,
                                    bool has_id_to_subset_map, BufferRef logits,
                                    BufferRef id_to_subset,
                                    BufferRef history_ring, BufferRef row,
                                    std::string& error) {
    if (!sampler_shape_supported(1u, n_vocab) ||
        !sampler_shape_supported(1u, sampled_vocab) ||
        sampled_vocab > n_vocab ||
        (!has_id_to_subset_map && n_vocab != sampled_vocab)) {
        error = "coupled sampler vocabulary must be a valid subset and needs an id map";
        return false;
    }
    if (!validate_sampler_buffers(
            context, {logits, id_to_subset, history_ring, row}, error)) {
        return false;
    }
    const std::size_t inverse_map_bytes =
        has_id_to_subset_map
            ? static_cast<std::size_t>(n_vocab) * sizeof(std::int32_t)
            : sizeof(std::int32_t);
    if (!validate_buffer(logits,
                         static_cast<std::size_t>(sampled_vocab) * sizeof(float),
                         alignof(float), "coupled sampler logits", error) ||
        !validate_buffer(id_to_subset, inverse_map_bytes,
                         alignof(std::int32_t), "coupled sampler inverse map",
                         error) ||
        !validate_buffer(history_ring, sizeof(std::int32_t),
                         alignof(std::int32_t), "coupled sampler history",
                         error) ||
        !validate_buffer(row, sizeof(SampRow), alignof(SampRow),
                         "coupled sampler row", error)) {
        return false;
    }
    if (row.contents == nullptr) {
        error = "coupled sampler row must be host-visible";
        return false;
    }
    const SampRow& row_data = *static_cast<const SampRow*>(row.contents);
    const std::uint64_t history_entries =
        history_ring.length / sizeof(std::int32_t);
    if (row_data.hist_len > kSamplerMaxHistory ||
        static_cast<std::uint64_t>(row_data.hist_off) + row_data.hist_len >
            history_entries) {
        error = "coupled sampler history exceeds its buffer or the 4096-token limit";
        return false;
    }

    const SampPenalizeArgs args{
        sampled_vocab, n_vocab, has_id_to_subset_map ? 1u : 0u};
    const std::array bindings{
        BufferBinding{logits, 0},
        BufferBinding{id_to_subset, 1},
        BufferBinding{history_ring, 2},
        BufferBinding{row, 3},
    };
    const std::size_t scratch = rounded_threadgroup_bytes(
        ((static_cast<std::size_t>(sampled_vocab) + 31u) / 32u) *
        sizeof(std::uint32_t));
    return encode_kernel(context, encoder,
                         "strata_samp_coupled_penalize", bindings.data(),
                         bindings.size(), &args, sizeof(args), 4, scratch,
                         {1, 1, 1}, {1024, 1, 1}, error);
}

}  // namespace strata::metal
