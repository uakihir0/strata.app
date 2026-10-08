// src/metal/engine/embed_dispatch.mm - validated embedding Metal encoders.
#import <Metal/Metal.h>

#include "strata/metal/encoder.hpp"
#include "strata/metal/embed_dispatch.hpp"
#include "embedding_encode_internal.hpp"
#include "metal_fence.hpp"
#include "../kernels/common/params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kEmbedImageThreads = 256u;
constexpr std::uint32_t kEmbedImageGroupsX = N_EMBD / kEmbedImageThreads;
constexpr std::uint32_t kEmbedRowsThreads = 256u;
constexpr std::uint32_t kEmbedRowsGroupsX = N_EMBD / kEmbedRowsThreads;
constexpr std::uint32_t kImageTokenFlag = 0x80000000u;
constexpr std::uint64_t kEmbedImageIndexCapacity =
    std::uint64_t{1u} << 32u;
constexpr std::uint32_t kEmbedImageMaxRows =
    static_cast<std::uint32_t>(kEmbedImageIndexCapacity / N_EMBD);
constexpr std::uint32_t kEmbedRowsMaxRows = kEmbedImageMaxRows;

struct EmbedRowsInfo {
    const char* kernel_name;
    const char* format_name;
    std::size_t row_bytes;
};

bool embed_rows_info(EmbedRowsFormat format, EmbedRowsInfo& info) {
    switch (format) {
        case EmbedRowsFormat::q3_k:
            info = {"strata_embed_rows_q3_k", "q3_k", 1100u};
            return true;
        case EmbedRowsFormat::iq3_s:
            info = {"strata_embed_rows_iq3_s", "iq3_s", 1100u};
            return true;
        case EmbedRowsFormat::iq4_xs:
            info = {"strata_embed_rows_iq4_xs", "iq4_xs", 1360u};
            return true;
        case EmbedRowsFormat::bf16:
            info = {"strata_embed_rows_bf16", "bf16", 5120u};
            return true;
    }
    return false;
}

bool checked_bytes(std::uint32_t rows, std::uint32_t elements_per_row,
                   std::size_t element_size, std::size_t& bytes) {
    if (element_size == 0u ||
        static_cast<std::size_t>(elements_per_row) >
            std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(elements_per_row) * element_size;
    if (row_bytes != 0u &&
        static_cast<std::size_t>(rows) >
            std::numeric_limits<std::size_t>::max() / row_bytes) {
        return false;
    }
    bytes = static_cast<std::size_t>(rows) * row_bytes;
    return true;
}

bool validate_buffer_for(Context& context, BufferRef buffer,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view family, std::string_view name,
                         std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(family) + " " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (alignment == 0u || buffer.length < required_bytes ||
        buffer.offset % alignment != 0u) {
        error = std::string(family) + " " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || buffer.offset > native.length ||
        buffer.length > native.length - buffer.offset) {
        error = std::string(family) + " " + std::string(name) +
                " view exceeds its Metal allocation";
        return false;
    }
    if (native.contents == nullptr) {
        error = std::string(family) + " " + std::string(name) +
                " buffer has no CPU-visible backing metadata";
        return false;
    }
    const std::uintptr_t native_contents =
        reinterpret_cast<std::uintptr_t>(native.contents);
    if (native_contents >
            std::numeric_limits<std::uintptr_t>::max() - buffer.offset ||
        buffer.contents !=
            reinterpret_cast<void*>(native_contents + buffer.offset)) {
        error = std::string(family) + " " + std::string(name) +
                " CPU backing metadata does not match its Metal view";
        return false;
    }
    return true;
}

bool validate_read_only_buffer_for(Context& context, ReadOnlyBufferRef buffer,
                                   std::size_t required_bytes,
                                   std::size_t alignment,
                                   std::string_view family,
                                   std::string_view name,
                                   std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(family) + " " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (alignment == 0u || buffer.length() < required_bytes ||
        buffer.offset() % alignment != 0u) {
        error = std::string(family) + " " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    if (buffer.contents() == nullptr) {
        error = std::string(family) + " " + std::string(name) +
                " buffer has no CPU-visible backing metadata";
        return false;
    }
    error.clear();
    return true;
}

bool validate_buffer(Context& context, BufferRef buffer,
                     std::size_t required_bytes, std::size_t alignment,
                     std::string_view name, std::string& error) {
    return validate_buffer_for(context, buffer, required_bytes, alignment,
                               "embedding image", name, error);
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                           Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_views_overlap(BufferRef left, BufferRef right) {
    const bool same_buffer_overlap =
        left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.length,
                               right.offset, right.length);
    const bool cpu_backing_overlap =
        left.contents != nullptr && right.contents != nullptr &&
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.contents), left.length,
            reinterpret_cast<std::uintptr_t>(right.contents), right.length);
    const bool gpu_backing_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        address_ranges_overlap(left.gpu_address, left.length,
                               right.gpu_address, right.length);
    return same_buffer_overlap || cpu_backing_overlap || gpu_backing_overlap;
}

bool read_only_and_output_overlap(ReadOnlyBufferRef input, BufferRef output) {
    const auto* const input_contents = input.contents();
    const bool cpu_backing_overlap =
        input_contents != nullptr && output.contents != nullptr &&
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(input_contents), input.length(),
            reinterpret_cast<std::uintptr_t>(output.contents), output.length);
    const bool gpu_backing_overlap =
        input.gpu_address() != 0u && output.gpu_address != 0u &&
        address_ranges_overlap(input.gpu_address(), input.length(),
                               output.gpu_address, output.length);
    return cpu_backing_overlap || gpu_backing_overlap;
}

bool validate_pipeline_for(Context& context, const char* kernel_name,
                           std::string_view family, std::uint32_t threads,
                           std::uint32_t threadgroup_x,
                           std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t kernel_count = 0u;
    const KernelInfo* const manifest = kernel_manifest(kernel_count);
    for (std::size_t index = 0u; index < kernel_count; ++index) {
        if (std::string_view(manifest[index].name) == kernel_name) {
            info = &manifest[index];
            break;
        }
    }
    if (info == nullptr || info->threads != threads ||
        info->threadgroup_x != threadgroup_x ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = std::string(family) +
                " threadgroup does not match its manifest";
        return false;
    }

    void* const pipeline_handle = context.pipeline(kernel_name);
    if (pipeline_handle == nullptr) {
        error = std::string(family) + " pipeline is unavailable";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_handle;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        error = std::string(family) + " threadgroup exceeds the device limit";
        return false;
    }
    return true;
}

bool validate_pipeline(Context& context, std::string& error) {
    return validate_pipeline_for(context, "strata_embed_image",
                                 "embedding image", kEmbedImageThreads,
                                 kEmbedImageThreads, error);
}

id<MTLCommandBuffer> validate_command_buffer(
    Context& context, CommandBuffer& command_buffer, std::string& error) {
    void* const command_handle = command_buffer.unsafe_native_handle();
    if (command_handle == nullptr) {
        error = "embedding image command buffer is unavailable";
        return nil;
    }
    if (!command_buffer.belongs_to(context)) {
        error = "embedding image command buffer belongs to a different Context";
        return nil;
    }
    id<MTLCommandBuffer> native_command =
        (__bridge id<MTLCommandBuffer>)command_handle;
    if (native_command == nil ||
        native_command.status != MTLCommandBufferStatusNotEnqueued) {
        error = "embedding image work must be encoded before command-buffer submission";
        return nil;
    }
    return native_command;
}

}  // namespace

namespace detail {

bool prepare_embed_rows(Context& context, std::uint32_t rows,
                        std::uint32_t vocabulary_size, EmbedRowsFormat format,
                        const EmbedRowsBuffers& buffers,
                        EmbedRowsDispatch& dispatch, std::string& error) {
    if (rows == 0u) {
        error = "embedding rows row count must be positive";
        return false;
    }
    if (rows > kEmbedRowsMaxRows) {
        error = "embedding rows count is outside the 32-bit shader index range";
        return false;
    }
    if (vocabulary_size == 0u || vocabulary_size > kImageTokenFlag) {
        error = "embedding rows vocabulary size must be in 1..2^31";
        return false;
    }

    EmbedRowsInfo info{};
    if (!embed_rows_info(format, info)) {
        error = "embedding rows format is unsupported";
        return false;
    }

    std::size_t table_bytes = 0u;
    std::size_t token_bytes = 0u;
    std::size_t embedding_bytes = 0u;
    if (!checked_bytes(vocabulary_size,
                       static_cast<std::uint32_t>(info.row_bytes), 1u,
                       table_bytes) ||
        !checked_bytes(rows, 1u, sizeof(std::uint32_t), token_bytes) ||
        !checked_bytes(rows, N_EMBD, sizeof(float), embedding_bytes)) {
        error = "embedding rows dimensions overflow the host byte range";
        return false;
    }

    const std::string_view family = "embedding rows";
    if (!validate_pipeline_for(context, info.kernel_name, family,
                               kEmbedRowsThreads, kEmbedRowsThreads, error)) {
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            info.kernel_name);
    if (pipeline == nil) {
        error = "embedding rows pipeline is unavailable";
        return false;
    }

    if (!validate_buffer_for(context, buffers.table, table_bytes,
                             alignof(std::uint16_t), family, "table", error) ||
        !validate_buffer_for(context, buffers.tokens, token_bytes,
                             alignof(std::uint32_t), family, "tokens", error) ||
        !validate_buffer_for(context, buffers.embedding, embedding_bytes,
                             alignof(float), family, "embedding", error)) {
        return false;
    }
    if (buffer_views_overlap(buffers.embedding, buffers.table) ||
        buffer_views_overlap(buffers.embedding, buffers.tokens)) {
        error = "embedding rows output must not overlap table or tokens";
        return false;
    }

    const auto* const token_data =
        static_cast<const std::uint32_t*>(buffers.tokens.contents);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::uint32_t token = token_data[row];
        if ((token & kImageTokenFlag) != 0u) continue;
        if (token >= vocabulary_size) {
            error = "embedding rows token id at row " + std::to_string(row) +
                    " is outside the vocabulary";
            return false;
        }
    }

    dispatch = {rows, buffers, info.kernel_name, info.format_name};
    error.clear();
    return true;
}

void encode_embed_rows(Context& context,
                       id<MTLComputeCommandEncoder> encoder,
                       const EmbedRowsDispatch& dispatch) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            dispatch.kernel_name);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.table.handle
                offset:dispatch.buffers.table.offset
               atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.tokens.handle
                offset:dispatch.buffers.tokens.offset
               atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.embedding.handle
                offset:dispatch.buffers.embedding.offset
               atIndex:2u];
    [encoder dispatchThreadgroups:
                 MTLSizeMake(kEmbedRowsGroupsX, dispatch.rows, 1u)
              threadsPerThreadgroup:
                  MTLSizeMake(kEmbedRowsThreads, 1u, 1u)];
}

bool prepare_embed_image(Context& context, std::uint32_t rows,
                         const EmbedImageBuffers& buffers,
                         EmbedImageDispatch& dispatch, std::string& error) {
    if (rows == 0u) {
        error = "embedding image row count must be positive";
        return false;
    }
    if (rows > kEmbedImageMaxRows) {
        error =
            "embedding image row count exceeds the 32-bit shader index range";
        return false;
    }
    if (!validate_pipeline(context, error)) return false;

    std::size_t token_bytes = 0u;
    std::size_t image_bytes = 0u;
    if (!checked_bytes(rows, 1u, sizeof(std::uint32_t), token_bytes) ||
        !checked_bytes(rows, N_EMBD, sizeof(float), image_bytes)) {
        error = "embedding image row count overflows the buffer byte range";
        return false;
    }
    const std::array bindings{
        buffers.tokens, buffers.image, buffers.embedding,
    };
    const std::array required_bytes{token_bytes, image_bytes, image_bytes};
    constexpr std::array<std::string_view, 3> names{
        "tokens", "image", "embedding",
    };
    constexpr std::array<std::size_t, 3> alignments{
        alignof(std::uint32_t), alignof(float), alignof(float),
    };
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index], required_bytes[index],
                             alignments[index], names[index], error)) {
            return false;
        }
    }
    if (buffer_views_overlap(buffers.embedding, buffers.tokens) ||
        buffer_views_overlap(buffers.embedding, buffers.image)) {
        error = "embedding image output must not overlap either input";
        return false;
    }

    dispatch = {rows, buffers};
    error.clear();
    return true;
}

void encode_embed_image(Context& context,
                        id<MTLComputeCommandEncoder> encoder,
                        const EmbedImageDispatch& dispatch) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_embed_image");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.tokens.handle
                offset:dispatch.buffers.tokens.offset
               atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.image.handle
                offset:dispatch.buffers.image.offset
               atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.embedding.handle
                offset:dispatch.buffers.embedding.offset
               atIndex:2u];
    [encoder dispatchThreadgroups:
                 MTLSizeMake(kEmbedImageGroupsX, dispatch.rows, 1u)
              threadsPerThreadgroup:
                  MTLSizeMake(kEmbedImageThreads, 1u, 1u)];
}

}  // namespace detail

bool encode_embed_rows(Context& context, CommandBuffer& command_buffer,
                       std::uint32_t rows, std::uint32_t vocabulary_size,
                       EmbedRowsFormat format,
                       const EmbedRowsBuffers& buffers,
                       std::string& error) {
    return command_buffer.encode_unsubmitted(
        [&](void*, std::string&) {
            detail::EmbedRowsDispatch dispatch;
            if (!detail::prepare_embed_rows(context, rows, vocabulary_size,
                                            format, buffers, dispatch, error)) {
                return false;
            }
            id<MTLCommandBuffer> native_command =
                validate_command_buffer(context, command_buffer, error);
            if (native_command == nil) return false;

            @autoreleasepool {
                id<MTLComputeCommandEncoder> native_encoder =
                    begin_compute_encoder(context, native_command);
                if (native_encoder == nil) {
                    error = "cannot create embedding rows compute encoder";
                    return false;
                }
                native_encoder.label =
                    [NSString stringWithFormat:@"Embedding gather (%s)",
                                               dispatch.format_name];
                detail::encode_embed_rows(context, native_encoder, dispatch);
                end_compute_encoder(context, native_encoder);
            }
            error.clear();
            return true;
        },
        error);
}

bool encode_embed_rows(Context& context, Encoder& encoder,
                       std::uint32_t rows, std::uint32_t vocabulary_size,
                       EmbedRowsFormat format,
                       const EmbedRowsBuffers& buffers,
                       std::string& error) {
    error.clear();
    detail::EmbedRowsDispatch dispatch;
    if (!detail::prepare_embed_rows(context, rows, vocabulary_size, format,
                                    buffers, dispatch, error)) {
        return false;
    }
    const std::array buffer_bindings{
        EncoderBufferBinding{buffers.table, 0u, buffers.table.length,
                             alignof(std::uint16_t)},
        EncoderBufferBinding{buffers.tokens, 1u, buffers.tokens.length,
                             alignof(std::uint32_t)},
        EncoderBufferBinding{buffers.embedding, 2u, buffers.embedding.length,
                             alignof(float)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        dispatch.kernel_name, {N_EMBD / kEmbedRowsThreads, rows, 1u},
        {kEmbedRowsThreads, 1u, 1u}, buffer_bindings, bytes,
        threadgroup_memory, error);
}

bool encode_embed_rows(Context& context, Encoder& encoder,
                       std::uint32_t rows, std::uint32_t vocabulary_size,
                       EmbedRowsFormat format,
                       const EmbedRowsReadOnlyBuffers& buffers,
                       std::string& error) {
    error.clear();
    if (rows == 0u || rows > kEmbedRowsMaxRows) {
        error = rows == 0u
                    ? "embedding rows row count must be positive"
                    : "embedding rows count is outside the 32-bit shader index range";
        return false;
    }
    if (vocabulary_size == 0u || vocabulary_size > kImageTokenFlag) {
        error = "embedding rows vocabulary size must be in 1..2^31";
        return false;
    }

    EmbedRowsInfo info{};
    if (!embed_rows_info(format, info)) {
        error = "embedding rows format is unsupported";
        return false;
    }

    std::size_t table_bytes = 0u;
    std::size_t token_bytes = 0u;
    std::size_t embedding_bytes = 0u;
    if (!checked_bytes(vocabulary_size,
                       static_cast<std::uint32_t>(info.row_bytes), 1u,
                       table_bytes) ||
        !checked_bytes(rows, 1u, sizeof(std::uint32_t), token_bytes) ||
        !checked_bytes(rows, N_EMBD, sizeof(float), embedding_bytes)) {
        error = "embedding rows dimensions overflow the host byte range";
        return false;
    }
    if (!validate_pipeline_for(context, info.kernel_name, "embedding rows",
                               kEmbedRowsThreads, kEmbedRowsThreads, error)) {
        return false;
    }
    if (!validate_read_only_buffer_for(context, buffers.table, table_bytes,
                                       alignof(std::uint16_t), "embedding rows",
                                       "table", error) ||
        !validate_read_only_buffer_for(context, buffers.tokens, token_bytes,
                                       alignof(std::uint32_t), "embedding rows",
                                       "tokens", error) ||
        !validate_buffer_for(context, buffers.embedding, embedding_bytes,
                             alignof(float), "embedding rows", "embedding",
                             error)) {
        return false;
    }
    if (read_only_and_output_overlap(buffers.table, buffers.embedding) ||
        read_only_and_output_overlap(buffers.tokens, buffers.embedding)) {
        error = "embedding rows output must not overlap table or tokens";
        return false;
    }

    const auto* const token_data =
        static_cast<const std::uint32_t*>(buffers.tokens.contents());
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::uint32_t token = token_data[row];
        if ((token & kImageTokenFlag) == 0u && token >= vocabulary_size) {
            error = "embedding rows token id at row " + std::to_string(row) +
                    " is outside the vocabulary";
            return false;
        }
    }

    const std::array writable_bindings{
        EncoderBufferBinding{buffers.embedding, 2u, embedding_bytes,
                             alignof(float)},
    };
    const std::array read_only_bindings{
        EncoderReadOnlyBufferBinding{buffers.table, 0u, table_bytes,
                                    alignof(std::uint16_t)},
        EncoderReadOnlyBufferBinding{buffers.tokens, 1u, token_bytes,
                                    alignof(std::uint32_t)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        info.kernel_name, {N_EMBD / kEmbedRowsThreads, rows, 1u},
        {kEmbedRowsThreads, 1u, 1u}, writable_bindings, read_only_bindings,
        bytes, threadgroup_memory, error);
}

bool encode_embed_image(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, const EmbedImageBuffers& buffers,
                        std::string& error) {
    return command_buffer.encode_unsubmitted(
        [&](void*, std::string&) {
            detail::EmbedImageDispatch dispatch;
            if (!detail::prepare_embed_image(context, rows, buffers, dispatch,
                                             error)) {
                return false;
            }
            id<MTLCommandBuffer> native_command =
                validate_command_buffer(context, command_buffer, error);
            if (native_command == nil) return false;

            @autoreleasepool {
                id<MTLComputeCommandEncoder> native_encoder =
                    begin_compute_encoder(context, native_command);
                if (native_encoder == nil) {
                    error = "cannot create embedding image compute encoder";
                    return false;
                }
                native_encoder.label = @"Image embedding copy";
                detail::encode_embed_image(context, native_encoder, dispatch);
                end_compute_encoder(context, native_encoder);
            }
            error.clear();
            return true;
        },
        error);
}

bool encode_embed_image(Context& context, Encoder& encoder,
                        std::uint32_t rows, const EmbedImageBuffers& buffers,
                        std::string& error) {
    error.clear();
    detail::EmbedImageDispatch dispatch;
    if (!detail::prepare_embed_image(context, rows, buffers, dispatch, error)) {
        return false;
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::array buffer_bindings{
        EncoderBufferBinding{buffers.tokens, 0u,
                             static_cast<std::size_t>(rows) *
                                 sizeof(std::uint32_t),
                             alignof(std::uint32_t)},
        EncoderBufferBinding{buffers.image, 1u, row_bytes, alignof(float)},
        EncoderBufferBinding{buffers.embedding, 2u, row_bytes, alignof(float)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_embed_image", {kEmbedImageGroupsX, rows, 1u},
        {kEmbedImageThreads, 1u, 1u}, buffer_bindings, bytes,
        threadgroup_memory, error);
}

}  // namespace strata::metal
