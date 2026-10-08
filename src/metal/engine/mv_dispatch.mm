// src/metal/engine/mv_dispatch.mm - validated Q8_1 and dense mat-vec encoders.
#import <Metal/Metal.h>

#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "metal_fence.hpp"
#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kQ8_1Threads = 256u;
constexpr std::uint32_t kMatVecThreadsX = 32u;
constexpr std::uint32_t kMatVecThreadsY = 4u;
constexpr std::uint32_t kMaxMatVecWidth = 16'640u;
constexpr std::uint32_t kQ8_1BlockValues = 32u;
constexpr std::uint32_t kQ8_1BlockBytes = 36u;
constexpr std::uint64_t kShaderIndexCapacity = std::uint64_t{1u} << 32u;

struct Q8_1Params {
    std::uint32_t n;
    std::uint32_t rows;
};

struct MatVecParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};

static_assert(sizeof(Q81) == kQ8_1BlockBytes);
static_assert(sizeof(Q8_1Params) == 2u * sizeof(std::uint32_t));
static_assert(sizeof(MatVecParams) == 2u * sizeof(std::uint32_t));

struct FormatInfo {
    const char* name = nullptr;
    std::uint32_t block_values = 0u;
    std::uint32_t block_bytes = 0u;
    std::uint32_t small_k_threshold = 0u;
    bool row_family = false;
    bool bf16 = false;
};

bool format_info(MatVecWeightFormat format, FormatInfo& info) {
    switch (format) {
        case MatVecWeightFormat::q8_0:
            info = {"q8_0", 32u, sizeof(block_q8_0), 32u};
            return true;
        case MatVecWeightFormat::q4_0:
            info = {"q4_0", 32u, sizeof(block_q4_0), 64u};
            return true;
        case MatVecWeightFormat::q5_0:
            info = {"q5_0", 32u, sizeof(block_q5_0), 64u};
            return true;
        case MatVecWeightFormat::q2_0:
            info = {"q2_0", 64u, sizeof(block_q2_0), 64u};
            return true;
        case MatVecWeightFormat::iq4_nl:
            info = {"iq4_nl", 32u, sizeof(block_iq4_nl), 64u};
            return true;
        case MatVecWeightFormat::iq4_xs:
            info = {"iq4_xs", 256u, sizeof(block_iq4_xs), 16u};
            return true;
        case MatVecWeightFormat::iq3_s:
            info = {"iq3_s", 256u, sizeof(block_iq3_s), 0u, true};
            return true;
        case MatVecWeightFormat::q4_k:
            info = {"q4_k", 256u, sizeof(block_q4_K), 8u};
            return true;
        case MatVecWeightFormat::q5_k:
            info = {"q5_k", 256u, sizeof(block_q5_K), 8u};
            return true;
        case MatVecWeightFormat::q3_k:
            info = {"q3_k", 256u, sizeof(block_q3_K), 8u};
            return true;
        case MatVecWeightFormat::q6_k:
            info = {"q6_k", 256u, sizeof(block_q6_K), 4u};
            return true;
        case MatVecWeightFormat::bf16:
            info = {"bf16", 1u, sizeof(std::uint16_t), 0u, false, true};
            return true;
    }
    return false;
}

bool checked_bytes(std::uint64_t count, std::uint64_t element_bytes,
                   std::size_t& bytes) {
    if (element_bytes == 0u ||
        count > std::numeric_limits<std::size_t>::max() / element_bytes) {
        return false;
    }
    bytes = static_cast<std::size_t>(count * element_bytes);
    return true;
}

bool checked_mul(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& result) {
    if (right != 0u &&
        left > std::numeric_limits<std::uint64_t>::max() / right) {
        return false;
    }
    result = left * right;
    return true;
}

bool validate_buffer(Context& context, BufferRef buffer,
                     std::size_t required_bytes, std::size_t alignment,
                     std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "mat-vec " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (alignment == 0u || buffer.length < required_bytes ||
        buffer.offset % alignment != 0u) {
        error = "mat-vec " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }

    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || buffer.offset > native.length ||
        buffer.length > native.length - buffer.offset ||
        native.contents == nullptr) {
        error = "mat-vec " + std::string(name) +
                " view exceeds its CPU-visible Metal allocation";
        return false;
    }

    const std::uintptr_t native_contents =
        reinterpret_cast<std::uintptr_t>(native.contents);
    if (native_contents >
            std::numeric_limits<std::uintptr_t>::max() - buffer.offset ||
        buffer.contents !=
            reinterpret_cast<void*>(native_contents + buffer.offset)) {
        error = "mat-vec " + std::string(name) +
                " CPU backing metadata does not match its Metal view";
        return false;
    }

    const std::uint64_t native_gpu_address =
        static_cast<std::uint64_t>(native.gpuAddress);
    std::uint64_t expected_gpu_address = 0;
    if (!detail::gpu_address_with_offset(native_gpu_address, buffer.offset,
                                         expected_gpu_address) ||
        buffer.gpu_address != expected_gpu_address) {
        error = "mat-vec " + std::string(name) +
                " GPU backing metadata does not match its Metal view";
        return false;
    }
    if (reinterpret_cast<std::uintptr_t>(buffer.contents) % alignment != 0u ||
        buffer.gpu_address % alignment != 0u) {
        error = "mat-vec " + std::string(name) +
                " backing address is misaligned";
        return false;
    }
    return true;
}

bool validate_read_only_buffer(Context& context, ReadOnlyBufferRef buffer,
                               std::size_t required_bytes,
                               std::size_t alignment, std::string_view name,
                               std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "read-only BF16 mat-vec " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (alignment == 0u || buffer.length() < required_bytes ||
        buffer.offset() % alignment != 0u) {
        error = "read-only BF16 mat-vec " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    const auto* const contents = buffer.contents();
    const std::uintptr_t cpu_address =
        reinterpret_cast<std::uintptr_t>(contents);
    const std::uint64_t gpu_address = buffer.gpu_address();
    if (contents == nullptr || cpu_address % alignment != 0u ||
        (gpu_address != 0u && gpu_address % alignment != 0u) ||
        required_bytes >
            std::numeric_limits<std::uintptr_t>::max() - cpu_address ||
        (gpu_address != 0u &&
         required_bytes >
             std::numeric_limits<std::uint64_t>::max() - gpu_address)) {
        error = "read-only BF16 mat-vec " + std::string(name) +
                " buffer has invalid or misaligned backing metadata";
        return false;
    }
    return true;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_ranges_overlap(BufferRef left, std::size_t left_bytes,
                           BufferRef right, std::size_t right_bytes) {
    const bool same_buffer_overlap =
        left.handle == right.handle &&
        address_ranges_overlap(left.offset, left_bytes,
                               right.offset, right_bytes);
    const bool cpu_backing_overlap =
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.contents), left_bytes,
            reinterpret_cast<std::uintptr_t>(right.contents), right_bytes);
    const bool gpu_backing_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        address_ranges_overlap(left.gpu_address, left_bytes,
                               right.gpu_address, right_bytes);
    return same_buffer_overlap || cpu_backing_overlap || gpu_backing_overlap;
}

// ReadOnlyBufferRef intentionally hides its MTLBuffer handle. Its immutable
// CPU/GPU view addresses are enough to detect aliasing with writable outputs.
bool read_only_buffer_ranges_overlap(ReadOnlyBufferRef input,
                                     std::size_t input_bytes,
                                     BufferRef output,
                                     std::size_t output_bytes) {
    const auto* const input_contents = input.contents();
    const bool cpu_backing_overlap =
        address_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(input_contents), input_bytes,
            reinterpret_cast<std::uintptr_t>(output.contents), output_bytes);
    const std::uint64_t input_gpu_address = input.gpu_address();
    const bool gpu_backing_overlap =
        input_gpu_address != 0u && output.gpu_address != 0u &&
        address_ranges_overlap(input_gpu_address, input_bytes,
                               output.gpu_address, output_bytes);
    return cpu_backing_overlap || gpu_backing_overlap;
}

const KernelInfo* find_kernel_info(const char* kernel_name) {
    std::size_t count = 0u;
    const KernelInfo* const manifest = kernel_manifest(count);
    for (std::size_t index = 0u; index < count; ++index) {
        if (std::string_view(manifest[index].name) == kernel_name) {
            return &manifest[index];
        }
    }
    return nullptr;
}

bool validate_pipeline(Context& context, const char* kernel_name,
                       std::uint32_t threads_x, std::uint32_t threads_y,
                       std::string_view operation,
                       id<MTLComputePipelineState> __strong& pipeline,
                       std::string& error) {
    const std::uint32_t total_threads = threads_x * threads_y;
    const KernelInfo* const info = find_kernel_info(kernel_name);
    if (info == nullptr || info->math != KernelMath::safe ||
        info->threads != total_threads ||
        info->threadgroup_x != threads_x ||
        info->threadgroup_y != threads_y || info->threadgroup_z != 1u) {
        error = std::string(operation) +
                " threadgroup does not match its safe-kernel manifest";
        return false;
    }

    void* const pipeline_handle = context.pipeline(kernel_name);
    if (pipeline_handle == nullptr) {
        error = std::string(operation) + " pipeline is unavailable";
        return false;
    }
    pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
    if (pipeline.maxTotalThreadsPerThreadgroup < total_threads ||
        pipeline.staticThreadgroupMemoryLength >
            context.device_facts().max_threadgroup_memory_length) {
        error = std::string(operation) + " pipeline exceeds device limits";
        return false;
    }
    return true;
}

id<MTLCommandBuffer> validate_command_buffer(
    Context& context, void* command_handle, CommandBuffer& command_buffer,
    std::string_view operation, std::string& error) {
    if (!command_buffer.belongs_to(context)) {
        error = std::string(operation) +
                " command buffer belongs to a different Context";
        return nil;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command_handle;
    if (native == nil ||
        native.status != MTLCommandBufferStatusNotEnqueued) {
        error = std::string(operation) +
                " work must be encoded before command-buffer submission";
        return nil;
    }
    return native;
}

bool encode_q8_1_impl(Context& context, void* command_handle,
                      CommandBuffer* command_buffer, Encoder* segment,
                      std::uint32_t rows,
                      std::uint32_t width, const Q8_1Buffers& buffers,
                      std::string& error) {
    if (rows == 0u || width < kQ8_1BlockValues ||
        width > kMaxMatVecWidth || width % kQ8_1BlockValues != 0u) {
        error = "Q8_1 shape requires positive rows and width in 32..16640 divisible by 32";
        return false;
    }
    if (static_cast<std::uint64_t>(rows) * width > kShaderIndexCapacity) {
        error = "Q8_1 row-major index exceeds the 32-bit shader range";
        return false;
    }

    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    const std::uint64_t block_count = width / kQ8_1BlockValues;
    if (!checked_bytes(static_cast<std::uint64_t>(rows) * width,
                       sizeof(float), input_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) * block_count,
                       kQ8_1BlockBytes, output_bytes)) {
        error = "Q8_1 buffer size overflows the host byte range";
        return false;
    }

    id<MTLComputePipelineState> pipeline = nil;
    if (!validate_pipeline(context, "strata_q_q8_1", kQ8_1Threads,
                           1u, "Q8_1", pipeline, error)) {
        return false;
    }
    if (!validate_buffer(context, buffers.input_f32, input_bytes,
                         alignof(float), "Q8_1 input", error) ||
        !validate_buffer(context, buffers.output_q8_1, output_bytes,
                         alignof(std::uint32_t), "Q8_1 output", error)) {
        return false;
    }
    if (buffer_ranges_overlap(buffers.input_f32, input_bytes,
                              buffers.output_q8_1, output_bytes)) {
        error = "Q8_1 output must not overlap its input";
        return false;
    }

    const Q8_1Params params{width, rows};
    if (segment != nullptr) {
        const std::array bindings{
            EncoderBufferBinding{buffers.input_f32, 0u, input_bytes,
                                 alignof(float)},
            EncoderBufferBinding{buffers.output_q8_1, 1u, output_bytes,
                                 alignof(std::uint32_t)},
        };
        const std::array bytes{
            EncoderBytesBinding{&params, sizeof(params), 2u},
        };
        return segment->dispatch(
            "strata_q_q8_1",
            {(width + kQ8_1Threads - 1u) / kQ8_1Threads, rows, 1u},
            {kQ8_1Threads, 1u, 1u}, bindings, bytes, {}, error);
    }
    if (command_buffer == nullptr) {
        error = "Q8_1 command buffer is unavailable";
        return false;
    }
    id<MTLCommandBuffer> native = validate_command_buffer(
        context, command_handle, *command_buffer, "Q8_1", error);
    if (native == nil) return false;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native);
        if (encoder == nil) {
            error = "cannot create Q8_1 compute encoder";
            return false;
        }
        encoder.label = @"Q8_1 activation quantization";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.input_f32.handle
                    offset:buffers.input_f32.offset
                   atIndex:0u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.output_q8_1.handle
                    offset:buffers.output_q8_1.offset
                   atIndex:1u];
        [encoder setBytes:&params length:sizeof(params) atIndex:2u];
        [encoder dispatchThreadgroups:
                     MTLSizeMake((width + kQ8_1Threads - 1u) / kQ8_1Threads,
                                 rows, 1u)
                  threadsPerThreadgroup:
                     MTLSizeMake(kQ8_1Threads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_bf16_matvec_impl(
    Context& context, Encoder& encoder, std::uint32_t n_in,
    std::uint32_t n_out, std::uint32_t active_columns,
    std::uint32_t compiled_columns,
    const ReadOnlyBF16MatVecBuffers& buffers, std::string& error) {
    error.clear();
    if (n_in != 320u && n_in != 2'560u && n_in != 10'240u) {
        error = "read-only BF16 mat-vec n_in must be 320, 2560 or 10240";
        return false;
    }
    if (n_out == 0u) {
        error = "read-only BF16 mat-vec n_out must be positive";
        return false;
    }
    if (compiled_columns != 1u && compiled_columns != 2u &&
        compiled_columns != 4u && compiled_columns != 8u) {
        error = "read-only BF16 mat-vec compiled column width must be 1, 2, 4 or 8";
        return false;
    }
    if (active_columns == 0u || active_columns > compiled_columns) {
        error = "read-only BF16 mat-vec active columns must be in 1..compiled column width";
        return false;
    }

    std::uint64_t weight_count = 0u;
    std::uint64_t input_count = 0u;
    std::uint64_t output_count = 0u;
    std::uint64_t padded_count = 0u;
    if (!checked_mul(n_out, n_in, weight_count) ||
        !checked_mul(active_columns, n_in, input_count) ||
        !checked_mul(active_columns, n_out, output_count) ||
        !checked_mul(compiled_columns - active_columns, n_out,
                     padded_count) ||
        weight_count > kShaderIndexCapacity ||
        input_count > kShaderIndexCapacity ||
        static_cast<std::uint64_t>(compiled_columns) * n_out >
            kShaderIndexCapacity) {
        error = "read-only BF16 mat-vec flattened index exceeds the 32-bit shader range";
        return false;
    }

    std::size_t activation_bytes = 0u;
    std::size_t weight_bytes = 0u;
    std::size_t output_bytes = 0u;
    std::size_t sink_bytes = 0u;
    const std::uint64_t required_sink_count =
        padded_count == 0u ? 1u : padded_count;
    if (!checked_bytes(input_count, sizeof(float), activation_bytes) ||
        !checked_bytes(weight_count, sizeof(std::uint16_t), weight_bytes) ||
        !checked_bytes(output_count, sizeof(float), output_bytes) ||
        !checked_bytes(required_sink_count, sizeof(float), sink_bytes)) {
        error = "read-only BF16 mat-vec buffer size overflows the host byte range";
        return false;
    }

    const std::uint32_t block_size = n_in == 320u ? 160u : 256u;
    const std::size_t activation_alignment = 2u * alignof(float);
    constexpr std::size_t weight_alignment = alignof(std::uint32_t);
    if (!validate_read_only_buffer(context, buffers.activations,
                                   activation_bytes, activation_alignment,
                                   "activations", error) ||
        !validate_read_only_buffer(context, buffers.weights, weight_bytes,
                                   weight_alignment, "weights", error) ||
        !validate_read_only_buffer(context, buffers.active_rows,
                                   sizeof(std::uint32_t),
                                   alignof(std::uint32_t), "active-row count",
                                   error) ||
        !validate_buffer(context, buffers.output, output_bytes,
                         alignof(float), "read-only BF16 output", error) ||
        !validate_buffer(context, buffers.sink, sink_bytes, alignof(float),
                         "read-only BF16 sink", error)) {
        return false;
    }

    std::uint32_t live_rows = 0u;
    std::memcpy(&live_rows, buffers.active_rows.contents(), sizeof(live_rows));
    if (live_rows != active_columns) {
        error = "read-only BF16 mat-vec active-row count does not match active columns";
        return false;
    }

    if (read_only_buffer_ranges_overlap(buffers.activations, activation_bytes,
                                        buffers.output, output_bytes) ||
        read_only_buffer_ranges_overlap(buffers.weights, weight_bytes,
                                        buffers.output, output_bytes) ||
        read_only_buffer_ranges_overlap(buffers.active_rows,
                                        sizeof(std::uint32_t), buffers.output,
                                        output_bytes) ||
        read_only_buffer_ranges_overlap(buffers.activations, activation_bytes,
                                        buffers.sink, sink_bytes) ||
        read_only_buffer_ranges_overlap(buffers.weights, weight_bytes,
                                        buffers.sink, sink_bytes) ||
        read_only_buffer_ranges_overlap(buffers.active_rows,
                                        sizeof(std::uint32_t), buffers.sink,
                                        sink_bytes) ||
        buffer_ranges_overlap(buffers.output, output_bytes, buffers.sink,
                              sink_bytes)) {
        error = "read-only BF16 mat-vec outputs must not overlap an input or each other";
        return false;
    }

    const std::string kernel_name =
        "strata_mv_bf16_b" + std::to_string(block_size) + "_n" +
        std::to_string(compiled_columns);
    id<MTLComputePipelineState> pipeline = nil;
    if (!validate_pipeline(context, kernel_name.c_str(), block_size, 1u,
                           "read-only BF16 mat-vec", pipeline, error) ||
        pipeline == nil) {
        return false;
    }

    const MatVecParams params{n_in, n_out};
    const std::array writable_bindings{
        EncoderBufferBinding{buffers.output, 2u, output_bytes, alignof(float)},
        EncoderBufferBinding{buffers.sink, 3u, sink_bytes, alignof(float)},
    };
    const std::array read_only_bindings{
        EncoderReadOnlyBufferBinding{buffers.activations, 0u,
                                    activation_bytes, activation_alignment},
        EncoderReadOnlyBufferBinding{buffers.weights, 1u, weight_bytes,
                                    weight_alignment},
        EncoderReadOnlyBufferBinding{buffers.active_rows, 4u,
                                    sizeof(std::uint32_t),
                                    alignof(std::uint32_t)},
    };
    const std::array inline_bytes{
        EncoderBytesBinding{&params, sizeof(params), 5u},
    };
    return encoder.dispatch(kernel_name, {n_out, 1u, 1u},
                            {block_size, 1u, 1u}, writable_bindings,
                            read_only_bindings, inline_bytes, {}, error);
}

bool encode_matvec_impl(Context& context, void* command_handle,
                        CommandBuffer* command_buffer, Encoder* segment,
                        MatVecWeightFormat format, std::uint32_t n_in,
                        std::uint32_t n_out, std::uint32_t active_columns,
                        std::uint32_t compiled_columns,
                        const MatVecBuffers& buffers,
                        const BufferRef* row_map,
                        std::uint32_t weight_rows, std::string& error) {
    const bool indexed = row_map != nullptr;
    FormatInfo info;
    if (!format_info(format, info)) {
        error = "mat-vec weight format is unsupported";
        return false;
    }
    if (indexed &&
        format != MatVecWeightFormat::q5_k &&
        format != MatVecWeightFormat::q6_k &&
        format != MatVecWeightFormat::iq4_xs) {
        error = "indexed head format must be Q5_K, Q6_K or IQ4_XS";
        return false;
    }
    if (indexed && (active_columns != 1u || compiled_columns != 1u)) {
        error = "indexed head mat-vec requires one active and compiled column";
        return false;
    }
    if (n_in == 0u || n_in > kMaxMatVecWidth || n_out == 0u) {
        error = "mat-vec dimensions must have n_in in 1..16640 and positive n_out";
        return false;
    }
    if (indexed &&
        (weight_rows == 0u || n_out > weight_rows ||
         static_cast<std::uint64_t>(weight_rows) >
             static_cast<std::uint64_t>(
                 std::numeric_limits<std::int32_t>::max()) +
                 1u)) {
        error = "indexed head row counts must fit signed indices and output rows";
        return false;
    }
    if (compiled_columns != 1u && compiled_columns != 2u &&
        compiled_columns != 4u && compiled_columns != 8u) {
        error = "mat-vec compiled column width must be 1, 2, 4 or 8";
        return false;
    }
    if (active_columns == 0u || active_columns > compiled_columns) {
        error = "mat-vec active columns must be in 1..compiled column width";
        return false;
    }
    if (info.bf16) {
        if (n_in != 320u && n_in != 2'560u && n_in != 10'240u) {
            error = "BF16 mat-vec n_in must be 320, 2560 or 10240";
            return false;
        }
    } else if (n_in % info.block_values != 0u) {
        error = "mat-vec n_in is not divisible by the weight block width";
        return false;
    }

    const std::uint64_t blocks_per_row =
        info.bf16 ? n_in : n_in / info.block_values;
    const std::uint64_t row_bytes =
        info.bf16 ? static_cast<std::uint64_t>(n_in) * info.block_bytes
                  : blocks_per_row * info.block_bytes;
    const std::uint64_t matrix_rows = indexed ? weight_rows : n_out;
    std::uint64_t weight_count = 0u;
    std::uint64_t output_count = 0u;
    std::uint64_t input_count = 0u;
    std::uint64_t weight_byte_count = 0u;
    if (!checked_mul(matrix_rows, blocks_per_row, weight_count) ||
        !checked_mul(matrix_rows, row_bytes, weight_byte_count) ||
        !checked_mul(active_columns, n_out, output_count) ||
        !checked_mul(active_columns,
                     info.bf16 ? n_in : n_in / kQ8_1BlockValues,
                     input_count) ||
        weight_count > kShaderIndexCapacity ||
        static_cast<std::uint64_t>(compiled_columns) * n_out >
            kShaderIndexCapacity ||
        input_count > kShaderIndexCapacity) {
        error = "mat-vec flattened index exceeds the 32-bit shader range";
        return false;
    }

    std::size_t weight_bytes = 0u;
    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    std::size_t sink_bytes = 0u;
    std::size_t row_map_bytes = 0u;
    const std::uint64_t sink_count =
        static_cast<std::uint64_t>(compiled_columns - active_columns) * n_out;
    const std::uint64_t input_element_bytes =
        info.bf16 ? sizeof(float) : kQ8_1BlockBytes;
    const std::uint64_t required_sink_count =
        sink_count == 0u ? 1u : sink_count;
    if (!checked_bytes(weight_byte_count, 1u, weight_bytes) ||
        !checked_bytes(input_count, input_element_bytes, input_bytes) ||
        !checked_bytes(output_count, sizeof(float), output_bytes) ||
        !checked_bytes(required_sink_count, sizeof(float), sink_bytes) ||
        (indexed &&
         !checked_bytes(n_out, sizeof(std::int32_t), row_map_bytes))) {
        error = "mat-vec buffer size overflows the host byte range";
        return false;
    }

    std::uint32_t rows_per_group = 1u;
    std::uint32_t threads_x = kMatVecThreadsX;
    std::uint32_t threads_y = kMatVecThreadsY;
    std::uint32_t block_size = 0u;
    std::string kernel_name = "strata_mv_";
    kernel_name += info.name;
    if (info.bf16) {
        block_size = n_in == 320u ? 160u : 256u;
        threads_x = block_size;
        threads_y = 1u;
        kernel_name += "_b" + std::to_string(block_size);
    } else {
        if (info.row_family) {
            rows_per_group = 4u;
        } else if (blocks_per_row < info.small_k_threshold) {
            rows_per_group = 4u;
        }
    }
    kernel_name += "_n" + std::to_string(compiled_columns);
    if (indexed) kernel_name += "_x";

    id<MTLComputePipelineState> pipeline = nil;
    if (!validate_pipeline(context, kernel_name.c_str(), threads_x,
                           threads_y, "mat-vec", pipeline, error)) {
        return false;
    }

    const std::size_t activation_alignment =
        info.bf16 ? alignof(float) * 2u : 256u;
    const std::size_t weight_alignment =
        info.bf16 ? alignof(std::uint32_t) : alignof(std::uint16_t);
    if (!validate_buffer(context, buffers.activations, input_bytes,
                         activation_alignment, "activations", error) ||
        !validate_buffer(context, buffers.weights, weight_bytes,
                         weight_alignment, "weights", error) ||
        !validate_buffer(context, buffers.output, output_bytes,
                         alignof(float), "output", error) ||
        !validate_buffer(context, buffers.sink, sink_bytes,
                         alignof(float), "sink", error) ||
        !validate_buffer(context, buffers.active_rows,
                         sizeof(std::uint32_t), alignof(std::uint32_t),
                         "active-row count", error)) {
        return false;
    }
    if (indexed &&
        !validate_buffer(context, *row_map, row_map_bytes,
                         alignof(std::int32_t), "row map", error)) {
        return false;
    }
    if (*static_cast<const std::uint32_t*>(buffers.active_rows.contents) !=
        active_columns) {
        error = "mat-vec active-row buffer does not match active_columns";
        return false;
    }
    if (indexed) {
        const auto* const indices =
            static_cast<const std::int32_t*>(row_map->contents);
        for (std::uint32_t output_row = 0u; output_row < n_out; ++output_row) {
            if (indices[output_row] < 0 ||
                static_cast<std::uint32_t>(indices[output_row]) >= weight_rows) {
                error = "mat-vec row map contains an index outside the full head";
                return false;
            }
        }
    }
    if (buffer_ranges_overlap(buffers.output, output_bytes,
                              buffers.activations, input_bytes) ||
        buffer_ranges_overlap(buffers.output, output_bytes,
                              buffers.weights, weight_bytes) ||
        buffer_ranges_overlap(buffers.output, output_bytes,
                              buffers.active_rows, sizeof(std::uint32_t)) ||
        buffer_ranges_overlap(buffers.sink, sink_bytes,
                              buffers.activations, input_bytes) ||
        buffer_ranges_overlap(buffers.sink, sink_bytes,
                              buffers.weights, weight_bytes) ||
        buffer_ranges_overlap(buffers.sink, sink_bytes,
                              buffers.active_rows, sizeof(std::uint32_t)) ||
        (indexed &&
         (buffer_ranges_overlap(buffers.output, output_bytes,
                                *row_map, row_map_bytes) ||
          buffer_ranges_overlap(buffers.sink, sink_bytes,
                                *row_map, row_map_bytes))) ||
        buffer_ranges_overlap(buffers.output, output_bytes,
                              buffers.sink, sink_bytes)) {
        error = "mat-vec outputs must not overlap an input or each other";
        return false;
    }

    const MatVecParams params{n_in, n_out};
    const std::uint32_t grid_x = info.bf16
        ? n_out
        : static_cast<std::uint32_t>(
              (static_cast<std::uint64_t>(n_out) + rows_per_group - 1u) /
              rows_per_group);
    if (segment != nullptr) {
        std::array<EncoderBufferBinding, 6> bindings{};
        bindings[0] = {buffers.activations, 0u, input_bytes,
                       activation_alignment};
        bindings[1] = {buffers.weights, 1u, weight_bytes, weight_alignment};
        bindings[2] = {buffers.output, 2u, output_bytes, alignof(float)};
        bindings[3] = {buffers.sink, 3u, sink_bytes, alignof(float)};
        bindings[4] = {buffers.active_rows, 4u, sizeof(std::uint32_t),
                       alignof(std::uint32_t)};
        std::size_t binding_count = 5u;
        if (indexed) {
            bindings[5] = {*row_map, 6u, row_map_bytes, alignof(std::int32_t)};
            binding_count = 6u;
        }
        const std::array inline_bytes{
            EncoderBytesBinding{&params, sizeof(params), 5u},
        };
        return segment->dispatch(
            kernel_name, {grid_x, 1u, 1u},
            {threads_x, threads_y, 1u},
            std::span<const EncoderBufferBinding>(bindings.data(),
                                                  binding_count),
            inline_bytes, {}, error);
    }
    if (command_buffer == nullptr) {
        error = "mat-vec command buffer is unavailable";
        return false;
    }
    id<MTLCommandBuffer> native = validate_command_buffer(
        context, command_handle, *command_buffer, "mat-vec", error);
    if (native == nil) return false;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native);
        if (encoder == nil) {
            error = "cannot create mat-vec compute encoder";
            return false;
        }
        encoder.label = [NSString stringWithFormat:@"Mat-vec (%s, NC=%u)",
                                                   info.name,
                                                   compiled_columns];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.activations.handle
                    offset:buffers.activations.offset
                   atIndex:0u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.weights.handle
                    offset:buffers.weights.offset
                   atIndex:1u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.output.handle
                    offset:buffers.output.offset
                   atIndex:2u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.sink.handle
                    offset:buffers.sink.offset
                   atIndex:3u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.active_rows.handle
                    offset:buffers.active_rows.offset
                   atIndex:4u];
        [encoder setBytes:&params length:sizeof(params) atIndex:5u];
        if (indexed) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)row_map->handle
                        offset:row_map->offset
                       atIndex:6u];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(grid_x, 1u, 1u)
                  threadsPerThreadgroup:
                      MTLSizeMake(threads_x, threads_y, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_q8_1(Context& context, CommandBuffer& command_buffer,
                 std::uint32_t rows, std::uint32_t width,
                 const Q8_1Buffers& buffers, std::string& error) {
    return command_buffer.encode_unsubmitted(
        [&](void* command_handle, std::string& encode_error) {
            return encode_q8_1_impl(context, command_handle, &command_buffer,
                                    nullptr, rows, width, buffers,
                                    encode_error);
        },
        error);
}

bool encode_q8_1(Context& context, Encoder& encoder,
                 std::uint32_t rows, std::uint32_t width,
                 const Q8_1Buffers& buffers, std::string& error) {
    return encode_q8_1_impl(context, nullptr, nullptr, &encoder, rows, width,
                            buffers, error);
}

bool encode_matvec(Context& context, CommandBuffer& command_buffer,
                   MatVecWeightFormat format, std::uint32_t n_in,
                   std::uint32_t n_out, std::uint32_t active_columns,
                   std::uint32_t compiled_columns,
                   const MatVecBuffers& buffers, std::string& error) {
    return command_buffer.encode_unsubmitted(
        [&](void* command_handle, std::string& encode_error) {
            return encode_matvec_impl(
                context, command_handle, &command_buffer, nullptr, format, n_in, n_out,
                active_columns, compiled_columns, buffers, nullptr, n_out,
                encode_error);
        },
        error);
}

bool encode_matvec(Context& context, Encoder& encoder,
                   MatVecWeightFormat format, std::uint32_t n_in,
                   std::uint32_t n_out, std::uint32_t active_columns,
                   std::uint32_t compiled_columns,
                   const MatVecBuffers& buffers, std::string& error) {
    return encode_matvec_impl(
        context, nullptr, nullptr, &encoder, format, n_in, n_out,
        active_columns, compiled_columns, buffers, nullptr, n_out, error);
}

bool encode_bf16_matvec(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t n_in, std::uint32_t n_out,
                        std::uint32_t active_columns,
                        std::uint32_t compiled_columns,
                        const ReadOnlyBF16MatVecBuffers& buffers,
                        std::string& error) {
    return Encoder::encode(
        context, command_buffer, "read-only BF16 mat-vec",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_bf16_matvec(context, encoder, n_in, n_out,
                                      active_columns, compiled_columns,
                                      buffers, encode_error);
        },
        error);
}

bool encode_bf16_matvec(Context& context, Encoder& encoder,
                        std::uint32_t n_in, std::uint32_t n_out,
                        std::uint32_t active_columns,
                        std::uint32_t compiled_columns,
                        const ReadOnlyBF16MatVecBuffers& buffers,
                        std::string& error) {
    return encode_bf16_matvec_impl(context, encoder, n_in, n_out,
                                    active_columns, compiled_columns,
                                    buffers, error);
}

bool encode_indexed_head_matvec(
    Context& context, CommandBuffer& command_buffer,
    MatVecWeightFormat format, std::uint32_t n_in,
    std::uint32_t weight_rows, std::uint32_t output_rows,
    const IndexedHeadMatVecBuffers& buffers, std::string& error) {
    return command_buffer.encode_unsubmitted(
        [&](void* command_handle, std::string& encode_error) {
            return encode_matvec_impl(
                context, command_handle, &command_buffer, nullptr, format,
                n_in, output_rows, 1u, 1u, buffers.matvec,
                &buffers.row_map, weight_rows, encode_error);
        },
        error);
}

}  // namespace strata::metal
