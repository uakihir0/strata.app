// src/metal/engine/gdn_dispatch.mm - validated GDN Metal encoders.
#import <Metal/Metal.h>

#include "strata/metal/gdn_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "metal_fence.hpp"
#include "../kernels/common/params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kConvThreads = 128u;
constexpr std::uint32_t kABThreads = 256u;
constexpr std::uint32_t kStepThreads = 512u;
constexpr std::uint32_t kStateElements =
    GDN_HEAD_DIM * GDN_V_HEADS * GDN_HEAD_DIM;
constexpr std::uint32_t kHistoryElements =
    GDN_CONV_CHANNELS * (GDN_CONV_KERNEL - 1u);
constexpr std::uint32_t kABWeightElements = N_EMBD * GDN_V_HEADS;
struct RowsParams {
    std::uint32_t rows;
};
static_assert(sizeof(RowsParams) == sizeof(std::uint32_t));

struct StepParams {
    std::uint32_t rows;
    std::uint32_t mode;
};
static_assert(sizeof(StepParams) == 2u * sizeof(std::uint32_t));

struct Binding {
    BufferRef buffer;
    NSUInteger index;
};

struct ThreadgroupMemory {
    NSUInteger index;
    NSUInteger bytes;
};

bool checked_float_bytes(std::uint32_t rows, std::uint32_t columns,
                         std::size_t& bytes) {
    const std::size_t max_elements =
        std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (columns != 0u &&
        static_cast<std::size_t>(rows) > max_elements / columns) {
        return false;
    }
    bytes = static_cast<std::size_t>(rows) * columns * sizeof(float);
    return true;
}

bool buffer_overlaps(BufferRef left, std::size_t left_bytes,
                     BufferRef right, std::size_t right_bytes) {
    const auto overlaps = [](std::uintptr_t a, std::size_t a_size,
                             std::uintptr_t b, std::size_t b_size) {
        if (a == 0u || b == 0u || a_size == 0u || b_size == 0u) return false;
        return a <= b ? b - a < a_size : a - b < b_size;
    };
    if (left.handle == right.handle &&
        overlaps(left.offset, left_bytes, right.offset, right_bytes)) {
        return true;
    }
    if (overlaps(reinterpret_cast<std::uintptr_t>(left.contents), left_bytes,
                 reinterpret_cast<std::uintptr_t>(right.contents), right_bytes)) {
        return true;
    }
    return overlaps(static_cast<std::uintptr_t>(left.gpu_address), left_bytes,
                    static_cast<std::uintptr_t>(right.gpu_address), right_bytes);
}

bool validate_buffer(Context& context, BufferRef buffer,
                     std::size_t required_bytes, std::size_t alignment,
                     std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "GDN " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes || buffer.offset % alignment != 0u) {
        error = "GDN " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || buffer.offset > native.length ||
        buffer.length > native.length - buffer.offset) {
        error = "GDN " + std::string(name) +
                " view exceeds its Metal allocation";
        return false;
    }

    const std::uintptr_t native_contents =
        reinterpret_cast<std::uintptr_t>(native.contents);
    if (native.contents == nullptr ||
        native_contents >
            std::numeric_limits<std::uintptr_t>::max() - buffer.offset ||
        buffer.contents !=
            reinterpret_cast<void*>(native_contents + buffer.offset)) {
        error = "GDN " + std::string(name) +
                " CPU backing metadata does not match its Metal view";
        return false;
    }

    const std::uint64_t native_gpu_address = native.gpuAddress;
    if (native_gpu_address == 0u) {
        if (buffer.gpu_address != 0u) {
            error = "GDN " + std::string(name) +
                    " GPU address metadata is forged";
            return false;
        }
    } else if (native_gpu_address >
                   std::numeric_limits<std::uint64_t>::max() - buffer.offset ||
               buffer.gpu_address != native_gpu_address + buffer.offset) {
        error = "GDN " + std::string(name) +
                " GPU address metadata does not match its Metal view";
        return false;
    }
    return true;
}

bool validate_pipeline(Context& context, const char* name,
                       std::uint32_t threads, std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != threads ||
        info->threadgroup_x != threads || info->threadgroup_y != 1u ||
        info->threadgroup_z != 1u) {
        error = std::string("GDN ") + name +
                " threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(name);
    if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < threads) {
        error = std::string("GDN ") + name +
                " pipeline is unavailable or below its threadgroup limit";
        return false;
    }
    return true;
}

id<MTLCommandBuffer> validate_command_buffer(
    Context& context, CommandBuffer& command_buffer, std::string& error) {
    void* handle = command_buffer.unsafe_native_handle();
    if (handle == nullptr) {
        error = "GDN command buffer is unavailable";
        return nil;
    }
    if (!command_buffer.belongs_to(context)) {
        error = "GDN command buffer belongs to a different Context";
        return nil;
    }
    id<MTLCommandBuffer> native = (__bridge id<MTLCommandBuffer>)handle;
    if (native == nil ||
        native.status != MTLCommandBufferStatusNotEnqueued) {
        error = "GDN work must be encoded before command-buffer submission";
        return nil;
    }
    return native;
}

bool encode_kernel(Context& context, CommandBuffer& command_buffer,
                   const char* kernel_name, NSString* label,
                   std::uint32_t threads, MTLSize grid,
                   const std::vector<Binding>& bindings,
                   const void* params, std::size_t params_bytes,
                   NSUInteger params_index,
                   const std::vector<ThreadgroupMemory>& memory,
                   std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_pipeline(context, kernel_name, threads, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;

    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = std::string("cannot create GDN ") + kernel_name + " encoder";
        return false;
    }
    encoder.label = label;
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(kernel_name)];
    for (const Binding& binding : bindings) {
        [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                    offset:binding.buffer.offset
                   atIndex:binding.index];
    }
    if (params != nullptr && params_bytes != 0u) {
        [encoder setBytes:params length:params_bytes atIndex:params_index];
    }
    for (const ThreadgroupMemory& item : memory) {
        [encoder setThreadgroupMemoryLength:item.bytes atIndex:item.index];
    }
    [encoder dispatchThreadgroups:grid
              threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool validate_rows(std::uint32_t rows, bool allow_zero, std::string& error) {
    if ((allow_zero && rows > VERIFY_MAX_T) ||
        (!allow_zero && (rows == 0u || rows > VERIFY_MAX_T))) {
        error = allow_zero ? "GDN row count must be in 0..8"
                           : "GDN row count must be in 1..8";
        return false;
    }
    return true;
}

bool validate_bindings(Context& context, const std::vector<Binding>& bindings,
                       const std::vector<std::size_t>& sizes,
                       const std::vector<std::size_t>& alignments,
                       std::string& error) {
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        if (!validate_buffer(context, bindings[i].buffer, sizes[i],
                             alignments[i], "bound", error)) {
            return false;
        }
    }
    return true;
}

bool reject_output_overlap(const BufferRef output, std::size_t output_bytes,
                           const std::vector<std::pair<BufferRef, std::size_t>>& inputs,
                           std::string_view name, std::string& error) {
    for (const auto& [input, input_bytes] : inputs) {
        if (buffer_overlaps(output, output_bytes, input, input_bytes)) {
            error = "GDN " + std::string(name) +
                    " output must not overlap an input";
            return false;
        }
    }
    return true;
}

}  // namespace

bool encode_gdn_conv_l2(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, const GDNConvL2Buffers& buffers,
                        std::string& error) {
    if (!validate_rows(rows, false, error)) return false;
    std::size_t qkv_bytes = 0u;
    if (!checked_float_bytes(rows, GDN_CONV_CHANNELS, qkv_bytes)) {
        error = "GDN convolution buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kHistoryElements) * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(GDN_CONV_CHANNELS) *
        GDN_CONV_KERNEL * sizeof(float);
    const std::size_t output_bytes = qkv_bytes;
    const std::vector<Binding> bindings{
        {buffers.history, 0u}, {buffers.qkv, 1u},
        {buffers.conv_weights, 2u}, {buffers.output, 3u}};
    const std::vector<std::size_t> sizes{
        history_bytes, qkv_bytes, weight_bytes, output_bytes};
    const std::vector<std::size_t> alignments(4u, alignof(float));
    if (!validate_bindings(context, bindings, sizes, alignments, error) ||
        !reject_output_overlap(
            buffers.output, output_bytes,
            {{buffers.history, history_bytes}, {buffers.qkv, qkv_bytes},
             {buffers.conv_weights, weight_bytes}},
            "convolution", error) ||
        !validate_pipeline(context, "strata_gdn_conv_l2", kConvThreads,
                           error)) {
        return false;
    }
    return encode_kernel(context, command_buffer, "strata_gdn_conv_l2",
                         @"GDN convolution and q/k L2", kConvThreads,
                         MTLSizeMake(80u, rows, 1u), bindings,
                         nullptr, 0u, 4u, {}, error);
}

bool encode_gdn_conv_l2(Context& context, Encoder& encoder,
                        std::uint32_t rows, const GDNConvL2Buffers& buffers,
                        std::string& error) {
    error.clear();
    if (!validate_rows(rows, false, error)) return false;
    std::size_t qkv_bytes = 0u;
    if (!checked_float_bytes(rows, GDN_CONV_CHANNELS, qkv_bytes)) {
        error = "GDN convolution buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kHistoryElements) * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(GDN_CONV_CHANNELS) *
        GDN_CONV_KERNEL * sizeof(float);
    const std::array<BufferRef, 4u> views{
        buffers.history, buffers.qkv, buffers.conv_weights, buffers.output};
    const std::array<std::size_t, 4u> required_bytes{
        history_bytes, qkv_bytes, weight_bytes, qkv_bytes};
    for (std::size_t i = 0u; i < views.size(); ++i) {
        if (!validate_buffer(context, views[i], required_bytes[i],
                             alignof(float), "convolution", error)) {
            return false;
        }
    }
    if (!reject_output_overlap(
            buffers.output, qkv_bytes,
            {{buffers.history, history_bytes}, {buffers.qkv, qkv_bytes},
             {buffers.conv_weights, weight_bytes}},
            "convolution", error) ||
        !validate_pipeline(context, "strata_gdn_conv_l2", kConvThreads,
                           error)) {
        return false;
    }

    const std::array bindings{
        EncoderBufferBinding{buffers.history, 0u, history_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.qkv, 1u, qkv_bytes, alignof(float)},
        EncoderBufferBinding{buffers.conv_weights, 2u, weight_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.output, 3u, qkv_bytes, alignof(float)},
    };
    return encoder.dispatch(
        "strata_gdn_conv_l2", {80u, rows, 1u}, {kConvThreads, 1u, 1u},
        bindings, {}, {}, error);
}

bool encode_gdn_ab(Context& context, CommandBuffer& command_buffer,
                   std::uint32_t rows, const GDNABBuffers& buffers,
                   std::string& error) {
    if (!validate_rows(rows, false, error)) return false;
    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!checked_float_bytes(rows, N_EMBD, input_bytes) ||
        !checked_float_bytes(rows, GDN_V_HEADS, output_bytes)) {
        error = "GDN gate buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kABWeightElements) * sizeof(std::uint16_t);
    constexpr std::size_t head_bytes =
        static_cast<std::size_t>(GDN_V_HEADS) * sizeof(float);
    const std::vector<Binding> bindings{
        {buffers.input, 0u}, {buffers.alpha, 1u},
        {buffers.beta_weights, 2u}, {buffers.dt_bias, 3u},
        {buffers.a, 4u}, {buffers.gate, 5u}, {buffers.beta, 6u}};
    const std::vector<std::size_t> sizes{
        input_bytes, weight_bytes, weight_bytes, head_bytes,
        head_bytes, output_bytes, output_bytes};
    const std::vector<std::size_t> alignments{
        alignof(float), 16u, 16u, alignof(float), alignof(float),
        alignof(float), alignof(float)};
    if (!validate_bindings(context, bindings, sizes, alignments, error) ||
        !reject_output_overlap(
            buffers.gate, output_bytes,
            {{buffers.input, input_bytes}, {buffers.alpha, weight_bytes},
             {buffers.beta_weights, weight_bytes}, {buffers.dt_bias, head_bytes},
             {buffers.a, head_bytes}, {buffers.beta, output_bytes}},
            "gate", error) ||
        !reject_output_overlap(
            buffers.beta, output_bytes,
            {{buffers.input, input_bytes}, {buffers.alpha, weight_bytes},
             {buffers.beta_weights, weight_bytes}, {buffers.dt_bias, head_bytes},
             {buffers.a, head_bytes}},
            "beta", error) ||
        !validate_pipeline(context, "strata_gdn_ab", kABThreads, error)) {
        return false;
    }
    const RowsParams params{rows};
    return encode_kernel(context, command_buffer, "strata_gdn_ab",
                         @"GDN alpha/beta gates", kABThreads,
                         MTLSizeMake(12u, 1u, 1u), bindings,
                         &params, sizeof(params), 7u, {}, error);
}

bool encode_gdn_ab(Context& context, Encoder& encoder, std::uint32_t rows,
                   const GDNABBuffers& buffers, std::string& error) {
    error.clear();
    if (!validate_rows(rows, false, error)) return false;
    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!checked_float_bytes(rows, N_EMBD, input_bytes) ||
        !checked_float_bytes(rows, GDN_V_HEADS, output_bytes)) {
        error = "GDN gate buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kABWeightElements) * sizeof(std::uint16_t);
    constexpr std::size_t head_bytes =
        static_cast<std::size_t>(GDN_V_HEADS) * sizeof(float);
    const std::array<BufferRef, 7u> views{
        buffers.input, buffers.alpha, buffers.beta_weights, buffers.dt_bias,
        buffers.a, buffers.gate, buffers.beta};
    const std::array<std::size_t, 7u> required_bytes{
        input_bytes, weight_bytes, weight_bytes, head_bytes,
        head_bytes, output_bytes, output_bytes};
    const std::array<std::size_t, 7u> alignments{
        alignof(float), 16u, 16u, alignof(float), alignof(float),
        alignof(float), alignof(float)};
    for (std::size_t i = 0u; i < views.size(); ++i) {
        if (!validate_buffer(context, views[i], required_bytes[i],
                             alignments[i], "gate", error)) {
            return false;
        }
    }
    if (!reject_output_overlap(
            buffers.gate, output_bytes,
            {{buffers.input, input_bytes}, {buffers.alpha, weight_bytes},
             {buffers.beta_weights, weight_bytes}, {buffers.dt_bias, head_bytes},
             {buffers.a, head_bytes}, {buffers.beta, output_bytes}},
            "gate", error) ||
        !reject_output_overlap(
            buffers.beta, output_bytes,
            {{buffers.input, input_bytes}, {buffers.alpha, weight_bytes},
             {buffers.beta_weights, weight_bytes}, {buffers.dt_bias, head_bytes},
             {buffers.a, head_bytes}},
            "beta", error) ||
        !validate_pipeline(context, "strata_gdn_ab", kABThreads, error)) {
        return false;
    }

    const RowsParams params{rows};
    const std::array bindings{
        EncoderBufferBinding{buffers.input, 0u, input_bytes, alignof(float)},
        EncoderBufferBinding{buffers.alpha, 1u, weight_bytes, 16u},
        EncoderBufferBinding{buffers.beta_weights, 2u, weight_bytes, 16u},
        EncoderBufferBinding{buffers.dt_bias, 3u, head_bytes, alignof(float)},
        EncoderBufferBinding{buffers.a, 4u, head_bytes, alignof(float)},
        EncoderBufferBinding{buffers.gate, 5u, output_bytes, alignof(float)},
        EncoderBufferBinding{buffers.beta, 6u, output_bytes, alignof(float)},
    };
    const EncoderBytesBinding params_binding{&params, sizeof(params), 7u};
    const std::array byte_bindings{params_binding};
    return encoder.dispatch(
        "strata_gdn_ab", {12u, 1u, 1u}, {kABThreads, 1u, 1u},
        bindings, byte_bindings, {}, error);
}

bool encode_gdn_step_norm(Context& context, CommandBuffer& command_buffer,
                          std::uint32_t rows, GDNStateMode mode,
                          const GDNStepNormBuffers& buffers,
                          std::string& error) {
    if (mode != GDNStateMode::window && mode != GDNStateMode::commit &&
        mode != GDNStateMode::write_through) {
        error = "GDN state mode must be window, commit or write-through";
        return false;
    }
    const bool commit = mode == GDNStateMode::commit;
    if (!validate_rows(rows, commit, error)) return false;
    if (commit && rows == 0u) {
        auto encoding_lock = command_buffer.lock_for_encoding();
        if (validate_command_buffer(context, command_buffer, error) == nil) {
            return false;
        }
        error.clear();
        return true;
    }

    std::size_t row_h_bytes = 0u;
    std::size_t row_v_bytes = 0u;
    if (!checked_float_bytes(rows, GDN_CONV_CHANNELS, row_h_bytes) ||
        !checked_float_bytes(rows, GDN_VALUE_DIM, row_v_bytes)) {
        error = "GDN state buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kStateElements) * sizeof(float);
    constexpr std::size_t head_bytes =
        static_cast<std::size_t>(GDN_V_HEADS) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(GDN_HEAD_DIM) * sizeof(float);
    const std::vector<Binding> bindings{
        {buffers.state, 0u}, {buffers.h, 1u}, {buffers.gate, 2u},
        {buffers.beta, 3u}, {buffers.z, 4u}, {buffers.gamma, 5u},
        {buffers.output, 6u}};
    const std::vector<std::size_t> sizes{
        state_bytes, row_h_bytes, head_bytes * rows, head_bytes * rows,
        row_v_bytes, gamma_bytes, row_v_bytes};
    const std::vector<std::size_t> alignments(7u, alignof(float));
    if (!validate_bindings(context, bindings, sizes, alignments, error) ||
        !reject_output_overlap(
            buffers.output, row_v_bytes,
            {{buffers.state, state_bytes}, {buffers.h, row_h_bytes},
             {buffers.gate, head_bytes * rows}, {buffers.beta, head_bytes * rows},
             {buffers.z, row_v_bytes}, {buffers.gamma, gamma_bytes}},
            "state norm", error) ||
        !validate_pipeline(context, "strata_gdn_step_norm", kStepThreads,
                           error)) {
        return false;
    }
    for (const Binding& binding : bindings) {
        if (binding.index == 0u) continue;
        if (buffer_overlaps(buffers.state, state_bytes,
                            binding.buffer, sizes[binding.index])) {
            error = "GDN state must not overlap another bound view";
            return false;
        }
    }

    const StepParams params{rows, static_cast<std::uint32_t>(mode)};
    const std::vector<ThreadgroupMemory> memory{
        {0u, 128u * sizeof(float)},
        {1u, 128u * sizeof(float)},
        {2u, 4u * 128u * sizeof(float)},
        {3u, 16u * sizeof(float)}};
    return encode_kernel(context, command_buffer, "strata_gdn_step_norm",
                         @"GDN recurrent step and output norm", kStepThreads,
                         MTLSizeMake(GDN_V_HEADS, 1u, 1u), bindings,
                         &params, sizeof(params), 7u, memory, error);
}

bool encode_gdn_step_norm(Context& context, Encoder& encoder,
                          std::uint32_t rows, GDNStateMode mode,
                          const GDNStepNormBuffers& buffers,
                          std::string& error) {
    error.clear();
    if (mode != GDNStateMode::window && mode != GDNStateMode::commit &&
        mode != GDNStateMode::write_through) {
        error = "GDN state mode must be window, commit or write-through";
        return false;
    }
    const bool commit = mode == GDNStateMode::commit;
    if (!validate_rows(rows, commit, error)) return false;
    if (commit && rows == 0u) {
        error.clear();
        return true;
    }

    std::size_t row_h_bytes = 0u;
    std::size_t row_v_bytes = 0u;
    if (!checked_float_bytes(rows, GDN_CONV_CHANNELS, row_h_bytes) ||
        !checked_float_bytes(rows, GDN_VALUE_DIM, row_v_bytes)) {
        error = "GDN state buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kStateElements) * sizeof(float);
    constexpr std::size_t head_bytes =
        static_cast<std::size_t>(GDN_V_HEADS) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(GDN_HEAD_DIM) * sizeof(float);
    const std::array<BufferRef, 7u> views{
        buffers.state, buffers.h, buffers.gate, buffers.beta, buffers.z,
        buffers.gamma, buffers.output};
    const std::array<std::size_t, 7u> required_bytes{
        state_bytes, row_h_bytes, head_bytes * rows, head_bytes * rows,
        row_v_bytes, gamma_bytes, row_v_bytes};
    constexpr std::array<std::string_view, 7u> view_names{
        "state", "h", "gate", "beta", "z", "gamma", "output"};
    for (std::size_t i = 0u; i < views.size(); ++i) {
        if (!validate_buffer(context, views[i], required_bytes[i],
                             alignof(float), view_names[i], error)) {
            return false;
        }
    }
    if (!reject_output_overlap(
            buffers.output, row_v_bytes,
            {{buffers.state, state_bytes}, {buffers.h, row_h_bytes},
             {buffers.gate, head_bytes * rows},
             {buffers.beta, head_bytes * rows}, {buffers.z, row_v_bytes},
             {buffers.gamma, gamma_bytes}},
            "state norm", error) ||
        !validate_pipeline(context, "strata_gdn_step_norm", kStepThreads,
                           error)) {
        return false;
    }
    for (std::size_t i = 1u; i < views.size(); ++i) {
        if (buffer_overlaps(buffers.state, state_bytes, views[i],
                            required_bytes[i])) {
            error = "GDN state must not overlap another bound view";
            return false;
        }
    }

    const StepParams params{rows, static_cast<std::uint32_t>(mode)};
    const std::array bindings{
        EncoderBufferBinding{buffers.state, 0u, state_bytes, alignof(float)},
        EncoderBufferBinding{buffers.h, 1u, row_h_bytes, alignof(float)},
        EncoderBufferBinding{buffers.gate, 2u, head_bytes * rows,
                             alignof(float)},
        EncoderBufferBinding{buffers.beta, 3u, head_bytes * rows,
                             alignof(float)},
        EncoderBufferBinding{buffers.z, 4u, row_v_bytes, alignof(float)},
        EncoderBufferBinding{buffers.gamma, 5u, gamma_bytes, alignof(float)},
        EncoderBufferBinding{buffers.output, 6u, row_v_bytes, alignof(float)},
    };
    const EncoderBytesBinding params_binding{&params, sizeof(params), 7u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{0u, 128u * sizeof(float)},
        EncoderThreadgroupMemory{1u, 128u * sizeof(float)},
        EncoderThreadgroupMemory{2u, 4u * 128u * sizeof(float)},
        EncoderThreadgroupMemory{3u, 16u * sizeof(float)},
    };
    return encoder.dispatch(
        "strata_gdn_step_norm", {GDN_V_HEADS, 1u, 1u},
        {kStepThreads, 1u, 1u}, bindings, byte_bindings,
        threadgroup_memory, error);
}

bool encode_gdn_conv_commit(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows,
                            const GDNConvCommitBuffers& buffers,
                            std::string& error) {
    if (!validate_rows(rows, true, error)) return false;
    if (rows == 0u) {
        auto encoding_lock = command_buffer.lock_for_encoding();
        if (validate_command_buffer(context, command_buffer, error) == nil) {
            return false;
        }
        error.clear();
        return true;
    }
    std::size_t qkv_bytes = 0u;
    if (!checked_float_bytes(rows, GDN_CONV_CHANNELS, qkv_bytes)) {
        error = "GDN commit buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kHistoryElements) * sizeof(float);
    const std::vector<Binding> bindings{
        {buffers.history, 0u}, {buffers.qkv, 1u}};
    const std::vector<std::size_t> sizes{history_bytes, qkv_bytes};
    const std::vector<std::size_t> alignments(2u, alignof(float));
    if (!validate_bindings(context, bindings, sizes, alignments, error)) {
        return false;
    }
    if (buffer_overlaps(buffers.history, history_bytes,
                        buffers.qkv, qkv_bytes)) {
        error = "GDN convolution history must not overlap qkv";
        return false;
    }
    if (!validate_pipeline(context, "strata_gdn_conv_commit",
                           kConvThreads, error)) {
        return false;
    }
    const RowsParams params{rows};
    return encode_kernel(context, command_buffer, "strata_gdn_conv_commit",
                         @"GDN convolution history commit", kConvThreads,
                         MTLSizeMake(80u, 1u, 1u), bindings,
                         &params, sizeof(params), 2u, {}, error);
}

bool encode_gdn_conv_commit(Context& context, Encoder& encoder,
                            std::uint32_t rows,
                            const GDNConvCommitBuffers& buffers,
                            std::string& error) {
    error.clear();
    if (!validate_rows(rows, true, error)) return false;
    if (rows == 0u) return true;

    std::size_t qkv_bytes = 0u;
    if (!checked_float_bytes(rows, GDN_CONV_CHANNELS, qkv_bytes)) {
        error = "GDN commit buffer size overflows host address space";
        return false;
    }
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kHistoryElements) * sizeof(float);
    if (!validate_buffer(context, buffers.history, history_bytes,
                         alignof(float), "convolution history", error) ||
        !validate_buffer(context, buffers.qkv, qkv_bytes, alignof(float),
                         "commit qkv", error)) {
        return false;
    }
    if (buffer_overlaps(buffers.history, history_bytes,
                        buffers.qkv, qkv_bytes)) {
        error = "GDN convolution history must not overlap qkv";
        return false;
    }
    if (!validate_pipeline(context, "strata_gdn_conv_commit",
                           kConvThreads, error)) {
        return false;
    }

    const RowsParams params{rows};
    const std::array bindings{
        EncoderBufferBinding{buffers.history, 0u, history_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.qkv, 1u, qkv_bytes, alignof(float)},
    };
    const EncoderBytesBinding params_binding{&params, sizeof(params), 2u};
    const std::array byte_bindings{params_binding};
    return encoder.dispatch("strata_gdn_conv_commit", {80u, 1u, 1u},
                            {kConvThreads, 1u, 1u}, bindings, byte_bindings,
                            {}, error);
}

}  // namespace strata::metal
