// src/metal/engine/ple_dispatch.mm - validated PLE Metal encoders.
#import <Metal/Metal.h>

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/ple_dispatch.hpp"
#include "metal_fence.hpp"
#include "../kernels/common/params.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kPLEBroadcastThreads = 256u;
constexpr std::uint32_t kPLEBroadcastGroups = HC_DIM / kPLEBroadcastThreads;
constexpr std::uint32_t kPLEGateThreads = 512u;
constexpr std::uint32_t kPLEConvResidualThreads = 256u;
constexpr std::uint32_t kPLEConvResidualGroups = HC_DIM / kPLEConvResidualThreads;
constexpr std::uint32_t kPLEHistoryThreads = 256u;
constexpr std::uint32_t kPLEHistoryGroups = HC_DIM / kPLEHistoryThreads;

bool validate_buffer(Context& context, BufferRef buffer,
                     std::size_t required_bytes, std::size_t alignment,
                     std::string_view family, std::string_view name,
                     std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(family) + " " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes ||
        alignment == 0u || buffer.offset % alignment != 0u) {
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
    return true;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
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

bool buffer_views_start_at_same_address(BufferRef left, BufferRef right) {
    return (left.handle == right.handle && left.offset == right.offset) ||
           (left.contents != nullptr && left.contents == right.contents) ||
           (left.gpu_address != 0u &&
            left.gpu_address == right.gpu_address);
}

bool validate_ple_post_projection_aliases(
    std::uint32_t rows, const PLEPostProjectionBuffers& buffers,
    std::string& error) {
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "PLE post-projection row count must be in 1..8";
        return false;
    }
    const std::array views{
        buffers.key,          buffers.residual,    buffers.query,
        buffers.ple_norm,     buffers.value,       buffers.gate,
        buffers.gated,        buffers.history,     buffers.conv1d,
        buffers.key_gamma,    buffers.query_gamma, buffers.conv_gamma,
    };
    constexpr std::size_t query_index = 2u;
    constexpr std::size_t norm_index = 3u;
    constexpr std::array<std::string_view, 12> names{
        "key",       "residual",   "query",     "ple_norm",
        "value",     "gate",       "gated",     "history",
        "conv1d",    "key gamma",  "query gamma", "conv gamma",
    };
    for (std::size_t left = 0; left < views.size(); ++left) {
        for (std::size_t right = left + 1u; right < views.size(); ++right) {
            if (!buffer_views_overlap(views[left], views[right])) continue;
            const bool documented_scratch_reuse =
                left == query_index && right == norm_index &&
                buffer_views_start_at_same_address(views[left], views[right]);
            if (documented_scratch_reuse) continue;
            error = "PLE post-projection " + std::string(names[left]) +
                    " and " + std::string(names[right]) +
                    " buffers must not overlap";
            return false;
        }
    }
    return true;
}

bool validate_ple_block_aliases(bool quantized_key,
                               const PLEBlockBuffers& buffers,
                               std::string& error) {
    struct Role {
        BufferRef view;
        std::string_view name;
        bool writes;
    };
    std::array<Role, 18> roles{};
    std::size_t count = 0u;
    const auto add = [&](BufferRef view, std::string_view name, bool writes) {
        roles[count++] = {view, name, writes};
    };
    add(buffers.input_f32, "input", false);
    add(buffers.key_weights, "key weights", false);
    add(buffers.value_weights, "value weights", false);
    add(buffers.active_rows, "active rows", false);
    if (quantized_key) add(buffers.key_q8_1, "key Q8_1", true);
    add(buffers.matvec_sink, "mat-vec sink", true);

    const std::size_t query_index = count + 2u;
    const std::size_t norm_index = count + 3u;
    add(buffers.post.key, "key output", true);
    add(buffers.post.residual, "residual", true);
    add(buffers.post.query, "query", true);
    add(buffers.post.ple_norm, "PLE norm", true);
    add(buffers.post.value, "value output", true);
    add(buffers.post.gate, "gate", true);
    add(buffers.post.gated, "gated value", true);
    add(buffers.post.history, "history", true);
    add(buffers.post.conv1d, "convolution weights", false);
    add(buffers.post.key_gamma, "key gamma", false);
    add(buffers.post.query_gamma, "query gamma", false);
    add(buffers.post.conv_gamma, "convolution gamma", false);

    for (std::size_t left = 0u; left < count; ++left) {
        for (std::size_t right = left + 1u; right < count; ++right) {
            if (!buffer_views_overlap(roles[left].view, roles[right].view)) {
                continue;
            }
            const bool documented_query_norm_reuse =
                left == query_index && right == norm_index &&
                buffer_views_start_at_same_address(roles[left].view,
                                                   roles[right].view);
            if (documented_query_norm_reuse) continue;
            if (!roles[left].writes && !roles[right].writes) continue;
            error = "PLE block " + std::string(roles[left].name) + " and " +
                    std::string(roles[right].name) + " buffers must not overlap";
            return false;
        }
    }
    return true;
}

bool validate_pipeline(Context& context, const char* kernel_name,
                       std::string_view family, std::uint32_t threads,
                       std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t kernel_count = 0;
    const KernelInfo* const manifest = kernel_manifest(kernel_count);
    for (std::size_t index = 0; index < kernel_count; ++index) {
        if (std::string_view(manifest[index].name) == kernel_name) {
            info = &manifest[index];
            break;
        }
    }
    if (info == nullptr || info->threads != threads ||
        info->threadgroup_x != threads ||
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

bool checked_float_bytes(std::uint32_t rows, std::uint32_t columns,
                         std::size_t& bytes) {
    const std::size_t maximum_elements =
        std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (columns != 0u &&
        static_cast<std::size_t>(rows) > maximum_elements / columns) {
        return false;
    }
    const std::size_t elements =
        static_cast<std::size_t>(rows) * columns;
    if (elements > maximum_elements) return false;
    bytes = elements * sizeof(float);
    return true;
}

bool validate_ple_gate(Context& context, std::uint32_t rows,
                       const PLEGateBuffers& buffers, std::string& error) {
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "PLE gate row count must be in 1..8";
        return false;
    }
    if (!validate_pipeline(context, "strata_ple_gate", "PLE gate",
                           kPLEGateThreads, error)) {
        return false;
    }
    std::size_t input_bytes = 0u;
    std::size_t gate_bytes = 0u;
    if (!checked_float_bytes(rows, HC_DIM, input_bytes) ||
        !checked_float_bytes(rows, HC, gate_bytes)) {
        error = "PLE gate row count overflows the buffer byte range";
        return false;
    }
    const std::array bindings{buffers.key, buffers.query, buffers.gate};
    const std::array required_bytes{input_bytes, input_bytes, gate_bytes};
    const std::array<std::string_view, 3> names{"key", "query", "gate"};
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index], required_bytes[index],
                             alignof(float), "PLE gate", names[index], error)) {
            return false;
        }
    }
    if (buffer_views_overlap(buffers.gate, buffers.key) ||
        buffer_views_overlap(buffers.gate, buffers.query)) {
        error = "PLE gate output must not overlap either input";
        return false;
    }
    return true;
}

bool validate_ple_broadcast(Context& context, std::uint32_t rows,
                            const PLEBroadcastBuffers& buffers,
                            std::string& error) {
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "PLE broadcast row count must be in 1..8";
        return false;
    }
    if (!validate_pipeline(context, "strata_ple_broadcast", "PLE broadcast",
                           kPLEBroadcastThreads, error)) {
        return false;
    }
    std::size_t value_bytes = 0u;
    std::size_t gate_bytes = 0u;
    std::size_t gated_bytes = 0u;
    if (!checked_float_bytes(rows, N_EMBD, value_bytes) ||
        !checked_float_bytes(rows, HC, gate_bytes) ||
        !checked_float_bytes(rows, HC_DIM, gated_bytes)) {
        error = "PLE broadcast row count overflows the buffer byte range";
        return false;
    }
    const std::array bindings{
        buffers.value, buffers.gate, buffers.gated,
    };
    const std::array required_bytes{value_bytes, gate_bytes, gated_bytes};
    const std::array<std::string_view, 3> names{"value", "gate", "gated"};
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index], required_bytes[index],
                             alignof(float), "PLE broadcast", names[index],
                             error)) {
            return false;
        }
    }
    if (buffer_views_overlap(buffers.gated, buffers.value) ||
        buffer_views_overlap(buffers.gated, buffers.gate)) {
        error = "PLE broadcast output must not overlap either input";
        return false;
    }
    return true;
}

bool validate_ple_conv_residual(
    Context& context, std::uint32_t rows,
    const PLEConvResidualBuffers& buffers, std::string& error) {
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "PLE conv-residual row count must be in 1..8";
        return false;
    }
    if (!validate_pipeline(context, "strata_ple_conv_residual",
                           "PLE conv-residual",
                           kPLEConvResidualThreads, error)) {
        return false;
    }
    std::size_t row_bytes = 0u;
    if (!checked_float_bytes(rows, HC_DIM, row_bytes)) {
        error = "PLE conv-residual row count overflows the buffer byte range";
        return false;
    }
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS * sizeof(float);
    constexpr std::size_t conv1d_bytes =
        static_cast<std::size_t>(HC_DIM) * 4u * sizeof(std::uint16_t);
    const std::array bindings{
        buffers.ple_norm, buffers.history, buffers.conv1d,
        buffers.gated, buffers.residual,
    };
    const std::array required_bytes{
        row_bytes, history_bytes, conv1d_bytes, row_bytes, row_bytes,
    };
    const std::array alignments{
        alignof(float), alignof(float), alignof(std::uint16_t),
        alignof(float), alignof(float),
    };
    const std::array<std::string_view, 5> names{
        "ple_norm", "history", "conv1d", "gated", "residual",
    };
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index], required_bytes[index],
                             alignments[index], "PLE conv-residual",
                             names[index], error)) {
            return false;
        }
    }
    for (std::size_t index = 0; index + 1u < bindings.size(); ++index) {
        if (buffer_views_overlap(buffers.residual, bindings[index])) {
            error = "PLE conv-residual output must not overlap an input";
            return false;
        }
    }
    return true;
}

bool validate_ple_history(Context& context, std::uint32_t n,
                          const PLEHistoryBuffers& buffers,
                          std::string& error) {
    if (n > static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max())) {
        error = "PLE history row count exceeds the signed Metal index range";
        return false;
    }
    if (!validate_pipeline(context, "strata_ple_history", "PLE history",
                           kPLEHistoryThreads, error)) {
        return false;
    }
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS * sizeof(float);
    std::size_t norm_bytes = 0u;
    if (!checked_float_bytes(n, HC_DIM, norm_bytes)) {
        error = "PLE history row count overflows the norm byte range";
        return false;
    }
    if (!validate_buffer(context, buffers.history, history_bytes,
                         alignof(float), "PLE history", "history", error) ||
        !validate_buffer(context, buffers.ple_norm, norm_bytes,
                         alignof(float), "PLE history", "ple_norm", error)) {
        return false;
    }
    if (buffer_views_overlap(buffers.history, buffers.ple_norm)) {
        error = "PLE history and ple_norm views must not overlap";
        return false;
    }
    return true;
}

}  // namespace

bool encode_ple_gate(Context& context, Encoder& encoder, std::uint32_t rows,
                     const PLEGateBuffers& buffers, std::string& error) {
    error.clear();
    if (!validate_ple_gate(context, rows, buffers, error)) return false;

    const std::array bindings{
        EncoderBufferBinding{buffers.key, 0u,
                             static_cast<std::size_t>(rows) * HC_DIM *
                                 sizeof(float),
                             alignof(float)},
        EncoderBufferBinding{buffers.query, 1u,
                             static_cast<std::size_t>(rows) * HC_DIM *
                                 sizeof(float),
                             alignof(float)},
        EncoderBufferBinding{buffers.gate, 2u,
                             static_cast<std::size_t>(rows) * HC *
                                 sizeof(float),
                             alignof(float)},
    };
    const float scale =
        1.0f / std::sqrt(static_cast<float>(N_EMBD));
    const std::array bytes{
        EncoderBytesBinding{&scale, sizeof(scale), 3u},
    };
    return encoder.dispatch(
        "strata_ple_gate", {static_cast<std::size_t>(rows) * HC, 1u, 1u},
        {kPLEGateThreads, 1u, 1u}, bindings, bytes, {}, error);
}

bool encode_ple_gate(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t rows, const PLEGateBuffers& buffers,
                     std::string& error) {
    if (!validate_ple_gate(context, rows, buffers, error)) {
        return false;
    }
    return Encoder::encode(
        context, command_buffer, "PLE gate reduction",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_ple_gate(context, encoder, rows, buffers,
                                   encode_error);
        },
        error);
}

bool encode_ple_broadcast(Context& context, Encoder& encoder,
                          std::uint32_t rows,
                          const PLEBroadcastBuffers& buffers,
                          std::string& error) {
    error.clear();
    if (!validate_ple_broadcast(context, rows, buffers, error)) {
        return false;
    }
    const std::array bindings{
        EncoderBufferBinding{
            buffers.value, 0u,
            static_cast<std::size_t>(rows) * N_EMBD * sizeof(float),
            alignof(float)},
        EncoderBufferBinding{
            buffers.gate, 1u,
            static_cast<std::size_t>(rows) * HC * sizeof(float),
            alignof(float)},
        EncoderBufferBinding{
            buffers.gated, 2u,
            static_cast<std::size_t>(rows) * HC_DIM * sizeof(float),
            alignof(float)},
    };
    return encoder.dispatch(
        "strata_ple_broadcast",
        {kPLEBroadcastGroups, rows, 1u},
        {kPLEBroadcastThreads, 1u, 1u}, bindings, {}, {}, error);
}

bool encode_ple_broadcast(Context& context, CommandBuffer& command_buffer,
                          std::uint32_t rows,
                          const PLEBroadcastBuffers& buffers,
                          std::string& error) {
    if (!validate_ple_broadcast(context, rows, buffers, error)) {
        return false;
    }
    return Encoder::encode(
        context, command_buffer, "PLE value broadcast",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_ple_broadcast(context, encoder, rows, buffers,
                                        encode_error);
        },
        error);
}

bool encode_ple_conv_residual(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PLEConvResidualBuffers& buffers, std::string& error) {
    error.clear();
    if (!validate_ple_conv_residual(context, rows, buffers, error)) {
        return false;
    }
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS * sizeof(float);
    constexpr std::size_t conv1d_bytes =
        static_cast<std::size_t>(HC_DIM) * 4u * sizeof(std::uint16_t);
    const std::size_t row_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::array bindings{
        EncoderBufferBinding{
            buffers.ple_norm, 0u, row_bytes, alignof(float)},
        EncoderBufferBinding{
            buffers.history, 1u, history_bytes, alignof(float)},
        EncoderBufferBinding{
            buffers.conv1d, 2u, conv1d_bytes, alignof(std::uint16_t)},
        EncoderBufferBinding{
            buffers.gated, 3u, row_bytes, alignof(float)},
        EncoderBufferBinding{
            buffers.residual, 4u, row_bytes, alignof(float)},
    };
    return encoder.dispatch(
        "strata_ple_conv_residual",
        {kPLEConvResidualGroups, rows, 1u},
        {kPLEConvResidualThreads, 1u, 1u}, bindings, {}, {}, error);
}

bool encode_ple_conv_residual(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PLEConvResidualBuffers& buffers, std::string& error) {
    if (!validate_ple_conv_residual(context, rows, buffers, error)) {
        return false;
    }
    return Encoder::encode(
        context, command_buffer, "PLE convolution and residual",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_ple_conv_residual(context, encoder, rows, buffers,
                                            encode_error);
        },
        error);
}

bool encode_ple_block(
    Context& context, Encoder& encoder, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const PLEBlockBuffers& buffers,
    std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "PLE block row count must be in 1..8";
        return false;
    }
    if (compiled_columns != 1u && compiled_columns != 2u &&
        compiled_columns != 4u && compiled_columns != 8u) {
        error = "PLE block compiled column width must be 1, 2, 4 or 8";
        return false;
    }
    if (rows > compiled_columns) {
        error = "PLE block token rows exceed the compiled column width";
        return false;
    }
    if (key_format != MatVecWeightFormat::q2_0 &&
        key_format != MatVecWeightFormat::q8_0 &&
        key_format != MatVecWeightFormat::bf16) {
        error = "PLE key format must be Q2_0, Q8_0 or BF16";
        return false;
    }
    if (history_update != PLEHistoryUpdate::defer &&
        history_update != PLEHistoryUpdate::advance_rows) {
        error = "PLE block history update policy is invalid";
        return false;
    }
    if (!validate_ple_post_projection_aliases(rows, buffers.post, error)) {
        return false;
    }
    const bool quantized_key = key_format != MatVecWeightFormat::bf16;
    if (!validate_ple_block_aliases(quantized_key, buffers, error)) {
        return false;
    }

    BufferRef key_activations = buffers.input_f32;
    if (quantized_key) {
        if (!encode_q8_1(
                context, encoder, rows, N_EMBD,
                {buffers.input_f32, buffers.key_q8_1}, error)) {
            return false;
        }
        key_activations = buffers.key_q8_1;
    }

    const MatVecBuffers key_buffers{
        key_activations, buffers.key_weights, buffers.post.key,
        buffers.matvec_sink, buffers.active_rows,
    };
    if (!encode_matvec(context, encoder, key_format, N_EMBD, HC_DIM, rows,
                       compiled_columns, key_buffers, error)) {
        return false;
    }

    const MatVecBuffers value_buffers{
        buffers.input_f32, buffers.value_weights, buffers.post.value,
        buffers.matvec_sink, buffers.active_rows,
    };
    if (!encode_matvec(context, encoder, MatVecWeightFormat::bf16, N_EMBD,
                       N_EMBD, rows, compiled_columns, value_buffers,
                       error)) {
        return false;
    }
    return encode_ple_post_projection(context, encoder, rows, history_update,
                                      buffers.post, error);
}

bool encode_ple_block(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const PLEBlockBuffers& buffers,
    std::string& error) {
    error.clear();
    return Encoder::encode(
        context, command_buffer, "PLE decode block",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_ple_block(context, encoder, rows, key_format,
                                    compiled_columns, history_update, buffers,
                                    encode_error);
        },
        error);
}

bool encode_ple_history(Context& context, Encoder& encoder, std::uint32_t n,
                        const PLEHistoryBuffers& buffers,
                        std::string& error) {
    error.clear();
    if (!validate_ple_history(context, n, buffers, error)) return false;

    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS * sizeof(float);
    std::size_t norm_bytes = 0u;
    if (!checked_float_bytes(n, HC_DIM, norm_bytes)) {
        error = "PLE history row count overflows the norm byte range";
        return false;
    }
    BufferRef norm_binding = buffers.ple_norm;
    std::size_t norm_binding_bytes = norm_bytes;
    if (n == 0u) {
        // The dispatcher requires a non-empty binding; the shader skips
        // ple_norm entirely for n=0, so use history as a valid placeholder.
        norm_binding = buffers.history;
        norm_binding_bytes = sizeof(float);
    }
    const std::array buffer_bindings{
        EncoderBufferBinding{
            buffers.history, 0u, history_bytes, alignof(float)},
        EncoderBufferBinding{
            norm_binding, 1u, norm_binding_bytes, alignof(float)},
    };
    const EncoderBytesBinding row_binding{&n, sizeof(n), 2u};
    const std::array byte_bindings{row_binding};
    return encoder.dispatch(
        "strata_ple_history",
        {kPLEHistoryGroups, 1u, 1u},
        {kPLEHistoryThreads, 1u, 1u}, buffer_bindings, byte_bindings, {},
        error);
}

bool encode_ple_history(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t n,
                        const PLEHistoryBuffers& buffers,
                        std::string& error) {
    if (!validate_ple_history(context, n, buffers, error)) return false;
    return Encoder::encode(
        context, command_buffer, "PLE history advancement",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_ple_history(context, encoder, n, buffers,
                                      encode_error);
        },
        error);
}

bool encode_ple_post_projection(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PLEHistoryUpdate history_update,
    const PLEPostProjectionBuffers& buffers, std::string& error) {
    error.clear();
    if (history_update != PLEHistoryUpdate::defer &&
        history_update != PLEHistoryUpdate::advance_rows) {
        error = "PLE post-projection history update policy is invalid";
        return false;
    }
    if (!validate_ple_post_projection_aliases(rows, buffers, error)) {
        return false;
    }
    constexpr float epsilon = 1.0e-6f;
    const std::uint32_t norm_rows = rows * HC;
    if (!encode_weighted_rms_norm(
            context, encoder, norm_rows, N_EMBD, HC, epsilon,
            {buffers.key, buffers.key_gamma, buffers.key}, error)) {
        return false;
    }
    if (!encode_weighted_rms_norm(
            context, encoder, norm_rows, N_EMBD, HC, epsilon,
            {buffers.residual, buffers.query_gamma, buffers.query}, error)) {
        return false;
    }
    if (!encode_ple_gate(
            context, encoder, rows,
            {buffers.key, buffers.query, buffers.gate}, error)) {
        return false;
    }
    if (!encode_ple_broadcast(
            context, encoder, rows,
            {buffers.value, buffers.gate, buffers.gated}, error)) {
        return false;
    }
    if (!encode_weighted_rms_norm(
            context, encoder, norm_rows, N_EMBD, HC, epsilon,
            {buffers.gated, buffers.conv_gamma, buffers.ple_norm}, error)) {
        return false;
    }
    if (!encode_ple_conv_residual(
            context, encoder, rows,
            {buffers.ple_norm, buffers.history, buffers.conv1d,
             buffers.gated, buffers.residual},
            error)) {
        return false;
    }
    if (history_update == PLEHistoryUpdate::defer) return true;
    return encode_ple_history(
        context, encoder, rows, {buffers.history, buffers.ple_norm}, error);
}

bool encode_ple_post_projection(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PLEHistoryUpdate history_update,
    const PLEPostProjectionBuffers& buffers, std::string& error) {
    error.clear();
    return Encoder::encode(
        context, command_buffer, "PLE post-projection sequence",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_ple_post_projection(
                context, encoder, rows, history_update, buffers,
                encode_error);
        },
        error);
}

}  // namespace strata::metal
