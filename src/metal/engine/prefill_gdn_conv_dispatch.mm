// Validated encoders for Prefill GDN convolution stages.
#include "strata/metal/prefill_gdn_conv_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include "../kernels/common/params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kChannels = GDN_CONV_CHANNELS;
constexpr std::uint32_t kKernelWidth = GDN_CONV_KERNEL;
constexpr std::uint32_t kThreads = 128u;
constexpr std::uint32_t kRowsPerTile = 64u;
constexpr std::size_t kFloatBytes = sizeof(float);

bool checked_float_bytes(std::uint32_t rows, std::uint32_t values_per_row,
                         std::size_t& bytes) {
    const std::size_t max_values =
        std::numeric_limits<std::size_t>::max() / kFloatBytes;
    if (values_per_row != 0u &&
        static_cast<std::size_t>(rows) > max_values / values_per_row) {
        return false;
    }
    bytes = static_cast<std::size_t>(rows) * values_per_row * kFloatBytes;
    return true;
}

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                    Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0 || right_bytes == 0) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool validate_read_view(Context& context, ReadOnlyBufferRef view,
                        std::size_t required_bytes, std::string_view name,
                        std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN convolution " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes ||
        view.offset() % kFloatBytes != 0u) {
        error = "Prefill GDN convolution " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error =
            "Prefill GDN convolution output is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes ||
        view.offset % kFloatBytes != 0u) {
        error = "Prefill GDN convolution output is undersized or misaligned";
        return false;
    }
    return true;
}

bool overlaps(ReadOnlyBufferRef input, BufferRef output) {
    const void* const input_contents = input.contents();
    const bool same_cpu_backing =
        input_contents != nullptr && output.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(input_contents),
                       input.length(),
                       reinterpret_cast<std::uintptr_t>(output.contents),
                       output.length);
    const bool same_gpu_backing =
        input.gpu_address() != 0u && output.gpu_address != 0u &&
        ranges_overlap(input.gpu_address(), input.length(), output.gpu_address,
                       output.length);
    return same_cpu_backing || same_gpu_backing;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       const PrefillGDNConvTiledBuffers& buffers,
                       std::size_t& history_bytes, std::size_t& row_bytes,
                       std::size_t& weight_bytes, std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error =
            "Prefill GDN convolution row count exceeds the shader-safe range";
        return false;
    }
    if (!checked_float_bytes(kKernelWidth - 1u, kChannels, history_bytes) ||
        !checked_float_bytes(rows, kChannels, row_bytes) ||
        !checked_float_bytes(kKernelWidth, kChannels, weight_bytes)) {
        error = "Prefill GDN convolution buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.history, history_bytes,
                            "history", error) ||
        !validate_read_view(context, buffers.qkv, row_bytes, "QKV", error) ||
        !validate_read_view(context, buffers.conv_weights, weight_bytes,
                            "convolution weights", error) ||
        !validate_write_view(context, buffers.output, row_bytes, error)) {
        return false;
    }
    if (overlaps(buffers.history, buffers.output) ||
        overlaps(buffers.qkv, buffers.output) ||
        overlaps(buffers.conv_weights, buffers.output)) {
        error = "Prefill GDN convolution output overlaps a read-only input";
        return false;
    }
    return true;
}

bool validate_history_dispatch(
    Context& context, std::uint32_t rows,
    const PrefillGDNConvHistoryBuffers& buffers,
    std::size_t& history_bytes, std::size_t& row_bytes,
    std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GDN convolution history row count exceeds the "
                "shader-safe range";
        return false;
    }
    if (!checked_float_bytes(kKernelWidth - 1u, kChannels, history_bytes) ||
        !checked_float_bytes(rows, kChannels, row_bytes)) {
        error = "Prefill GDN convolution history buffer size overflows size_t";
        return false;
    }
    if (!buffers.history || !context.buffers().owns(buffers.history)) {
        error = "Prefill GDN convolution history buffer is not owned by this "
                "Context";
        return false;
    }
    if (buffers.history.length < history_bytes ||
        buffers.history.offset % kFloatBytes != 0u) {
        error = "Prefill GDN convolution history buffer is undersized or "
                "misaligned";
        return false;
    }
    if (!validate_read_view(context, buffers.qkv, row_bytes, "QKV", error)) {
        return false;
    }
    if (overlaps(buffers.qkv, buffers.history)) {
        error = "Prefill GDN convolution history overlaps QKV";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_gdn_conv_tiled(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNConvTiledBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GDN convolution requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    std::size_t history_bytes = 0u;
    std::size_t row_bytes = 0u;
    std::size_t weight_bytes = 0u;
    if (!validate_dispatch(context, rows, buffers, history_bytes, row_bytes,
                           weight_bytes, error)) {
        return false;
    }

    const EncoderReadOnlyBufferBinding inputs[] = {
        {buffers.history, 0u, history_bytes, kFloatBytes},
        {buffers.qkv, 1u, row_bytes, kFloatBytes},
        {buffers.conv_weights, 2u, weight_bytes, kFloatBytes},
    };
    const EncoderBufferBinding outputs[] = {
        {buffers.output, 3u, row_bytes, kFloatBytes},
    };
    const EncoderBytesBinding row_binding{&rows, sizeof(rows), 4u};
    const EncoderThreadgroupMemory no_threadgroup_memory{};
    const std::uint32_t row_tiles =
        (rows + kRowsPerTile - 1u) / kRowsPerTile;
    return encoder.dispatch(
        "strata_pf_gdn_conv_tiled",
        {kChannels / kThreads, row_tiles, 1u}, {kThreads, 1u, 1u}, outputs,
        inputs, std::span<const EncoderBytesBinding>(&row_binding, 1u),
        std::span<const EncoderThreadgroupMemory>(&no_threadgroup_memory, 0u),
        error);
}

bool encode_prefill_gdn_conv_tiled(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNConvTiledBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN tiled convolution",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_conv_tiled(
                context, encoder, rows, buffers, encode_error);
        },
        error);
}

bool encode_prefill_gdn_conv_history(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNConvHistoryBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GDN convolution history requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    std::size_t history_bytes = 0u;
    std::size_t row_bytes = 0u;
    if (!validate_history_dispatch(context, rows, buffers, history_bytes,
                                   row_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding outputs[] = {
        {buffers.history, 0u, history_bytes, kFloatBytes},
    };
    const EncoderReadOnlyBufferBinding inputs[] = {
        {buffers.qkv, 1u, row_bytes, kFloatBytes},
    };
    const EncoderBytesBinding row_binding{&rows, sizeof(rows), 2u};
    const EncoderThreadgroupMemory no_threadgroup_memory{};
    return encoder.dispatch(
        "strata_pf_gdn_conv_hist", {kChannels / kThreads, 1u, 1u},
        {kThreads, 1u, 1u}, outputs, inputs,
        std::span<const EncoderBytesBinding>(&row_binding, 1u),
        std::span<const EncoderThreadgroupMemory>(&no_threadgroup_memory, 0u),
        error);
}

bool encode_prefill_gdn_conv_history(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNConvHistoryBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN convolution history",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_conv_history(
                context, encoder, rows, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
