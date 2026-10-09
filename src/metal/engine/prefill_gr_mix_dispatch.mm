// Validated encoder for Prefill's distinct four-stream GR mixer.
#include "strata/metal/prefill_gr_mix_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kStreams = 4u;
constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kTileWidth = 256u;
constexpr std::uint32_t kTiles = kWidth / kTileWidth;
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kBFloatBytes = sizeof(std::uint16_t);
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);

bool checked_bytes(std::uint32_t rows, std::size_t values_per_row,
                   std::size_t value_bytes, std::size_t& bytes) {
    if (rows != 0u && values_per_row >
            std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    const std::size_t values =
        static_cast<std::size_t>(rows) * values_per_row;
    if (values != 0u && value_bytes >
            std::numeric_limits<std::size_t>::max() / values) {
        return false;
    }
    bytes = values * value_bytes;
    return true;
}

bool validate_read_view(Context& context, ReadOnlyBufferRef buffer,
                        std::size_t required_bytes, std::size_t alignment,
                        std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "Prefill GR mix " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length() < required_bytes || alignment == 0u ||
        buffer.offset() % alignment != 0u) {
        error = "Prefill GR mix " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef buffer,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "Prefill GR mix " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes || alignment == 0u ||
        buffer.offset % alignment != 0u) {
        error = "Prefill GR mix " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                    Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool write_views_overlap(BufferRef left, BufferRef right) {
    const bool same_allocation =
        left.handle == right.handle &&
        ranges_overlap(left.offset, left.length, right.offset, right.length);
    const bool same_cpu_backing =
        left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.contents),
                       left.length,
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length);
    const bool same_gpu_backing =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address, left.length,
                       right.gpu_address, right.length);
    return same_allocation || same_cpu_backing || same_gpu_backing;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       PrefillBF16X2Mode mode,
                       const PrefillGRMixBuffers& buffers,
                       std::size_t& stream_input_bytes,
                       std::size_t& scale_input_bytes,
                       std::size_t& output_bytes, std::string& error) {
    const std::uint32_t mode_value = static_cast<std::uint32_t>(mode);
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GR mix row count exceeds the shader-safe range";
        return false;
    }
    if (mode_value > static_cast<std::uint32_t>(
                         PrefillBF16X2Mode::except_gr)) {
        error = "Prefill GR mix BF16X2 mode is invalid";
        return false;
    }
    const bool write_low = mode != PrefillBF16X2Mode::off;
    if (write_low != static_cast<bool>(buffers.mixed_bf_lo)) {
        error = write_low
                    ? "Prefill GR mix modes 1 and 2 require mixed_bf_lo"
                    : "Prefill GR mix mixed_bf_lo is only valid in modes 1 and 2";
        return false;
    }
    if (!checked_bytes(rows,
                       static_cast<std::size_t>(kStreams) * kWidth,
                       kFloatBytes, stream_input_bytes) ||
        !checked_bytes(rows, kStreams, kFloatBytes, scale_input_bytes) ||
        !checked_bytes(rows, kWidth, kBFloatBytes, output_bytes)) {
        error = "Prefill GR mix buffer size overflows size_t";
        return false;
    }
    constexpr std::size_t norm_bytes =
        static_cast<std::size_t>(kStreams) * kWidth * kFloatBytes;
    if (!validate_read_view(context, buffers.residual, stream_input_bytes,
                            kFloatBytes, "residual", error) ||
        !validate_read_view(context, buffers.rs, scale_input_bytes,
                            kFloatBytes, "scale", error) ||
        !validate_read_view(context, buffers.norm, norm_bytes,
                            kFloatBytes, "norm", error) ||
        !validate_read_view(context, buffers.gated, stream_input_bytes,
                            kFloatBytes, "gated", error) ||
        !validate_write_view(context, buffers.mixed_bf, output_bytes,
                             kBFloatBytes, "BF16 output", error) ||
        !validate_write_view(context, buffers.mixed_h, output_bytes,
                             kHalfBytes, "F16 output", error)) {
        return false;
    }
    if (write_low &&
        !validate_write_view(context, buffers.mixed_bf_lo, output_bytes,
                             kBFloatBytes, "BF16 remainder output", error)) {
        return false;
    }

    const std::array<BufferRef, 3> outputs{
        buffers.mixed_bf, buffers.mixed_h,
        write_low ? buffers.mixed_bf_lo : BufferRef{},
    };
    const std::size_t output_count = write_low ? outputs.size()
                                               : outputs.size() - 1u;
    for (std::size_t out = 0u; out < output_count; ++out) {
        for (std::size_t earlier = 0u; earlier < out; ++earlier) {
            if (write_views_overlap(outputs[out], outputs[earlier])) {
                error = "Prefill GR mix output buffer views overlap";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_gr_mix(Context& context, Encoder& encoder,
                           std::uint32_t rows, PrefillBF16X2Mode mode,
                           const PrefillGRMixBuffers& buffers,
                           std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GR mix requires an ordered concurrent-group boundary";
        return false;
    }

    std::size_t stream_input_bytes = 0u;
    std::size_t scale_input_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!validate_dispatch(context, rows, mode, buffers, stream_input_bytes,
                           scale_input_bytes, output_bytes, error)) {
        return false;
    }

    const bool write_low = mode != PrefillBF16X2Mode::off;
    const std::array<EncoderBufferBinding, 3> writable{{
        EncoderBufferBinding{buffers.mixed_bf, 4u, output_bytes,
                             kBFloatBytes},
        EncoderBufferBinding{buffers.mixed_h, 5u, output_bytes, kHalfBytes},
        EncoderBufferBinding{buffers.mixed_bf_lo, 6u, output_bytes,
                             kBFloatBytes},
    }};
    constexpr std::size_t norm_bytes =
        static_cast<std::size_t>(kStreams) * kWidth * kFloatBytes;
    const std::array<EncoderReadOnlyBufferBinding, 4> read_only{{
        EncoderReadOnlyBufferBinding{buffers.residual, 0u,
                                     stream_input_bytes, kFloatBytes},
        EncoderReadOnlyBufferBinding{buffers.rs, 1u, scale_input_bytes,
                                     kFloatBytes},
        EncoderReadOnlyBufferBinding{buffers.norm, 2u, norm_bytes,
                                     kFloatBytes},
        EncoderReadOnlyBufferBinding{buffers.gated, 3u,
                                     stream_input_bytes, kFloatBytes},
    }};
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        write_low ? "strata_pf_gr_mix_lo" : "strata_pf_gr_mix",
        {kTiles, rows, 1u}, {kTileWidth, 1u, 1u},
        std::span<const EncoderBufferBinding>(
            writable.data(), write_low ? writable.size()
                                       : writable.size() - 1u),
        read_only, bytes, threadgroup_memory, error);
}

bool encode_prefill_gr_mix(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, PrefillBF16X2Mode mode,
                           const PrefillGRMixBuffers& buffers,
                           std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GR mixer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_mix(context, encoder, rows, mode,
                                         buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
