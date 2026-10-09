// Validated encoder for Prefill's fused GR write and next-read normalization.
#include "strata/metal/prefill_gr_write_norm_dispatch.hpp"
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
constexpr std::uint32_t kThreads = 256u;
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kBFloatBytes = sizeof(std::uint16_t);

bool checked_bytes(std::uint32_t rows, std::size_t values_per_row,
                   std::size_t value_bytes, std::size_t& bytes) {
    if (rows != 0u && values_per_row >
            std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    const std::size_t values = static_cast<std::size_t>(rows) * values_per_row;
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
        error = "Prefill GR write-norm " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length() < required_bytes || alignment == 0u ||
        buffer.offset() % alignment != 0u) {
        error = "Prefill GR write-norm " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef buffer,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "Prefill GR write-norm " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes || alignment == 0u ||
        buffer.offset % alignment != 0u) {
        error = "Prefill GR write-norm " + std::string(name) +
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

bool views_overlap(BufferRef left, BufferRef right) {
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

bool views_overlap(ReadOnlyBufferRef left, BufferRef right) {
    const void* const left_contents = left.contents();
    const bool same_cpu_backing =
        left_contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left.length(),
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length);
    const bool same_gpu_backing =
        left.gpu_address() != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address(), left.length(), right.gpu_address,
                       right.length);
    return same_cpu_backing || same_gpu_backing;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       PrefillBF16X2Mode mode,
                       const PrefillGRWriteNormBuffers& buffers,
                       std::size_t& residual_bytes,
                       std::size_t& block_out_bytes,
                       std::size_t& inject_bytes,
                       std::size_t& scale_bytes,
                       std::size_t& output_bytes,
                       std::string& error) {
    const std::uint32_t mode_value = static_cast<std::uint32_t>(mode);
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GR write-norm row count exceeds the shader-safe range";
        return false;
    }
    if (mode_value > static_cast<std::uint32_t>(
                         PrefillBF16X2Mode::except_gr)) {
        error = "Prefill GR write-norm BF16X2 mode is invalid";
        return false;
    }
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    if (write_low != static_cast<bool>(buffers.xn16_lo)) {
        error = write_low
                    ? "Prefill GR write-norm mode 1 requires xn16_lo"
                    : "Prefill GR write-norm xn16_lo is only valid in mode 1";
        return false;
    }

    if (!checked_bytes(rows, static_cast<std::size_t>(kStreams) * kWidth,
                       kFloatBytes, residual_bytes) ||
        !checked_bytes(rows, kWidth, kFloatBytes, block_out_bytes) ||
        !checked_bytes(rows, kStreams, kFloatBytes, inject_bytes) ||
        !checked_bytes(rows, kStreams, kFloatBytes, scale_bytes) ||
        !checked_bytes(rows, static_cast<std::size_t>(kStreams) * kWidth,
                       kBFloatBytes, output_bytes)) {
        error = "Prefill GR write-norm buffer size overflows size_t";
        return false;
    }
    constexpr std::size_t norm_bytes =
        static_cast<std::size_t>(kStreams) * kWidth * kFloatBytes;
    if (!validate_write_view(context, buffers.residual, residual_bytes,
                             kFloatBytes, "residual", error) ||
        !validate_read_view(context, buffers.block_out, block_out_bytes,
                            kFloatBytes, "block output", error) ||
        !validate_read_view(context, buffers.inject, inject_bytes, kFloatBytes,
                            "injection", error) ||
        !validate_read_view(context, buffers.next_norm, norm_bytes,
                            kFloatBytes, "next norm", error) ||
        !validate_write_view(context, buffers.rs, scale_bytes, kFloatBytes,
                             "scale output", error) ||
        !validate_write_view(context, buffers.xn16, output_bytes,
                             kBFloatBytes, "BF16 output", error)) {
        return false;
    }
    if (write_low &&
        !validate_write_view(context, buffers.xn16_lo, output_bytes,
                             kBFloatBytes, "BF16 remainder output", error)) {
        return false;
    }

    const std::array<ReadOnlyBufferRef, 3> inputs{
        buffers.block_out, buffers.inject, buffers.next_norm};
    const std::array<BufferRef, 4> outputs{
        buffers.residual, buffers.rs, buffers.xn16,
        write_low ? buffers.xn16_lo : BufferRef{}};
    const std::size_t output_count = write_low ? outputs.size()
                                               : outputs.size() - 1u;
    for (std::size_t out = 0u; out < output_count; ++out) {
        if (views_overlap(inputs[0], outputs[out]) ||
            views_overlap(inputs[1], outputs[out]) ||
            views_overlap(inputs[2], outputs[out])) {
            error = "Prefill GR write-norm output overlaps a read-only input";
            return false;
        }
        for (std::size_t earlier = 0u; earlier < out; ++earlier) {
            if (views_overlap(outputs[out], outputs[earlier])) {
                error = "Prefill GR write-norm output buffer views overlap";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_gr_write_norm(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRWriteNormBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "Prefill GR write-norm requires an ordered concurrent-group boundary";
        return false;
    }

    std::size_t residual_bytes = 0u;
    std::size_t block_out_bytes = 0u;
    std::size_t inject_bytes = 0u;
    std::size_t scale_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!validate_dispatch(context, rows, mode, buffers, residual_bytes,
                           block_out_bytes, inject_bytes, scale_bytes,
                           output_bytes, error)) {
        return false;
    }

    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    constexpr std::size_t norm_bytes =
        static_cast<std::size_t>(kStreams) * kWidth * kFloatBytes;
    const std::array<EncoderBufferBinding, 4> writable{{
        {buffers.residual, 0u, residual_bytes, kFloatBytes},
        {buffers.rs, 4u, scale_bytes, kFloatBytes},
        {buffers.xn16, 5u, output_bytes, kBFloatBytes},
        {buffers.xn16_lo, 6u, output_bytes, kBFloatBytes},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 3> read_only{{
        {buffers.block_out, 1u, block_out_bytes, kFloatBytes},
        {buffers.inject, 2u, inject_bytes, kFloatBytes},
        {buffers.next_norm, 3u, norm_bytes, kFloatBytes},
    }};
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        write_low ? "strata_pf_gr_write_norm_lo"
                  : "strata_pf_gr_write_norm",
        {static_cast<std::size_t>(rows) * kStreams, 1u, 1u},
        {kThreads, 1u, 1u},
        std::span<const EncoderBufferBinding>(
            writable.data(), write_low ? writable.size()
                                       : writable.size() - 1u),
        read_only, bytes, threadgroup_memory, error);
}

bool encode_prefill_gr_write_norm(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRWriteNormBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GR write and next-read norm",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_write_norm(context, encoder, rows, mode,
                                                 buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
