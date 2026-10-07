// Validated encoder for Prefill's distinct GR normalization reduction.
#include "strata/metal/prefill_gr_norm_dispatch.hpp"
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

bool checked_bytes(std::uint32_t rows, std::size_t row_bytes,
                   std::size_t& bytes) {
    if (rows != 0u && row_bytes >
            std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    bytes = static_cast<std::size_t>(rows) * row_bytes;
    return true;
}

bool validate_view(Context& context, BufferRef buffer,
                   std::size_t required_bytes, std::size_t alignment,
                   std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "Prefill GR norm " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes || alignment == 0u ||
        buffer.offset % alignment != 0u) {
        error = "Prefill GR norm " + std::string(name) +
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

bool validate_dispatch(Context& context, std::uint32_t rows,
                       PrefillBF16X2Mode mode,
                       const PrefillGRNormBuffers& buffers,
                       std::size_t& residual_bytes,
                       std::size_t& output_bytes, std::string& error) {
    const std::uint32_t mode_value = static_cast<std::uint32_t>(mode);
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GR norm row count exceeds the shader-safe range";
        return false;
    }
    if (mode_value > static_cast<std::uint32_t>(
                         PrefillBF16X2Mode::except_gr)) {
        error = "Prefill GR norm BF16X2 mode is invalid";
        return false;
    }
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    if (write_low != static_cast<bool>(buffers.xn16_lo)) {
        error = write_low
                    ? "Prefill GR norm mode 1 requires xn16_lo"
                    : "Prefill GR norm xn16_lo is only valid in mode 1";
        return false;
    }

    const std::size_t row_values =
        static_cast<std::size_t>(kStreams) * kWidth;
    if (!checked_bytes(rows, row_values * kFloatBytes, residual_bytes) ||
        !checked_bytes(rows, row_values * kBFloatBytes, output_bytes)) {
        error = "Prefill GR norm buffer size overflows size_t";
        return false;
    }
    const std::size_t scale_bytes =
        static_cast<std::size_t>(rows) * kStreams * kFloatBytes;
    const std::size_t norm_bytes = row_values * kFloatBytes;

    if (!validate_view(context, buffers.residual, residual_bytes, kFloatBytes,
                       "residual", error) ||
        !validate_view(context, buffers.norm, norm_bytes, kFloatBytes,
                       "norm", error) ||
        !validate_view(context, buffers.rs, scale_bytes, kFloatBytes,
                       "scale output", error) ||
        !validate_view(context, buffers.xn16, output_bytes, kBFloatBytes,
                       "BF16 output", error)) {
        return false;
    }
    if (write_low &&
        !validate_view(context, buffers.xn16_lo, output_bytes, kBFloatBytes,
                       "BF16 remainder output", error)) {
        return false;
    }

    const std::array inputs{buffers.residual, buffers.norm};
    std::array<BufferRef, 3> outputs{
        buffers.rs, buffers.xn16,
        write_low ? buffers.xn16_lo : BufferRef{},
    };
    const std::size_t output_count = write_low ? 3u : 2u;
    for (std::size_t out = 0u; out < output_count; ++out) {
        for (const BufferRef& input : inputs) {
            if (views_overlap(outputs[out], input)) {
                error = "Prefill GR norm output overlaps a read-only input";
                return false;
            }
        }
        for (std::size_t earlier = 0u; earlier < out; ++earlier) {
            if (views_overlap(outputs[out], outputs[earlier])) {
                error = "Prefill GR norm output buffer views overlap";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_gr_norm(Context& context, Encoder& encoder,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRNormBuffers& buffers,
                            std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GR norm requires an ordered concurrent-group boundary";
        return false;
    }

    std::size_t residual_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!validate_dispatch(context, rows, mode, buffers, residual_bytes,
                           output_bytes, error)) {
        return false;
    }

    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const std::size_t row_values =
        static_cast<std::size_t>(kStreams) * kWidth;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(rows) * kStreams * kFloatBytes;
    const std::size_t norm_bytes = row_values * kFloatBytes;
    const std::array<EncoderBufferBinding, 5> bindings{{
        {buffers.residual, 0u, residual_bytes, kFloatBytes},
        {buffers.norm, 1u, norm_bytes, kFloatBytes},
        {buffers.rs, 2u, scale_bytes, kFloatBytes},
        {buffers.xn16, 3u, output_bytes, kBFloatBytes},
        {buffers.xn16_lo, 4u, output_bytes, kBFloatBytes},
    }};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        write_low ? "strata_pf_gr_norm_lo" : "strata_pf_gr_norm",
        {static_cast<std::size_t>(rows) * kStreams, 1u, 1u},
        {kThreads, 1u, 1u},
        std::span<const EncoderBufferBinding>(bindings.data(),
                                              write_low ? bindings.size()
                                                        : bindings.size() - 1u),
        {}, threadgroup_memory, error);
}

bool encode_prefill_gr_norm(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRNormBuffers& buffers,
                            std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GR normalization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_norm(context, encoder, rows, mode,
                                          buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
