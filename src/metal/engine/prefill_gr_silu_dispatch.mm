// Validated encoder for Prefill's GR hyper-connection SiLU activation.
#include "strata/metal/prefill_gr_silu_dispatch.hpp"
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

constexpr std::uint32_t kLowRankWidth = 320u;
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

bool validate_read_view(Context& context, ReadOnlyBufferRef buffer,
                        std::size_t required_bytes, std::size_t alignment,
                        std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "Prefill GR SiLU " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length() < required_bytes || alignment == 0u ||
        buffer.offset() % alignment != 0u) {
        error = "Prefill GR SiLU " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef buffer,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "Prefill GR SiLU " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes || alignment == 0u ||
        buffer.offset % alignment != 0u) {
        error = "Prefill GR SiLU " + std::string(name) +
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
                       const PrefillGRSiluBuffers& buffers,
                       std::size_t& input_bytes,
                       std::size_t& output_bytes, std::string& error) {
    const std::uint32_t mode_value = static_cast<std::uint32_t>(mode);
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GR SiLU row count exceeds the shader-safe range";
        return false;
    }
    if (mode_value > static_cast<std::uint32_t>(
                         PrefillBF16X2Mode::except_gr)) {
        error = "Prefill GR SiLU BF16X2 mode is invalid";
        return false;
    }
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    if (write_low != static_cast<bool>(buffers.low_bf16)) {
        error = write_low
                    ? "Prefill GR SiLU mode 1 requires low_bf16"
                    : "Prefill GR SiLU low_bf16 is only valid in mode 1";
        return false;
    }

    const std::size_t row_values =
        static_cast<std::size_t>(kLowRankWidth);
    if (!checked_bytes(rows, row_values * kFloatBytes, input_bytes) ||
        !checked_bytes(rows, row_values * kBFloatBytes, output_bytes)) {
        error = "Prefill GR SiLU buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.low_rank, input_bytes,
                            kFloatBytes, "low-rank input", error) ||
        !validate_write_view(context, buffers.high_bf16, output_bytes,
                             kBFloatBytes, "BF16 output", error)) {
        return false;
    }
    if (write_low &&
        !validate_write_view(context, buffers.low_bf16, output_bytes,
                             kBFloatBytes, "BF16 remainder output", error)) {
        return false;
    }
    if (write_low &&
        write_views_overlap(buffers.high_bf16, buffers.low_bf16)) {
        error = "Prefill GR SiLU output buffer views overlap";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_gr_silu(Context& context, Encoder& encoder,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRSiluBuffers& buffers,
                            std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GR SiLU requires an ordered concurrent-group boundary";
        return false;
    }

    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!validate_dispatch(context, rows, mode, buffers, input_bytes,
                           output_bytes, error)) {
        return false;
    }

    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const std::array<EncoderBufferBinding, 2> writable{{
        EncoderBufferBinding{buffers.high_bf16, 1u, output_bytes,
                             kBFloatBytes},
        EncoderBufferBinding{buffers.low_bf16, 2u, output_bytes,
                             kBFloatBytes},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 1> read_only{{
        EncoderReadOnlyBufferBinding{buffers.low_rank, 0u, input_bytes,
                                     kFloatBytes},
    }};
    const std::uint32_t row_count = rows;
    const std::array bytes{
        EncoderBytesBinding{&row_count, sizeof(row_count), 3u},
    };
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    const std::size_t values =
        static_cast<std::size_t>(rows) * kLowRankWidth;
    return encoder.dispatch(
        write_low ? "strata_pf_gr_silu_lo" : "strata_pf_gr_silu",
        {(values + kThreads - 1u) / kThreads, 1u, 1u},
        {kThreads, 1u, 1u},
        std::span<const EncoderBufferBinding>(writable.data(),
                                              write_low ? writable.size()
                                                        : writable.size() - 1u),
        read_only, bytes, threadgroup_memory, error);
}

bool encode_prefill_gr_silu(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRSiluBuffers& buffers,
                            std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GR SiLU",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_silu(context, encoder, rows, mode,
                                          buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
