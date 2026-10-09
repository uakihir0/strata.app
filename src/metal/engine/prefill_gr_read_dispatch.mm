// Ordered Prefill GR normalization and hyper-connection read projections.
#include "strata/metal/prefill_gr_read_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_gemm_dispatch.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_gr_silu_dispatch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::size_t kStreams = 4u;
constexpr std::size_t kWidth = 2560u;
constexpr std::size_t kLowRankWidth = 320u;
constexpr std::size_t kBFloatBytes = sizeof(std::uint16_t);
constexpr std::size_t kFloatBytes = sizeof(float);

struct MemoryRange {
    void* handle = nullptr;
    std::uintptr_t cpu_address = 0u;
    std::uint64_t gpu_address = 0u;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
};

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t& result) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool matrix_bytes(std::uint32_t rows, std::size_t columns,
                 std::size_t element_bytes, std::size_t& bytes) {
    std::size_t elements = 0u;
    return checked_multiply(rows, columns, elements) &&
           checked_multiply(elements, element_bytes, bytes);
}

bool validate_view(Context& context, BufferRef view,
                   std::size_t required_bytes, std::size_t alignment,
                   std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GR read " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || alignment == 0u ||
        view.offset % alignment != 0u) {
        error = "Prefill GR read " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_view(Context& context, ReadOnlyBufferRef view,
                   std::size_t required_bytes, std::size_t alignment,
                   std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GR read " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || alignment == 0u ||
        view.offset() % alignment != 0u) {
        error = "Prefill GR read " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

MemoryRange memory_range(BufferRef view, std::size_t bytes) {
    return {view.handle, reinterpret_cast<std::uintptr_t>(view.contents),
            view.gpu_address, view.offset, bytes};
}

MemoryRange memory_range(ReadOnlyBufferRef view, std::size_t bytes) {
    const void* const contents = view.contents();
    return {nullptr, reinterpret_cast<std::uintptr_t>(contents),
            view.gpu_address(), view.offset(), bytes};
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool ranges_overlap(const MemoryRange& left, const MemoryRange& right) {
    const bool same_allocation =
        left.handle != nullptr && left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.bytes, right.offset,
                               right.bytes);
    const bool same_cpu_backing =
        address_ranges_overlap(left.cpu_address, left.bytes,
                               right.cpu_address, right.bytes);
    const bool same_gpu_backing =
        address_ranges_overlap(left.gpu_address, left.bytes,
                               right.gpu_address, right.bytes);
    return same_allocation || same_cpu_backing || same_gpu_backing;
}

bool validate_buffers(Context& context, std::uint32_t rows,
                      PrefillBF16X2Mode mode,
                      const PrefillGRReadBuffers& buffers,
                      std::array<std::size_t, 8>& output_bytes,
                      std::array<std::size_t, 5>& input_bytes,
                      std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GR read row count exceeds the shader-safe range";
        return false;
    }
    const std::uint32_t mode_value = static_cast<std::uint32_t>(mode);
    if (mode_value > static_cast<std::uint32_t>(
                         PrefillBF16X2Mode::except_gr)) {
        error = "Prefill GR read BF16X2 mode is invalid";
        return false;
    }
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    if (write_low != static_cast<bool>(buffers.xn16_lo) ||
        write_low != static_cast<bool>(buffers.low_rank_bf16_lo)) {
        error = write_low
                    ? "Prefill GR read mode 1 requires both remainder outputs"
                    : "Prefill GR read remainder outputs are only valid in mode 1";
        return false;
    }

    constexpr std::size_t normalized_values = kStreams * kWidth;
    constexpr std::size_t gated_values = kStreams * kWidth;
    if (!matrix_bytes(rows, normalized_values, kFloatBytes, input_bytes[0]) ||
        !checked_multiply(normalized_values, kFloatBytes, input_bytes[1]) ||
        !checked_multiply(320u * 10240u, kBFloatBytes, input_bytes[2]) ||
        !checked_multiply(10240u * 320u, kBFloatBytes, input_bytes[3]) ||
        !checked_multiply(4u * 10240u, kBFloatBytes, input_bytes[4]) ||
        !matrix_bytes(rows, kStreams, kFloatBytes, output_bytes[0]) ||
        !matrix_bytes(rows, normalized_values, kBFloatBytes,
                      output_bytes[1]) ||
        !matrix_bytes(rows, normalized_values, kBFloatBytes,
                      output_bytes[2]) ||
        !matrix_bytes(rows, kLowRankWidth, kFloatBytes, output_bytes[3]) ||
        !matrix_bytes(rows, kLowRankWidth, kBFloatBytes, output_bytes[4]) ||
        !matrix_bytes(rows, kLowRankWidth, kBFloatBytes, output_bytes[5]) ||
        !matrix_bytes(rows, gated_values, kFloatBytes, output_bytes[6]) ||
        !matrix_bytes(rows, kStreams, kFloatBytes, output_bytes[7])) {
        error = "Prefill GR read buffer size overflows size_t";
        return false;
    }

    const std::array<BufferRef, 8> writable{
        buffers.rs, buffers.xn16, buffers.xn16_lo, buffers.low_rank,
        buffers.low_rank_bf16, buffers.low_rank_bf16_lo, buffers.gated,
        buffers.inject};
    const std::array<std::size_t, 8> output_alignments{
        kFloatBytes, kBFloatBytes, kBFloatBytes, kFloatBytes,
        kBFloatBytes, kBFloatBytes, kFloatBytes, kFloatBytes};
    const std::array<std::string_view, 8> output_names{
        "scale output", "normalized BF16 output",
        "normalized BF16 remainder", "down projection output",
        "SiLU BF16 output", "SiLU BF16 remainder", "up projection output",
        "injection output"};
    constexpr std::array<std::size_t, 6> kOutputsWithoutRemainders{
        0u, 1u, 3u, 4u, 6u, 7u};
    const std::size_t active_outputs =
        write_low ? output_bytes.size() : kOutputsWithoutRemainders.size();
    const auto output_index = [&](std::size_t active_index) {
        return write_low ? active_index
                         : kOutputsWithoutRemainders[active_index];
    };
    for (std::size_t active_index = 0u;
         active_index < active_outputs; ++active_index) {
        const std::size_t index = output_index(active_index);
        if (!validate_view(context, writable[index], output_bytes[index],
                           output_alignments[index], output_names[index],
                           error)) {
            return false;
        }
    }
    if (!validate_view(context, buffers.residual, input_bytes[0], kFloatBytes,
                       "residual", error) ||
        !validate_view(context, buffers.norm, input_bytes[1], kFloatBytes,
                       "norm weights", error) ||
        !validate_view(context, buffers.down_weights, input_bytes[2],
                       kBFloatBytes, "down weights", error) ||
        !validate_view(context, buffers.up_weights, input_bytes[3],
                       kBFloatBytes, "up weights", error) ||
        !validate_view(context, buffers.inject_weights, input_bytes[4],
                       kBFloatBytes, "injection weights", error)) {
        return false;
    }

    const std::array<MemoryRange, 5> read_only{
        memory_range(buffers.residual, input_bytes[0]),
        memory_range(buffers.norm, input_bytes[1]),
        memory_range(buffers.down_weights, input_bytes[2]),
        memory_range(buffers.up_weights, input_bytes[3]),
        memory_range(buffers.inject_weights, input_bytes[4])};
    std::array<MemoryRange, 8> outputs{};
    for (std::size_t active_index = 0u;
         active_index < active_outputs; ++active_index) {
        const std::size_t index = output_index(active_index);
        outputs[active_index] =
            memory_range(writable[index], output_bytes[index]);
    }

    for (std::size_t output = 0u; output < active_outputs; ++output) {
        for (const MemoryRange& input : read_only) {
            if (ranges_overlap(outputs[output], input)) {
                error = "Prefill GR read writable output overlaps a read-only input";
                return false;
            }
        }
        for (std::size_t earlier = 0u; earlier < output; ++earlier) {
            if (ranges_overlap(outputs[output], outputs[earlier])) {
                error = "Prefill GR read writable output views overlap";
                return false;
            }
        }
    }
    return true;
}

bool encode_projection(Context& context, Encoder& encoder,
                       const PrefillGemmMPPParams& params,
                       ReadOnlyBufferRef high, ReadOnlyBufferRef low,
                       ReadOnlyBufferRef weights, BufferRef output,
                       PrefillBF16X2Mode mode, std::string& error) {
    if (mode == PrefillBF16X2Mode::all_projections) {
        return encode_prefill_gemm_bf16x2(
            context, encoder, params,
            {high, low, weights, output}, error);
    }
    return encode_prefill_gemm(
        context, encoder, params, {high, weights, output},
        PrefillGemmFormat::BF16, error);
}

}  // namespace

bool encode_prefill_gr_read(Context& context, Encoder& encoder,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRReadBuffers& buffers,
                            std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GR read requires an ordered concurrent-group boundary";
        return false;
    }

    std::array<std::size_t, 8> output_bytes{};
    std::array<std::size_t, 5> input_bytes{};
    if (!validate_buffers(context, rows, mode, buffers, output_bytes,
                          input_bytes, error)) {
        return false;
    }

    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const auto xn16 = context.buffers().read_only(buffers.xn16);
    const auto xn16_lo = write_low
                             ? context.buffers().read_only(buffers.xn16_lo)
                             : ReadOnlyBufferRef{};
    const auto low_rank_bf16 =
        context.buffers().read_only(buffers.low_rank_bf16);
    const auto low_rank_bf16_lo =
        write_low ? context.buffers().read_only(buffers.low_rank_bf16_lo)
                  : ReadOnlyBufferRef{};

    const PrefillGRNormBuffers norm_buffers{
        buffers.residual, buffers.norm, buffers.rs, buffers.xn16,
        write_low ? buffers.xn16_lo : BufferRef{}};
    if (!encode_prefill_gr_norm(context, encoder, rows, mode, norm_buffers,
                                error)) {
        return false;
    }

    const PrefillGemmMPPParams down_params{
        rows, 320u, 10240u, 320u, 0u, false};
    if (!encode_projection(context, encoder, down_params, xn16, xn16_lo,
                           buffers.down_weights, buffers.low_rank, mode,
                           error)) {
        return false;
    }

    const PrefillGRSiluBuffers silu_buffers{
        context.buffers().read_only(buffers.low_rank),
        buffers.low_rank_bf16,
        write_low ? buffers.low_rank_bf16_lo : BufferRef{}};
    if (!encode_prefill_gr_silu(context, encoder, rows, mode, silu_buffers,
                                error)) {
        return false;
    }

    const PrefillGemmMPPParams up_params{
        rows, 10240u, 320u, 10240u, 0u, false};
    if (!encode_projection(context, encoder, up_params, low_rank_bf16,
                           low_rank_bf16_lo, buffers.up_weights,
                           buffers.gated, mode, error)) {
        return false;
    }

    const PrefillGemmMPPParams inject_params{
        rows, 4u, 10240u, 4u, 0u, false};
    const auto inject_high =
        context.buffers().read_only(buffers.xn16);
    const auto inject_low =
        write_low ? context.buffers().read_only(buffers.xn16_lo)
                  : ReadOnlyBufferRef{};
    return encode_projection(context, encoder, inject_params, inject_high,
                             inject_low, buffers.inject_weights,
                             buffers.inject, mode, error);
}

bool encode_prefill_gr_read(Context& context,
                            CommandBuffer& command_buffer,
                            std::uint32_t rows, PrefillBF16X2Mode mode,
                            const PrefillGRReadBuffers& buffers,
                            std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GR read",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_read(context, encoder, rows, mode,
                                          buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
