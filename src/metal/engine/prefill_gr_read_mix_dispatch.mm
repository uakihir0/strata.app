// Ordered Prefill GR normalization, projections and four-stream mixer.
#include "strata/metal/prefill_gr_read_mix_dispatch.hpp"

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

MemoryRange memory_range(BufferRef view, std::size_t bytes) {
    return {view.handle, reinterpret_cast<std::uintptr_t>(view.contents),
            view.gpu_address, view.offset, bytes};
}

MemoryRange memory_range(ReadOnlyBufferRef view, std::size_t bytes) {
    return {nullptr, reinterpret_cast<std::uintptr_t>(view.contents()),
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

bool validate_mixer_input(Context& context, BufferRef view,
                          std::size_t required_bytes, std::size_t alignment,
                          std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GR read/mix " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || alignment == 0u ||
        view.offset % alignment != 0u) {
        error = "Prefill GR read/mix " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_mix_outputs(Context& context, std::uint32_t rows,
                          PrefillBF16X2Mode mode,
                          const PrefillGRReadMixBuffers& buffers,
                          std::string& error) {
    const std::uint32_t mode_value = static_cast<std::uint32_t>(mode);
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GR read/mix row count exceeds the shader-safe range";
        return false;
    }
    if (mode_value >
        static_cast<std::uint32_t>(PrefillBF16X2Mode::except_gr)) {
        error = "Prefill GR read/mix BF16X2 mode is invalid";
        return false;
    }
    const bool write_low = mode != PrefillBF16X2Mode::off;
    if (write_low != static_cast<bool>(buffers.mixed_bf_lo)) {
        error = write_low
                    ? "Prefill GR read/mix modes 1 and 2 require mixed_bf_lo"
                    : "Prefill GR read/mix mixed_bf_lo is only valid in modes 1 and 2";
        return false;
    }

    std::size_t output_values = 0u;
    std::size_t output_bytes = 0u;
    std::size_t stream_values = 0u;
    std::size_t stream_bytes = 0u;
    std::size_t low_rank_values = 0u;
    std::size_t low_rank_float_bytes = 0u;
    std::size_t low_rank_bf16_bytes = 0u;
    std::size_t scale_values = 0u;
    std::size_t scale_bytes = 0u;
    std::size_t norm_float_bytes = 0u;
    std::size_t down_weight_values = 0u;
    std::size_t down_weight_bytes = 0u;
    std::size_t up_weight_values = 0u;
    std::size_t up_weight_bytes = 0u;
    std::size_t inject_weight_values = 0u;
    std::size_t inject_weight_bytes = 0u;
    if (!checked_multiply(rows, kWidth, output_values) ||
        !checked_multiply(output_values, kBFloatBytes, output_bytes) ||
        !checked_multiply(rows, kStreams * kWidth, stream_values) ||
        !checked_multiply(stream_values, kFloatBytes, stream_bytes) ||
        !checked_multiply(rows, kLowRankWidth, low_rank_values) ||
        !checked_multiply(low_rank_values, kFloatBytes, low_rank_float_bytes) ||
        !checked_multiply(low_rank_values, kBFloatBytes,
                          low_rank_bf16_bytes) ||
        !checked_multiply(rows, kStreams, scale_values) ||
        !checked_multiply(scale_values, kFloatBytes, scale_bytes) ||
        !checked_multiply(kStreams * kWidth, kFloatBytes, norm_float_bytes) ||
        !checked_multiply(kLowRankWidth, kStreams * kWidth,
                          down_weight_values) ||
        !checked_multiply(down_weight_values, kBFloatBytes,
                          down_weight_bytes) ||
        !checked_multiply(kStreams * kWidth, kLowRankWidth,
                          up_weight_values) ||
        !checked_multiply(up_weight_values, kBFloatBytes, up_weight_bytes) ||
        !checked_multiply(kStreams, kStreams * kWidth,
                          inject_weight_values) ||
        !checked_multiply(inject_weight_values, kBFloatBytes,
                          inject_weight_bytes)) {
        error = "Prefill GR read/mix buffer size overflows size_t";
        return false;
    }

    const std::array<BufferRef, 3> mix_outputs{
        buffers.mixed_bf, buffers.mixed_bf_lo, buffers.mixed_h};
    const std::array<std::size_t, 3> alignments{
        kBFloatBytes, kBFloatBytes, kBFloatBytes};
    const std::array<std::string_view, 3> names{
        "mixed BF16 output", "mixed BF16 remainder output", "mixed F16 output"};
    const std::array<std::size_t, 3> active_without_low{0u, 2u, 0u};
    const std::size_t output_count = write_low ? 3u : 2u;
    const auto output_index = [&](std::size_t active) {
        return write_low ? active : active_without_low[active];
    };
    std::array<MemoryRange, 3> output_ranges{};
    for (std::size_t active = 0u; active < output_count; ++active) {
        const std::size_t index = output_index(active);
        const BufferRef output = mix_outputs[index];
        if (!output || !context.buffers().owns(output)) {
            error = "Prefill GR read/mix " + std::string(names[index]) +
                    " buffer is not owned by this Context";
            return false;
        }
        if (output.length < output_bytes ||
            output.offset % alignments[index] != 0u) {
            error = "Prefill GR read/mix " + std::string(names[index]) +
                    " buffer is undersized or misaligned";
            return false;
        }
        output_ranges[active] = memory_range(output, output_bytes);
        for (std::size_t earlier = 0u; earlier < active; ++earlier) {
            if (ranges_overlap(output_ranges[active], output_ranges[earlier])) {
                error = "Prefill GR read/mix writable output views overlap";
                return false;
            }
        }
    }

    const PrefillGRReadBuffers& read = buffers.read;
    if (!validate_mixer_input(context, read.residual, stream_bytes,
                              kFloatBytes, "residual", error) ||
        !validate_mixer_input(context, read.norm, norm_float_bytes,
                              kFloatBytes, "norm weights", error)) {
        return false;
    }
    const std::array<BufferRef, 8> read_outputs{
        read.rs, read.xn16, read.xn16_lo, read.low_rank,
        read.low_rank_bf16, read.low_rank_bf16_lo, read.gated, read.inject};
    const std::array<std::size_t, 8> read_output_bytes{
        scale_bytes, stream_values * kBFloatBytes,
        stream_values * kBFloatBytes, low_rank_float_bytes,
        low_rank_bf16_bytes, low_rank_bf16_bytes,
        stream_bytes, static_cast<std::size_t>(rows) * kStreams * kFloatBytes};
    constexpr std::array<std::size_t, 6> kOutputsWithoutRemainders{
        0u, 1u, 3u, 4u, 6u, 7u};
    const std::size_t read_output_count =
        mode == PrefillBF16X2Mode::all_projections
            ? read_outputs.size()
            : kOutputsWithoutRemainders.size();
    const auto read_output_index = [&](std::size_t active) {
        return mode == PrefillBF16X2Mode::all_projections
                   ? active
                   : kOutputsWithoutRemainders[active];
    };
    constexpr std::array<std::size_t, 5> kProjectionOutputIndices{
        3u, 4u, 5u, 6u, 7u};
    constexpr std::array<std::size_t, 4>
        kProjectionOutputsWithoutRemainders{3u, 4u, 6u, 7u};
    const std::size_t projection_output_count =
        mode == PrefillBF16X2Mode::all_projections
            ? kProjectionOutputIndices.size()
            : kProjectionOutputsWithoutRemainders.size();
    const std::array<MemoryRange, 2> mixer_inputs{
        memory_range(read.residual, stream_bytes),
        memory_range(read.norm, norm_float_bytes)};
    for (std::size_t active = 0u; active < projection_output_count; ++active) {
        const std::size_t index =
            mode == PrefillBF16X2Mode::all_projections
                ? kProjectionOutputIndices[active]
                : kProjectionOutputsWithoutRemainders[active];
        if (!read_outputs[index]) continue;
        const MemoryRange projection_output =
            memory_range(read_outputs[index], read_output_bytes[index]);
        for (const MemoryRange& mixer_input : mixer_inputs) {
            if (ranges_overlap(projection_output, mixer_input)) {
                error =
                    "Prefill GR read projection output overlaps a mixer read input";
                return false;
            }
        }
    }
    for (std::size_t active = 0u; active < output_count; ++active) {
        const MemoryRange& mix_output = output_ranges[active];
        for (std::size_t read_active = 0u;
             read_active < read_output_count; ++read_active) {
            const std::size_t index = read_output_index(read_active);
            const BufferRef read_output = read_outputs[index];
            if (read_output &&
                ranges_overlap(mix_output,
                               memory_range(read_output,
                                            read_output_bytes[index]))) {
                error = "Prefill GR read/mix mixer output overlaps a GR read output";
                return false;
            }
        }
    }

    const std::array<MemoryRange, 5> read_inputs{
        memory_range(read.residual, stream_bytes),
        memory_range(read.norm, norm_float_bytes),
        memory_range(read.down_weights, down_weight_bytes),
        memory_range(read.up_weights, up_weight_bytes),
        memory_range(read.inject_weights, inject_weight_bytes)};
    for (std::size_t active = 0u; active < output_count; ++active) {
        for (const MemoryRange& input : read_inputs) {
            if (ranges_overlap(output_ranges[active], input)) {
                error = "Prefill GR read/mix mixer output overlaps a read input";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

namespace {

bool encode_prefill_gr_read_mix_impl(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    bool pre_normalized, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "Prefill GR read/mix requires an ordered concurrent-group boundary";
        return false;
    }
    if (!validate_mix_outputs(context, rows, mode, buffers, error)) {
        return false;
    }
    const bool read_encoded =
        pre_normalized
            ? encode_prefill_gr_read_pre_normalized(
                  context, encoder, rows, mode, buffers.read, error)
            : encode_prefill_gr_read(context, encoder, rows, mode,
                                     buffers.read, error);
    if (!read_encoded) {
        return false;
    }

    const PrefillGRMixBuffers mix_buffers{
        context.buffers().read_only(buffers.read.residual),
        context.buffers().read_only(buffers.read.rs),
        context.buffers().read_only(buffers.read.norm),
        context.buffers().read_only(buffers.read.gated),
        buffers.mixed_bf,
        buffers.mixed_bf_lo,
        buffers.mixed_h};
    return encode_prefill_gr_mix(context, encoder, rows, mode, mix_buffers,
                                 error);
}

}  // namespace

bool encode_prefill_gr_read_mix(Context& context, Encoder& encoder,
                                std::uint32_t rows,
                                PrefillBF16X2Mode mode,
                                const PrefillGRReadMixBuffers& buffers,
                                std::string& error) {
    return encode_prefill_gr_read_mix_impl(context, encoder, rows, mode,
                                           buffers, false, error);
}

bool encode_prefill_gr_read_mix_pre_normalized(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    std::string& error) {
    return encode_prefill_gr_read_mix_impl(context, encoder, rows, mode,
                                           buffers, true, error);
}

bool encode_prefill_gr_read_mix(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GR read and mix",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_read_mix(context, encoder, rows, mode,
                                              buffers, encode_error);
        },
        error);
}

bool encode_prefill_gr_read_mix_pre_normalized(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadMixBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GR read from prepared norm and mix",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_read_mix_pre_normalized(
                context, encoder, rows, mode, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
