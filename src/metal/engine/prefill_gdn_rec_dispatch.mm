// Validated encoder for Prefill's GDN recurrent state update.
#include "strata/metal/prefill_gdn_rec_dispatch.hpp"

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

constexpr std::uint32_t kThreads = 128u;
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kStateElements = GDN_STATE_ELEMENTS;

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

bool overlaps(BufferRef left, BufferRef right, std::size_t left_bytes,
              std::size_t right_bytes) {
    const bool same_allocation =
        left.handle == right.handle &&
        ranges_overlap(left.offset, left_bytes, right.offset, right_bytes);
    const bool same_cpu_backing =
        left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.contents),
                       left_bytes,
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right_bytes);
    const bool same_gpu_backing =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address, left_bytes, right.gpu_address,
                       right_bytes);
    return same_allocation || same_cpu_backing || same_gpu_backing;
}

bool overlaps(ReadOnlyBufferRef input, BufferRef output,
              std::size_t input_bytes, std::size_t output_bytes) {
    const void* const input_contents = input.contents();
    const bool same_cpu_backing =
        input_contents != nullptr && output.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(input_contents),
                       input_bytes,
                       reinterpret_cast<std::uintptr_t>(output.contents),
                       output_bytes);
    const bool same_gpu_backing =
        input.gpu_address() != 0u && output.gpu_address != 0u &&
        ranges_overlap(input.gpu_address(), input_bytes, output.gpu_address,
                       output_bytes);
    return same_cpu_backing || same_gpu_backing;
}

bool validate_read_view(Context& context, ReadOnlyBufferRef view,
                        std::size_t required_bytes, std::string_view name,
                        std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN recurrence " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes ||
        view.offset() % kFloatBytes != 0u) {
        error = "Prefill GDN recurrence " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::string_view name,
                         std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN recurrence " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % kFloatBytes != 0u) {
        error = "Prefill GDN recurrence " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       const PrefillGDNRecBuffers& buffers,
                       std::size_t& state_bytes, std::size_t& h_bytes,
                       std::size_t& head_bytes, std::size_t& output_bytes,
                       std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error =
            "Prefill GDN recurrence row count exceeds the shader-safe range";
        return false;
    }
    if (!checked_float_bytes(1u, static_cast<std::uint32_t>(kStateElements),
                             state_bytes) ||
        !checked_float_bytes(rows, GDN_CONV_CHANNELS, h_bytes) ||
        !checked_float_bytes(rows, GDN_V_HEADS, head_bytes) ||
        !checked_float_bytes(rows, GDN_VALUE_DIM, output_bytes)) {
        error = "Prefill GDN recurrence buffer size overflows size_t";
        return false;
    }
    if (!validate_write_view(context, buffers.state, state_bytes, "state",
                             error) ||
        !validate_read_view(context, buffers.h, h_bytes, "h", error) ||
        !validate_read_view(context, buffers.gate, head_bytes, "gate",
                            error) ||
        !validate_read_view(context, buffers.beta, head_bytes, "beta",
                            error) ||
        !validate_write_view(context, buffers.output, output_bytes, "output",
                             error)) {
        return false;
    }

    const std::array<ReadOnlyBufferRef, 3u> inputs{
        buffers.h, buffers.gate, buffers.beta};
    const std::array<std::size_t, 3u> input_bytes{
        h_bytes, head_bytes, head_bytes};
    const std::array<BufferRef, 2u> outputs{buffers.state, buffers.output};
    const std::array<std::size_t, 2u> output_bytes_by_view{
        state_bytes, output_bytes};
    for (std::size_t out = 0u; out < outputs.size(); ++out) {
        for (std::size_t in = 0u; in < inputs.size(); ++in) {
            if (overlaps(inputs[in], outputs[out], input_bytes[in],
                         output_bytes_by_view[out])) {
                error = "Prefill GDN recurrence writable view overlaps a "
                        "read-only input";
                return false;
            }
        }
        for (std::size_t earlier = 0u; earlier < out; ++earlier) {
            if (overlaps(outputs[earlier], outputs[out],
                         output_bytes_by_view[earlier],
                         output_bytes_by_view[out])) {
                error = "Prefill GDN recurrence writable views overlap";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_gdn_rec(Context& context, Encoder& encoder,
                            std::uint32_t rows,
                            const PrefillGDNRecBuffers& buffers,
                            std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GDN recurrence requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    std::size_t state_bytes = 0u;
    std::size_t h_bytes = 0u;
    std::size_t head_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!validate_dispatch(context, rows, buffers, state_bytes, h_bytes,
                           head_bytes, output_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding outputs[] = {
        {buffers.state, 0u, state_bytes, kFloatBytes},
        {buffers.output, 4u, output_bytes, kFloatBytes},
    };
    const EncoderReadOnlyBufferBinding inputs[] = {
        {buffers.h, 1u, h_bytes, kFloatBytes},
        {buffers.gate, 2u, head_bytes, kFloatBytes},
        {buffers.beta, 3u, head_bytes, kFloatBytes},
    };
    const EncoderBytesBinding row_binding{&rows, sizeof(rows), 5u};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{0u, GDN_HEAD_DIM * kFloatBytes},
        EncoderThreadgroupMemory{1u, GDN_HEAD_DIM * kFloatBytes},
        EncoderThreadgroupMemory{2u, 4u * 32u * kFloatBytes},
    };
    return encoder.dispatch(
        "strata_pf_gdn_rec", {GDN_V_HEADS * 4u, 1u, 1u},
        {kThreads, 1u, 1u}, outputs, inputs,
        std::span<const EncoderBytesBinding>(&row_binding, 1u),
        threadgroup_memory, error);
}

bool encode_prefill_gdn_rec(Context& context, CommandBuffer& command_buffer,
                            std::uint32_t rows,
                            const PrefillGDNRecBuffers& buffers,
                            std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN recurrence",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_rec(context, encoder, rows, buffers,
                                          encode_error);
        },
        error);
}

}  // namespace strata::metal
