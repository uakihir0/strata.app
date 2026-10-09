// Validated encoder for Prefill GDN output normalization.
#include "strata/metal/prefill_gdn_out_norm_dispatch.hpp"

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
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);

bool checked_bytes(std::uint32_t rows, std::uint32_t values_per_row,
                   std::size_t element_bytes, std::size_t& bytes) {
    const std::size_t max_values =
        std::numeric_limits<std::size_t>::max() / element_bytes;
    if (values_per_row != 0u &&
        static_cast<std::size_t>(rows) > max_values / values_per_row) {
        return false;
    }
    bytes = static_cast<std::size_t>(rows) * values_per_row * element_bytes;
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
                        std::size_t required_bytes, std::size_t alignment,
                        std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN output normalization " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || view.offset() % alignment != 0u) {
        error = "Prefill GDN output normalization " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN output normalization y_h buffer is not owned "
                "by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % alignment != 0u) {
        error = "Prefill GDN output normalization y_h buffer is undersized "
                "or misaligned";
        return false;
    }
    return true;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       const PrefillGDNOutNormBuffers& buffers,
                       std::size_t& oc_bytes, std::size_t& z_bytes,
                       std::size_t& gamma_bytes, std::size_t& y_h_bytes,
                       std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GDN output normalization row count exceeds the "
                "shader-safe range";
        return false;
    }
    if (!checked_bytes(rows, GDN_VALUE_DIM, kFloatBytes, oc_bytes) ||
        !checked_bytes(rows, GDN_VALUE_DIM, kFloatBytes, z_bytes) ||
        !checked_bytes(1u, GDN_HEAD_DIM, kFloatBytes, gamma_bytes) ||
        !checked_bytes(rows, GDN_VALUE_DIM, kHalfBytes, y_h_bytes)) {
        error = "Prefill GDN output normalization buffer size overflows "
                "size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.oc, oc_bytes, kFloatBytes, "oc",
                            error) ||
        !validate_read_view(context, buffers.z, z_bytes, kFloatBytes, "z",
                            error) ||
        !validate_read_view(context, buffers.gamma, gamma_bytes, kFloatBytes,
                            "gamma", error) ||
        !validate_write_view(context, buffers.y_h, y_h_bytes, kHalfBytes,
                             error)) {
        return false;
    }

    const std::array<ReadOnlyBufferRef, 3u> inputs{
        buffers.oc, buffers.z, buffers.gamma};
    const std::array<std::size_t, 3u> input_bytes{
        oc_bytes, z_bytes, gamma_bytes};
    for (std::size_t i = 0u; i < inputs.size(); ++i) {
        if (overlaps(inputs[i], buffers.y_h, input_bytes[i], y_h_bytes)) {
            error = "Prefill GDN output normalization y_h overlaps a "
                    "read-only input";
            return false;
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_gdn_out_norm(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNOutNormBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GDN output normalization requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    std::size_t oc_bytes = 0u;
    std::size_t z_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    std::size_t y_h_bytes = 0u;
    if (!validate_dispatch(context, rows, buffers, oc_bytes, z_bytes,
                           gamma_bytes, y_h_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding output{
        buffers.y_h, 3u, y_h_bytes, kHalfBytes};
    const EncoderReadOnlyBufferBinding inputs[] = {
        {buffers.oc, 0u, oc_bytes, kFloatBytes},
        {buffers.z, 1u, z_bytes, kFloatBytes},
        {buffers.gamma, 2u, gamma_bytes, kFloatBytes},
    };
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{0u, 4u * kFloatBytes}};
    return encoder.dispatch(
        "strata_pf_gdn_out_norm", {rows, GDN_V_HEADS, 1u},
        {kThreads, 1u, 1u}, std::span<const EncoderBufferBinding>(&output, 1u),
        inputs, {}, threadgroup_memory, error);
}

bool encode_prefill_gdn_out_norm(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNOutNormBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN output normalization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_out_norm(context, encoder, rows, buffers,
                                               encode_error);
        },
        error);
}

}  // namespace strata::metal
