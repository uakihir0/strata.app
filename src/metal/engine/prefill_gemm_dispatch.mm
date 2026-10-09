// Validated encoders for isolated Prefill F16/BF16 GEMM components.
#include "strata/metal/prefill_gemm_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kTile = 64u;
constexpr std::uint32_t kSimdRows = 32u;
constexpr std::uint32_t kThreads = 128u;
constexpr std::size_t kHalfBytes = 2u;
constexpr std::size_t kFloatBytes = 4u;
constexpr std::uint32_t kMaxSignedExtent =
    static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());

bool supports_apple10(const Context& context) {
    const auto& families = context.device_facts().families;
    return std::find(families.begin(), families.end(), "apple10") !=
           families.end();
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t& result) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_matrix_bytes(std::uint32_t rows, std::uint32_t columns,
                          std::size_t element_bytes, std::size_t& result) {
    std::size_t elements = 0u;
    return checked_multiply(static_cast<std::size_t>(rows),
                            static_cast<std::size_t>(columns), elements) &&
           checked_multiply(elements, element_bytes, result);
}

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                    Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
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
                        std::size_t required_bytes, std::string_view name,
                        std::string_view component, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = std::string(component) + " " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || view.offset() % kHalfBytes != 0u) {
        error = std::string(component) + " " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes,
                         std::string_view component, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = std::string(component) +
                " y buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % kFloatBytes != 0u) {
        error = std::string(component) +
                " y buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_dispatch(Context& context,
                       const PrefillGemmMPPParams& params,
                       const PrefillGemmMPPBuffers& buffers,
                       std::string_view component,
                       bool require_apple10,
                       std::size_t input_element_bytes,
                       std::size_t& x_bytes, std::size_t& w_bytes,
                       std::size_t& y_bytes, std::string& error) {
    if (require_apple10 && !supports_apple10(context)) {
        error = std::string(component) + " requires an Apple10 GPU";
        return false;
    }
    if (params.rows == 0u || params.rows > kPrefillMaxShaderRows) {
        error = std::string(component) +
                " row count exceeds the shader-safe range";
        return false;
    }
    if (params.n == 0u || params.k < 32u || params.k % 32u != 0u) {
        error = std::string(component) +
                " requires n > 0 and K divisible by 32";
        return false;
    }
    const std::uint64_t output_column_end =
        static_cast<std::uint64_t>(params.col0) + params.n;
    if (params.n > kMaxSignedExtent || params.k > kMaxSignedExtent ||
        params.ldy > kMaxSignedExtent ||
        output_column_end > kMaxSignedExtent ||
        params.ldy < output_column_end) {
        error = std::string(component) + " output extent or stride is invalid";
        return false;
    }
    if (!checked_matrix_bytes(params.rows, params.k, input_element_bytes,
                              x_bytes) ||
        !checked_matrix_bytes(params.n, params.k, input_element_bytes,
                              w_bytes)) {
        error = std::string(component) +
                " input buffer size overflows size_t";
        return false;
    }
    std::size_t last_row_offset = 0u;
    if (!checked_multiply(static_cast<std::size_t>(params.rows - 1u),
                          static_cast<std::size_t>(params.ldy),
                          last_row_offset) ||
        output_column_end >
            std::numeric_limits<std::size_t>::max() - last_row_offset ||
        !checked_multiply(
            last_row_offset + static_cast<std::size_t>(output_column_end),
            kFloatBytes, y_bytes)) {
        error = std::string(component) +
                " output buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.x, x_bytes, "X", component,
                            error) ||
        !validate_read_view(context, buffers.w, w_bytes, "W", component,
                            error) ||
        !validate_write_view(context, buffers.y, y_bytes, component, error)) {
        return false;
    }
    if (overlaps(buffers.x, buffers.y, x_bytes, y_bytes) ||
        overlaps(buffers.w, buffers.y, w_bytes, y_bytes)) {
        error = std::string(component) + " y overlaps a read-only input";
        return false;
    }
    return true;
}

bool encode_prefill_gemm_component(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string_view component,
    std::string_view pipeline, bool require_apple10,
    std::size_t input_element_bytes, std::uint32_t tile_rows,
    std::string& error) {
    if (encoder.concurrent_group_open()) {
        error = std::string(component) +
                " requires an ordered concurrent-group boundary";
        return false;
    }

    std::size_t x_bytes = 0u;
    std::size_t w_bytes = 0u;
    std::size_t y_bytes = 0u;
    if (!validate_dispatch(context, params, buffers, component,
                           require_apple10, input_element_bytes, x_bytes,
                           w_bytes, y_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding output{buffers.y, 2u, y_bytes, kFloatBytes};
    const EncoderReadOnlyBufferBinding inputs[] = {
        {buffers.x, 0u, x_bytes, input_element_bytes},
        {buffers.w, 1u, w_bytes, input_element_bytes},
    };
    const std::array<std::uint32_t, 6u> scalar_values{
        params.rows, params.n, params.k, params.ldy, params.col0,
        params.accumulate ? 1u : 0u};
    std::array<EncoderBytesBinding, scalar_values.size()> scalar_bindings{};
    for (std::size_t i = 0u; i < scalar_values.size(); ++i) {
        scalar_bindings[i] = {
            &scalar_values[i], sizeof(scalar_values[i]),
            static_cast<std::uint32_t>(i + 3u)};
    }
    const std::size_t grid_x =
        params.n / kTile + static_cast<std::size_t>(params.n % kTile != 0u);
    const std::size_t grid_y =
        params.rows / tile_rows +
        static_cast<std::size_t>(params.rows % tile_rows != 0u);
    return encoder.dispatch(
        pipeline, {grid_x, grid_y, 1u}, {kThreads, 1u, 1u},
        std::span<const EncoderBufferBinding>(&output, 1u), inputs,
        scalar_bindings, {}, error);
}

bool encode_prefill_gemm_component(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string_view component,
    std::string_view pipeline, bool require_apple10,
    std::size_t input_element_bytes, std::uint32_t tile_rows,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, component,
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gemm_component(
                context, encoder, params, buffers, component, pipeline,
                require_apple10, input_element_bytes, tile_rows, encode_error);
        },
        error);
}

}  // namespace

bool encode_prefill_gemm_bf16_mpp(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, encoder, params, buffers, "Prefill BF16 MPP GEMM",
        "strata_pf_gemm_bf16_mpp", true, kHalfBytes, kTile, error);
}

bool encode_prefill_gemm_bf16_mpp(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, command_buffer, params, buffers,
        "Prefill BF16 MPP GEMM", "strata_pf_gemm_bf16_mpp", true,
        kHalfBytes, kTile, error);
}

bool encode_prefill_gemm_f16_mpp(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, encoder, params, buffers, "Prefill F16 MPP GEMM",
        "strata_pf_gemm_f16_mpp", true, kHalfBytes, kTile, error);
}

bool encode_prefill_gemm_f16_mpp(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, command_buffer, params, buffers, "Prefill F16 MPP GEMM",
        "strata_pf_gemm_f16_mpp", true, kHalfBytes, kTile, error);
}

bool encode_prefill_gemm_f16_simd(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, encoder, params, buffers, "Prefill F16 SIMD GEMM",
        "strata_pf_gemm_f16_simd", false, kHalfBytes, kSimdRows, error);
}

bool encode_prefill_gemm_f16_simd(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, command_buffer, params, buffers, "Prefill F16 SIMD GEMM",
        "strata_pf_gemm_f16_simd", false, kHalfBytes, kSimdRows, error);
}

bool encode_prefill_gemm_bf16_simd(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, encoder, params, buffers, "Prefill BF16 SIMD GEMM",
        "strata_pf_gemm_bf16_simd", false, kHalfBytes, kSimdRows, error);
}

bool encode_prefill_gemm_bf16_simd(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params,
    const PrefillGemmMPPBuffers& buffers, std::string& error) {
    error.clear();
    return encode_prefill_gemm_component(
        context, command_buffer, params, buffers, "Prefill BF16 SIMD GEMM",
        "strata_pf_gemm_bf16_simd", false, kHalfBytes, kSimdRows, error);
}

}  // namespace strata::metal
