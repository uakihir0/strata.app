// Validated encoder for the isolated Prefill BF16 MPP GEMM component.
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
constexpr std::uint32_t kThreads = 128u;
constexpr std::size_t kBF16Bytes = 2u;
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
                        std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill BF16 MPP GEMM " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes ||
        view.offset() % kBF16Bytes != 0u) {
        error = "Prefill BF16 MPP GEMM " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill BF16 MPP GEMM y buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % kFloatBytes != 0u) {
        error = "Prefill BF16 MPP GEMM y buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_dispatch(Context& context,
                       const PrefillGemmBF16MPPParams& params,
                       const PrefillGemmBF16MPPBuffers& buffers,
                       std::size_t& x_bytes, std::size_t& w_bytes,
                       std::size_t& y_bytes, std::string& error) {
    if (!supports_apple10(context)) {
        error = "Prefill BF16 MPP GEMM requires an Apple10 GPU";
        return false;
    }
    if (params.rows == 0u || params.rows > kPrefillMaxShaderRows) {
        error = "Prefill BF16 MPP GEMM row count exceeds the shader-safe "
                "range";
        return false;
    }
    if (params.n == 0u || params.k < 32u || params.k % 32u != 0u) {
        error = "Prefill BF16 MPP GEMM requires n > 0 and K divisible by 32";
        return false;
    }
    const std::uint64_t output_column_end =
        static_cast<std::uint64_t>(params.col0) + params.n;
    if (params.n > kMaxSignedExtent || params.k > kMaxSignedExtent ||
        params.ldy > kMaxSignedExtent ||
        output_column_end > kMaxSignedExtent ||
        params.ldy < output_column_end) {
        error = "Prefill BF16 MPP GEMM output extent or stride is invalid";
        return false;
    }
    if (!checked_matrix_bytes(params.rows, params.k, kBF16Bytes, x_bytes) ||
        !checked_matrix_bytes(params.n, params.k, kBF16Bytes, w_bytes)) {
        error = "Prefill BF16 MPP GEMM input buffer size overflows size_t";
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
        error = "Prefill BF16 MPP GEMM output buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.x, x_bytes, "X", error) ||
        !validate_read_view(context, buffers.w, w_bytes, "W", error) ||
        !validate_write_view(context, buffers.y, y_bytes, error)) {
        return false;
    }
    if (overlaps(buffers.x, buffers.y, x_bytes, y_bytes) ||
        overlaps(buffers.w, buffers.y, w_bytes, y_bytes)) {
        error = "Prefill BF16 MPP GEMM y overlaps a read-only input";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_gemm_bf16_mpp(
    Context& context, Encoder& encoder,
    const PrefillGemmBF16MPPParams& params,
    const PrefillGemmBF16MPPBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill BF16 MPP GEMM requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    std::size_t x_bytes = 0u;
    std::size_t w_bytes = 0u;
    std::size_t y_bytes = 0u;
    if (!validate_dispatch(context, params, buffers, x_bytes, w_bytes, y_bytes,
                           error)) {
        return false;
    }

    const EncoderBufferBinding output{buffers.y, 2u, y_bytes, kFloatBytes};
    const EncoderReadOnlyBufferBinding inputs[] = {
        {buffers.x, 0u, x_bytes, kBF16Bytes},
        {buffers.w, 1u, w_bytes, kBF16Bytes},
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
        params.rows / kTile +
        static_cast<std::size_t>(params.rows % kTile != 0u);
    return encoder.dispatch(
        "strata_pf_gemm_bf16_mpp", {grid_x, grid_y, 1u},
        {kThreads, 1u, 1u}, std::span<const EncoderBufferBinding>(&output, 1u),
        inputs, scalar_bindings, {}, error);
}

bool encode_prefill_gemm_bf16_mpp(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmBF16MPPParams& params,
    const PrefillGemmBF16MPPBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill BF16 MPP GEMM",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gemm_bf16_mpp(
                context, encoder, params, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
