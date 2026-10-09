// Ordered native-weight dequantization and F16 Prefill projection.
#include "strata/metal/prefill_native_projection.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_dequant_dispatch.hpp"

#include "../kernels/common/types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

struct NativeFormatLayout {
    std::uint32_t values_per_block;
    std::size_t bytes_per_block;
};

bool format_layout(PrefillNativeWeightFormat format,
                   NativeFormatLayout& layout) {
    switch (format) {
        case PrefillNativeWeightFormat::Q2_0:
            layout = {64u, sizeof(block_q2_0)};
            return true;
        case PrefillNativeWeightFormat::IQ4_NL:
            layout = {32u, sizeof(block_iq4_nl)};
            return true;
        case PrefillNativeWeightFormat::Q4_0:
            layout = {32u, sizeof(block_q4_0)};
            return true;
        case PrefillNativeWeightFormat::Q5_0:
            layout = {32u, sizeof(block_q5_0)};
            return true;
        case PrefillNativeWeightFormat::Q8_0:
            layout = {32u, sizeof(block_q8_0)};
            return true;
        case PrefillNativeWeightFormat::Q3_K:
            layout = {256u, sizeof(block_q3_K)};
            return true;
        case PrefillNativeWeightFormat::IQ3_S:
            layout = {256u, sizeof(block_iq3_s)};
            return true;
        case PrefillNativeWeightFormat::IQ4_XS:
            layout = {256u, sizeof(block_iq4_xs)};
            return true;
        case PrefillNativeWeightFormat::Q4_K:
            layout = {256u, sizeof(block_q4_K)};
            return true;
        case PrefillNativeWeightFormat::Q5_K:
            layout = {256u, sizeof(block_q5_K)};
            return true;
        case PrefillNativeWeightFormat::Q6_K:
            layout = {256u, sizeof(block_q6_K)};
            return true;
    }
    return false;
}

bool encode_dequant(Context& context, Encoder& encoder,
                    const PrefillDequantParams& params,
                    const PrefillDequantBuffers& buffers,
                    PrefillNativeWeightFormat format, std::string& error) {
    switch (format) {
        case PrefillNativeWeightFormat::Q2_0:
            return encode_prefill_dequant_q2_0(context, encoder, params,
                                               buffers, error);
        case PrefillNativeWeightFormat::IQ4_NL:
            return encode_prefill_dequant_iq4_nl(context, encoder, params,
                                                  buffers, error);
        case PrefillNativeWeightFormat::Q4_0:
            return encode_prefill_dequant_q4_0(context, encoder, params,
                                               buffers, error);
        case PrefillNativeWeightFormat::Q5_0:
            return encode_prefill_dequant_q5_0(context, encoder, params,
                                               buffers, error);
        case PrefillNativeWeightFormat::Q8_0:
            return encode_prefill_dequant_q8_0(context, encoder, params,
                                               buffers, error);
        case PrefillNativeWeightFormat::Q3_K:
            return encode_prefill_dequant_q3_k(context, encoder, params,
                                               buffers, error);
        case PrefillNativeWeightFormat::IQ4_XS:
            return encode_prefill_dequant_iq4_xs(context, encoder, params,
                                                 buffers, error);
        case PrefillNativeWeightFormat::IQ3_S:
            return encode_prefill_dequant_iq3_s(context, encoder, params,
                                                buffers, error);
        case PrefillNativeWeightFormat::Q4_K:
            return encode_prefill_dequant_q4_k(context, encoder, params,
                                               buffers, error);
        case PrefillNativeWeightFormat::Q5_K:
            return encode_prefill_dequant_q5_k(context, encoder, params,
                                               buffers, error);
        case PrefillNativeWeightFormat::Q6_K:
            return encode_prefill_dequant_q6_k(context, encoder, params,
                                               buffers, error);
    }
    error = "Prefill native projection format is invalid";
    return false;
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

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                    Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

const void* cpu_contents(ReadOnlyBufferRef view) { return view.contents(); }
const void* cpu_contents(BufferRef view) {
    return view ? view.contents : nullptr;
}
std::uint64_t gpu_address(ReadOnlyBufferRef view) {
    return view.gpu_address();
}
std::uint64_t gpu_address(BufferRef view) {
    return view ? view.gpu_address : 0u;
}

template <class Left, class Right>
bool views_overlap(Left left, std::size_t left_bytes, Right right,
                   std::size_t right_bytes) {
    if (!left || !right) return false;
    const void* const left_contents = cpu_contents(left);
    const void* const right_contents = cpu_contents(right);
    const bool same_cpu_backing =
        left_contents != nullptr && right_contents != nullptr &&
        ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left_contents), left_bytes,
            reinterpret_cast<std::uintptr_t>(right_contents), right_bytes);
    const std::uint64_t left_gpu_address = gpu_address(left);
    const std::uint64_t right_gpu_address = gpu_address(right);
    const bool same_gpu_backing =
        left_gpu_address != 0u && right_gpu_address != 0u &&
        ranges_overlap(left_gpu_address, left_bytes, right_gpu_address,
                       right_bytes);
    return same_cpu_backing || same_gpu_backing;
}

bool validate_aliases(const PrefillGemmMPPParams& params,
                      NativeFormatLayout layout,
                      const PrefillNativeProjectionBuffers& buffers,
                      std::string& error) {
    if (params.rows == 0u || params.n == 0u || params.k == 0u ||
        params.k % layout.values_per_block != 0u) {
        error = "Prefill native projection requires positive extents and K "
                "divisible by the native block size";
        return false;
    }

    std::size_t x_elements = 0u;
    std::size_t x_bytes = 0u;
    std::size_t blocks = 0u;
    std::size_t weight_bytes = 0u;
    std::size_t scratch_elements = 0u;
    std::size_t scratch_bytes = 0u;
    if (!checked_multiply(params.rows, params.k, x_elements) ||
        !checked_multiply(x_elements, sizeof(std::uint16_t), x_bytes) ||
        !checked_multiply(params.n, params.k / layout.values_per_block,
                          blocks) ||
        !checked_multiply(blocks, layout.bytes_per_block, weight_bytes) ||
        !checked_multiply(params.n, params.k, scratch_elements) ||
        !checked_multiply(scratch_elements, sizeof(std::uint16_t),
                          scratch_bytes)) {
        error = "Prefill native projection buffer size overflows size_t";
        return false;
    }

    const std::uint64_t column_end =
        static_cast<std::uint64_t>(params.col0) + params.n;
    std::size_t last_row_offset = 0u;
    std::size_t y_bytes = 0u;
    if (column_end > std::numeric_limits<std::size_t>::max() ||
        !checked_multiply(params.rows - 1u, params.ldy, last_row_offset) ||
        static_cast<std::size_t>(column_end) >
            std::numeric_limits<std::size_t>::max() - last_row_offset ||
        !checked_multiply(
            last_row_offset + static_cast<std::size_t>(column_end),
            sizeof(float), y_bytes)) {
        error = "Prefill native projection output size overflows size_t";
        return false;
    }

    if (views_overlap(buffers.x_f16, x_bytes, buffers.dequantized_f16,
                     scratch_bytes) ||
        views_overlap(buffers.weights, weight_bytes,
                      buffers.dequantized_f16, scratch_bytes)) {
        error = "Prefill native projection scratch overlaps an input";
        return false;
    }
    if (views_overlap(buffers.x_f16, x_bytes, buffers.y_f32, y_bytes) ||
        views_overlap(buffers.weights, weight_bytes, buffers.y_f32, y_bytes) ||
        views_overlap(buffers.dequantized_f16, scratch_bytes,
                      buffers.y_f32, y_bytes)) {
        error = "Prefill native projection output overlaps another buffer";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_native_projection(
    Context& context, Encoder& encoder,
    const PrefillGemmMPPParams& params, PrefillNativeWeightFormat format,
    const PrefillNativeProjectionBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend) {
    error.clear();
    NativeFormatLayout layout{};
    if (!format_layout(format, layout)) {
        error = "Prefill native projection format is invalid";
        return false;
    }
    if (encoder.concurrent_group_open()) {
        error = "Prefill native projection requires an ordered "
                "concurrent-group boundary";
        return false;
    }
    if (!validate_aliases(params, layout, buffers, error)) return false;

    const PrefillDequantParams dequant_params{params.n, params.k};
    const PrefillDequantBuffers dequant_buffers{
        buffers.weights, buffers.dequantized_f16};
    if (!encode_dequant(context, encoder, dequant_params, dequant_buffers,
                        format, error)) {
        return false;
    }

    const PrefillGemmMPPBuffers gemm_buffers{
        buffers.x_f16,
        context.buffers().read_only(buffers.dequantized_f16),
        buffers.y_f32};
    return encode_prefill_gemm(context, encoder, params, gemm_buffers,
                               PrefillGemmFormat::F16, error,
                               selected_backend);
}

bool encode_prefill_native_projection(
    Context& context, CommandBuffer& command_buffer,
    const PrefillGemmMPPParams& params, PrefillNativeWeightFormat format,
    const PrefillNativeProjectionBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend) {
    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    const bool encoded = Encoder::encode(
        context, command_buffer, "Prefill native projection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_native_projection(
                context, encoder, params, format, buffers, encode_error,
                &selected);
        },
        error);
    if (encoded && selected_backend != nullptr) {
        *selected_backend = selected;
    }
    return encoded;
}

}  // namespace strata::metal
