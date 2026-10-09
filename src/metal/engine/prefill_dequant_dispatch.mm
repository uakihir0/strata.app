// Validated Prefill dequantization for native Q8_0 weights.
#include "strata/metal/prefill_dequant_dispatch.hpp"

#include "strata/metal/encoder.hpp"

#include "../kernels/common/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace strata::metal {
namespace {

constexpr std::uint32_t kValuesPerQ8Block = 32u;
constexpr std::uint32_t kThreads = 256u;
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::uint32_t kMaxSignedExtent =
    static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
static_assert(sizeof(block_q8_0) == 34u);
static_assert(sizeof(PrefillDequantParams) == 2u * sizeof(std::uint32_t));

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

bool overlaps(ReadOnlyBufferRef input, std::size_t input_bytes,
              BufferRef output, std::size_t output_bytes) {
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

bool validate_dispatch(Context& context,
                       const PrefillDequantParams& params,
                       const PrefillDequantBuffers& buffers,
                       std::size_t& weight_bytes,
                       std::size_t& scratch_bytes, std::string& error) {
    if (params.n_out == 0u || params.k < kValuesPerQ8Block ||
        params.k % kValuesPerQ8Block != 0u ||
        params.n_out > kMaxSignedExtent || params.k > kMaxSignedExtent) {
        error = "Prefill Q8_0 dequant requires positive signed extents and "
                "K divisible by 32";
        return false;
    }

    const std::size_t blocks_per_row = params.k / kValuesPerQ8Block;
    std::size_t block_count = 0u;
    if (!checked_multiply(params.n_out, blocks_per_row, block_count) ||
        !checked_multiply(block_count, sizeof(block_q8_0), weight_bytes) ||
        !checked_multiply(params.n_out, params.k, scratch_bytes) ||
        !checked_multiply(scratch_bytes, kHalfBytes, scratch_bytes)) {
        error = "Prefill Q8_0 dequant buffer size overflows size_t";
        return false;
    }

    if (!buffers.weights || !context.buffers().owns(buffers.weights)) {
        error = "Prefill Q8_0 dequant weights are not owned by this Context";
        return false;
    }
    if (buffers.weights.length() < weight_bytes ||
        buffers.weights.offset() % alignof(block_q8_0) != 0u) {
        error = "Prefill Q8_0 dequant weight buffer is undersized or misaligned";
        return false;
    }
    if (!buffers.k_major_f16 ||
        !context.buffers().owns(buffers.k_major_f16)) {
        error = "Prefill Q8_0 dequant output is not owned by this Context";
        return false;
    }
    if (buffers.k_major_f16.length < scratch_bytes ||
        buffers.k_major_f16.offset % kHalfBytes != 0u) {
        error = "Prefill Q8_0 dequant output is undersized or misaligned";
        return false;
    }
    if (overlaps(buffers.weights, weight_bytes, buffers.k_major_f16,
                 scratch_bytes)) {
        error = "Prefill Q8_0 dequant output overlaps its weights";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_dequant_q8_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    error.clear();
    std::size_t weight_bytes = 0u;
    std::size_t scratch_bytes = 0u;
    if (!validate_dispatch(context, params, buffers, weight_bytes,
                           scratch_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding output{
        buffers.k_major_f16, 1u, scratch_bytes, kHalfBytes};
    const EncoderReadOnlyBufferBinding weights{
        buffers.weights, 0u, weight_bytes, alignof(block_q8_0)};
    const EncoderBytesBinding parameter_bytes{
        &params, sizeof(params), 2u};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_pf_dequant_q8_0",
        {(static_cast<std::size_t>(params.n_out) + kThreads - 1u) / kThreads,
         params.k / kValuesPerQ8Block, 1u},
        {kThreads, 1u, 1u}, std::span(&output, 1u),
        std::span(&weights, 1u), std::span(&parameter_bytes, 1u),
        threadgroup_memory, error);
}

bool encode_prefill_dequant_q8_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q8_0 dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q8_0(context, encoder, params,
                                               buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
