// Validated Prefill dequantization for native dense weights.
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

constexpr std::uint32_t kValuesPerQ2Block = 64u;
constexpr std::uint32_t kValuesPer32ValueBlock = 32u;
constexpr std::uint32_t kValuesPerQ4KBlock = 256u;
constexpr std::uint32_t kValuesPerQ5KBlock = 256u;
constexpr std::uint32_t kValuesPerQ6KBlock = 256u;
constexpr std::uint32_t kValuesPerQ3KBlock = 256u;
constexpr std::uint32_t kThreads = 256u;
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::uint32_t kMaxSignedExtent =
    static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
static_assert(sizeof(block_q2_0) == 18u);
static_assert(sizeof(block_iq4_nl) == 18u);
static_assert(sizeof(block_q4_0) == 18u);
static_assert(sizeof(block_q5_0) == 22u);
static_assert(sizeof(block_q8_0) == 34u);
static_assert(sizeof(block_q4_K) == 144u);
static_assert(sizeof(block_q5_K) == 176u);
static_assert(sizeof(block_q6_K) == 210u);
static_assert(sizeof(block_q3_K) == 110u);
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

template <typename Block>
bool validate_dispatch(Context& context,
                       const PrefillDequantParams& params,
                       const PrefillDequantBuffers& buffers,
                       const char* format, std::uint32_t values_per_block,
                       std::size_t& weight_bytes,
                       std::size_t& scratch_bytes, std::string& error) {
    const std::string prefix = std::string("Prefill ") + format + " dequant ";
    if (params.n_out == 0u || params.k < values_per_block ||
        params.k % values_per_block != 0u ||
        params.n_out > kMaxSignedExtent || params.k > kMaxSignedExtent) {
        error = prefix + "requires positive signed extents and K divisible "
                          "by " +
                std::to_string(values_per_block);
        return false;
    }

    const std::size_t blocks_per_row = params.k / values_per_block;
    std::size_t block_count = 0u;
    if (!checked_multiply(params.n_out, blocks_per_row, block_count) ||
        !checked_multiply(block_count, sizeof(Block), weight_bytes) ||
        !checked_multiply(params.n_out, params.k, scratch_bytes) ||
        !checked_multiply(scratch_bytes, kHalfBytes, scratch_bytes)) {
        error = prefix + "buffer size overflows size_t";
        return false;
    }

    if (!buffers.weights || !context.buffers().owns(buffers.weights)) {
        error = prefix + "weights are not owned by this Context";
        return false;
    }
    if (buffers.weights.length() < weight_bytes ||
        buffers.weights.offset() % alignof(Block) != 0u) {
        error = prefix + "weight buffer is undersized or misaligned";
        return false;
    }
    if (!buffers.k_major_f16 ||
        !context.buffers().owns(buffers.k_major_f16)) {
        error = prefix + "output is not owned by this Context";
        return false;
    }
    if (buffers.k_major_f16.length < scratch_bytes ||
        buffers.k_major_f16.offset % kHalfBytes != 0u) {
        error = prefix + "output is undersized or misaligned";
        return false;
    }
    if (overlaps(buffers.weights, weight_bytes, buffers.k_major_f16,
                 scratch_bytes)) {
        error = prefix + "output overlaps its weights";
        return false;
    }
    return true;
}

template <typename Block>
bool encode_dequant_dispatch(Context& context, Encoder& encoder,
                             const PrefillDequantParams& params,
                             const PrefillDequantBuffers& buffers,
                             const char* format, const char* kernel,
                             std::uint32_t values_per_block,
                             std::string& error) {
    error.clear();
    std::size_t weight_bytes = 0u;
    std::size_t scratch_bytes = 0u;
    if (!validate_dispatch<Block>(context, params, buffers, format,
                                  values_per_block,
                                  weight_bytes, scratch_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding output{
        buffers.k_major_f16, 1u, scratch_bytes, kHalfBytes};
    const EncoderReadOnlyBufferBinding weights{
        buffers.weights, 0u, weight_bytes, alignof(Block)};
    const EncoderBytesBinding parameter_bytes{
        &params, sizeof(params), 2u};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        kernel,
        {(static_cast<std::size_t>(params.n_out) + kThreads - 1u) / kThreads,
         params.k / values_per_block, 1u},
        {kThreads, 1u, 1u}, std::span(&output, 1u),
        std::span(&weights, 1u), std::span(&parameter_bytes, 1u),
        threadgroup_memory, error);
}

}  // namespace

bool encode_prefill_dequant_q2_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q2_0>(
        context, encoder, params, buffers, "Q2_0",
        "strata_pf_dequant_q2_0", kValuesPerQ2Block, error);
}

bool encode_prefill_dequant_q2_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q2_0 dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q2_0(context, encoder, params,
                                               buffers, encode_error);
        },
        error);
}

bool encode_prefill_dequant_q8_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q8_0>(
        context, encoder, params, buffers, "Q8_0",
        "strata_pf_dequant_q8_0", kValuesPer32ValueBlock, error);
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

bool encode_prefill_dequant_iq4_nl(Context& context, Encoder& encoder,
                                   const PrefillDequantParams& params,
                                   const PrefillDequantBuffers& buffers,
                                   std::string& error) {
    return encode_dequant_dispatch<block_iq4_nl>(
        context, encoder, params, buffers, "IQ4_NL",
        "strata_pf_dequant_iq4_nl", kValuesPer32ValueBlock, error);
}

bool encode_prefill_dequant_iq4_nl(
    Context& context, CommandBuffer& command_buffer,
    const PrefillDequantParams& params, const PrefillDequantBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill IQ4_NL dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_iq4_nl(
                context, encoder, params, buffers, encode_error);
        },
        error);
}

bool encode_prefill_dequant_q4_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q4_0>(
        context, encoder, params, buffers, "Q4_0",
        "strata_pf_dequant_q4_0", kValuesPer32ValueBlock, error);
}

bool encode_prefill_dequant_q4_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q4_0 dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q4_0(context, encoder, params,
                                               buffers, encode_error);
        },
        error);
}

bool encode_prefill_dequant_q5_0(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q5_0>(
        context, encoder, params, buffers, "Q5_0",
        "strata_pf_dequant_q5_0", kValuesPer32ValueBlock, error);
}

bool encode_prefill_dequant_q5_0(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q5_0 dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q5_0(context, encoder, params,
                                               buffers, encode_error);
        },
        error);
}

bool encode_prefill_dequant_q4_k(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q4_K>(
        context, encoder, params, buffers, "Q4_K",
        "strata_pf_dequant_q4_k", kValuesPerQ4KBlock, error);
}

bool encode_prefill_dequant_q4_k(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q4_K dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q4_k(
                context, encoder, params, buffers, encode_error);
        },
        error);
}

bool encode_prefill_dequant_q5_k(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q5_K>(
        context, encoder, params, buffers, "Q5_K",
        "strata_pf_dequant_q5_k", kValuesPerQ5KBlock, error);
}

bool encode_prefill_dequant_q5_k(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q5_K dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q5_k(
                context, encoder, params, buffers, encode_error);
        },
        error);
}

bool encode_prefill_dequant_q6_k(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q6_K>(
        context, encoder, params, buffers, "Q6_K",
        "strata_pf_dequant_q6_k", kValuesPerQ6KBlock, error);
}

bool encode_prefill_dequant_q6_k(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q6_K dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q6_k(
                context, encoder, params, buffers, encode_error);
        },
        error);
}

bool encode_prefill_dequant_q3_k(Context& context, Encoder& encoder,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return encode_dequant_dispatch<block_q3_K>(
        context, encoder, params, buffers, "Q3_K",
        "strata_pf_dequant_q3_k", kValuesPerQ3KBlock, error);
}

bool encode_prefill_dequant_q3_k(Context& context,
                                 CommandBuffer& command_buffer,
                                 const PrefillDequantParams& params,
                                 const PrefillDequantBuffers& buffers,
                                 std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill Q3_K dequantization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_dequant_q3_k(
                context, encoder, params, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
