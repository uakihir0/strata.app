// Validated encoder for Prefill's QSA attention FWHT gate stage.
#include "strata/metal/prefill_qsa_gate_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include "../kernels/common/params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::uint32_t kAttentionRowWidth = N_HEAD_Q * HEAD_DIM;
constexpr std::uint32_t kQueryRowWidth = N_HEAD_Q * 2u * HEAD_DIM;
constexpr std::uint32_t kThreads = 256u;

bool checked_bytes(std::uint32_t rows, std::uint32_t values_per_row,
                   std::size_t element_bytes, std::size_t& bytes) {
    if (element_bytes == 0u ||
        static_cast<std::size_t>(values_per_row) >
            std::numeric_limits<std::size_t>::max() / element_bytes) {
        return false;
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(values_per_row) * element_bytes;
    if (rows != 0u &&
        row_bytes > std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    bytes = static_cast<std::size_t>(rows) * row_bytes;
    return true;
}

bool ranges_overlap(std::uintptr_t left, std::size_t left_bytes,
                   std::uintptr_t right, std::size_t right_bytes) {
    if (left == 0u || right == 0u || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool overlaps(ReadOnlyBufferRef input, BufferRef output,
              std::size_t input_bytes, std::size_t output_bytes) {
    if (ranges_overlap(reinterpret_cast<std::uintptr_t>(input.contents()),
                       input_bytes,
                       reinterpret_cast<std::uintptr_t>(output.contents),
                       output_bytes)) {
        return true;
    }
    return ranges_overlap(static_cast<std::uintptr_t>(input.gpu_address()),
                          input_bytes,
                          static_cast<std::uintptr_t>(output.gpu_address),
                          output_bytes);
}

bool validate_read_view(Context& context, ReadOnlyBufferRef view,
                        std::size_t required_bytes, std::size_t alignment,
                        std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill QSA gate " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || view.offset() % alignment != 0u) {
        error = "Prefill QSA gate " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill QSA gate attn_h buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % alignment != 0u) {
        error = "Prefill QSA gate attn_h buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       const PrefillQsaGateAttnBuffers& buffers,
                       std::size_t& attn_bytes, std::size_t& q_full_bytes,
                       std::size_t& attn_h_bytes, std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill QSA gate row count exceeds the shader-safe range";
        return false;
    }
    if (!checked_bytes(rows, kAttentionRowWidth, kFloatBytes, attn_bytes) ||
        !checked_bytes(rows, kQueryRowWidth, kFloatBytes, q_full_bytes) ||
        !checked_bytes(rows, kAttentionRowWidth, kHalfBytes,
                       attn_h_bytes)) {
        error = "Prefill QSA gate buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.attn, attn_bytes, kFloatBytes,
                            "attention", error) ||
        !validate_read_view(context, buffers.q_full, q_full_bytes, kFloatBytes,
                            "Qf", error) ||
        !validate_write_view(context, buffers.attn_h, attn_h_bytes, kHalfBytes,
                             error)) {
        return false;
    }
    if (overlaps(buffers.attn, buffers.attn_h, attn_bytes, attn_h_bytes) ||
        overlaps(buffers.q_full, buffers.attn_h, q_full_bytes, attn_h_bytes)) {
        error = "Prefill QSA gate attn_h overlaps a read-only input";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_qsa_gate_attn(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillQsaGateAttnBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA gate requires an ordered concurrent-group boundary";
        return false;
    }

    std::size_t attn_bytes = 0u;
    std::size_t q_full_bytes = 0u;
    std::size_t attn_h_bytes = 0u;
    if (!validate_dispatch(context, rows, buffers, attn_bytes, q_full_bytes,
                           attn_h_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding output{
        buffers.attn_h, 2u, attn_h_bytes, kHalfBytes};
    const std::array inputs{
        EncoderReadOnlyBufferBinding{buffers.attn, 0u, attn_bytes, kFloatBytes},
        EncoderReadOnlyBufferBinding{buffers.q_full, 1u, q_full_bytes,
                                     kFloatBytes},
    };
    const std::uint32_t row_count = rows;
    const std::array bytes{
        EncoderBytesBinding{&row_count, sizeof(row_count), 3u},
    };
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    const std::size_t groups =
        static_cast<std::size_t>(rows) * N_HEAD_Q;
    return encoder.dispatch(
        "strata_pf_gate_attn", {static_cast<std::uint32_t>(groups), 1u, 1u},
        {kThreads, 1u, 1u},
        std::span<const EncoderBufferBinding>(&output, 1u), inputs, bytes,
        threadgroup_memory, error);
}

bool encode_prefill_qsa_gate_attn(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillQsaGateAttnBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill QSA attention gate",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_gate_attn(context, encoder, rows,
                                                buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
