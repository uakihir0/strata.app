// Ordered Prefill QSA gate and native attention-output projection.
#include "strata/metal/prefill_qsa_output_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/qsa_dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::metal {
namespace {

constexpr std::uint32_t kQsaAttentionWidth = 24u * 256u;
constexpr std::uint32_t kQsaHeadCount = 24u;
constexpr std::uint32_t kQsaHeadWidth = 256u;
constexpr std::uint32_t kQFullHeadWidth = 512u;
constexpr std::uint32_t kModelWidth = 2560u;

PrefillGemmMPPParams projection_params(std::uint32_t rows) {
    return {rows, kModelWidth, kQsaAttentionWidth, kModelWidth, 0u, false};
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

bool same_view_origin(BufferRef writable, ReadOnlyBufferRef readable) {
    return writable && readable &&
           writable.contents == readable.contents() &&
           writable.gpu_address == readable.gpu_address() &&
           writable.offset == readable.offset();
}

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes, Address right,
                    std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
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

bool validate_projection_buffers(
    Context& context, const PrefillQsaOutputBuffers& buffers,
    const PrefillNativeProjectionBufferSizes& sizes, std::size_t attention_bytes,
    std::size_t q_full_bytes,
    PrefillQsaAttentionTransform attention_transform, std::string& error) {
    if (!buffers.gate.attn || !context.buffers().owns(buffers.gate.attn)) {
        error = "Prefill QSA output attention is not owned by this Context";
        return false;
    }
    if (buffers.gate.attn.length() < attention_bytes ||
        buffers.gate.attn.offset() % alignof(float) != 0u) {
        error = "Prefill QSA output attention is undersized or misaligned";
        return false;
    }
    if (attention_transform == PrefillQsaAttentionTransform::fwht256) {
        if (!buffers.attention_transform ||
            !context.buffers().owns(buffers.attention_transform)) {
            error = "Prefill QSA output FWHT attention is not owned by this "
                    "Context";
            return false;
        }
        if (buffers.attention_transform.length < attention_bytes ||
            buffers.attention_transform.offset % alignof(float) != 0u ||
            !same_view_origin(buffers.attention_transform, buffers.gate.attn)) {
            error = "Prefill QSA output FWHT attention does not match the "
                    "gate attention view";
            return false;
        }
    } else if (attention_transform != PrefillQsaAttentionTransform::none) {
        error = "Prefill QSA output attention transform is invalid";
        return false;
    }
    if (!buffers.gate.q_full || !context.buffers().owns(buffers.gate.q_full)) {
        error = "Prefill QSA output Qf is not owned by this Context";
        return false;
    }
    if (buffers.gate.q_full.length() < q_full_bytes ||
        buffers.gate.q_full.offset() % alignof(float) != 0u) {
        error = "Prefill QSA output Qf is undersized or misaligned";
        return false;
    }
    if (!buffers.gate.attn_h ||
        !context.buffers().owns(buffers.gate.attn_h)) {
        error = "Prefill QSA output gate buffer is not owned by this Context";
        return false;
    }
    if (buffers.gate.attn_h.length < sizes.x_f16_bytes ||
        buffers.gate.attn_h.offset % sizeof(std::uint16_t) != 0u) {
        error = "Prefill QSA output gate buffer is undersized or misaligned";
        return false;
    }
    if (!buffers.native_weights ||
        !context.buffers().owns(buffers.native_weights)) {
        error = "Prefill QSA output weights are not owned by this Context";
        return false;
    }
    if (buffers.native_weights.length() < sizes.weights_bytes ||
        sizes.weights_alignment == 0u ||
        buffers.native_weights.offset() % sizes.weights_alignment != 0u) {
        error = "Prefill QSA output weights are undersized or misaligned";
        return false;
    }
    if (!buffers.dequantized_f16 ||
        !context.buffers().owns(buffers.dequantized_f16)) {
        error =
            "Prefill QSA output dequantized weights are not owned by this "
            "Context";
        return false;
    }
    if (buffers.dequantized_f16.length < sizes.dequantized_f16_bytes ||
        buffers.dequantized_f16.offset % sizeof(std::uint16_t) != 0u) {
        error = "Prefill QSA output dequantized weights are undersized or "
                "misaligned";
        return false;
    }
    if (!buffers.y_f32 || !context.buffers().owns(buffers.y_f32)) {
        error = "Prefill QSA output Y is not owned by this Context";
        return false;
    }
    if (buffers.y_f32.length < sizes.y_f32_bytes ||
        buffers.y_f32.offset % alignof(float) != 0u) {
        error = "Prefill QSA output Y is undersized or misaligned";
        return false;
    }

    const bool gate_alias =
        views_overlap(buffers.gate.attn_h, sizes.x_f16_bytes,
                      buffers.gate.attn, attention_bytes) ||
        views_overlap(buffers.gate.attn_h, sizes.x_f16_bytes,
                      buffers.gate.q_full, q_full_bytes) ||
        views_overlap(buffers.gate.attn_h, sizes.x_f16_bytes,
                      buffers.native_weights, sizes.weights_bytes) ||
        views_overlap(buffers.gate.attn_h, sizes.x_f16_bytes,
                      buffers.dequantized_f16,
                      sizes.dequantized_f16_bytes) ||
        views_overlap(buffers.gate.attn_h, sizes.x_f16_bytes, buffers.y_f32,
                      sizes.y_f32_bytes);
    const bool dequant_alias =
        views_overlap(buffers.dequantized_f16,
                      sizes.dequantized_f16_bytes, buffers.gate.attn,
                      attention_bytes) ||
        views_overlap(buffers.dequantized_f16,
                      sizes.dequantized_f16_bytes, buffers.gate.q_full,
                      q_full_bytes) ||
        views_overlap(buffers.dequantized_f16,
                      sizes.dequantized_f16_bytes, buffers.native_weights,
                      sizes.weights_bytes) ||
        views_overlap(buffers.dequantized_f16,
                      sizes.dequantized_f16_bytes, buffers.y_f32,
                      sizes.y_f32_bytes);
    const bool output_alias =
        views_overlap(buffers.y_f32, sizes.y_f32_bytes, buffers.gate.attn,
                      attention_bytes) ||
        views_overlap(buffers.y_f32, sizes.y_f32_bytes, buffers.gate.q_full,
                      q_full_bytes) ||
        views_overlap(buffers.y_f32, sizes.y_f32_bytes,
                      buffers.native_weights, sizes.weights_bytes);
    if (gate_alias || dequant_alias || output_alias) {
        error = "Prefill QSA output writable buffers overlap another active "
                "stage buffer";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_qsa_output(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillNativeWeightFormat format,
    const PrefillQsaOutputBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend,
    PrefillQsaAttentionTransform attention_transform) {
    error.clear();
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill QSA output row count exceeds the shader-safe range";
        return false;
    }
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA output requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    const PrefillGemmMPPParams params = projection_params(rows);
    PrefillNativeProjectionBufferSizes sizes;
    if (!prefill_native_projection_buffer_sizes(params, format, sizes,
                                                error)) {
        return false;
    }
    const std::size_t attention_bytes =
        static_cast<std::size_t>(rows) * kQsaAttentionWidth * sizeof(float);
    const std::size_t q_full_bytes =
        static_cast<std::size_t>(rows) * kQsaHeadCount * kQFullHeadWidth *
        sizeof(float);
    static_assert(kQsaHeadCount * kQsaHeadWidth == kQsaAttentionWidth);
    static_assert(kQsaHeadWidth * 2u == kQFullHeadWidth);
    if (!validate_projection_buffers(context, buffers, sizes, attention_bytes,
                                     q_full_bytes, attention_transform,
                                     error)) {
        return false;
    }

    if (attention_transform == PrefillQsaAttentionTransform::fwht256) {
        if (views_overlap(buffers.attention_transform, attention_bytes,
                          buffers.gate.q_full, q_full_bytes) ||
            views_overlap(buffers.attention_transform, attention_bytes,
                          buffers.native_weights, sizes.weights_bytes)) {
            error = "Prefill QSA output FWHT attention overlaps another active "
                    "read-only input";
            return false;
        }
        if (!encode_qsa_fwht256(context, encoder, rows * kQsaHeadCount,
                                buffers.attention_transform, error)) {
            return false;
        }
    }

    if (!encode_prefill_qsa_gate_attn(context, encoder, rows, buffers.gate,
                                      error)) {
        return false;
    }

    const PrefillNativeProjectionBuffers projection_buffers{
        context.buffers().read_only(buffers.gate.attn_h),
        buffers.native_weights,
        buffers.dequantized_f16,
        buffers.y_f32};
    return encode_prefill_native_projection(
        context, encoder, params, format, projection_buffers, error,
        selected_backend);
}

bool encode_prefill_qsa_output(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillNativeWeightFormat format,
    const PrefillQsaOutputBuffers& buffers, std::string& error,
    PrefillGemmBackend* selected_backend,
    PrefillQsaAttentionTransform attention_transform) {
    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    const bool encoded = Encoder::encode(
        context, command_buffer, "Prefill QSA output",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_output(
                context, encoder, rows, format, buffers, encode_error,
                &selected, attention_transform);
        },
        error);
    if (encoded && selected_backend != nullptr) {
        *selected_backend = selected;
    }
    return encoded;
}

}  // namespace strata::metal
