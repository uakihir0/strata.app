// Ordered one-chunk Prefill GDN projection and transform coordinator.
#include "strata/metal/prefill_gdn_block_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_gdn_dispatch.hpp"
#include "strata/metal/prefill_gemm_dispatch.hpp"

#include "../kernels/common/params.h"

#include <algorithm>
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

struct MemoryRange {
    void* handle = nullptr;
    std::uintptr_t cpu_address = 0u;
    std::uint64_t gpu_address = 0u;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
};

struct DispatchSizes {
    PrefillGemmMPPParams qkv_params;
    PrefillGemmMPPParams gate_params;
    PrefillGemmMPPParams alpha_params;
    PrefillGemmMPPParams beta_params;
    PrefillGemmMPPParams out_params;
    PrefillNativeProjectionBufferSizes qkv_projection;
    PrefillNativeProjectionBufferSizes gate_projection;
    PrefillNativeProjectionBufferSizes out_projection;
    std::size_t mixed_bf_bytes = 0u;
    std::size_t low_mixed_bf_bytes = 0u;
    std::size_t alpha_weights_bytes = 0u;
    std::size_t beta_weights_bytes = 0u;
    std::size_t dequantized_f16_bytes = 0u;
    std::size_t alpha_beta_bytes = 0u;
    std::size_t dt_bias_bytes = 0u;
    std::size_t history_bytes = 0u;
    std::size_t conv_weights_bytes = 0u;
    std::size_t gate_beta_bytes = 0u;
    std::size_t state_bytes = 0u;
    std::size_t value_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    std::size_t y_h_bytes = 0u;
};

bool checked_bytes(std::size_t rows, std::size_t values_per_row,
                   std::size_t element_bytes, std::size_t& bytes) {
    if (element_bytes == 0u ||
        values_per_row >
            std::numeric_limits<std::size_t>::max() / element_bytes) {
        return false;
    }
    const std::size_t row_bytes = values_per_row * element_bytes;
    if (rows != 0u &&
        row_bytes > std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    bytes = rows * row_bytes;
    return true;
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

MemoryRange memory_range(BufferRef view, std::size_t bytes,
                         std::string_view name) {
    return {view.handle, reinterpret_cast<std::uintptr_t>(view.contents),
            view.gpu_address, view.offset, bytes, name};
}

MemoryRange memory_range(ReadOnlyBufferRef view, std::size_t bytes,
                         std::string_view name) {
    return {nullptr, reinterpret_cast<std::uintptr_t>(view.contents()),
            view.gpu_address(), view.offset(), bytes, name};
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

template <class View>
bool validate_view(Context& context, View view, std::size_t required_bytes,
                   std::size_t alignment, std::string_view name,
                   std::string& error);

template <>
bool validate_view<BufferRef>(Context& context, BufferRef view,
                              std::size_t required_bytes,
                              std::size_t alignment, std::string_view name,
                              std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN block " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % alignment != 0u) {
        error = "Prefill GDN block " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

template <>
bool validate_view<ReadOnlyBufferRef>(Context& context, ReadOnlyBufferRef view,
                                      std::size_t required_bytes,
                                      std::size_t alignment,
                                      std::string_view name,
                                      std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN block " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || view.offset() % alignment != 0u) {
        error = "Prefill GDN block " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       PrefillBF16X2Mode mode,
                       const PrefillGDNBlockBuffers& buffers,
                       DispatchSizes& sizes, std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GDN block row count exceeds the shader-safe range";
        return false;
    }
    if (mode != PrefillBF16X2Mode::off &&
        mode != PrefillBF16X2Mode::all_projections &&
        mode != PrefillBF16X2Mode::except_gr) {
        error = "Prefill GDN block BF16X2 mode is invalid";
        return false;
    }
    const bool use_bf16x2 = mode != PrefillBF16X2Mode::off;
    if (use_bf16x2 != static_cast<bool>(buffers.mixed_bf_lo)) {
        error = use_bf16x2
                    ? "Prefill GDN block BF16X2 mode requires mixed_bf_lo"
                    : "Prefill GDN block mode off requires an empty "
                      "mixed_bf_lo view";
        return false;
    }

    sizes.qkv_params = {rows, GDN_CONV_CHANNELS, N_EMBD,
                        GDN_CONV_CHANNELS, 0u, false};
    sizes.gate_params = {rows, GDN_VALUE_DIM, N_EMBD, GDN_VALUE_DIM, 0u,
                         false};
    sizes.alpha_params = {rows, GDN_V_HEADS, N_EMBD, 2u * GDN_V_HEADS, 0u,
                          false};
    sizes.beta_params = {rows, GDN_V_HEADS, N_EMBD, 2u * GDN_V_HEADS,
                         GDN_V_HEADS, false};
    sizes.out_params = {rows, N_EMBD, GDN_VALUE_DIM, N_EMBD, 0u, false};

    if (!prefill_native_projection_buffer_sizes(
            sizes.qkv_params, buffers.attn_qkv_format, sizes.qkv_projection,
            error) ||
        !prefill_native_projection_buffer_sizes(
            sizes.gate_params, buffers.attn_gate_format,
            sizes.gate_projection, error) ||
        !prefill_native_projection_buffer_sizes(
            sizes.out_params, buffers.ssm_out_format, sizes.out_projection,
            error)) {
        return false;
    }

    const std::size_t rows_size = rows;
    if (!checked_bytes(rows_size, N_EMBD, kHalfBytes,
                       sizes.mixed_bf_bytes) ||
        !checked_bytes(rows_size, N_EMBD, kHalfBytes,
                       sizes.low_mixed_bf_bytes) ||
        !checked_bytes(GDN_V_HEADS, N_EMBD, kHalfBytes,
                       sizes.alpha_weights_bytes) ||
        !checked_bytes(GDN_V_HEADS, N_EMBD, kHalfBytes,
                       sizes.beta_weights_bytes) ||
        !checked_bytes(rows_size, 2u * GDN_V_HEADS, kFloatBytes,
                       sizes.alpha_beta_bytes) ||
        !checked_bytes(1u, GDN_V_HEADS, kFloatBytes,
                       sizes.dt_bias_bytes) ||
        !checked_bytes(GDN_CONV_KERNEL - 1u, GDN_CONV_CHANNELS, kFloatBytes,
                       sizes.history_bytes) ||
        !checked_bytes(GDN_CONV_CHANNELS, GDN_CONV_KERNEL, kFloatBytes,
                       sizes.conv_weights_bytes) ||
        !checked_bytes(rows_size, GDN_V_HEADS, kFloatBytes,
                       sizes.gate_beta_bytes) ||
        !checked_bytes(1u, GDN_STATE_ELEMENTS, kFloatBytes,
                       sizes.state_bytes) ||
        !checked_bytes(rows_size, GDN_VALUE_DIM, kFloatBytes,
                       sizes.value_bytes) ||
        !checked_bytes(1u, GDN_HEAD_DIM, kFloatBytes, sizes.gamma_bytes) ||
        !checked_bytes(rows_size, GDN_VALUE_DIM, kHalfBytes,
                       sizes.y_h_bytes)) {
        error = "Prefill GDN block buffer size overflows size_t";
        return false;
    }
    sizes.dequantized_f16_bytes =
        std::max({sizes.qkv_projection.dequantized_f16_bytes,
                  sizes.gate_projection.dequantized_f16_bytes,
                  sizes.out_projection.dequantized_f16_bytes});

    if (!validate_view(context, buffers.mixed_h,
                       sizes.qkv_projection.x_f16_bytes, kHalfBytes,
                       "mixed_h", error) ||
        !validate_view(context, buffers.mixed_bf, sizes.mixed_bf_bytes,
                       kHalfBytes, "mixed_bf", error) ||
        (use_bf16x2 &&
         !validate_view(context, buffers.mixed_bf_lo,
                        sizes.low_mixed_bf_bytes, kHalfBytes, "mixed_bf_lo",
                        error)) ||
        !validate_view(context, buffers.attn_qkv_weights,
                       sizes.qkv_projection.weights_bytes,
                       sizes.qkv_projection.weights_alignment,
                       "attn_qkv weights", error) ||
        !validate_view(context, buffers.attn_gate_weights,
                       sizes.gate_projection.weights_bytes,
                       sizes.gate_projection.weights_alignment,
                       "attn_gate weights", error) ||
        !validate_view(context, buffers.ssm_alpha_weights,
                       sizes.alpha_weights_bytes, kHalfBytes,
                       "ssm_alpha weights", error) ||
        !validate_view(context, buffers.ssm_beta_weights,
                       sizes.beta_weights_bytes, kHalfBytes,
                       "ssm_beta weights", error) ||
        !validate_view(context, buffers.ssm_out_weights,
                       sizes.out_projection.weights_bytes,
                       sizes.out_projection.weights_alignment,
                       "ssm_out weights", error) ||
        !validate_view(context, buffers.dt_bias, sizes.dt_bias_bytes,
                       kFloatBytes, "ssm_dt.bias", error) ||
        !validate_view(context, buffers.a, sizes.dt_bias_bytes, kFloatBytes,
                       "ssm_a", error) ||
        !validate_view(context, buffers.conv_weights,
                       sizes.conv_weights_bytes, kFloatBytes, "ssm_conv1d",
                       error) ||
        !validate_view(context, buffers.gamma, sizes.gamma_bytes, kFloatBytes,
                       "ssm_norm", error) ||
        !validate_view(context, buffers.dequantized_f16,
                       sizes.dequantized_f16_bytes, kHalfBytes,
                       "dequantized_f16 scratch", error) ||
        !validate_view(context, buffers.qkv,
                       sizes.qkv_projection.y_f32_bytes, kFloatBytes, "qkv",
                       error) ||
        !validate_view(context, buffers.z,
                       sizes.gate_projection.y_f32_bytes, kFloatBytes, "z",
                       error) ||
        !validate_view(context, buffers.alpha_beta,
                       sizes.alpha_beta_bytes, kFloatBytes, "alpha_beta",
                       error) ||
        !validate_view(context, buffers.history, sizes.history_bytes,
                       kFloatBytes, "history", error) ||
        !validate_view(context, buffers.gate, sizes.gate_beta_bytes,
                       kFloatBytes, "gate", error) ||
        !validate_view(context, buffers.beta, sizes.gate_beta_bytes,
                       kFloatBytes, "beta", error) ||
        !validate_view(context, buffers.h,
                       sizes.qkv_projection.y_f32_bytes, kFloatBytes, "h",
                       error) ||
        !validate_view(context, buffers.state, sizes.state_bytes, kFloatBytes,
                       "state", error) ||
        !validate_view(context, buffers.oc, sizes.value_bytes, kFloatBytes,
                       "oc", error) ||
        !validate_view(context, buffers.y_h, sizes.y_h_bytes, kHalfBytes,
                       "y_h", error) ||
        !validate_view(context, buffers.bo,
                       sizes.out_projection.y_f32_bytes, kFloatBytes, "bo",
                       error)) {
        return false;
    }

    const std::array<MemoryRange, 12u> writable{
        memory_range(buffers.dequantized_f16, sizes.dequantized_f16_bytes,
                     "dequantized_f16 scratch"),
        memory_range(buffers.qkv, sizes.qkv_projection.y_f32_bytes, "qkv"),
        memory_range(buffers.z, sizes.gate_projection.y_f32_bytes, "z"),
        memory_range(buffers.alpha_beta, sizes.alpha_beta_bytes, "alpha_beta"),
        memory_range(buffers.history, sizes.history_bytes, "history"),
        memory_range(buffers.gate, sizes.gate_beta_bytes, "gate"),
        memory_range(buffers.beta, sizes.gate_beta_bytes, "beta"),
        memory_range(buffers.h, sizes.qkv_projection.y_f32_bytes, "h"),
        memory_range(buffers.state, sizes.state_bytes, "state"),
        memory_range(buffers.oc, sizes.value_bytes, "oc"),
        memory_range(buffers.y_h, sizes.y_h_bytes, "y_h"),
        memory_range(buffers.bo, sizes.out_projection.y_f32_bytes, "bo")};
    const std::array<MemoryRange, 12u> read_only{
        memory_range(buffers.mixed_h, sizes.qkv_projection.x_f16_bytes,
                     "mixed_h"),
        memory_range(buffers.mixed_bf, sizes.mixed_bf_bytes, "mixed_bf"),
        memory_range(buffers.mixed_bf_lo,
                     use_bf16x2 ? sizes.low_mixed_bf_bytes : 0u,
                     "mixed_bf_lo"),
        memory_range(buffers.attn_qkv_weights,
                     sizes.qkv_projection.weights_bytes, "attn_qkv weights"),
        memory_range(buffers.attn_gate_weights,
                     sizes.gate_projection.weights_bytes,
                     "attn_gate weights"),
        memory_range(buffers.ssm_alpha_weights, sizes.alpha_weights_bytes,
                     "ssm_alpha weights"),
        memory_range(buffers.ssm_beta_weights, sizes.beta_weights_bytes,
                     "ssm_beta weights"),
        memory_range(buffers.ssm_out_weights,
                     sizes.out_projection.weights_bytes, "ssm_out weights"),
        memory_range(buffers.dt_bias, sizes.dt_bias_bytes, "ssm_dt.bias"),
        memory_range(buffers.a, sizes.dt_bias_bytes, "ssm_a"),
        memory_range(buffers.conv_weights, sizes.conv_weights_bytes,
                     "ssm_conv1d"),
        memory_range(buffers.gamma, sizes.gamma_bytes, "ssm_norm")};

    for (std::size_t output = 0u; output < writable.size(); ++output) {
        for (std::size_t earlier = 0u; earlier < output; ++earlier) {
            if (ranges_overlap(writable[output], writable[earlier])) {
                error = "Prefill GDN block writable stages overlap: " +
                        std::string(writable[earlier].name) + " and " +
                        std::string(writable[output].name);
                return false;
            }
        }
        for (const MemoryRange& input : read_only) {
            if (ranges_overlap(writable[output], input)) {
                error = "Prefill GDN block writable " +
                        std::string(writable[output].name) +
                        " overlaps read-only " + std::string(input.name);
                return false;
            }
        }
    }
    return true;
}

BufferRef active_view(BufferRef view, std::size_t bytes) {
    return view.slice(0u, bytes);
}

ReadOnlyBufferRef active_view(ReadOnlyBufferRef view, std::size_t bytes) {
    return view.slice(0u, bytes);
}

}  // namespace

bool encode_prefill_gdn_block(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGDNBlockBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GDN block requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    DispatchSizes sizes;
    if (!validate_dispatch(context, rows, mode, buffers, sizes, error)) {
        return false;
    }

    const ReadOnlyBufferRef mixed_h =
        active_view(buffers.mixed_h, sizes.qkv_projection.x_f16_bytes);
    const ReadOnlyBufferRef mixed_bf =
        active_view(buffers.mixed_bf, sizes.mixed_bf_bytes);
    const BufferRef dequantized_f16 = active_view(
        buffers.dequantized_f16, sizes.dequantized_f16_bytes);
    const BufferRef qkv =
        active_view(buffers.qkv, sizes.qkv_projection.y_f32_bytes);
    const BufferRef z =
        active_view(buffers.z, sizes.gate_projection.y_f32_bytes);
    const BufferRef alpha_beta =
        active_view(buffers.alpha_beta, sizes.alpha_beta_bytes);
    const ReadOnlyBufferRef qkv_weights =
        active_view(buffers.attn_qkv_weights,
                    sizes.qkv_projection.weights_bytes);
    const ReadOnlyBufferRef gate_weights =
        active_view(buffers.attn_gate_weights,
                    sizes.gate_projection.weights_bytes);
    const ReadOnlyBufferRef alpha_weights =
        active_view(buffers.ssm_alpha_weights, sizes.alpha_weights_bytes);
    const ReadOnlyBufferRef beta_weights =
        active_view(buffers.ssm_beta_weights, sizes.beta_weights_bytes);

    if (!encode_prefill_native_projection(
            context, encoder, sizes.qkv_params, buffers.attn_qkv_format,
            {mixed_h, qkv_weights,
             active_view(dequantized_f16,
                         sizes.qkv_projection.dequantized_f16_bytes),
             qkv},
            error) ||
        !encode_prefill_native_projection(
            context, encoder, sizes.gate_params, buffers.attn_gate_format,
            {mixed_h, gate_weights,
             active_view(dequantized_f16,
                         sizes.gate_projection.dequantized_f16_bytes),
             z},
            error)) {
        return false;
    }

    if (mode == PrefillBF16X2Mode::off) {
        const PrefillGemmMPPBuffers alpha_buffers{
            mixed_bf, alpha_weights, alpha_beta};
        if (!encode_prefill_gemm(context, encoder, sizes.alpha_params,
                                 alpha_buffers, PrefillGemmFormat::BF16,
                                 error) ||
            !encode_prefill_gemm(
                context, encoder, sizes.beta_params,
                {mixed_bf, beta_weights, alpha_beta},
                PrefillGemmFormat::BF16, error)) {
            return false;
        }
    } else {
        const ReadOnlyBufferRef mixed_bf_lo =
            active_view(buffers.mixed_bf_lo, sizes.low_mixed_bf_bytes);
        if (!encode_prefill_gemm_bf16x2(
                context, encoder, sizes.alpha_params,
                {mixed_bf, mixed_bf_lo, alpha_weights, alpha_beta}, error) ||
            !encode_prefill_gemm_bf16x2(
                context, encoder, sizes.beta_params,
                {mixed_bf, mixed_bf_lo, beta_weights, alpha_beta}, error)) {
            return false;
        }
    }

    const BufferRef history =
        active_view(buffers.history, sizes.history_bytes);
    const BufferRef gate = active_view(buffers.gate, sizes.gate_beta_bytes);
    const BufferRef beta = active_view(buffers.beta, sizes.gate_beta_bytes);
    const BufferRef h =
        active_view(buffers.h, sizes.qkv_projection.y_f32_bytes);
    const BufferRef state = active_view(buffers.state, sizes.state_bytes);
    const BufferRef oc = active_view(buffers.oc, sizes.value_bytes);
    const BufferRef y_h = active_view(buffers.y_h, sizes.y_h_bytes);
    const ReadOnlyBufferRef dt_bias =
        active_view(buffers.dt_bias, sizes.dt_bias_bytes);
    const ReadOnlyBufferRef a =
        active_view(buffers.a, sizes.dt_bias_bytes);
    const ReadOnlyBufferRef conv_weights =
        active_view(buffers.conv_weights, sizes.conv_weights_bytes);
    const ReadOnlyBufferRef gamma =
        active_view(buffers.gamma, sizes.gamma_bytes);
    const PrefillGDNTransformBuffers transform_buffers{
        context.buffers().read_only(alpha_beta),
        dt_bias,
        a,
        history,
        context.buffers().read_only(qkv),
        conv_weights,
        gate,
        beta,
        h,
        state,
        oc,
        context.buffers().read_only(z),
        gamma,
        y_h};
    if (!encode_prefill_gdn_transforms(context, encoder, rows,
                                       transform_buffers, error)) {
        return false;
    }

    return encode_prefill_native_projection(
        context, encoder, sizes.out_params, buffers.ssm_out_format,
        {context.buffers().read_only(y_h),
         active_view(buffers.ssm_out_weights,
                     sizes.out_projection.weights_bytes),
         active_view(dequantized_f16,
                     sizes.out_projection.dequantized_f16_bytes),
         active_view(buffers.bo, sizes.out_projection.y_f32_bytes)},
        error);
}

bool encode_prefill_gdn_block(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGDNBlockBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN block",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_block(context, encoder, rows, mode,
                                            buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
