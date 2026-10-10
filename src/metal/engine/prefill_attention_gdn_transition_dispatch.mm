// Ordered Prefill attention-half GR-read/GDN transition.
#include "strata/metal/prefill_attention_gdn_transition_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_gr_write_norm_dispatch.hpp"

#include "../kernels/common/params.h"

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

struct LayerRanges {
    std::array<MemoryRange, 11u> gr_outputs{};
    std::size_t gr_output_count = 0u;
    std::array<MemoryRange, 4u> gr_weights{};
    std::array<MemoryRange, 9u> gdn_weights{};
    std::array<MemoryRange, 2u> gdn_persistent{};
    std::array<MemoryRange, 12u> gdn_outputs{};
    std::array<MemoryRange, 6u> carried_values{};
    std::size_t residual_bytes = 0u;
    std::size_t inject_bytes = 0u;
    std::size_t next_norm_bytes = 0u;
    std::size_t scale_bytes = 0u;
    std::size_t stream_bf16_bytes = 0u;
    std::size_t block_out_bytes = 0u;
    std::size_t stream_f32_bytes = 0u;
    std::size_t norm_weight_bytes = 0u;
    std::size_t down_weight_bytes = 0u;
    std::size_t up_weight_bytes = 0u;
    std::size_t inject_weight_bytes = 0u;
    std::size_t low_rank_f32_bytes = 0u;
    std::size_t low_rank_bf16_bytes = 0u;
    std::size_t mixer_bytes = 0u;
};

bool build_ranges(Context& context, std::uint32_t rows,
                  PrefillBF16X2Mode mode,
                  const PrefillAttentionGDNTransitionBuffers& buffers,
                  LayerRanges& ranges, std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error =
            "Prefill attention GDN transition row count exceeds the shader-safe range";
        return false;
    }
    if (mode != PrefillBF16X2Mode::off &&
        mode != PrefillBF16X2Mode::all_projections &&
        mode != PrefillBF16X2Mode::except_gr) {
        error = "Prefill attention GDN transition BF16X2 mode is invalid";
        return false;
    }
    if (buffers.gdn.mixed_h || buffers.gdn.mixed_bf ||
        buffers.gdn.mixed_bf_lo) {
        error =
            "Prefill attention GDN transition GDN mixer inputs are bound by the "
            "coordinator and must be empty";
        return false;
    }

    constexpr std::size_t stream_count =
        static_cast<std::size_t>(HC) * N_EMBD;
    constexpr std::size_t low_rank_width = HC_LOW_RANK;
    const bool read_low = mode == PrefillBF16X2Mode::all_projections;
    const bool mix_low = mode != PrefillBF16X2Mode::off;
    std::size_t read_scale_bytes = 0u;
    std::size_t read_stream_f32_bytes = 0u;
    std::size_t read_stream_f16_bytes = 0u;
    std::size_t read_low_rank_f32_bytes = 0u;
    std::size_t read_low_rank_f16_bytes = 0u;
    std::size_t mixer_f16_bytes = 0u;
    std::size_t gr_norm_weight_bytes = 0u;
    std::size_t gr_down_weight_bytes = 0u;
    std::size_t gr_up_weight_bytes = 0u;
    std::size_t gr_inject_weight_bytes = 0u;
    std::size_t residual_bytes = 0u;
    std::size_t inject_bytes = 0u;
    std::size_t next_norm_bytes = 0u;
    if (!checked_bytes(rows, HC, kFloatBytes, read_scale_bytes) ||
        !checked_bytes(rows, stream_count, kFloatBytes,
                       read_stream_f32_bytes) ||
        !checked_bytes(rows, stream_count, kHalfBytes,
                       read_stream_f16_bytes) ||
        !checked_bytes(rows, low_rank_width, kFloatBytes,
                       read_low_rank_f32_bytes) ||
        !checked_bytes(rows, low_rank_width, kHalfBytes,
                       read_low_rank_f16_bytes) ||
        !checked_bytes(rows, N_EMBD, kHalfBytes, mixer_f16_bytes) ||
        !checked_bytes(1u, stream_count, kFloatBytes,
                       gr_norm_weight_bytes) ||
        !checked_bytes(1u, low_rank_width * stream_count, kHalfBytes,
                       gr_down_weight_bytes) ||
        !checked_bytes(1u, stream_count * low_rank_width, kHalfBytes,
                       gr_up_weight_bytes) ||
        !checked_bytes(1u, HC * stream_count, kHalfBytes,
                       gr_inject_weight_bytes) ||
        !checked_bytes(rows, stream_count, kFloatBytes, residual_bytes) ||
        !checked_bytes(rows, HC, kFloatBytes, inject_bytes) ||
        !checked_bytes(HC, N_EMBD, kFloatBytes, next_norm_bytes)) {
        error = "Prefill attention GDN transition buffer size overflows size_t";
        return false;
    }

    if (!buffers.next_norm || !context.buffers().owns(buffers.next_norm) ||
        buffers.next_norm.length() < next_norm_bytes ||
        buffers.next_norm.offset() % kFloatBytes != 0u) {
        error =
            "Prefill attention GDN transition next norm is unavailable, undersized, "
            "misaligned, or not owned by this Context";
        return false;
    }

    const auto add_gr_output = [&](BufferRef view, std::size_t bytes,
                                   std::string_view name) {
        ranges.gr_outputs[ranges.gr_output_count++] =
            memory_range(view, bytes, name);
    };
    const PrefillGRReadBuffers& read = buffers.gr_read_mix.read;
    ranges.gr_weights = {
        memory_range(read.norm, gr_norm_weight_bytes, "GR norm weights"),
        memory_range(read.down_weights, gr_down_weight_bytes,
                     "GR down weights"),
        memory_range(read.up_weights, gr_up_weight_bytes, "GR up weights"),
        memory_range(read.inject_weights, gr_inject_weight_bytes,
                     "GR injection weights")};
    add_gr_output(read.rs, read_scale_bytes, "GR read scale");
    add_gr_output(read.xn16, read_stream_f16_bytes, "GR read xn16");
    if (read_low) {
        add_gr_output(read.xn16_lo, read_stream_f16_bytes,
                      "GR read xn16 remainder");
    }
    add_gr_output(read.low_rank, read_low_rank_f32_bytes,
                  "GR read low rank");
    add_gr_output(read.low_rank_bf16, read_low_rank_f16_bytes,
                  "GR read low rank BF16");
    if (read_low) {
        add_gr_output(read.low_rank_bf16_lo, read_low_rank_f16_bytes,
                      "GR read low rank remainder");
    }
    add_gr_output(read.gated, read_stream_f32_bytes, "GR read gated");
    add_gr_output(read.inject, inject_bytes, "GR read inject");
    add_gr_output(buffers.gr_read_mix.mixed_bf, mixer_f16_bytes,
                  "GR mixer BF16");
    if (mix_low) {
        add_gr_output(buffers.gr_read_mix.mixed_bf_lo, mixer_f16_bytes,
                      "GR mixer BF16 remainder");
    }
    add_gr_output(buffers.gr_read_mix.mixed_h, mixer_f16_bytes,
                  "GR mixer F16");

    const PrefillGDNBlockBuffers& gdn = buffers.gdn;
    const PrefillGemmMPPParams qkv_params{
        rows, GDN_CONV_CHANNELS, N_EMBD, GDN_CONV_CHANNELS, 0u, false};
    const PrefillGemmMPPParams gate_params{
        rows, GDN_VALUE_DIM, N_EMBD, GDN_VALUE_DIM, 0u, false};
    const PrefillGemmMPPParams out_params{
        rows, N_EMBD, GDN_VALUE_DIM, N_EMBD, 0u, false};
    PrefillNativeProjectionBufferSizes qkv_projection;
    PrefillNativeProjectionBufferSizes gate_projection;
    PrefillNativeProjectionBufferSizes out_projection;
    if (!prefill_native_projection_buffer_sizes(
            qkv_params, gdn.attn_qkv_format, qkv_projection, error) ||
        !prefill_native_projection_buffer_sizes(
            gate_params, gdn.attn_gate_format, gate_projection, error) ||
        !prefill_native_projection_buffer_sizes(
            out_params, gdn.ssm_out_format, out_projection, error)) {
        return false;
    }

    std::size_t alpha_weights_bytes = 0u;
    std::size_t beta_weights_bytes = 0u;
    std::size_t alpha_beta_bytes = 0u;
    std::size_t head_parameter_bytes = 0u;
    std::size_t history_bytes = 0u;
    std::size_t conv_weights_bytes = 0u;
    std::size_t gate_beta_bytes = 0u;
    std::size_t state_bytes = 0u;
    std::size_t value_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    std::size_t y_h_bytes = 0u;
    if (!checked_bytes(GDN_V_HEADS, N_EMBD, kHalfBytes,
                       alpha_weights_bytes) ||
        !checked_bytes(GDN_V_HEADS, N_EMBD, kHalfBytes,
                       beta_weights_bytes) ||
        !checked_bytes(rows, 2u * GDN_V_HEADS, kFloatBytes,
                       alpha_beta_bytes) ||
        !checked_bytes(GDN_V_HEADS, 1u, kFloatBytes,
                       head_parameter_bytes) ||
        !checked_bytes(GDN_CONV_KERNEL - 1u, GDN_CONV_CHANNELS,
                       kFloatBytes, history_bytes) ||
        !checked_bytes(GDN_CONV_CHANNELS, GDN_CONV_KERNEL, kFloatBytes,
                       conv_weights_bytes) ||
        !checked_bytes(rows, GDN_V_HEADS, kFloatBytes, gate_beta_bytes) ||
        !checked_bytes(1u, GDN_STATE_ELEMENTS, kFloatBytes, state_bytes) ||
        !checked_bytes(rows, GDN_VALUE_DIM, kFloatBytes, value_bytes) ||
        !checked_bytes(1u, GDN_HEAD_DIM, kFloatBytes, gamma_bytes) ||
        !checked_bytes(rows, GDN_VALUE_DIM, kHalfBytes, y_h_bytes)) {
        error = "Prefill attention GDN transition GDN buffer size overflows size_t";
        return false;
    }

    ranges.gdn_weights = {
        memory_range(gdn.attn_qkv_weights, qkv_projection.weights_bytes,
                     "GDN attn_qkv weights"),
        memory_range(gdn.attn_gate_weights, gate_projection.weights_bytes,
                     "GDN attn_gate weights"),
        memory_range(gdn.ssm_alpha_weights, alpha_weights_bytes,
                     "GDN alpha weights"),
        memory_range(gdn.ssm_beta_weights, beta_weights_bytes,
                     "GDN beta weights"),
        memory_range(gdn.ssm_out_weights, out_projection.weights_bytes,
                     "GDN output weights"),
        memory_range(gdn.dt_bias, head_parameter_bytes, "GDN dt bias"),
        memory_range(gdn.a, head_parameter_bytes, "GDN decay"),
        memory_range(gdn.conv_weights, conv_weights_bytes,
                     "GDN convolution weights"),
        memory_range(gdn.gamma, gamma_bytes, "GDN gamma")};
    ranges.gdn_persistent = {
        memory_range(gdn.history, history_bytes, "GDN history"),
        memory_range(gdn.state, state_bytes, "GDN recurrent state")};

    const std::size_t scratch_bytes =
        std::max({qkv_projection.dequantized_f16_bytes,
                  gate_projection.dequantized_f16_bytes,
                  out_projection.dequantized_f16_bytes});
    ranges.gdn_outputs = {
        memory_range(gdn.dequantized_f16, scratch_bytes,
                     "GDN native dequant scratch"),
        memory_range(gdn.qkv, qkv_projection.y_f32_bytes, "GDN qkv"),
        memory_range(gdn.z, gate_projection.y_f32_bytes, "GDN z"),
        memory_range(gdn.alpha_beta, alpha_beta_bytes, "GDN alpha beta"),
        ranges.gdn_persistent[0],
        memory_range(gdn.gate, gate_beta_bytes, "GDN gate"),
        memory_range(gdn.beta, gate_beta_bytes, "GDN beta"),
        memory_range(gdn.h, qkv_projection.y_f32_bytes, "GDN convolution"),
        ranges.gdn_persistent[1],
        memory_range(gdn.oc, value_bytes, "GDN recurrence output"),
        memory_range(gdn.y_h, y_h_bytes, "GDN normalized output"),
        memory_range(gdn.bo, out_projection.y_f32_bytes, "GDN block output")};
    ranges.carried_values = {
        memory_range(read.residual, residual_bytes, "GR residual"),
        memory_range(read.inject, inject_bytes, "GR injection"),
        memory_range(buffers.next_norm, next_norm_bytes, "next GR norm"),
        memory_range(read.rs, read_scale_bytes, "next GR scale output"),
        memory_range(read.xn16, read_stream_f16_bytes,
                     "next GR BF16 output"),
        read_low ? memory_range(read.xn16_lo, read_stream_f16_bytes,
                                "next GR BF16 remainder")
                 : MemoryRange{}};
    ranges.residual_bytes = residual_bytes;
    ranges.inject_bytes = inject_bytes;
    ranges.next_norm_bytes = next_norm_bytes;
    ranges.scale_bytes = read_scale_bytes;
    ranges.stream_bf16_bytes = read_stream_f16_bytes;
    ranges.block_out_bytes = out_projection.y_f32_bytes;
    ranges.stream_f32_bytes = read_stream_f32_bytes;
    ranges.norm_weight_bytes = gr_norm_weight_bytes;
    ranges.down_weight_bytes = gr_down_weight_bytes;
    ranges.up_weight_bytes = gr_up_weight_bytes;
    ranges.inject_weight_bytes = gr_inject_weight_bytes;
    ranges.low_rank_f32_bytes = read_low_rank_f32_bytes;
    ranges.low_rank_bf16_bytes = read_low_rank_f16_bytes;
    ranges.mixer_bytes = mixer_f16_bytes;

    for (const MemoryRange& gr_output :
         std::span<const MemoryRange>(ranges.gr_outputs.data(),
                                      ranges.gr_output_count)) {
        if (ranges_overlap(gr_output, ranges.carried_values[2])) {
            error =
                "Prefill attention GDN transition next GR norm overlaps an earlier "
                "GR output";
            return false;
        }
        for (const MemoryRange& weight : ranges.gr_weights) {
            if (ranges_overlap(gr_output, weight)) {
                error = "Prefill attention GDN transition " +
                        std::string(weight.name) +
                        " overlaps an earlier GR output";
                return false;
            }
        }
        for (const MemoryRange& weight : ranges.gdn_weights) {
            if (ranges_overlap(gr_output, weight)) {
                error = "Prefill attention GDN transition " +
                        std::string(weight.name) +
                        " overlaps an earlier GR output";
                return false;
            }
        }
        for (const MemoryRange& persistent : ranges.gdn_persistent) {
            if (ranges_overlap(gr_output, persistent)) {
                error = "Prefill attention GDN transition earlier GR output overlaps "
                        "persistent " +
                        std::string(persistent.name);
                return false;
            }
        }
    }

    for (const MemoryRange& output : ranges.gdn_outputs) {
        for (const MemoryRange& weight : ranges.gr_weights) {
            if (ranges_overlap(output, weight)) {
                error = "Prefill attention GDN transition " +
                        std::string(output.name) + " overlaps " +
                        std::string(weight.name);
                return false;
            }
        }
        for (const MemoryRange& weight : ranges.gdn_weights) {
            if (ranges_overlap(output, weight)) {
                error = "Prefill attention GDN transition " +
                        std::string(output.name) + " overlaps " +
                        std::string(weight.name);
                return false;
            }
        }
        for (const MemoryRange& carried : ranges.carried_values) {
            if (ranges_overlap(output, carried)) {
                error = "Prefill attention GDN transition " +
                        std::string(output.name) + " overlaps carried " +
                        std::string(carried.name);
                return false;
            }
        }
    }
    for (const MemoryRange& weight : ranges.gdn_weights) {
        if (ranges_overlap(ranges.carried_values[0], weight)) {
            error = "Prefill attention GDN transition residual overlaps " +
                    std::string(weight.name);
            return false;
        }
    }
    for (const MemoryRange& weight : ranges.gr_weights) {
        if (ranges_overlap(ranges.carried_values[0], weight)) {
            error = "Prefill attention GDN transition residual overlaps " +
                    std::string(weight.name);
            return false;
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_attention_gdn_transition(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillAttentionGDNTransitionBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill attention GDN transition requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    LayerRanges ranges;
    if (!build_ranges(context, rows, mode, buffers, ranges, error)) {
        return false;
    }

    PrefillGRReadMixBuffers active_read_mix = buffers.gr_read_mix;
    PrefillGRReadBuffers& read = active_read_mix.read;
    read.residual = read.residual.slice(0u, ranges.residual_bytes);
    read.norm = read.norm.slice(0u, ranges.norm_weight_bytes);
    read.rs = read.rs.slice(0u, ranges.scale_bytes);
    read.xn16 = read.xn16.slice(0u, ranges.stream_bf16_bytes);
    if (read.xn16_lo) {
        read.xn16_lo = read.xn16_lo.slice(0u, ranges.stream_bf16_bytes);
    }
    read.down_weights =
        read.down_weights.slice(0u, ranges.down_weight_bytes);
    read.low_rank = read.low_rank.slice(0u, ranges.low_rank_f32_bytes);
    read.low_rank_bf16 =
        read.low_rank_bf16.slice(0u, ranges.low_rank_bf16_bytes);
    if (read.low_rank_bf16_lo) {
        read.low_rank_bf16_lo =
            read.low_rank_bf16_lo.slice(0u, ranges.low_rank_bf16_bytes);
    }
    read.up_weights = read.up_weights.slice(0u, ranges.up_weight_bytes);
    read.gated = read.gated.slice(0u, ranges.stream_f32_bytes);
    read.inject_weights =
        read.inject_weights.slice(0u, ranges.inject_weight_bytes);
    read.inject = read.inject.slice(0u, ranges.inject_bytes);
    active_read_mix.mixed_bf =
        active_read_mix.mixed_bf.slice(0u, ranges.mixer_bytes);
    if (active_read_mix.mixed_bf_lo) {
        active_read_mix.mixed_bf_lo =
            active_read_mix.mixed_bf_lo.slice(0u, ranges.mixer_bytes);
    }
    active_read_mix.mixed_h =
        active_read_mix.mixed_h.slice(0u, ranges.mixer_bytes);
    const bool active_read_views_valid =
        read.residual && read.norm && read.rs && read.xn16 &&
        read.down_weights && read.low_rank && read.low_rank_bf16 &&
        read.up_weights && read.gated && read.inject_weights &&
        read.inject && active_read_mix.mixed_bf && active_read_mix.mixed_h &&
        (mode != PrefillBF16X2Mode::all_projections ||
         (read.xn16_lo && read.low_rank_bf16_lo)) &&
        (mode == PrefillBF16X2Mode::off ||
         active_read_mix.mixed_bf_lo);
    if (!active_read_views_valid) {
        error = "Prefill attention GDN transition could not create active "
                "GR read/mix views";
        return false;
    }

    PrefillGDNBlockBuffers gdn = buffers.gdn;
    gdn.mixed_h = context.buffers().read_only(active_read_mix.mixed_h);
    gdn.mixed_bf = context.buffers().read_only(active_read_mix.mixed_bf);
    if (mode != PrefillBF16X2Mode::off) {
        gdn.mixed_bf_lo =
            context.buffers().read_only(active_read_mix.mixed_bf_lo);
    }
    const ReadOnlyBufferRef block_out =
        context.buffers().read_only(gdn.bo)
            .slice(0u, ranges.block_out_bytes);
    const ReadOnlyBufferRef inject =
        context.buffers().read_only(read.inject)
            .slice(0u, ranges.inject_bytes);
    const ReadOnlyBufferRef next_norm =
        buffers.next_norm.slice(0u, ranges.next_norm_bytes);
    const BufferRef residual = read.residual.slice(0u, ranges.residual_bytes);
    const BufferRef rs = read.rs.slice(0u, ranges.scale_bytes);
    const BufferRef xn16 = read.xn16.slice(0u, ranges.stream_bf16_bytes);
    const BufferRef xn16_lo =
        mode == PrefillBF16X2Mode::all_projections
            ? read.xn16_lo.slice(0u, ranges.stream_bf16_bytes)
            : BufferRef{};
    if (!block_out || !inject || !next_norm || !residual || !rs || !xn16 ||
        (mode == PrefillBF16X2Mode::all_projections && !xn16_lo)) {
        error = "Prefill attention GDN transition could not create active "
                "buffer views";
        return false;
    }
    if (!encode_prefill_gr_read_mix(context, encoder, rows, mode,
                                    active_read_mix, error) ||
        !encode_prefill_gdn_block(context, encoder, rows, mode, gdn, error)) {
        return false;
    }

    const PrefillGRWriteNormBuffers write_norm{
        residual, block_out, inject, next_norm, rs, xn16, xn16_lo};
    return encode_prefill_gr_write_norm(context, encoder, rows, mode,
                                        write_norm, error);
}

bool encode_prefill_attention_gdn_transition(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillAttentionGDNTransitionBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill attention GDN transition",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_attention_gdn_transition(
                context, encoder, rows, mode, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
