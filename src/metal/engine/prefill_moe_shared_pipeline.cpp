// Ordered shared-expert Prefill MoE projections and gate composition.
#include "strata/metal/prefill_moe_shared_pipeline.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_moe_post.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kHiddenWidth = 2560u;
constexpr std::uint32_t kFFNWidth = 640u;
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::size_t kFloatBytes = sizeof(float);

struct MemoryRange {
    void* handle = nullptr;
    std::uintptr_t cpu_address = 0u;
    std::uint64_t gpu_address = 0u;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
};

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t& result) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool matrix_bytes(std::uint32_t rows, std::uint32_t columns,
                 std::size_t element_bytes, std::size_t& bytes) {
    std::size_t elements = 0u;
    return checked_multiply(rows, columns, elements) &&
           checked_multiply(elements, element_bytes, bytes);
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0u || right == 0u || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
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

bool add_range(Context& context, BufferRef view, std::size_t bytes,
               std::size_t alignment, std::string_view name,
               std::vector<MemoryRange>& ranges, std::string& error) {
    if (!view || !context.buffers().owns(view) || bytes == 0u ||
        view.length < bytes || alignment == 0u ||
        view.offset % alignment != 0u) {
        error = "Prefill MoE shared " + std::string(name) +
                " buffer is foreign, undersized or misaligned";
        return false;
    }
    ranges.push_back({view.handle,
                      reinterpret_cast<std::uintptr_t>(view.contents),
                      view.gpu_address, view.offset, bytes, name});
    return true;
}

bool add_range(Context& context, ReadOnlyBufferRef view, std::size_t bytes,
               std::size_t alignment, std::string_view name,
               std::vector<MemoryRange>& ranges, std::string& error) {
    if (!view || !context.buffers().owns(view) || bytes == 0u ||
        view.length() < bytes || alignment == 0u ||
        view.offset() % alignment != 0u) {
        error = "Prefill MoE shared " + std::string(name) +
                " buffer is foreign, undersized or misaligned";
        return false;
    }
    ranges.push_back({nullptr,
                      reinterpret_cast<std::uintptr_t>(view.contents()),
                      view.gpu_address(), view.offset(), bytes, name});
    return true;
}

bool validate_ranges(Context& context, std::uint32_t rows,
                     const PrefillMoeSharedFFNFormats& formats,
                     const PrefillMoeSharedFFNBuffers& buffers,
                     PrefillNativeProjectionBufferSizes& gate_sizes,
                     PrefillNativeProjectionBufferSizes& up_sizes,
                     PrefillNativeProjectionBufferSizes& down_sizes,
                     std::string& error) {
    if (rows == 0u || rows > 8u) {
        error = "Prefill MoE shared FFN supports 1..8 rows";
        return false;
    }
    const PrefillGemmMPPParams gate_params{
        rows, kFFNWidth, kHiddenWidth, kFFNWidth, 0u, false};
    const PrefillGemmMPPParams up_params = gate_params;
    const PrefillGemmMPPParams down_params{
        rows, kHiddenWidth, kFFNWidth, kHiddenWidth, 0u, false};
    if (!prefill_native_projection_buffer_sizes(
            gate_params, formats.gate, gate_sizes, error) ||
        !prefill_native_projection_buffer_sizes(
            up_params, formats.up, up_sizes, error) ||
        !prefill_native_projection_buffer_sizes(
            down_params, formats.down, down_sizes, error)) {
        return false;
    }

    std::size_t hidden_input_bytes = 0u;
    std::size_t gate_up_output_bytes = 0u;
    std::size_t hidden_output_bytes = 0u;
    std::size_t down_output_bytes = 0u;
    std::size_t router_input_bytes = 0u;
    std::size_t router_weight_bytes = 0u;
    std::size_t router_output_bytes = 0u;
    if (!matrix_bytes(rows, kHiddenWidth, kHalfBytes, hidden_input_bytes) ||
        !matrix_bytes(rows, kFFNWidth, kFloatBytes,
                      gate_up_output_bytes) ||
        !matrix_bytes(rows, kFFNWidth, kHalfBytes, hidden_output_bytes) ||
        !matrix_bytes(rows, kHiddenWidth, kFloatBytes, down_output_bytes) ||
        !matrix_bytes(rows, kHiddenWidth, kHalfBytes, router_input_bytes) ||
        !matrix_bytes(1u, kHiddenWidth, kHalfBytes, router_weight_bytes) ||
        !matrix_bytes(rows, 1u, kFloatBytes, router_output_bytes)) {
        error = "Prefill MoE shared FFN extent overflows size_t";
        return false;
    }

    std::vector<MemoryRange> inputs;
    std::vector<MemoryRange> outputs;
    inputs.reserve(7u);
    outputs.reserve(8u);
    if (!add_range(context, buffers.mixed_h, hidden_input_bytes, 4u,
                   "mixed F16 input", inputs, error) ||
        !add_range(context, buffers.gate_weights, gate_sizes.weights_bytes,
                   gate_sizes.weights_alignment, "gate weights", inputs,
                   error) ||
        !add_range(context, buffers.up_weights, up_sizes.weights_bytes,
                   up_sizes.weights_alignment, "up weights", inputs, error) ||
        !add_range(context, buffers.down_weights, down_sizes.weights_bytes,
                   down_sizes.weights_alignment, "down weights", inputs,
                   error) ||
        !add_range(context, buffers.router_gate.x_hi, router_input_bytes, 4u,
                   "shared gate high input", inputs, error) ||
        !add_range(context, buffers.router_gate.x_lo, router_input_bytes, 4u,
                   "shared gate low input", inputs, error) ||
        !add_range(context, buffers.router_gate.w, router_weight_bytes, 2u,
                   "shared gate weights", inputs, error) ||
        !add_range(context, buffers.gate_dequantized_f16,
                   gate_sizes.dequantized_f16_bytes, 2u,
                   "gate dequantization scratch", outputs, error) ||
        !add_range(context, buffers.gate_f32, gate_sizes.y_f32_bytes, 4u,
                   "gate output", outputs, error) ||
        !add_range(context, buffers.up_dequantized_f16,
                   up_sizes.dequantized_f16_bytes, 2u,
                   "up dequantization scratch", outputs, error) ||
        !add_range(context, buffers.up_f32, up_sizes.y_f32_bytes, 4u,
                   "up output", outputs, error) ||
        !add_range(context, buffers.hidden_f16, hidden_output_bytes, 2u,
                   "SwiGLU output", outputs, error) ||
        !add_range(context, buffers.down_dequantized_f16,
                   down_sizes.dequantized_f16_bytes, 2u,
                   "down dequantization scratch", outputs, error) ||
        !add_range(context, buffers.shared_f32, down_output_bytes, 4u,
                   "shared down output", outputs, error) ||
        !add_range(context, buffers.router_gate.y, router_output_bytes, 4u,
                   "shared gate output", outputs, error)) {
        return false;
    }

    for (std::size_t output = 0u; output < outputs.size(); ++output) {
        for (const auto& input : inputs) {
            if (ranges_overlap(outputs[output], input)) {
                error = "Prefill MoE shared " +
                        std::string(outputs[output].name) +
                        " overlaps input " + std::string(input.name);
                return false;
            }
        }
        for (std::size_t previous = 0u; previous < output; ++previous) {
            if (ranges_overlap(outputs[output], outputs[previous])) {
                error = "Prefill MoE shared " +
                        std::string(outputs[output].name) + " overlaps " +
                        std::string(outputs[previous].name);
                return false;
            }
        }
    }
    return true;
}

bool encode_shared_ffn(Context& context, Encoder& encoder,
                       std::uint32_t rows,
                       PrefillBF16X2Mode projection_mode,
                       const PrefillMoeSharedFFNFormats& formats,
                       const PrefillMoeSharedFFNBuffers& buffers,
                       std::string& error) {
    if (projection_mode != PrefillBF16X2Mode::all_projections &&
        projection_mode != PrefillBF16X2Mode::except_gr) {
        error = "Prefill MoE shared gate requires BF16X2 projection mode 1 or 2";
        return false;
    }
    PrefillNativeProjectionBufferSizes gate_sizes;
    PrefillNativeProjectionBufferSizes up_sizes;
    PrefillNativeProjectionBufferSizes down_sizes;
    if (!validate_ranges(context, rows, formats, buffers, gate_sizes,
                         up_sizes, down_sizes, error)) {
        return false;
    }

    const PrefillGemmMPPParams gate_params{
        rows, kFFNWidth, kHiddenWidth, kFFNWidth, 0u, false};
    const PrefillNativeProjectionBuffers gate_buffers{
        buffers.mixed_h, buffers.gate_weights,
        buffers.gate_dequantized_f16, buffers.gate_f32};
    if (!encode_prefill_native_projection(
            context, encoder, gate_params, formats.gate, gate_buffers,
            error)) {
        return false;
    }

    const PrefillNativeProjectionBuffers up_buffers{
        buffers.mixed_h, buffers.up_weights,
        buffers.up_dequantized_f16, buffers.up_f32};
    if (!encode_prefill_native_projection(
            context, encoder, gate_params, formats.up, up_buffers, error)) {
        return false;
    }

    const PrefillMoeSwiGLUBuffers swiglu_buffers{
        context.buffers().read_only(buffers.gate_f32),
        context.buffers().read_only(buffers.up_f32), buffers.hidden_f16};
    if (!encode_prefill_moe_swiglu(context, encoder, rows, swiglu_buffers,
                                   error)) {
        return false;
    }

    const PrefillGemmMPPParams down_params{
        rows, kHiddenWidth, kFFNWidth, kHiddenWidth, 0u, false};
    const PrefillNativeProjectionBuffers down_buffers{
        context.buffers().read_only(buffers.hidden_f16),
        buffers.down_weights, buffers.down_dequantized_f16,
        buffers.shared_f32};
    if (!encode_prefill_native_projection(
            context, encoder, down_params, formats.down, down_buffers,
            error)) {
        return false;
    }

    const PrefillGemmMPPParams router_gate_params{
        rows, 1u, kHiddenWidth, 1u, 0u, false};
    if (!encode_prefill_gemm_bf16x2(context, encoder, router_gate_params,
                                    buffers.router_gate, error)) {
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_prefill_moe_shared_ffn(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeSharedFFNFormats& formats,
    const PrefillMoeSharedFFNBuffers& buffers, std::string& error) {
    return encode_shared_ffn(context, encoder, rows, projection_mode, formats,
                              buffers, error);
}

bool encode_prefill_moe_shared_ffn(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode projection_mode,
    const PrefillMoeSharedFFNFormats& formats,
    const PrefillMoeSharedFFNBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE shared FFN",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_shared_ffn(context, encoder, rows, projection_mode,
                                     formats, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
