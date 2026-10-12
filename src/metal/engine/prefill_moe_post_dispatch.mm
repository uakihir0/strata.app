// Validated Prefill MoE SwiGLU and weighted-combine encoders.
#include "strata/metal/prefill_moe_post.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::size_t kThreads = 256u;
constexpr std::size_t kSwiGLUWidth = 640u;
constexpr std::size_t kModelWidth = 2560u;
constexpr std::size_t kTopK = 10u;
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);

struct ActiveRange {
    const void* cpu = nullptr;
    std::uint64_t gpu = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
};

bool checked_product(std::initializer_list<std::size_t> factors,
                     std::size_t& result) {
    result = 1u;
    for (const std::size_t factor : factors) {
        if (factor != 0u &&
            result > std::numeric_limits<std::size_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

std::size_t round_up_div(std::size_t value, std::size_t divisor) {
    return value / divisor +
           static_cast<std::size_t>(value % divisor != 0u);
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

bool overlaps(const ActiveRange& left, const ActiveRange& right) {
    const bool cpu_overlap =
        left.cpu != nullptr && right.cpu != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.cpu), left.bytes,
                       reinterpret_cast<std::uintptr_t>(right.cpu),
                       right.bytes);
    const bool gpu_overlap =
        left.gpu != 0u && right.gpu != 0u &&
        ranges_overlap(left.gpu, left.bytes, right.gpu, right.bytes);
    return cpu_overlap || gpu_overlap;
}

ActiveRange read_range(ReadOnlyBufferRef view, std::size_t bytes,
                       std::string_view name) {
    return ActiveRange{view.contents(), view.gpu_address(), bytes, name};
}

ActiveRange write_range(BufferRef view, std::size_t bytes,
                        std::string_view name) {
    return ActiveRange{view.contents, view.gpu_address, bytes, name};
}

bool validate_read_view(Context& context, ReadOnlyBufferRef view,
                        std::size_t required_bytes, std::size_t alignment,
                        std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill MoE post " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || alignment == 0u ||
        view.offset() % alignment != 0u) {
        error = "Prefill MoE post " + std::string(name) +
                " view is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill MoE post " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || alignment == 0u ||
        view.offset % alignment != 0u) {
        error = "Prefill MoE post " + std::string(name) +
                " view is undersized or misaligned";
        return false;
    }
    return true;
}

ReadOnlyBufferRef active_view(ReadOnlyBufferRef view,
                              std::size_t active_bytes) {
    return view.slice(0u, active_bytes);
}

BufferRef active_view(BufferRef view, std::size_t active_bytes) {
    return view.slice(0u, active_bytes);
}

bool validate_swiglu(Context& context, std::uint32_t rows,
                     const PrefillMoeSwiGLUBuffers& buffers,
                     std::size_t& input_bytes, std::size_t& output_bytes,
                     std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill MoE SwiGLU row count exceeds the shader-safe range";
        return false;
    }
    if (!checked_product({rows, kSwiGLUWidth, kFloatBytes}, input_bytes) ||
        !checked_product({rows, kSwiGLUWidth, kHalfBytes}, output_bytes)) {
        error = "Prefill MoE SwiGLU buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.gate, input_bytes, kFloatBytes,
                            "gate", error) ||
        !validate_read_view(context, buffers.up, input_bytes, kFloatBytes,
                            "up", error) ||
        !validate_write_view(context, buffers.output, output_bytes,
                             kHalfBytes, "SwiGLU output", error)) {
        return false;
    }

    const ActiveRange output =
        write_range(buffers.output, output_bytes, "SwiGLU output");
    const std::array inputs{read_range(buffers.gate, input_bytes, "gate"),
                            read_range(buffers.up, input_bytes, "up")};
    for (const ActiveRange& input : inputs) {
        if (overlaps(output, input)) {
            error = "Prefill MoE SwiGLU output overlaps active " +
                    std::string(input.name);
            return false;
        }
    }
    return true;
}

bool validate_combine(Context& context, std::uint32_t rows,
                      const PrefillMoeCombineBuffers& buffers,
                      std::size_t& parts_bytes, std::size_t& weights_bytes,
                      std::size_t& row_bytes, std::size_t& gate_bytes,
                      std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill MoE combine row count exceeds the shader-safe range";
        return false;
    }
    if (!checked_product({rows, kTopK, kModelWidth, kFloatBytes},
                         parts_bytes) ||
        !checked_product({rows, kTopK, kFloatBytes}, weights_bytes) ||
        !checked_product({rows, kModelWidth, kFloatBytes}, row_bytes) ||
        !checked_product({rows, kFloatBytes}, gate_bytes)) {
        error = "Prefill MoE combine buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.parts, parts_bytes, kFloatBytes,
                            "routed parts", error) ||
        !validate_read_view(context, buffers.weights, weights_bytes,
                            kFloatBytes, "route weights", error) ||
        !validate_read_view(context, buffers.shared, row_bytes, kFloatBytes,
                            "shared expert output", error) ||
        !validate_read_view(context, buffers.gate, gate_bytes, kFloatBytes,
                            "shared gate", error) ||
        !validate_write_view(context, buffers.output, row_bytes, kFloatBytes,
                             "combined output", error)) {
        return false;
    }

    const ActiveRange output =
        write_range(buffers.output, row_bytes, "combined output");
    const std::array inputs{
        read_range(buffers.parts, parts_bytes, "routed parts"),
        read_range(buffers.weights, weights_bytes, "route weights"),
        read_range(buffers.shared, row_bytes, "shared expert output"),
        read_range(buffers.gate, gate_bytes, "shared gate"),
    };
    for (const ActiveRange& input : inputs) {
        if (overlaps(output, input)) {
            error = "Prefill MoE combine output overlaps active " +
                    std::string(input.name);
            return false;
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_moe_swiglu(Context& context, Encoder& encoder,
                               std::uint32_t rows,
                               const PrefillMoeSwiGLUBuffers& buffers,
                               std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill MoE SwiGLU requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!validate_swiglu(context, rows, buffers, input_bytes, output_bytes,
                         error)) {
        return false;
    }

    const std::size_t values =
        static_cast<std::size_t>(rows) * kSwiGLUWidth;
    const std::uint32_t row_parameter = rows;
    const auto gate = active_view(buffers.gate, input_bytes);
    const auto up = active_view(buffers.up, input_bytes);
    const auto output = active_view(buffers.output, output_bytes);
    const std::array<EncoderBufferBinding, 1u> outputs{{
        {output, 2u, output_bytes, kHalfBytes},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 2u> inputs{{
        {gate, 0u, input_bytes, kFloatBytes},
        {up, 1u, input_bytes, kFloatBytes},
    }};
    const std::array<EncoderBytesBinding, 1u> bytes{{
        {&row_parameter, sizeof(row_parameter), 3u},
    }};
    constexpr std::array<EncoderThreadgroupMemory, 0u> no_threadgroup{};
    return encoder.dispatch(
        "strata_pf_swiglu_pair", {round_up_div(values, kThreads), 1u, 1u},
        {kThreads, 1u, 1u}, outputs, inputs, bytes, no_threadgroup, error);
}

bool encode_prefill_moe_swiglu(Context& context,
                               CommandBuffer& command_buffer,
                               std::uint32_t rows,
                               const PrefillMoeSwiGLUBuffers& buffers,
                               std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE SwiGLU",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_moe_swiglu(context, encoder, rows, buffers,
                                              encode_error);
        },
        error);
}

bool encode_prefill_moe_combine(Context& context, Encoder& encoder,
                                std::uint32_t rows,
                                const PrefillMoeCombineBuffers& buffers,
                                std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill MoE combine requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    std::size_t parts_bytes = 0u;
    std::size_t weights_bytes = 0u;
    std::size_t row_bytes = 0u;
    std::size_t gate_bytes = 0u;
    if (!validate_combine(context, rows, buffers, parts_bytes, weights_bytes,
                          row_bytes, gate_bytes, error)) {
        return false;
    }

    const std::size_t route_weight_bytes = weights_bytes;
    const auto parts = active_view(buffers.parts, parts_bytes);
    const auto weights = active_view(buffers.weights, route_weight_bytes);
    const auto shared = active_view(buffers.shared, row_bytes);
    const auto gate = active_view(buffers.gate, gate_bytes);
    const auto output = active_view(buffers.output, row_bytes);
    const std::array<EncoderBufferBinding, 1u> outputs{{
        {output, 4u, row_bytes, kFloatBytes},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 4u> inputs{{
        {parts, 0u, parts_bytes, kFloatBytes},
        {weights, 1u, route_weight_bytes, kFloatBytes},
        {shared, 2u, row_bytes, kFloatBytes},
        {gate, 3u, gate_bytes, kFloatBytes},
    }};
    constexpr std::array<EncoderBytesBinding, 0u> no_bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0u> no_threadgroup{};
    return encoder.dispatch(
        "strata_pf_combine", {kTopK, rows, 1u}, {kThreads, 1u, 1u},
        outputs, inputs, no_bytes, no_threadgroup, error);
}

bool encode_prefill_moe_combine(Context& context,
                                CommandBuffer& command_buffer,
                                std::uint32_t rows,
                                const PrefillMoeCombineBuffers& buffers,
                                std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE combine",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_moe_combine(context, encoder, rows, buffers,
                                               encode_error);
        },
        error);
}

}  // namespace strata::metal
