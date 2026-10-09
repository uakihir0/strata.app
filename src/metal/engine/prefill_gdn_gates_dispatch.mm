// Validated encoder for Prefill's post-projection GDN gates.
#include "strata/metal/prefill_gdn_gates_dispatch.hpp"
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

constexpr std::uint32_t kHeads = GDN_V_HEADS;
constexpr std::uint32_t kABValuesPerRow = 2u * kHeads;
constexpr std::uint32_t kThreads = 256u;
constexpr std::size_t kFloatBytes = sizeof(float);

bool checked_float_bytes(std::uint32_t rows, std::uint32_t values_per_row,
                         std::size_t& bytes) {
    const std::size_t max_values =
        std::numeric_limits<std::size_t>::max() / kFloatBytes;
    if (values_per_row != 0u &&
        static_cast<std::size_t>(rows) > max_values / values_per_row) {
        return false;
    }
    bytes = static_cast<std::size_t>(rows) * values_per_row * kFloatBytes;
    return true;
}

bool validate_read_view(Context& context, ReadOnlyBufferRef view,
                        std::size_t required_bytes, std::string_view name,
                        std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN gates " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes ||
        view.offset() % kFloatBytes != 0u) {
        error = "Prefill GDN gates " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::string_view name,
                         std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN gates " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes ||
        view.offset % kFloatBytes != 0u) {
        error = "Prefill GDN gates " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                    Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0 || right_bytes == 0) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool views_overlap(BufferRef left, BufferRef right) {
    const bool same_allocation =
        left.handle == right.handle &&
        ranges_overlap(left.offset, left.length, right.offset, right.length);
    const bool same_cpu_backing =
        left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.contents),
                       left.length,
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length);
    const bool same_gpu_backing =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address, left.length, right.gpu_address,
                       right.length);
    return same_allocation || same_cpu_backing || same_gpu_backing;
}

bool views_overlap(ReadOnlyBufferRef left, BufferRef right) {
    const void* const left_contents = left.contents();
    const bool same_cpu_backing =
        left_contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left.length(),
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length);
    const bool same_gpu_backing =
        left.gpu_address() != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address(), left.length(), right.gpu_address,
                       right.length);
    return same_cpu_backing || same_gpu_backing;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       const PrefillGDNGatesBuffers& buffers,
                       std::size_t& ab_bytes, std::size_t& head_bytes,
                       std::size_t& output_bytes, std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GDN gates row count exceeds the shader-safe range";
        return false;
    }
    if (!checked_float_bytes(rows, kABValuesPerRow, ab_bytes) ||
        !checked_float_bytes(1u, kHeads, head_bytes) ||
        !checked_float_bytes(rows, kHeads, output_bytes)) {
        error = "Prefill GDN gates buffer size overflows size_t";
        return false;
    }
    if (!validate_read_view(context, buffers.alpha_beta, ab_bytes,
                            "alpha/beta projection", error) ||
        !validate_read_view(context, buffers.dt_bias, head_bytes,
                            "dt bias", error) ||
        !validate_read_view(context, buffers.a, head_bytes, "A", error) ||
        !validate_write_view(context, buffers.gate, output_bytes, "gate",
                             error) ||
        !validate_write_view(context, buffers.beta, output_bytes, "beta",
                             error)) {
        return false;
    }

    const std::array<ReadOnlyBufferRef, 3> inputs{
        buffers.alpha_beta, buffers.dt_bias, buffers.a};
    const std::array<BufferRef, 2> outputs{buffers.gate, buffers.beta};
    for (std::size_t out = 0u; out < outputs.size(); ++out) {
        for (const ReadOnlyBufferRef& input : inputs) {
            if (views_overlap(input, outputs[out])) {
                error = "Prefill GDN gates output overlaps a read-only input";
                return false;
            }
        }
        for (std::size_t earlier = 0u; earlier < out; ++earlier) {
            if (views_overlap(outputs[out], outputs[earlier])) {
                error = "Prefill GDN gates output buffer views overlap";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_gdn_gates(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNGatesBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "Prefill GDN gates require an ordered concurrent-group boundary";
        return false;
    }

    std::size_t ab_bytes = 0u;
    std::size_t head_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!validate_dispatch(context, rows, buffers, ab_bytes, head_bytes,
                           output_bytes, error)) {
        return false;
    }

    const EncoderReadOnlyBufferBinding inputs[] = {
        {buffers.alpha_beta, 0u, ab_bytes, kFloatBytes},
        {buffers.dt_bias, 1u, head_bytes, kFloatBytes},
        {buffers.a, 2u, head_bytes, kFloatBytes},
    };
    const EncoderBufferBinding outputs[] = {
        {buffers.gate, 3u, output_bytes, kFloatBytes},
        {buffers.beta, 4u, output_bytes, kFloatBytes},
    };
    const EncoderBytesBinding row_binding{&rows, sizeof(rows), 5u};
    const std::size_t value_count = static_cast<std::size_t>(rows) * kHeads;
    constexpr EncoderThreadgroupMemory kNoThreadgroupMemory{};
    return encoder.dispatch(
        "strata_pf_gdn_gates",
        {(value_count + kThreads - 1u) / kThreads, 1u, 1u},
        {kThreads, 1u, 1u}, outputs, inputs,
        std::span<const EncoderBytesBinding>(&row_binding, 1u),
        std::span<const EncoderThreadgroupMemory>(&kNoThreadgroupMemory, 0u),
        error);
}

bool encode_prefill_gdn_gates(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNGatesBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN gates",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_gates(context, encoder, rows, buffers,
                                            encode_error);
        },
        error);
}

}  // namespace strata::metal
