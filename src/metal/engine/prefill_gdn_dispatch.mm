// Ordered encoder for Prefill's post-projection GDN transform chain.
#include "strata/metal/prefill_gdn_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_gdn_conv_dispatch.hpp"
#include "strata/metal/prefill_gdn_gates_dispatch.hpp"
#include "strata/metal/prefill_gdn_l2_dispatch.hpp"
#include "strata/metal/prefill_gdn_out_norm_dispatch.hpp"
#include "strata/metal/prefill_gdn_rec_dispatch.hpp"

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

struct MemoryRange {
    void* handle = nullptr;
    std::uintptr_t cpu_address = 0u;
    std::uint64_t gpu_address = 0u;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
};

struct DispatchSizes {
    std::size_t alpha_beta_bytes = 0u;
    std::size_t head_bytes = 0u;
    std::size_t history_bytes = 0u;
    std::size_t qkv_bytes = 0u;
    std::size_t conv_weight_bytes = 0u;
    std::size_t head_rows_bytes = 0u;
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

BufferRef active_view(BufferRef view, std::size_t bytes) {
    return view.slice(0u, bytes);
}

ReadOnlyBufferRef active_view(ReadOnlyBufferRef view, std::size_t bytes) {
    return view.slice(0u, bytes);
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
                   std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN transforms " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % alignment != 0u) {
        error = "Prefill GDN transforms " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_view(Context& context, ReadOnlyBufferRef view,
                   std::size_t required_bytes, std::size_t alignment,
                   std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill GDN transforms " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || view.offset() % alignment != 0u) {
        error = "Prefill GDN transforms " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       const PrefillGDNTransformBuffers& buffers,
                       DispatchSizes& sizes, std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error =
            "Prefill GDN transforms row count exceeds the shader-safe range";
        return false;
    }

    if (!checked_bytes(rows, 2u * GDN_V_HEADS, kFloatBytes,
                       sizes.alpha_beta_bytes) ||
        !checked_bytes(1u, GDN_V_HEADS, kFloatBytes, sizes.head_bytes) ||
        !checked_bytes(GDN_CONV_KERNEL - 1u, GDN_CONV_CHANNELS, kFloatBytes,
                       sizes.history_bytes) ||
        !checked_bytes(rows, GDN_CONV_CHANNELS, kFloatBytes,
                       sizes.qkv_bytes) ||
        !checked_bytes(GDN_CONV_KERNEL, GDN_CONV_CHANNELS, kFloatBytes,
                       sizes.conv_weight_bytes) ||
        !checked_bytes(rows, GDN_V_HEADS, kFloatBytes,
                       sizes.head_rows_bytes) ||
        !checked_bytes(1u, GDN_STATE_ELEMENTS, kFloatBytes,
                       sizes.state_bytes) ||
        !checked_bytes(rows, GDN_VALUE_DIM, kFloatBytes,
                       sizes.value_bytes) ||
        !checked_bytes(1u, GDN_HEAD_DIM, kFloatBytes, sizes.gamma_bytes) ||
        !checked_bytes(rows, GDN_VALUE_DIM, kHalfBytes, sizes.y_h_bytes)) {
        error = "Prefill GDN transforms buffer size overflows size_t";
        return false;
    }

    if (!validate_view(context, buffers.alpha_beta, sizes.alpha_beta_bytes,
                       kFloatBytes, "alpha/beta projection", error) ||
        !validate_view(context, buffers.dt_bias, sizes.head_bytes, kFloatBytes,
                       "dt bias", error) ||
        !validate_view(context, buffers.a, sizes.head_bytes, kFloatBytes, "A",
                       error) ||
        !validate_view(context, buffers.history, sizes.history_bytes,
                       kFloatBytes,
                       "history", error) ||
        !validate_view(context, buffers.qkv, sizes.qkv_bytes, kFloatBytes,
                       "QKV",
                       error) ||
        !validate_view(context, buffers.conv_weights, sizes.conv_weight_bytes,
                       kFloatBytes, "convolution weights", error) ||
        !validate_view(context, buffers.gate, sizes.head_rows_bytes,
                       kFloatBytes, "gate", error) ||
        !validate_view(context, buffers.beta, sizes.head_rows_bytes,
                       kFloatBytes, "beta", error) ||
        !validate_view(context, buffers.h, sizes.qkv_bytes, kFloatBytes, "h",
                       error) ||
        !validate_view(context, buffers.state, sizes.state_bytes, kFloatBytes,
                       "state", error) ||
        !validate_view(context, buffers.oc, sizes.value_bytes, kFloatBytes, "oc",
                       error) ||
        !validate_view(context, buffers.z, sizes.value_bytes, kFloatBytes, "z",
                       error) ||
        !validate_view(context, buffers.gamma, sizes.gamma_bytes, kFloatBytes,
                       "gamma", error) ||
        !validate_view(context, buffers.y_h, sizes.y_h_bytes, kHalfBytes, "y_h",
                       error)) {
        return false;
    }

    const std::array<MemoryRange, 7u> writable{
        memory_range(buffers.history, sizes.history_bytes, "history"),
        memory_range(buffers.gate, sizes.head_rows_bytes, "gate"),
        memory_range(buffers.beta, sizes.head_rows_bytes, "beta"),
        memory_range(buffers.h, sizes.qkv_bytes, "h"),
        memory_range(buffers.state, sizes.state_bytes, "state"),
        memory_range(buffers.oc, sizes.value_bytes, "oc"),
        memory_range(buffers.y_h, sizes.y_h_bytes, "y_h")};
    const std::array<MemoryRange, 7u> read_only{
        memory_range(buffers.alpha_beta, sizes.alpha_beta_bytes, "alpha/beta"),
        memory_range(buffers.dt_bias, sizes.head_bytes, "dt bias"),
        memory_range(buffers.a, sizes.head_bytes, "A"),
        memory_range(buffers.qkv, sizes.qkv_bytes, "QKV"),
        memory_range(buffers.conv_weights, sizes.conv_weight_bytes,
                     "convolution weights"),
        memory_range(buffers.z, sizes.value_bytes, "z"),
        memory_range(buffers.gamma, sizes.gamma_bytes, "gamma")};
    for (std::size_t output = 0u; output < writable.size(); ++output) {
        for (std::size_t earlier = 0u; earlier < output; ++earlier) {
            if (ranges_overlap(writable[output], writable[earlier])) {
                error = "Prefill GDN transforms writable stages overlap: " +
                        std::string(writable[earlier].name) + " and " +
                        std::string(writable[output].name);
                return false;
            }
        }
        for (const MemoryRange& input : read_only) {
            if (ranges_overlap(writable[output], input)) {
                error = "Prefill GDN transforms writable " +
                        std::string(writable[output].name) +
                        " overlaps read-only " + std::string(input.name);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool encode_prefill_gdn_transforms(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const PrefillGDNTransformBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GDN transforms require an ordered "
                "concurrent-group boundary";
        return false;
    }
    DispatchSizes sizes;
    if (!validate_dispatch(context, rows, buffers, sizes, error)) return false;

    const ReadOnlyBufferRef alpha_beta =
        active_view(buffers.alpha_beta, sizes.alpha_beta_bytes);
    const ReadOnlyBufferRef dt_bias =
        active_view(buffers.dt_bias, sizes.head_bytes);
    const ReadOnlyBufferRef a = active_view(buffers.a, sizes.head_bytes);
    const BufferRef history =
        active_view(buffers.history, sizes.history_bytes);
    const ReadOnlyBufferRef qkv =
        active_view(buffers.qkv, sizes.qkv_bytes);
    const ReadOnlyBufferRef conv_weights =
        active_view(buffers.conv_weights, sizes.conv_weight_bytes);
    const BufferRef gate =
        active_view(buffers.gate, sizes.head_rows_bytes);
    const BufferRef beta =
        active_view(buffers.beta, sizes.head_rows_bytes);
    const BufferRef h = active_view(buffers.h, sizes.qkv_bytes);
    const BufferRef state =
        active_view(buffers.state, sizes.state_bytes);
    const BufferRef oc = active_view(buffers.oc, sizes.value_bytes);
    const ReadOnlyBufferRef z =
        active_view(buffers.z, sizes.value_bytes);
    const ReadOnlyBufferRef gamma =
        active_view(buffers.gamma, sizes.gamma_bytes);
    const BufferRef y_h = active_view(buffers.y_h, sizes.y_h_bytes);

    if (!encode_prefill_gdn_gates(
            context, encoder, rows, {alpha_beta, dt_bias, a, gate, beta},
            error) ||
        !encode_prefill_gdn_conv_tiled(
            context, encoder, rows,
            {context.buffers().read_only(history), qkv, conv_weights, h},
            error) ||
        !encode_prefill_gdn_conv_history(context, encoder, rows,
                                         {history, qkv}, error) ||
        !encode_prefill_gdn_l2(context, encoder, rows, h, error) ||
        !encode_prefill_gdn_rec(
            context, encoder, rows,
            {state, context.buffers().read_only(h),
             context.buffers().read_only(gate),
             context.buffers().read_only(beta), oc},
            error)) {
        return false;
    }
    return encode_prefill_gdn_out_norm(
        context, encoder, rows,
        {context.buffers().read_only(oc), z, gamma, y_h},
        error);
}

bool encode_prefill_gdn_transforms(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const PrefillGDNTransformBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN transforms",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_transforms(
                context, encoder, rows, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
