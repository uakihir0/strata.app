// Validated encoder for Prefill GDN q/k head normalization.
#include "strata/metal/prefill_gdn_l2_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include "../kernels/common/params.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace strata::metal {
namespace {

constexpr std::uint32_t kHeadGroups = 32u;
constexpr std::uint32_t kThreads = 128u;
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

bool validate_dispatch(Context& context, std::uint32_t rows, BufferRef h,
                       std::size_t& required_bytes, std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill GDN L2 row count exceeds the shader-safe range";
        return false;
    }
    if (!checked_float_bytes(rows, GDN_CONV_CHANNELS, required_bytes)) {
        error = "Prefill GDN L2 buffer size overflows size_t";
        return false;
    }
    if (!h || !context.buffers().owns(h)) {
        error = "Prefill GDN L2 buffer is not owned by this Context";
        return false;
    }
    if (h.length < required_bytes || h.offset % kFloatBytes != 0u) {
        error = "Prefill GDN L2 buffer is undersized or misaligned";
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_gdn_l2(Context& context, Encoder& encoder,
                           std::uint32_t rows, BufferRef h,
                           std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill GDN L2 requires an ordered concurrent-group boundary";
        return false;
    }

    std::size_t required_bytes = 0u;
    if (!validate_dispatch(context, rows, h, required_bytes, error)) {
        return false;
    }

    const EncoderBufferBinding output{h, 0u, required_bytes, kFloatBytes};
    return encoder.dispatch(
        "strata_pf_gdn_l2", {kHeadGroups, rows, 1u}, {kThreads, 1u, 1u},
        std::span<const EncoderBufferBinding>(&output, 1u), {}, {}, error);
}

bool encode_prefill_gdn_l2(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, BufferRef h,
                           std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill GDN L2",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_l2(context, encoder, rows, h,
                                         encode_error);
        },
        error);
}

}  // namespace strata::metal
