// src/metal/engine/mtp_dispatch.cpp - validated MTP stage encoders.
#include "strata/metal/mtp_dispatch.hpp"

#include "strata/metal/encoder.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

namespace strata::metal {
namespace {

constexpr std::uint32_t kMtpAddStreamsThreads = 256u;
constexpr std::uint32_t kMtpAddStreamsGroupsX =
    HC_DIM / kMtpAddStreamsThreads;
constexpr std::uint32_t kMtpMaxRows =
    static_cast<std::uint32_t>(sizeof(MtpParams{}.tokens) /
                               sizeof(MtpParams{}.tokens[0]));

static_assert(HC_DIM % kMtpAddStreamsThreads == 0u);
static_assert(sizeof(MtpParams) == 236u);
static_assert(offsetof(MtpParams, T) == 0u);

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                   Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool overlaps(ReadOnlyBufferRef input, std::size_t input_bytes,
              BufferRef output, std::size_t output_bytes) {
    const void* const input_contents = input.contents();
    const bool cpu_overlap =
        input_contents != nullptr && output.contents != nullptr &&
        ranges_overlap(
            reinterpret_cast<std::uintptr_t>(input_contents), input_bytes,
            reinterpret_cast<std::uintptr_t>(output.contents), output_bytes);
    const std::uint64_t input_address = input.gpu_address();
    const bool gpu_overlap =
        input_address != 0u && output.gpu_address != 0u &&
        ranges_overlap(input_address, input_bytes, output.gpu_address,
                       output_bytes);
    return cpu_overlap || gpu_overlap;
}

bool validate_read_only_buffer(Context& context, ReadOnlyBufferRef buffer,
                               std::size_t required_bytes,
                               std::size_t alignment, const char* name,
                               std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string("MTP add streams ") + name +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length() < required_bytes ||
        buffer.offset() % alignment != 0u) {
        error = std::string("MTP add streams ") + name +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_output_buffer(Context& context, BufferRef buffer,
                            std::size_t required_bytes, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "MTP add streams output buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes ||
        buffer.offset % alignof(float) != 0u) {
        error = "MTP add streams output buffer is undersized or misaligned";
        return false;
    }
    return true;
}

}  // namespace

bool encode_mtp_add_streams(Context& context, Encoder& encoder,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error) {
    error.clear();
    if (!validate_read_only_buffer(context, buffers.params, sizeof(MtpParams),
                                   alignof(MtpParams), "parameter", error)) {
        return false;
    }

    const void* const parameter_contents = buffers.params.contents();
    if (parameter_contents == nullptr) {
        error = "MTP add streams parameter block has no CPU-visible contents";
        return false;
    }
    std::uint32_t rows = 0u;
    std::memcpy(&rows, parameter_contents, sizeof(rows));
    if (rows == 0u || rows > kMtpMaxRows) {
        error = "MTP add streams T must be in 1..8";
        return false;
    }

    const std::size_t h2_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t e2_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    if (!validate_read_only_buffer(context, buffers.h2, h2_bytes,
                                   alignof(float), "h2", error) ||
        !validate_read_only_buffer(context, buffers.e2, e2_bytes,
                                   alignof(float), "e2", error) ||
        !validate_output_buffer(context, buffers.output, h2_bytes, error)) {
        return false;
    }

    if (overlaps(buffers.params, sizeof(MtpParams), buffers.output, h2_bytes) ||
        overlaps(buffers.h2, h2_bytes, buffers.output, h2_bytes) ||
        overlaps(buffers.e2, e2_bytes, buffers.output, h2_bytes)) {
        error = "MTP add streams output overlaps a read-only input";
        return false;
    }

    const std::array read_only_bindings{
        EncoderReadOnlyBufferBinding{buffers.params, 0u, sizeof(MtpParams),
                                     alignof(MtpParams)},
        EncoderReadOnlyBufferBinding{buffers.h2, 1u, h2_bytes,
                                     alignof(float)},
        EncoderReadOnlyBufferBinding{buffers.e2, 2u, e2_bytes,
                                     alignof(float)},
    };
    const std::array writable_bindings{
        EncoderBufferBinding{buffers.output, 3u, h2_bytes, alignof(float)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_mtp_add_streams", {kMtpAddStreamsGroupsX, rows, 1u},
        {kMtpAddStreamsThreads, 1u, 1u}, writable_bindings,
        read_only_bindings, bytes, threadgroup_memory, error);
}

bool encode_mtp_add_streams(Context& context, CommandBuffer& command_buffer,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP add streams",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_add_streams(context, encoder, buffers,
                                          encode_error);
        },
        error);
}

}  // namespace strata::metal
