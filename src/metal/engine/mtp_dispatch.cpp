// src/metal/engine/mtp_dispatch.cpp - validated MTP stage encoders.
#include "strata/metal/mtp_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

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

bool read_mtp_rows(Context& context, ReadOnlyBufferRef params,
                   const char* operation, std::uint32_t& rows,
                   std::string& error) {
    if (!params || !context.buffers().owns(params)) {
        error = std::string(operation) +
                " parameter block is not owned by this Context";
        return false;
    }
    if (params.length() < sizeof(MtpParams) ||
        params.offset() % alignof(MtpParams) != 0u) {
        error = std::string(operation) +
                " parameter block is undersized or misaligned";
        return false;
    }
    const void* const contents = params.contents();
    if (contents == nullptr) {
        error = std::string(operation) +
                " parameter block has no CPU-visible contents";
        return false;
    }
    std::memcpy(&rows, contents, sizeof(rows));
    if (rows == 0u || rows > kMtpMaxRows) {
        error = std::string(operation) + " T must be in 1..8";
        return false;
    }
    return true;
}

}  // namespace

bool encode_mtp_add_streams(Context& context, Encoder& encoder,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error) {
    error.clear();
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, "MTP add streams", rows,
                       error)) {
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

bool encode_mtp_attention_read(Context& context, Encoder& encoder,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MTP attention read requires an ordered concurrent-group boundary";
        return false;
    }

    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, "MTP attention read", rows,
                       error)) {
        return false;
    }

    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(rows) * HC_LOW_RANK * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
        sizeof(std::uint16_t);
    const std::array<std::pair<BufferRef, std::size_t>, 11> active_views{{
        {buffers.residual, residual_bytes},
        {buffers.dead_h2_scratch, block_bytes + injection_bytes},
        {buffers.norm, static_cast<std::size_t>(HC_DIM) * sizeof(float)},
        {buffers.normalized,
         residual_bytes},
        {buffers.rs, injection_bytes},
        {buffers.hc_down, weight_bytes},
        {buffers.hc_inject,
         static_cast<std::size_t>(HC) * HC_DIM * sizeof(std::uint16_t)},
        {buffers.lo, low_rank_bytes},
        {buffers.inject_out, injection_bytes},
        {buffers.hc_up, weight_bytes},
        {buffers.mixed, block_bytes},
    }};
    for (const auto& [view, active_bytes] : active_views) {
        if (overlaps(buffers.params, sizeof(MtpParams), view,
                     active_bytes)) {
            error = "MTP attention read view overlaps the parameter block";
            return false;
        }
    }

    if (!buffers.dead_h2_scratch ||
        !context.buffers().owns(buffers.dead_h2_scratch)) {
        error = "MTP attention read dead h2 scratch is not owned by this Context";
        return false;
    }
    if (buffers.dead_h2_scratch.length < block_bytes + injection_bytes ||
        buffers.dead_h2_scratch.offset % alignof(float) != 0u) {
        error = "MTP attention read dead h2 scratch is undersized or misaligned";
        return false;
    }
    const BufferRef unused_block_out =
        buffers.dead_h2_scratch.slice(0u, block_bytes);
    const BufferRef unused_injection =
        buffers.dead_h2_scratch.slice(block_bytes, injection_bytes);
    if (!unused_block_out || !unused_injection) {
        error = "MTP attention read cannot form its dead h2 scratch views";
        return false;
    }

    const GRReadBuffers gr_buffers{
        buffers.residual,
        unused_block_out,
        unused_injection,
        buffers.norm,
        buffers.normalized,
        buffers.rs,
        buffers.hc_down,
        buffers.hc_inject,
        buffers.lo,
        buffers.inject_out,
        buffers.hc_up,
        buffers.mixed,
    };
    return encode_gr_read(context, encoder, rows, false, gr_buffers, error);
}

bool encode_mtp_attention_read(Context& context,
                               CommandBuffer& command_buffer,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention read",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_read(context, encoder, buffers,
                                            encode_error);
        },
        error);
}

}  // namespace strata::metal
