// src/metal/engine/mtp_dispatch.cpp - validated MTP stage encoders.
#include "strata/metal/mtp_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"

#include <algorithm>
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

constexpr std::uint32_t kMtpKvProjectionWidth = 512u;
constexpr std::uint32_t kMtpQ8BlockValues = 32u;
constexpr std::uint32_t kMtpQ81BlockBytes = 36u;
constexpr std::uint32_t kMtpAddStreamsThreads = 256u;
constexpr std::uint32_t kMtpAddStreamsGroupsX =
    HC_DIM / kMtpAddStreamsThreads;
constexpr std::uint32_t kMtpMaxRows =
    static_cast<std::uint32_t>(sizeof(MtpParams{}.tokens) /
                               sizeof(MtpParams{}.tokens[0]));

static_assert(HC_DIM % kMtpAddStreamsThreads == 0u);
static_assert(sizeof(MtpParams) == 236u);
static_assert(offsetof(MtpParams, T) == 0u);
static_assert(sizeof(Q81) == kMtpQ81BlockBytes);

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

bool overlaps(BufferRef left, std::size_t left_bytes, BufferRef right,
              std::size_t right_bytes) {
    const bool cpu_overlap =
        left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.contents), left_bytes,
            reinterpret_cast<std::uintptr_t>(right.contents), right_bytes);
    const bool gpu_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address, left_bytes, right.gpu_address,
                       right_bytes);
    return cpu_overlap || gpu_overlap;
}

bool overlaps(ReadOnlyBufferRef left, std::size_t left_bytes,
              ReadOnlyBufferRef right, std::size_t right_bytes) {
    const void* const left_contents = left.contents();
    const void* const right_contents = right.contents();
    const bool cpu_overlap =
        left_contents != nullptr && right_contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left_bytes,
                       reinterpret_cast<std::uintptr_t>(right_contents),
                       right_bytes);
    const std::uint64_t left_address = left.gpu_address();
    const std::uint64_t right_address = right.gpu_address();
    const bool gpu_overlap =
        left_address != 0u && right_address != 0u &&
        ranges_overlap(left_address, left_bytes, right_address, right_bytes);
    return cpu_overlap || gpu_overlap;
}

bool validate_read_only_buffer(Context& context, ReadOnlyBufferRef buffer,
                               std::size_t required_bytes,
                               std::size_t alignment, const char* name,
                               const char* operation, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(operation) + " " + name +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length() < required_bytes ||
        buffer.offset() % alignment != 0u) {
        error = std::string(operation) + " " + name +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_output_buffer(Context& context, BufferRef buffer,
                            std::size_t required_bytes,
                            const char* operation, const char* name,
                            std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(operation) + " " + name +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes ||
        buffer.offset % alignof(float) != 0u) {
        error = std::string(operation) + " " + name +
                " buffer is undersized or misaligned";
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

std::uint32_t compiled_mtp_columns(std::uint32_t rows) {
    if (rows <= 1u) return 1u;
    if (rows <= 2u) return 2u;
    if (rows <= 4u) return 4u;
    return 8u;
}

bool validate_mtp_kv_projection(
    Context& context, const MtpAttentionKvProjectionBuffers& buffers,
    std::uint32_t& rows, std::uint32_t& compiled_columns,
    std::string& error) {
    constexpr char operation[] = "MTP attention K/V projection";
    if (!read_mtp_rows(context, buffers.params, operation, rows, error)) {
        return false;
    }

    compiled_columns = compiled_mtp_columns(rows);
    const std::size_t input_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t q8_1_bytes =
        static_cast<std::size_t>(rows) * (N_EMBD / kMtpQ8BlockValues) *
        kMtpQ81BlockBytes;
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kMtpKvProjectionWidth) *
        (N_EMBD / kMtpQ8BlockValues) * sizeof(block_q8_0);
    const std::size_t output_bytes = static_cast<std::size_t>(rows) *
                                     kMtpKvProjectionWidth * sizeof(float);
    const std::size_t sink_floats =
        std::max<std::size_t>(
            1u, static_cast<std::size_t>(compiled_columns - rows) *
                    kMtpKvProjectionWidth);
    const std::size_t sink_bytes = sink_floats * sizeof(float);

    if (!validate_read_only_buffer(context, buffers.params,
                                   sizeof(MtpParams), alignof(MtpParams),
                                   "params", operation, error) ||
        !validate_output_buffer(context, buffers.mixed, input_bytes,
                                operation, "mixed", error) ||
        !validate_output_buffer(context, buffers.xq, q8_1_bytes,
                                operation, "xq", error) ||
        !validate_read_only_buffer(context, buffers.key_weights, weight_bytes,
                                   alignof(std::uint16_t), "key weights",
                                   operation,
                                   error) ||
        !validate_read_only_buffer(context, buffers.value_weights,
                                   weight_bytes, alignof(std::uint16_t),
                                   "value weights", operation, error) ||
        !validate_output_buffer(context, buffers.kcur, output_bytes,
                                operation, "kcur", error) ||
        !validate_output_buffer(context, buffers.vcur, output_bytes,
                                operation, "vcur", error) ||
        !validate_output_buffer(context, buffers.sink, sink_bytes,
                                operation, "sink", error)) {
        return false;
    }

    const ReadOnlyBufferRef active_rows =
        buffers.params.slice(offsetof(MtpParams, T), sizeof(std::uint32_t));
    if (!active_rows) {
        error = std::string(operation) +
                " cannot form its active-row metadata view";
        return false;
    }

    const std::array<std::pair<const char*, BufferRef>, 4> writes{{
        {"xq", buffers.xq},
        {"kcur", buffers.kcur},
        {"vcur", buffers.vcur},
        {"sink", buffers.sink},
    }};
    const std::array<std::size_t, 4> write_bytes{
        q8_1_bytes, output_bytes, output_bytes, sink_bytes};
    const std::array<std::pair<const char*, ReadOnlyBufferRef>, 3> reads{{
        {"mixed", context.buffers().read_only(buffers.mixed)},
        {"key weights", buffers.key_weights},
        {"value weights", buffers.value_weights},
    }};
    const std::array<std::size_t, 3> read_bytes{
        input_bytes, weight_bytes, weight_bytes};

    for (std::size_t write_index = 0u; write_index < writes.size();
         ++write_index) {
        for (std::size_t other_write = write_index + 1u;
             other_write < writes.size(); ++other_write) {
            if (overlaps(writes[write_index].second, write_bytes[write_index],
                         writes[other_write].second,
                         write_bytes[other_write])) {
                error = std::string(operation) + " writable views " +
                        writes[write_index].first + " and " +
                        writes[other_write].first + " overlap";
                return false;
            }
        }
        if (overlaps(buffers.params, sizeof(MtpParams),
                     writes[write_index].second, write_bytes[write_index])) {
            error = std::string(operation) + " parameter block overlaps " +
                    writes[write_index].first;
            return false;
        }
        if (overlaps(active_rows, sizeof(std::uint32_t),
                     writes[write_index].second,
                     write_bytes[write_index])) {
            error = std::string(operation) + " writable view " +
                    writes[write_index].first +
                    " overlaps active-row metadata";
            return false;
        }
        for (std::size_t read_index = 0u; read_index < reads.size();
             ++read_index) {
            if (overlaps(reads[read_index].second, read_bytes[read_index],
                         writes[write_index].second,
                         write_bytes[write_index])) {
                error = std::string(operation) + " writable view " +
                        writes[write_index].first + " overlaps read-only input " +
                        reads[read_index].first;
                return false;
            }
        }
    }
    for (std::size_t read_index = 0u; read_index < reads.size(); ++read_index) {
        if (overlaps(buffers.params, sizeof(MtpParams),
                     reads[read_index].second, read_bytes[read_index])) {
            error = std::string(operation) +
                    " parameter block overlaps read-only input " +
                    reads[read_index].first;
            return false;
        }
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
                                   alignof(float), "h2",
                                   "MTP add streams", error) ||
        !validate_read_only_buffer(context, buffers.e2, e2_bytes,
                                   alignof(float), "e2",
                                   "MTP add streams", error) ||
        !validate_output_buffer(context, buffers.output, h2_bytes,
                                "MTP add streams", "output", error)) {
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

bool encode_mtp_attention_kv_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "MTP attention K/V projection requires an ordered concurrent-group boundary";
        return false;
    }

    std::uint32_t rows = 0u;
    std::uint32_t compiled_columns = 0u;
    if (!validate_mtp_kv_projection(
            context, buffers, rows, compiled_columns, error)) {
        return false;
    }

    const ReadOnlyBufferRef active_rows =
        buffers.params.slice(offsetof(MtpParams, T), sizeof(std::uint32_t));
    const ReadOnlyBufferRef activations =
        context.buffers().read_only(buffers.xq);
    if (!encode_q8_1(context, encoder, rows, N_EMBD,
                     {buffers.mixed, buffers.xq}, error)) {
        return false;
    }
    if (!encode_q8_0_matvec(
            context, encoder, N_EMBD, kMtpKvProjectionWidth, rows,
            compiled_columns,
            {activations, buffers.key_weights, active_rows, buffers.kcur,
             buffers.sink},
            error)) {
        return false;
    }
    return encode_q8_0_matvec(
        context, encoder, N_EMBD, kMtpKvProjectionWidth, rows,
        compiled_columns,
        {activations, buffers.value_weights, active_rows, buffers.vcur,
         buffers.sink},
        error);
}

bool encode_mtp_attention_kv_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention K/V projection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_kv_projection(
                context, encoder, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
