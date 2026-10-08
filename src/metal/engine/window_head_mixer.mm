// Ordered GPU composition for the Window's final GR head mixer.
#include "strata/metal/window_head_mixer.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/session_state.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kHeadMixerLowRank = 320u;
constexpr float kHeadMixerEpsilon = 1.0e-6f;

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_views_overlap(BufferRef left, BufferRef right) {
    return (left.handle == right.handle &&
            address_ranges_overlap(left.offset, left.length,
                                   right.offset, right.length)) ||
           address_ranges_overlap(
               reinterpret_cast<std::uintptr_t>(left.contents), left.length,
               reinterpret_cast<std::uintptr_t>(right.contents), right.length) ||
           address_ranges_overlap(left.gpu_address, left.length,
                                  right.gpu_address, right.length);
}

bool buffer_views_overlap(ReadOnlyBufferRef left, BufferRef right) {
    const void* const left_contents = left.contents();
    return address_ranges_overlap(
               reinterpret_cast<std::uintptr_t>(left_contents), left.length(),
               reinterpret_cast<std::uintptr_t>(right.contents), right.length) ||
           address_ranges_overlap(left.gpu_address(), left.length(),
                                  right.gpu_address, right.length);
}

bool exact_view(BufferRef left, BufferRef right) {
    return left.handle == right.handle && left.contents == right.contents &&
           left.gpu_address == right.gpu_address &&
           left.offset == right.offset && left.length == right.length;
}

std::uint32_t compiled_columns(std::uint32_t rows) {
    std::uint32_t columns = 1u;
    while (columns < rows) columns <<= 1u;
    return columns;
}

bool validate_buffers(Context& context, std::uint32_t rows,
                      const WindowHeadMixerBuffers& buffers,
                      std::string& error) {
    if (rows == 0u || rows > 8u) {
        error = "Window head mixer row count must be in 1..8";
        return false;
    }

    const std::array writable{
        buffers.normalized, buffers.low_rank, buffers.gate, buffers.mixed,
        buffers.matvec_sink,
    };
    for (const BufferRef& view : writable) {
        if (!context.buffers().owns(view)) {
            error = "Window head mixer scratch view is not owned by this Context";
            return false;
        }
    }
    if (!context.buffers().owns(buffers.residual) ||
        !context.buffers().owns(buffers.norm_gamma) ||
        !context.buffers().owns(buffers.down_weights) ||
        !context.buffers().owns(buffers.up_weights) ||
        !context.buffers().owns(buffers.active_rows)) {
        error = "Window head mixer input is not owned by this Context";
        return false;
    }

    const bool residual_normalized_overlap =
        buffer_views_overlap(buffers.residual, buffers.normalized);
    if (residual_normalized_overlap &&
        !exact_view(buffers.residual, buffers.normalized)) {
        error = "Window head mixer residual and normalized views may only alias exactly";
        return false;
    }
    for (std::size_t index = 0u; index < writable.size(); ++index) {
        for (std::size_t earlier = 0u; earlier < index; ++earlier) {
            if (buffer_views_overlap(writable[index], writable[earlier])) {
                error = "Window head mixer writable scratch views must be disjoint";
                return false;
            }
        }
    }

    // The final norm consumes residual once, while every other stage output
    // must preserve the original row until that read has completed.
    for (std::size_t index = 1u; index < writable.size(); ++index) {
        if (buffer_views_overlap(buffers.residual, writable[index])) {
            error = "Window head mixer scratch overlaps the residual input";
            return false;
        }
    }

    const std::array read_only_inputs{
        buffers.norm_gamma, buffers.down_weights, buffers.up_weights,
        buffers.active_rows,
    };
    for (const ReadOnlyBufferRef& input : read_only_inputs) {
        for (const BufferRef& output : writable) {
            if (buffer_views_overlap(input, output)) {
                error = "Window head mixer scratch overlaps a read-only input";
                return false;
            }
        }
    }

    if (buffers.active_rows.length() < sizeof(std::uint32_t) ||
        buffers.active_rows.contents() == nullptr) {
        error = "Window head mixer active-row input is undersized";
        return false;
    }
    std::uint32_t live_rows = 0u;
    std::memcpy(&live_rows, buffers.active_rows.contents(), sizeof(live_rows));
    if (live_rows != rows) {
        error = "Window head mixer active-row value does not match its row count";
        return false;
    }

    const std::uint32_t columns = compiled_columns(rows);
    const std::uint64_t sink_values =
        columns == rows
            ? 1u
            : static_cast<std::uint64_t>(columns - rows) *
                  kSessionHyperconnections * kSessionEmbeddingWidth;
    if (sink_values > std::numeric_limits<std::size_t>::max() /
                          sizeof(float) ||
        buffers.matvec_sink.length <
            static_cast<std::size_t>(sink_values) * sizeof(float)) {
        error = "Window head mixer mat-vec sink is undersized";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_window_head_mixer(Context& context, Encoder& encoder,
                              std::uint32_t rows,
                              const WindowHeadMixerBuffers& buffers,
                              std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window head mixer requires an ordered concurrent-group boundary";
        return false;
    }
    if (!validate_buffers(context, rows, buffers, error)) return false;

    const std::uint32_t columns = compiled_columns(rows);
    const ReadOnlyWeightedRMSNormBuffers norm_buffers{
        buffers.residual, buffers.norm_gamma, buffers.normalized};
    if (!encode_weighted_rms_norm(
            context, encoder, rows * kSessionHyperconnections,
            kSessionEmbeddingWidth, kSessionHyperconnections,
            kHeadMixerEpsilon, norm_buffers, error)) {
        return false;
    }

    const ReadOnlyBufferRef normalized =
        context.buffers().read_only(buffers.normalized);
    if (!normalized) {
        error = "Window head mixer normalized scratch is not a readable Context view";
        return false;
    }
    if (!encode_bf16_matvec(
            context, encoder,
            kSessionHyperconnections * kSessionEmbeddingWidth,
            kHeadMixerLowRank, rows, columns,
            {normalized, buffers.down_weights, buffers.active_rows,
             buffers.low_rank, buffers.matvec_sink},
            error)) {
        return false;
    }
    if (!encode_gr_down_silu(context, encoder, rows, buffers.low_rank,
                             error)) {
        return false;
    }

    const ReadOnlyBufferRef low_rank =
        context.buffers().read_only(buffers.low_rank);
    if (!low_rank) {
        error = "Window head mixer low-rank scratch is not a readable Context view";
        return false;
    }
    if (!encode_bf16_matvec(
            context, encoder, kHeadMixerLowRank,
            kSessionHyperconnections * kSessionEmbeddingWidth, rows, columns,
            {low_rank, buffers.up_weights, buffers.active_rows, buffers.gate,
             buffers.matvec_sink},
            error)) {
        return false;
    }
    return encode_gr_pre_gated(
        context, encoder, rows,
        {buffers.normalized, buffers.gate, buffers.mixed}, error);
}

bool encode_window_head_mixer(Context& context,
                              CommandBuffer& command_buffer,
                              std::uint32_t rows,
                              const WindowHeadMixerBuffers& buffers,
                              std::string& error) {
    error.clear();
    return Encoder::encode(
        context, command_buffer, "Window final GR head mixer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_head_mixer(context, encoder, rows, buffers,
                                            encode_error);
        },
        error);
}

}  // namespace strata::metal
