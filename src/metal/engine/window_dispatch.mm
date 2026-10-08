// Ordered T=1 Window embedding and GR initialization encoding.
#import <Metal/Metal.h>

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/window_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "../kernels/common/params.h"

namespace strata::metal {
namespace {

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                   Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_views_overlap(BufferRef left, BufferRef right) {
    const bool same_buffer_overlap =
        left.handle == right.handle &&
        ranges_overlap(left.offset, left.length, right.offset, right.length);
    const bool cpu_backing_overlap =
        left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.contents),
                       left.length,
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length);
    const bool gpu_backing_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address, left.length, right.gpu_address,
                       right.length);
    return same_buffer_overlap || cpu_backing_overlap || gpu_backing_overlap;
}

bool validate_window_params(Context& context, BufferRef params_buffer,
                            std::uint32_t vocabulary_size,
                            WindowParams& params, std::string& error) {
    if (!params_buffer || !context.buffers().owns(params_buffer)) {
        error = "Window T=1 parameters are not owned by this Context";
        return false;
    }
    if (params_buffer.length < sizeof(WindowParams) ||
        params_buffer.offset % alignof(WindowParams) != 0u ||
        params_buffer.contents == nullptr) {
        error = "Window T=1 parameters are undersized or misaligned";
        return false;
    }
    if (vocabulary_size == 0u || vocabulary_size > 0x80000000u) {
        error = "Window T=1 vocabulary size must be in 1..2^31";
        return false;
    }

    std::memcpy(&params, params_buffer.contents, sizeof(params));
    if (params.T != 1u || params.nc != 1u) {
        error = "Window stage requires a T=1 parameter block";
        return false;
    }
    if ((params.tokens[0] & 0x80000000u) != 0u ||
        params.tokens[0] >= vocabulary_size) {
        error = "Window T=1 token id is outside the text vocabulary";
        return false;
    }
    if (params.pos0 >= static_cast<std::uint32_t>(INT32_MAX)) {
        error = "Window T=1 position exceeds the signed StepRow range";
        return false;
    }

    const std::int32_t n_kv = static_cast<std::int32_t>(params.pos0 + 1u);
    const WindowStepRow expected{
        static_cast<std::int32_t>(params.pos0), n_kv, n_kv / 4,
        std::min(n_kv, static_cast<std::int32_t>(kWindowMaxSelectedCells))};
    const WindowStepRow& actual = params.step[0];
    if (actual.pos != expected.pos || actual.n_kv != expected.n_kv ||
        actual.n_bid != expected.n_bid || actual.width != expected.width) {
        error = "Window T=1 StepRow does not match its position";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_window_embedding(Context& context, Encoder& encoder,
                             std::uint32_t vocabulary_size,
                             EmbedRowsFormat format,
                             const WindowEmbeddingBuffers& buffers,
                             std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window T=1 embedding requires an ordered dispatch boundary";
        return false;
    }

    WindowParams params{};
    if (!validate_window_params(context, buffers.params, vocabulary_size,
                                params, error)) {
        return false;
    }
    if (buffer_views_overlap(buffers.params, buffers.embedding) ||
        buffer_views_overlap(buffers.params, buffers.streams)) {
        error = "Window T=1 scratch outputs must not overlap the parameter block";
        return false;
    }

    const ReadOnlyBufferRef read_only_params =
        context.buffers().read_only(buffers.params);
    const ReadOnlyBufferRef token_view = read_only_params.slice(
        offsetof(WindowParams, tokens), sizeof(std::uint32_t));
    if (!token_view) {
        error = "Window T=1 token field is not a live read-only view";
        return false;
    }
    if (!encode_embed_rows(
            context, encoder, 1u, vocabulary_size, format,
            {buffers.table, token_view, buffers.embedding}, error)) {
        return false;
    }

    const ReadOnlyBufferRef embedding_input =
        context.buffers().read_only(buffers.embedding);
    if (!embedding_input) {
        error = "Window T=1 embedding scratch is not a live read-only view";
        return false;
    }
    return encode_gr_broadcast(context, encoder, 1u,
                               {embedding_input, buffers.streams}, error);
}

namespace {

std::array<BufferRef, 12> gr_read_views(const GRReadBuffers& buffers) {
    return {buffers.residual, buffers.block_out, buffers.inject_prev,
            buffers.norm, buffers.normalized, buffers.rs, buffers.hc_down,
            buffers.hc_inject, buffers.lo, buffers.inject_out, buffers.hc_up,
            buffers.mixed};
}

std::array<BufferRef, 23> gdn_block_views(
    const WindowGDNBlockBuffers& buffers) {
    return {buffers.mixed, buffers.history, buffers.state, buffers.qkv_weights,
            buffers.conv_weights, buffers.alpha, buffers.beta_weights,
            buffers.dt_bias, buffers.a, buffers.gate_weights, buffers.gamma,
            buffers.output_weights, buffers.xq, buffers.qkv, buffers.h,
            buffers.gate, buffers.beta, buffers.z, buffers.y, buffers.yq,
            buffers.block_out, buffers.sink, buffers.active_rows};
}

bool same_view(BufferRef left, BufferRef right) {
    return left && right && left.handle == right.handle &&
           left.contents == right.contents &&
           left.gpu_address == right.gpu_address &&
           left.length == right.length && left.offset == right.offset;
}

bool validate_window_t1_attention_gdn(
    Context& context, std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNBuffers& buffers, std::string& error) {
    WindowParams params{};
    if (!validate_window_params(context, buffers.params, vocabulary_size,
                                params, error)) {
        return false;
    }
    if (params.n_keep != 0u) {
        error = "Window T=1 attention/GDN stage requires n_keep=0";
        return false;
    }

    const auto attention_views = gr_read_views(buffers.attention);
    const auto gdn_views = gdn_block_views(buffers.gdn);
    if (!same_view(buffers.attention.mixed, buffers.gdn.mixed)) {
        error = "Window T=1 attention mixed output must be the GDN mixed input";
        return false;
    }
    for (const BufferRef& view : attention_views) {
        if (buffer_views_overlap(buffers.params, view)) {
            error = "Window T=1 parameters overlap an attention buffer";
            return false;
        }
    }
    for (const BufferRef& view : gdn_views) {
        if (buffer_views_overlap(buffers.params, view)) {
            error = "Window T=1 parameters overlap a GDN buffer";
            return false;
        }
    }

    for (std::size_t attention_index = 0u;
         attention_index < attention_views.size(); ++attention_index) {
        for (std::size_t gdn_index = 0u; gdn_index < gdn_views.size();
             ++gdn_index) {
            if (attention_index == attention_views.size() - 1u &&
                gdn_index == 0u) {
                continue;
            }
            if (buffer_views_overlap(attention_views[attention_index],
                                     gdn_views[gdn_index])) {
                error = "Window T=1 attention/GDN buffers overlap outside the mixed handoff";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_window_t1_attention_gdn(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window T=1 attention/GDN requires an ordered dispatch boundary";
        return false;
    }
    if (!validate_window_t1_attention_gdn(context, vocabulary_size, buffers,
                                          error)) {
        return false;
    }
    if (!encode_gr_read(context, encoder, 1u, false, buffers.attention,
                        error)) {
        return false;
    }
    return encode_window_gdn_block(context, encoder, 1u, buffers.gdn, error);
}

bool encode_window_t1_attention_gdn(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window T=1 layer-0 attention and GDN",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_attention_gdn(
                context, encoder, vocabulary_size, buffers, encode_error);
        },
        error);
}

bool encode_window_embedding(Context& context, CommandBuffer& command_buffer,
                             std::uint32_t vocabulary_size,
                             EmbedRowsFormat format,
                             const WindowEmbeddingBuffers& buffers,
                             std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window T=1 embedding and GR initialization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_embedding(context, encoder, vocabulary_size,
                                           format, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
