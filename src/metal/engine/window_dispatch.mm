// Ordered T=1 Window embedding and GR initialization encoding.
#import <Metal/Metal.h>

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/tiered.hpp"
#include "strata/metal/window_dispatch.hpp"
#include "strata/metal/window.hpp"
#include "window_dispatch_internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

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

bool buffer_views_overlap(ReadOnlyBufferRef left, BufferRef right) {
    const void* left_contents = left.contents();
    const bool cpu_backing_overlap =
        left_contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left.length(),
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length);
    const std::uint64_t left_gpu_address = left.gpu_address();
    const bool gpu_backing_overlap =
        left_gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left_gpu_address, left.length(), right.gpu_address,
                       right.length);
    return cpu_backing_overlap || gpu_backing_overlap;
}

bool buffer_views_overlap(ReadOnlyBufferRef left, ReadOnlyBufferRef right) {
    const void* left_contents = left.contents();
    const void* right_contents = right.contents();
    const bool cpu_backing_overlap =
        left_contents != nullptr && right_contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left.length(),
                       reinterpret_cast<std::uintptr_t>(right_contents),
                       right.length());
    const std::uint64_t left_gpu_address = left.gpu_address();
    const std::uint64_t right_gpu_address = right.gpu_address();
    const bool gpu_backing_overlap =
        left_gpu_address != 0u && right_gpu_address != 0u &&
        ranges_overlap(left_gpu_address, left.length(), right_gpu_address,
                       right.length());
    return cpu_backing_overlap || gpu_backing_overlap;
}

bool same_buffer_start(const void* left_contents, std::uint64_t left_gpu,
                       const void* right_contents,
                       std::uint64_t right_gpu) {
    bool comparable = false;
    if (left_contents != nullptr && right_contents != nullptr) {
        if (left_contents != right_contents) return false;
        comparable = true;
    }
    if (left_gpu != 0u && right_gpu != 0u) {
        if (left_gpu != right_gpu) return false;
        comparable = true;
    }
    return comparable;
}

bool same_buffer_start(ReadOnlyBufferRef left, ReadOnlyBufferRef right) {
    return same_buffer_start(left.contents(), left.gpu_address(),
                             right.contents(), right.gpu_address());
}

bool same_buffer_start(BufferRef left, ReadOnlyBufferRef right) {
    return same_buffer_start(left.contents, left.gpu_address,
                             right.contents(), right.gpu_address());
}

bool same_buffer_start(ReadOnlyBufferRef left, BufferRef right) {
    return same_buffer_start(left.contents(), left.gpu_address(),
                             right.contents, right.gpu_address);
}

bool same_buffer_start(BufferRef left, BufferRef right) {
    return same_buffer_start(left.contents, left.gpu_address,
                             right.contents, right.gpu_address);
}

bool validate_prefix_buffer(Context& context, BufferRef buffer,
                            std::size_t required_bytes,
                            std::size_t alignment, std::string_view name,
                            std::string& error) {
    if (!buffer || !context.buffers().owns(buffer) ||
        buffer.length < required_bytes || buffer.offset % alignment != 0u) {
        error = "Window T=1 MoE prefix " + std::string(name) +
                " is stale, foreign, undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_prefix_buffer(Context& context, ReadOnlyBufferRef buffer,
                            std::size_t required_bytes,
                            std::size_t alignment, std::string_view name,
                            std::string& error) {
    if (!buffer || !context.buffers().owns(buffer) ||
        buffer.length() < required_bytes || buffer.offset() % alignment != 0u) {
        error = "Window T=1 MoE prefix " + std::string(name) +
                " is stale, foreign, undersized or misaligned";
        return false;
    }
    return true;
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

bool validate_speculative_window_params(
    Context& context, BufferRef params_buffer, std::uint32_t vocabulary_size,
    WindowParams& params, std::string& error) {
    if (!params_buffer || !context.buffers().owns(params_buffer)) {
        error = "Window speculative parameters are not owned by this Context";
        return false;
    }
    if (params_buffer.length < sizeof(WindowParams) ||
        params_buffer.offset % alignof(WindowParams) != 0u ||
        params_buffer.contents == nullptr) {
        error = "Window speculative parameters are undersized or misaligned";
        return false;
    }
    if (vocabulary_size == 0u || vocabulary_size > 0x80000000u) {
        error = "Window speculative vocabulary size must be in 1..2^31";
        return false;
    }

    std::memcpy(&params, params_buffer.contents, sizeof(params));
    if (params.T == 0u || params.T > kWindowMaxRows) {
        error = "Window speculative row count must be in 1..8";
        return false;
    }
    std::uint32_t expected_nc = 1u;
    while (expected_nc < params.T) expected_nc <<= 1u;
    if (params.nc != expected_nc) {
        error = "Window speculative parameter columns do not match T";
        return false;
    }
    if (static_cast<std::uint64_t>(params.pos0) + params.T >
        static_cast<std::uint64_t>(INT32_MAX)) {
        error = "Window speculative positions exceed the signed StepRow range";
        return false;
    }

    for (std::uint32_t row = 0u; row < params.T; ++row) {
        if ((params.tokens[row] & 0x80000000u) != 0u ||
            params.tokens[row] >= vocabulary_size) {
            error = "Window speculative token id is outside the text vocabulary";
            return false;
        }
        const std::uint32_t position = params.pos0 + row;
        const std::int32_t n_kv = static_cast<std::int32_t>(position + 1u);
        const WindowStepRow expected{
            static_cast<std::int32_t>(position), n_kv, n_kv / 4,
            std::min(n_kv, static_cast<std::int32_t>(kWindowMaxSelectedCells))};
        const WindowStepRow& actual = params.step[row];
        if (actual.pos != expected.pos || actual.n_kv != expected.n_kv ||
            actual.n_bid != expected.n_bid || actual.width != expected.width) {
            error = "Window speculative StepRow does not match its position";
            return false;
        }
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

template <std::size_t LeftCount, std::size_t RightCount, class Allowed>
bool validate_cross_stage_views(
    const std::array<BufferRef, LeftCount>& left,
    const std::array<BufferRef, RightCount>& right, Allowed allowed,
    std::string_view stage_pair, std::string& error) {
    for (std::size_t left_index = 0u; left_index < left.size(); ++left_index) {
        for (std::size_t right_index = 0u; right_index < right.size();
             ++right_index) {
            if (!buffer_views_overlap(left[left_index], right[right_index])) {
                continue;
            }
            if (allowed(left_index, right_index) &&
                same_view(left[left_index], right[right_index])) {
                continue;
            }
            error = "Window layer-0 buffers overlap across " +
                    std::string(stage_pair) + " outside an exact scratch handoff";
            return false;
        }
    }
    return true;
}

bool validate_window_t1_attention_gdn_ffn(
    Context& context, std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNFFNBuffers& buffers, std::string& error) {
    WindowParams params{};
    if (!validate_window_params(context, buffers.params, vocabulary_size,
                                params, error)) {
        return false;
    }
    if (params.n_keep != 0u) {
        error = "Window T=1 attention/GDN/FFN stage requires n_keep=0";
        return false;
    }

    const auto attention_views = gr_read_views(buffers.attention);
    const auto gdn_views = gdn_block_views(buffers.gdn);
    const auto ffn_views = gr_read_views(buffers.ffn);
    const GRReadBuffers& attention = buffers.attention;
    const GRReadBuffers& ffn = buffers.ffn;
    if (!same_view(attention.mixed, buffers.gdn.mixed) ||
        !same_view(buffers.gdn.mixed, ffn.mixed) ||
        !same_view(buffers.gdn.block_out, ffn.block_out)) {
        error = "Window T=1 layer-0 mixed and GDN block-output handoffs must reuse exact views";
        return false;
    }
    if (!same_view(attention.residual, ffn.residual) ||
        !same_view(attention.inject_out, ffn.inject_prev) ||
        !same_view(attention.inject_prev, ffn.inject_out) ||
        !same_view(attention.normalized, ffn.normalized) ||
        !same_view(attention.rs, ffn.rs) ||
        !same_view(attention.lo, ffn.lo)) {
        error = "Window T=1 layer-0 GR reads must reuse residual, scratch and swapped injection views";
        return false;
    }

    const auto parameters_are_disjoint = [&](const auto& views) {
        for (const BufferRef& view : views) {
            if (buffer_views_overlap(buffers.params, view)) return false;
        }
        return true;
    };
    if (!parameters_are_disjoint(attention_views) ||
        !parameters_are_disjoint(gdn_views) ||
        !parameters_are_disjoint(ffn_views)) {
        error = "Window T=1 parameters overlap a layer-0 stage buffer";
        return false;
    }

    const auto attention_gdn_handoff = [](std::size_t left,
                                          std::size_t right) {
        return left == 11u && right == 0u;
    };
    const auto gdn_ffn_handoffs = [](std::size_t left, std::size_t right) {
        return (left == 0u && right == 11u) ||
               (left == 20u && right == 1u);
    };
    const auto attention_ffn_handoffs = [](std::size_t left,
                                           std::size_t right) {
        switch (left) {
            case 0u: return right == 0u;
            case 2u: return right == 9u;
            case 4u: return right == 4u;
            case 5u: return right == 5u;
            case 8u: return right == 8u;
            case 9u: return right == 2u;
            case 11u: return right == 11u;
            default: return false;
        }
    };
    if (!validate_cross_stage_views(attention_views, gdn_views,
                                    attention_gdn_handoff, "attention/GDN",
                                    error) ||
        !validate_cross_stage_views(gdn_views, ffn_views, gdn_ffn_handoffs,
                                    "GDN/FFN", error) ||
        !validate_cross_stage_views(attention_views, ffn_views,
                                    attention_ffn_handoffs, "attention/FFN",
                                    error)) {
        return false;
    }
    error.clear();
    return true;
}

bool validate_window_attention_gdn_ffn(
    Context& context, std::uint32_t vocabulary_size,
    const WindowAttentionGDNFFNBuffers& buffers, WindowParams& params,
    std::string& error) {
    if (!validate_speculative_window_params(context, buffers.params,
                                            vocabulary_size, params, error)) {
        return false;
    }
    if (params.n_keep != 0u) {
        error = "Window speculative attention/GDN/FFN stage requires n_keep=0";
        return false;
    }

    const auto attention_views = gr_read_views(buffers.attention);
    const auto gdn_views = gdn_block_views(buffers.gdn);
    const auto ffn_views = gr_read_views(buffers.ffn);
    const GRReadBuffers& attention = buffers.attention;
    const GRReadBuffers& ffn = buffers.ffn;
    if (!same_view(attention.mixed, buffers.gdn.mixed) ||
        !same_view(buffers.gdn.mixed, ffn.mixed) ||
        !same_view(buffers.gdn.block_out, ffn.block_out)) {
        error = "Window speculative mixed and GDN block-output handoffs must reuse exact views";
        return false;
    }
    if (!same_view(attention.residual, ffn.residual) ||
        !same_view(attention.inject_out, ffn.inject_prev) ||
        !same_view(attention.inject_prev, ffn.inject_out) ||
        !same_view(attention.normalized, ffn.normalized) ||
        !same_view(attention.rs, ffn.rs) ||
        !same_view(attention.lo, ffn.lo)) {
        error = "Window speculative GR reads must reuse residual, scratch and swapped injection views";
        return false;
    }

    const auto parameters_are_disjoint = [&](const auto& views) {
        for (const BufferRef& view : views) {
            if (buffer_views_overlap(buffers.params, view)) return false;
        }
        return true;
    };
    if (!parameters_are_disjoint(attention_views) ||
        !parameters_are_disjoint(gdn_views) ||
        !parameters_are_disjoint(ffn_views)) {
        error = "Window speculative parameters overlap a layer-0 stage buffer";
        return false;
    }

    const auto attention_gdn_handoff = [](std::size_t left,
                                          std::size_t right) {
        return left == 11u && right == 0u;
    };
    const auto gdn_ffn_handoffs = [](std::size_t left, std::size_t right) {
        return (left == 0u && right == 11u) ||
               (left == 20u && right == 1u);
    };
    const auto attention_ffn_handoffs = [](std::size_t left,
                                           std::size_t right) {
        switch (left) {
            case 0u: return right == 0u;
            case 2u: return right == 9u;
            case 4u: return right == 4u;
            case 5u: return right == 5u;
            case 8u: return right == 8u;
            case 9u: return right == 2u;
            case 11u: return right == 11u;
            default: return false;
        }
    };
    if (!validate_cross_stage_views(attention_views, gdn_views,
                                    attention_gdn_handoff, "attention/GDN",
                                    error) ||
        !validate_cross_stage_views(gdn_views, ffn_views, gdn_ffn_handoffs,
                                    "GDN/FFN", error) ||
        !validate_cross_stage_views(attention_views, ffn_views,
                                    attention_ffn_handoffs, "attention/FFN",
                                    error)) {
        return false;
    }
    error.clear();
    return true;
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

bool validate_window_attention_gdn(
    Context& context, std::uint32_t vocabulary_size,
    const WindowAttentionGDNBuffers& buffers, WindowParams& params,
    std::string& error) {
    if (!validate_speculative_window_params(context, buffers.params,
                                            vocabulary_size, params, error)) {
        return false;
    }
    if (params.n_keep != 0u) {
        error = "Window speculative attention/GDN stage requires n_keep=0";
        return false;
    }

    const auto attention_views = gr_read_views(buffers.attention);
    const auto gdn_views = gdn_block_views(buffers.gdn);
    if (!same_view(buffers.attention.mixed, buffers.gdn.mixed)) {
        error = "Window attention mixed output must be the GDN mixed input";
        return false;
    }
    for (const BufferRef& view : attention_views) {
        if (buffer_views_overlap(buffers.params, view)) {
            error = "Window speculative parameters overlap an attention buffer";
            return false;
        }
    }
    for (const BufferRef& view : gdn_views) {
        if (buffer_views_overlap(buffers.params, view)) {
            error = "Window speculative parameters overlap a GDN buffer";
            return false;
        }
    }

    const auto attention_gdn_handoff = [](std::size_t left,
                                          std::size_t right) {
        return left == 11u && right == 0u;
    };
    return validate_cross_stage_views(
        attention_views, gdn_views, attention_gdn_handoff, "attention/GDN",
        error);
}

bool validate_window_t1_moe_router_prefix(
    Context& context, std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoeRouterPrefixBuffers& buffers, std::string& error) {
    if (experts != 256u && experts != 512u) {
        error = "Window T=1 MoE prefix requires 256 or 512 experts";
        return false;
    }
    if (!validate_window_t1_attention_gdn_ffn(
            context, vocabulary_size, buffers.layer0, error)) {
        return false;
    }

    constexpr std::size_t kRouterInputWidth = N_EMBD;
    constexpr std::size_t kRouteTopK = 10u;
    const std::size_t router_weight_bytes =
        static_cast<std::size_t>(experts) * kRouterInputWidth *
        sizeof(std::uint16_t);
    const std::size_t router_logit_bytes =
        static_cast<std::size_t>(experts) * sizeof(float);
    const std::size_t route_id_bytes = kRouteTopK * sizeof(std::int32_t);
    const std::size_t route_weight_bytes = kRouteTopK * sizeof(float);

    if (!validate_prefix_buffer(context, buffers.router_tensor,
                                router_weight_bytes, alignof(std::uint32_t),
                                "BF16 router tensor", error) ||
        !validate_prefix_buffer(context, buffers.router_logits,
                                router_logit_bytes, alignof(float),
                                "router logits", error) ||
        !validate_prefix_buffer(context, buffers.route_ids, route_id_bytes,
                                alignof(std::int32_t), "route IDs", error) ||
        !validate_prefix_buffer(context, buffers.route_weights,
                                route_weight_bytes, alignof(float),
                                "route weights", error) ||
        !validate_prefix_buffer(context, buffers.router_matvec_sink,
                                sizeof(float), alignof(float),
                                "router mat-vec sink", error)) {
        return false;
    }

    const std::array<BufferRef, 4u> prefix_outputs{
        buffers.router_logits, buffers.route_ids, buffers.route_weights,
        buffers.router_matvec_sink};
    for (std::size_t left = 0u; left < prefix_outputs.size(); ++left) {
        for (std::size_t right = left + 1u; right < prefix_outputs.size();
             ++right) {
            if (buffer_views_overlap(prefix_outputs[left],
                                     prefix_outputs[right])) {
                error = "Window T=1 MoE prefix scratch outputs overlap";
                return false;
            }
        }
    }

    const auto disjoint_from_prefix = [&](const auto& views) {
        for (const BufferRef& view : views) {
            if (buffer_views_overlap(buffers.layer0.params, view)) {
                error = "Window T=1 MoE prefix scratch overlaps its parameters";
                return false;
            }
            if (buffer_views_overlap(buffers.router_tensor, view)) {
                error = "Window T=1 MoE prefix scratch overlaps the router tensor";
                return false;
            }
            for (const BufferRef& output : prefix_outputs) {
                if (buffer_views_overlap(view, output)) {
                    error = "Window T=1 MoE prefix scratch overlaps a layer-0 stage buffer";
                    return false;
                }
            }
        }
        return true;
    };
    const auto attention_views = gr_read_views(buffers.layer0.attention);
    const auto gdn_views = gdn_block_views(buffers.layer0.gdn);
    const auto ffn_views = gr_read_views(buffers.layer0.ffn);
    if (!disjoint_from_prefix(attention_views) ||
        !disjoint_from_prefix(gdn_views) ||
        !disjoint_from_prefix(ffn_views)) {
        return false;
    }

    for (const BufferRef& output : prefix_outputs) {
        if (buffer_views_overlap(buffers.layer0.params, output) ||
            buffer_views_overlap(buffers.router_tensor, output)) {
            error = "Window T=1 MoE prefix output overlaps parameters or router weights";
            return false;
        }
    }
    if (buffer_views_overlap(buffers.router_tensor,
                             buffers.layer0.params)) {
        error = "Window T=1 MoE router tensor overlaps its parameters";
        return false;
    }
    const ReadOnlyBufferRef read_only_params =
        context.buffers().read_only(buffers.layer0.params);
    const ReadOnlyBufferRef active_rows = read_only_params.slice(
        offsetof(WindowParams, nc), sizeof(std::uint32_t));
    if (!active_rows) {
        error = "Window T=1 active-row count is not a live read-only parameter view";
        return false;
    }

    error.clear();
    return true;
}

bool validate_window_t1_moe_plan_prefix(
    Context& context, std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoePlanPrefixBuffers& buffers, std::string& error) {
    if (!validate_window_t1_moe_router_prefix(
            context, vocabulary_size, experts, buffers.router, error)) {
        return false;
    }

    const std::size_t resident_bytes =
        static_cast<std::size_t>(experts) * sizeof(std::int32_t);
    const std::size_t usage_bytes =
        static_cast<std::size_t>(experts) * sizeof(float);
    if (!validate_prefix_buffer(context, buffers.resident, resident_bytes,
                                alignof(std::int32_t), "resident table",
                                error) ||
        !validate_prefix_buffer(context, buffers.usage, usage_bytes,
                                alignof(float), "usage table", error) ||
        !validate_prefix_buffer(context, buffers.plans,
                                sizeof(TieredMoePlan),
                                alignof(TieredMoePlan), "plan output",
                                error)) {
        return false;
    }

    const std::array<BufferRef, 2u> plan_outputs{buffers.usage,
                                                buffers.plans};
    for (std::size_t left = 0u; left < plan_outputs.size(); ++left) {
        for (std::size_t right = left + 1u; right < plan_outputs.size();
             ++right) {
            if (buffer_views_overlap(plan_outputs[left],
                                     plan_outputs[right])) {
                error = "Window T=1 MoE plan outputs overlap";
                return false;
            }
        }
        if (buffer_views_overlap(buffers.resident, plan_outputs[left])) {
            error = "Window T=1 MoE plan output overlaps its resident table";
            return false;
        }
        if (buffer_views_overlap(buffers.router.router_tensor,
                                 plan_outputs[left])) {
            error = "Window T=1 MoE plan output overlaps the router tensor";
            return false;
        }
        if (buffer_views_overlap(buffers.router.layer0.params,
                                 plan_outputs[left])) {
            error = "Window T=1 MoE plan output overlaps its parameters";
            return false;
        }
    }
    if (buffer_views_overlap(buffers.resident,
                             buffers.router.router_tensor)) {
        error = "Window T=1 MoE resident table overlaps the router tensor";
        return false;
    }
    if (buffer_views_overlap(buffers.resident,
                             buffers.router.layer0.params)) {
        error = "Window T=1 MoE resident table overlaps its parameters";
        return false;
    }

    const auto disjoint_from_prefix = [&](const auto& views) {
        for (const BufferRef& view : views) {
            if (buffer_views_overlap(buffers.resident, view)) {
                error = "Window T=1 MoE resident table overlaps a prefix buffer";
                return false;
            }
            for (const BufferRef& output : plan_outputs) {
                if (buffer_views_overlap(view, output)) {
                    error = "Window T=1 MoE plan storage overlaps a prefix buffer";
                    return false;
                }
            }
        }
        return true;
    };
    const auto attention_views =
        gr_read_views(buffers.router.layer0.attention);
    const auto gdn_views = gdn_block_views(buffers.router.layer0.gdn);
    const auto ffn_views = gr_read_views(buffers.router.layer0.ffn);
    const std::array<BufferRef, 4u> router_views{
        buffers.router.router_logits, buffers.router.route_ids,
        buffers.router.route_weights, buffers.router.router_matvec_sink};
    if (!disjoint_from_prefix(attention_views) ||
        !disjoint_from_prefix(gdn_views) ||
        !disjoint_from_prefix(ffn_views) ||
        !disjoint_from_prefix(router_views)) {
        return false;
    }

    error.clear();
    return true;
}

bool validate_window_t1_moe_layer0_consumer_handoffs(
    Context& context, std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoeLayer0FullBuffers& buffers, std::string& error) {
    if (!validate_window_t1_moe_plan_prefix(
            context, vocabulary_size, experts, buffers.producer, error)) {
        return false;
    }
    if (experts != 256u && experts != 512u) {
        error = "Window T=1 MoE FFN requires 256 or 512 experts";
        return false;
    }

    const WindowT1MoeRouterPrefixBuffers& producer =
        buffers.producer.router;
    const WindowT1AttentionGDNFFNBuffers& layer0 = producer.layer0;
    WindowParams params{};
    if (!validate_window_params(context, layer0.params, vocabulary_size,
                                params, error)) {
        return false;
    }
    if (params.n_keep != 0u) {
        error = "Window T=1 MoE FFN requires an ordinary decode step";
        return false;
    }

    const ReadOnlyBufferRef producer_input =
        context.buffers().read_only(layer0.gdn.xq);
    const ReadOnlyBufferRef producer_plan =
        context.buffers().read_only(buffers.producer.plans);
    const ReadOnlyBufferRef producer_route_weights =
        context.buffers().read_only(producer.route_weights);
    const ReadOnlyBufferRef params_read_only =
        context.buffers().read_only(layer0.params);
    const ReadOnlyBufferRef active_rows = params_read_only.slice(
        offsetof(WindowParams, nc), sizeof(std::uint32_t));
    const MoeRoutedFFNBuffers& routed = buffers.ffn.routed;
    const MoeSharedFFNBuffers& shared = buffers.ffn.shared;
    if (!producer_input || !producer_plan || !producer_route_weights ||
        !active_rows) {
        error = "Window T=1 MoE FFN could not form producer views";
        return false;
    }
    if (!same_buffer_start(producer_input, routed.input_activations) ||
        !same_buffer_start(producer_input, shared.input_activations)) {
        error = "Window T=1 MoE FFN activations must use the producer Q8_1 input";
        return false;
    }
    if (!same_buffer_start(producer_plan, routed.plans)) {
        error = "Window T=1 MoE FFN plan must use the producer plan output";
        return false;
    }
    if (!same_buffer_start(producer_route_weights, shared.route_weights)) {
        error = "Window T=1 MoE FFN route weights must use the producer output";
        return false;
    }
    if (!same_buffer_start(layer0.ffn.mixed, shared.mixed)) {
        error = "Window T=1 MoE FFN mixed input must use the producer output";
        return false;
    }
    if (!same_buffer_start(active_rows, shared.active_rows)) {
        error = "Window T=1 MoE FFN live rows must use WindowParams.nc";
        return false;
    }

    struct ConsumerInputView {
        ReadOnlyBufferRef view;
        BufferRef allowed_handoff;
    };
    const std::array<ConsumerInputView, 11u> consumer_inputs{{
        {routed.addresses, {}},
        {routed.input_activations, layer0.gdn.xq},
        {routed.plans, buffers.producer.plans},
        {shared.input_activations, layer0.gdn.xq},
        {shared.mixed, layer0.ffn.mixed},
        {shared.gate_weights, {}},
        {shared.up_weights, {}},
        {shared.down_weights, {}},
        {shared.shared_gate_weights, {}},
        {shared.routed_parts, {}},
        {shared.route_weights, producer.route_weights},
    }};
    const std::array<BufferRef, 12u> consumer_outputs{
        routed.gate,
        routed.up,
        routed.hidden,
        routed.down_activations,
        routed.parts,
        shared.gate,
        shared.up,
        shared.hidden,
        shared.down_activations,
        shared.shared_output,
        shared.shared_gate_logits,
        shared.output,
    };
    const std::array<BufferRef, 4u> consumer_sinks{
        shared.gate_sink,
        shared.up_sink,
        shared.down_sink,
        shared.shared_gate_sink,
    };
    const bool consumer_inputs_disjoint_from_params =
        std::all_of(consumer_inputs.begin(), consumer_inputs.end(),
                    [&](const ConsumerInputView& input) {
                        return !buffer_views_overlap(input.view,
                                                     layer0.params);
                    });
    if (!consumer_inputs_disjoint_from_params ||
        !std::all_of(consumer_outputs.begin(), consumer_outputs.end(),
                     [&](BufferRef view) {
                         return !buffer_views_overlap(layer0.params, view);
                     }) ||
        !std::all_of(consumer_sinks.begin(), consumer_sinks.end(),
                     [&](BufferRef view) {
                         return !buffer_views_overlap(layer0.params, view);
                     })) {
        error = "Window T=1 MoE FFN consumer overlaps its parameters";
        return false;
    }

    const auto producer_attention =
        gr_read_views(layer0.attention);
    const auto producer_gdn = gdn_block_views(layer0.gdn);
    const auto producer_ffn = gr_read_views(layer0.ffn);
    const std::array<BufferRef, 6u> route_plan_outputs{
        producer.router_logits,
        producer.route_ids,
        producer.route_weights,
        producer.router_matvec_sink,
        buffers.producer.usage,
        buffers.producer.plans,
    };
    const std::array<ReadOnlyBufferRef, 2u> producer_read_only_tables{
        producer.router_tensor, buffers.producer.resident};
    const auto consumer_disjoint_from = [&](const auto& producer_views) {
        for (const ConsumerInputView& input : consumer_inputs) {
            for (const BufferRef& producer_view : producer_views) {
                const bool allowed_handoff =
                    input.allowed_handoff &&
                    same_buffer_start(input.view, input.allowed_handoff) &&
                    same_buffer_start(producer_view,
                                      input.allowed_handoff);
                if (buffer_views_overlap(input.view, producer_view) &&
                    !allowed_handoff) {
                    return false;
                }
            }
            for (const ReadOnlyBufferRef& producer_view :
                 producer_read_only_tables) {
                if (buffer_views_overlap(input.view, producer_view)) {
                    return false;
                }
            }
        }
        for (const BufferRef& output : consumer_outputs) {
            for (const BufferRef& input : producer_views) {
                if (buffer_views_overlap(output, input)) {
                    return false;
                }
            }
            for (const ReadOnlyBufferRef& input :
                 producer_read_only_tables) {
                if (buffer_views_overlap(input, output)) return false;
            }
        }
        for (const BufferRef& sink : consumer_sinks) {
            for (const BufferRef& input : producer_views) {
                if (buffer_views_overlap(sink, input)) return false;
            }
            for (const ReadOnlyBufferRef& input :
                 producer_read_only_tables) {
                if (buffer_views_overlap(input, sink)) return false;
            }
        }
        return true;
    };
    if (!consumer_disjoint_from(producer_attention) ||
        !consumer_disjoint_from(producer_gdn) ||
        !consumer_disjoint_from(producer_ffn) ||
        !consumer_disjoint_from(route_plan_outputs)) {
        error = "Window T=1 MoE consumer views overlap producer storage outside a declared handoff";
        return false;
    }

    error.clear();
    return true;
}

bool validate_window_t1_moe_layer0_full_mode(
    Context& context, const FullModeStorage& full_mode,
    std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoeLayer0FullBuffers& buffers, std::string& error) {
    const ReadOnlyBufferRef full_addresses =
        full_mode.expert_address_table();
    const ReadOnlyBufferRef full_resident = full_mode.resident_table();
    const BufferRef full_usage = full_mode.usage_table();
    if (!full_addresses || !context.buffers().owns(full_addresses) ||
        !full_resident || !context.buffers().owns(full_resident) ||
        !full_usage || !context.buffers().owns(full_usage) ||
        !same_buffer_start(full_addresses, buffers.ffn.routed.addresses) ||
        !same_buffer_start(full_resident, buffers.producer.resident) ||
        !same_buffer_start(full_usage, buffers.producer.usage)) {
        error = "Window T=1 full-mode MoE must use its FullModeStorage address, resident and usage tables";
        return false;
    }
    return validate_window_t1_moe_layer0_consumer_handoffs(
        context, vocabulary_size, experts, buffers, error);
}

}  // namespace

bool detail::validate_window_t1_moe_consumer_handoffs(
    Context& context, std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoeLayer0FullBuffers& buffers, std::string& error) {
    return validate_window_t1_moe_layer0_consumer_handoffs(
        context, vocabulary_size, experts, buffers, error);
}

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

bool encode_window_attention_gdn(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    const WindowAttentionGDNBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window speculative attention/GDN requires an ordered dispatch boundary";
        return false;
    }
    WindowParams params{};
    if (!validate_window_attention_gdn(context, vocabulary_size, buffers,
                                       params, error)) {
        return false;
    }
    if (!encode_gr_read(context, encoder, params.T, false, buffers.attention,
                        error)) {
        return false;
    }
    return encode_window_gdn_block(context, encoder, params.T, buffers.gdn,
                                   error);
}

bool encode_window_attention_gdn(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size,
    const WindowAttentionGDNBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window speculative attention and GDN",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_attention_gdn(
                context, encoder, vocabulary_size, buffers, encode_error);
        },
        error);
}

bool encode_window_t1_attention_gdn_ffn(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNFFNBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window T=1 layer-0 GR/GDN chain requires ordered dispatch boundaries";
        return false;
    }
    if (!validate_window_t1_attention_gdn_ffn(context, vocabulary_size,
                                              buffers, error)) {
        return false;
    }
    if (!encode_gr_read(context, encoder, 1u, false, buffers.attention,
                        error) ||
        !encode_window_gdn_block(context, encoder, 1u, buffers.gdn, error)) {
        return false;
    }
    return encode_gr_read(context, encoder, 1u, true, buffers.ffn, error);
}

bool encode_window_t1_attention_gdn_ffn(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size,
    const WindowT1AttentionGDNFFNBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window T=1 layer-0 attention, GDN and FFN read",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_attention_gdn_ffn(
                context, encoder, vocabulary_size, buffers, encode_error);
        },
        error);
}

bool encode_window_attention_gdn_ffn(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    const WindowAttentionGDNFFNBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window speculative layer-0 GR/GDN chain requires ordered dispatch boundaries";
        return false;
    }
    WindowParams params{};
    if (!validate_window_attention_gdn_ffn(context, vocabulary_size, buffers,
                                           params, error)) {
        return false;
    }
    if (!encode_gr_read(context, encoder, params.T, false, buffers.attention,
                        error) ||
        !encode_window_gdn_block(context, encoder, params.T, buffers.gdn,
                                 error)) {
        return false;
    }
    return encode_gr_read(context, encoder, params.T, true, buffers.ffn, error);
}

bool encode_window_attention_gdn_ffn(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size,
    const WindowAttentionGDNFFNBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window speculative attention, GDN and FFN read",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_attention_gdn_ffn(
                context, encoder, vocabulary_size, buffers, encode_error);
        },
        error);
}

bool encode_window_t1_moe_router_prefix(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    std::uint32_t experts,
    const WindowT1MoeRouterPrefixBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window T=1 MoE router prefix requires ordered dispatch boundaries";
        return false;
    }
    if (!validate_window_t1_moe_router_prefix(
            context, vocabulary_size, experts, buffers, error)) {
        return false;
    }

    const ReadOnlyBufferRef active_rows = context.buffers()
        .read_only(buffers.layer0.params)
        .slice(offsetof(WindowParams, nc), sizeof(std::uint32_t));
    const ReadOnlyBufferRef router_activations =
        context.buffers().read_only(buffers.layer0.ffn.mixed);
    const ReadOnlyBufferRef router_logits =
        context.buffers().read_only(buffers.router_logits);
    if (!active_rows || !router_activations || !router_logits) {
        error = "Window T=1 MoE router prefix could not form live read-only input views";
        return false;
    }

    if (!encode_window_t1_attention_gdn_ffn(
            context, encoder, vocabulary_size, buffers.layer0, error)) {
        return false;
    }

    const ReadOnlyBF16MatVecBuffers router_buffers{
        router_activations, buffers.router_tensor, active_rows,
        buffers.router_logits, buffers.router_matvec_sink};
    if (!encode_bf16_matvec(context, encoder, N_EMBD, experts, 1u, 1u,
                            router_buffers, error)) {
        return false;
    }

    const MoeRouteBuffers route_buffers{
        router_logits, buffers.route_ids, buffers.route_weights};
    if (!encode_moe_route(context, encoder, 1u, experts, false, route_buffers,
                          error)) {
        return false;
    }

    return encode_q8_1(context, encoder, 1u, N_EMBD,
                       {buffers.layer0.ffn.mixed, buffers.layer0.gdn.xq},
                       error);
}

bool encode_window_t1_moe_router_prefix(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoeRouterPrefixBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window T=1 layer-0 MoE router prefix",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_moe_router_prefix(
                context, encoder, vocabulary_size, experts, buffers,
                encode_error);
        },
        error);
}

bool encode_window_t1_moe_plan_prefix(
    Context& context, Encoder& encoder, std::uint32_t vocabulary_size,
    std::uint32_t experts, bool tiered,
    const WindowT1MoePlanPrefixBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window T=1 MoE plan prefix requires ordered dispatch boundaries";
        return false;
    }
    if (!validate_window_t1_moe_plan_prefix(
            context, vocabulary_size, experts, buffers, error)) {
        return false;
    }
    if (!encode_window_t1_moe_router_prefix(
            context, encoder, vocabulary_size, experts, buffers.router,
            error)) {
        return false;
    }

    const ReadOnlyBufferRef route_ids =
        context.buffers().read_only(buffers.router.route_ids);
    if (!route_ids) {
        error = "Window T=1 MoE plan could not form a live route ID input view";
        return false;
    }
    const MoePlanBuffers plan_buffers{route_ids, buffers.resident,
                                      buffers.usage, buffers.plans};
    return encode_moe_plan(context, encoder, 1u, experts, tiered, 0u,
                           plan_buffers, error);
}

bool encode_window_t1_moe_plan_prefix(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size, std::uint32_t experts, bool tiered,
    const WindowT1MoePlanPrefixBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window T=1 layer-0 MoE plan prefix",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_moe_plan_prefix(
                context, encoder, vocabulary_size, experts, tiered, buffers,
                encode_error);
        },
        error);
}

bool encode_window_t1_moe_layer0_full_mode(
    Context& context, const FullModeStorage& full_mode, Encoder& encoder,
    std::uint32_t vocabulary_size,
    std::uint32_t experts, MoeExpertFormat gate_up_format,
    MoeDownFormat down_format,
    const WindowT1MoeLayer0FullBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window T=1 MoE layer requires ordered dispatch boundaries";
        return false;
    }
    if (!validate_window_t1_moe_layer0_full_mode(
            context, full_mode, vocabulary_size, experts, buffers, error)) {
        return false;
    }

    if (!encode_window_t1_moe_plan_prefix(
            context, encoder, vocabulary_size, experts, false,
            buffers.producer, error)) {
        return false;
    }
    return encode_moe_routed_shared_ffn(
        context, encoder, 1u, experts, 0u, gate_up_format, down_format,
        buffers.ffn, error);
}

bool encode_window_t1_moe_layer0_full_mode(
    Context& context, const FullModeStorage& full_mode,
    CommandBuffer& command_buffer,
    std::uint32_t vocabulary_size, std::uint32_t experts,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const WindowT1MoeLayer0FullBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window T=1 layer-0 full-mode MoE",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_moe_layer0_full_mode(
                context, full_mode, encoder, vocabulary_size, experts,
                gate_up_format, down_format, buffers, encode_error);
        },
        error);
}

bool encode_window_moe_attention_entry(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowMoeAttentionEntryBuffers& buffers, std::string& error) {
    error.clear();
    if (!same_view(buffers.previous_gr_read.residual,
                   buffers.attention.residual)) {
        error = "Window MoE attention entry must reuse the previous GR residual";
        return false;
    }

    GRReadBuffers attention = buffers.attention;
    attention.block_out = buffers.previous_ffn.shared.output;
    attention.inject_prev = buffers.previous_gr_read.inject_out;
    return encode_gr_read(context, encoder, rows, true, attention, error);
}

bool encode_window_moe_attention_entry(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowMoeAttentionEntryBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window pending MoE attention entry",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_moe_attention_entry(
                context, encoder, rows, buffers, encode_error);
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
