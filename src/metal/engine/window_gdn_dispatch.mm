// Ordered speculative Window GDN block encoding.
#include "strata/metal/window_gdn_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/window_scratch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "../kernels/common/params.h"

namespace strata::metal {
namespace {

struct NamedView {
    BufferRef view;
    std::string_view name;
};

bool ranges_overlap(std::uint64_t left, std::size_t left_length,
                    std::uint64_t right, std::size_t right_length) {
    if (left_length == 0u || right_length == 0u) return false;
    if (left <= right) {
        return right - left < left_length;
    }
    return left - right < right_length;
}

bool views_overlap(BufferRef left, BufferRef right) {
    if (left.handle != nullptr && left.handle == right.handle &&
        ranges_overlap(left.offset, left.length, right.offset, right.length)) {
        return true;
    }
    if (left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.contents),
                       left.length,
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length)) {
        return true;
    }
    return left.gpu_address != 0u && right.gpu_address != 0u &&
           ranges_overlap(left.gpu_address, left.length, right.gpu_address,
                          right.length);
}

std::uint32_t compiled_columns(std::uint32_t rows) {
    if (rows <= 1u) return 1u;
    if (rows <= 2u) return 2u;
    if (rows <= 4u) return 4u;
    return 8u;
}

bool validate_views(Context& context, const WindowGDNBlockBuffers& buffers,
                    std::string& error) {
    const std::array views{
        NamedView{buffers.mixed, "mixed"},
        NamedView{buffers.history, "history"},
        NamedView{buffers.state, "state"},
        NamedView{buffers.qkv_weights, "qkv weights"},
        NamedView{buffers.conv_weights, "convolution weights"},
        NamedView{buffers.alpha, "alpha weights"},
        NamedView{buffers.beta_weights, "beta weights"},
        NamedView{buffers.dt_bias, "dt bias"},
        NamedView{buffers.a, "a"},
        NamedView{buffers.gate_weights, "gate weights"},
        NamedView{buffers.gamma, "gamma"},
        NamedView{buffers.output_weights, "output weights"},
        NamedView{buffers.xq, "xq"},
        NamedView{buffers.qkv, "qkv"},
        NamedView{buffers.h, "h"},
        NamedView{buffers.gate, "gate"},
        NamedView{buffers.beta, "beta"},
        NamedView{buffers.z, "z"},
        NamedView{buffers.y, "y"},
        NamedView{buffers.yq, "yq"},
        NamedView{buffers.block_out, "block output"},
        NamedView{buffers.sink, "matvec sink"},
        NamedView{buffers.active_rows, "active rows"},
    };
    for (std::size_t i = 0u; i < views.size(); ++i) {
        if (!views[i].view || !context.buffers().owns(views[i].view)) {
            error = "Window GDN " + std::string(views[i].name) +
                    " buffer is not owned by this Context";
            return false;
        }
        for (std::size_t j = 0u; j < i; ++j) {
            if (views_overlap(views[i].view, views[j].view)) {
                error = "Window GDN " + std::string(views[i].name) +
                        " buffer overlaps " + std::string(views[j].name);
                return false;
            }
        }
    }
    return true;
}

bool q8_1_weight_format(MatVecWeightFormat format) {
    return format != MatVecWeightFormat::bf16;
}

bool validate_commit_views(Context& context,
                           const WindowGDNCommitBuffers& buffers,
                           std::string& error) {
    const std::array views{
        NamedView{buffers.history, "history"},
        NamedView{buffers.qkv, "qkv"},
        NamedView{buffers.state, "state"},
        NamedView{buffers.h, "h"},
        NamedView{buffers.gate, "gate"},
        NamedView{buffers.beta, "beta"},
        NamedView{buffers.z, "z"},
        NamedView{buffers.gamma, "gamma"},
        NamedView{buffers.output_scratch, "output scratch"},
    };
    for (std::size_t i = 0u; i < views.size(); ++i) {
        if (!views[i].view || !context.buffers().owns(views[i].view)) {
            error = "Window GDN commit " + std::string(views[i].name) +
                    " buffer is not owned by this Context";
            return false;
        }
        for (std::size_t j = 0u; j < i; ++j) {
            if (views_overlap(views[i].view, views[j].view)) {
                error = "Window GDN commit " +
                        std::string(views[i].name) + " buffer overlaps " +
                        std::string(views[j].name);
                return false;
            }
        }
    }
    return true;
}

bool multiply_size(std::size_t left, std::size_t right,
                   std::size_t& result) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

}  // namespace

bool encode_window_gdn_commit(Context& context, Encoder& encoder,
                              std::uint32_t rows,
                              const WindowGDNCommitBuffers& buffers,
                              std::string& error) {
    error.clear();
    if (rows > 8u) {
        error = "Window GDN commit row count must be in 0..8";
        return false;
    }
    if (rows == 0u) return true;
    if (encoder.concurrent_group_open()) {
        error = "Window GDN commit requires an ordered concurrent-group boundary";
        return false;
    }
    if (!validate_commit_views(context, buffers, error)) return false;

    if (!encode_gdn_conv_commit(context, encoder, rows,
                                {buffers.history, buffers.qkv}, error)) {
        return false;
    }
    return encode_gdn_step_norm(
        context, encoder, rows, GDNStateMode::commit,
        {buffers.state, buffers.h, buffers.gate, buffers.beta, buffers.z,
         buffers.gamma, buffers.output_scratch},
        error);
}

bool encode_window_gdn_commit(Context& context, CommandBuffer& command_buffer,
                              std::uint32_t rows,
                              const WindowGDNCommitBuffers& buffers,
                              std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window GDN commit",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_gdn_commit(context, encoder, rows, buffers,
                                            encode_error);
        },
        error);
}

bool encode_window_gdn_commit_batch(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowGDNCommitBatchBuffers& buffers, std::string& error) {
    error.clear();
    if (rows > kWindowScratchTokenRows) {
        error = "Window GDN commit batch row count must be in 0..8";
        return false;
    }
    if (rows == 0u) return true;
    if (encoder.concurrent_group_open()) {
        error =
            "Window GDN commit batch requires an ordered concurrent-group boundary";
        return false;
    }
    if (buffers.states.size() != kSessionGdnLayers ||
        buffers.gamma.size() != kSessionGdnLayers) {
        error = "Window GDN commit batch requires one state and gamma per GDN layer";
        return false;
    }

    constexpr std::size_t kFloatBytes = sizeof(float);
    constexpr std::size_t kReplayRows = kWindowScratchTokenRows;
    std::size_t qkv_layer_bytes = 0u;
    std::size_t gate_layer_bytes = 0u;
    std::size_t qkv_replay_bytes = 0u;
    std::size_t gate_replay_bytes = 0u;
    std::size_t active_qkv_bytes = 0u;
    std::size_t active_gate_bytes = 0u;
    std::size_t active_value_bytes = 0u;
    std::size_t history_bytes = 0u;
    std::size_t state_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    if (!multiply_size(kReplayRows, GDN_CONV_CHANNELS, qkv_layer_bytes) ||
        !multiply_size(qkv_layer_bytes, kFloatBytes, qkv_layer_bytes) ||
        !multiply_size(kReplayRows, GDN_V_HEADS, gate_layer_bytes) ||
        !multiply_size(gate_layer_bytes, kFloatBytes, gate_layer_bytes) ||
        !multiply_size(qkv_layer_bytes, kSessionGdnLayers,
                       qkv_replay_bytes) ||
        !multiply_size(gate_layer_bytes, kSessionGdnLayers,
                       gate_replay_bytes) ||
        !multiply_size(rows, GDN_CONV_CHANNELS, active_qkv_bytes) ||
        !multiply_size(active_qkv_bytes, kFloatBytes, active_qkv_bytes) ||
        !multiply_size(rows, GDN_V_HEADS, active_gate_bytes) ||
        !multiply_size(active_gate_bytes, kFloatBytes, active_gate_bytes) ||
        !multiply_size(rows, GDN_VALUE_DIM, active_value_bytes) ||
        !multiply_size(active_value_bytes, kFloatBytes,
                       active_value_bytes) ||
        !multiply_size(GDN_HISTORY_ELEMENTS, kFloatBytes, history_bytes) ||
        !multiply_size(GDN_STATE_ELEMENTS, kFloatBytes, state_bytes) ||
        !multiply_size(GDN_VALUE_DIM, kFloatBytes, gamma_bytes)) {
        error = "Window GDN commit batch scratch extent overflows";
        return false;
    }

    const std::array bases{
        NamedView{buffers.qkv_replay, "qkv replay"},
        NamedView{buffers.h_replay, "h replay"},
        NamedView{buffers.gate_replay, "gate replay"},
        NamedView{buffers.beta_replay, "beta replay"},
        NamedView{buffers.z, "z scratch"},
        NamedView{buffers.output_scratch, "output scratch"},
    };
    for (std::size_t i = 0u; i < bases.size(); ++i) {
        if (!bases[i].view || !context.buffers().owns(bases[i].view)) {
            error = "Window GDN commit batch " +
                    std::string(bases[i].name) +
                    " buffer is not owned by this Context";
            return false;
        }
        const std::size_t required =
            i == 0u || i == 1u
                ? qkv_replay_bytes
                : i == 2u || i == 3u ? gate_replay_bytes
                                      : active_value_bytes;
        if (bases[i].view.length < required) {
            error = "Window GDN commit batch " +
                    std::string(bases[i].name) +
                    " buffer is shorter than its required extent";
            return false;
        }
        for (std::size_t j = 0u; j < i; ++j) {
            if (views_overlap(bases[i].view, bases[j].view)) {
                error = "Window GDN commit batch " +
                        std::string(bases[i].name) + " buffer overlaps " +
                        std::string(bases[j].name);
                return false;
            }
        }
    }

    std::vector<WindowGDNCommitBuffers> commits;
    commits.reserve(kSessionGdnLayers);
    std::vector<NamedView> all_layer_views;
    all_layer_views.reserve(static_cast<std::size_t>(kSessionGdnLayers) * 7u);
    for (std::uint32_t layer = 0u; layer < kSessionGdnLayers; ++layer) {
        std::size_t qkv_offset = 0u;
        std::size_t gate_offset = 0u;
        if (!multiply_size(layer, qkv_layer_bytes, qkv_offset) ||
            !multiply_size(layer, gate_layer_bytes, gate_offset)) {
            error = "Window GDN commit batch layer offset overflows";
            return false;
        }
        const GdnSessionStateViews& state = buffers.states[layer];
        const BufferRef gamma = buffers.gamma[layer];
        const WindowGDNCommitBuffers commit{
            state.conv_history,
            buffers.qkv_replay.slice(qkv_offset, active_qkv_bytes),
            state.recurrent,
            buffers.h_replay.slice(qkv_offset, active_qkv_bytes),
            buffers.gate_replay.slice(gate_offset, active_gate_bytes),
            buffers.beta_replay.slice(gate_offset, active_gate_bytes),
            buffers.z,
            gamma,
            buffers.output_scratch,
        };
        if (!validate_commit_views(context, commit, error)) {
            error = "Window GDN commit batch layer " +
                    std::to_string(layer) + ": " + error;
            return false;
        }
        if (commit.history.length < history_bytes ||
            commit.qkv.length < active_qkv_bytes ||
            commit.state.length < state_bytes ||
            commit.h.length < active_qkv_bytes ||
            commit.gate.length < active_gate_bytes ||
            commit.beta.length < active_gate_bytes ||
            commit.z.length < active_value_bytes ||
            commit.gamma.length < gamma_bytes ||
            commit.output_scratch.length < active_value_bytes) {
            error = "Window GDN commit batch layer " +
                    std::to_string(layer) +
                    " has a buffer shorter than its active extent";
            return false;
        }
        commits.push_back(commit);
        all_layer_views.push_back({commit.history, "history"});
        all_layer_views.push_back({commit.qkv, "qkv replay"});
        all_layer_views.push_back({commit.state, "state"});
        all_layer_views.push_back({commit.h, "h replay"});
        all_layer_views.push_back({commit.gate, "gate replay"});
        all_layer_views.push_back({commit.beta, "beta replay"});
        all_layer_views.push_back({commit.gamma, "gamma"});
    }

    for (const NamedView& base : bases) {
        for (const NamedView& layer_view : all_layer_views) {
            const bool is_own_replay_slice =
                (base.name == "qkv replay" &&
                 layer_view.name == "qkv replay") ||
                (base.name == "h replay" &&
                 layer_view.name == "h replay") ||
                (base.name == "gate replay" &&
                 layer_view.name == "gate replay") ||
                (base.name == "beta replay" &&
                 layer_view.name == "beta replay");
            if (is_own_replay_slice) continue;
            if (views_overlap(base.view, layer_view.view)) {
                error = "Window GDN commit batch " +
                        std::string(base.name) + " buffer overlaps a layer " +
                        std::string(layer_view.name) + " view";
                return false;
            }
        }
    }
    for (std::size_t i = 0u; i < all_layer_views.size(); ++i) {
        for (std::size_t j = 0u; j < i; ++j) {
            if (views_overlap(all_layer_views[i].view,
                              all_layer_views[j].view)) {
                error = "Window GDN commit batch layer view " +
                        std::string(all_layer_views[i].name) + " overlaps " +
                        std::string(all_layer_views[j].name);
                return false;
            }
        }
    }

    for (const WindowGDNCommitBuffers& commit : commits) {
        if (!encode_window_gdn_commit(context, encoder, rows, commit, error)) {
            return false;
        }
    }
    return true;
}

bool encode_window_gdn_commit_batch(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowGDNCommitBatchBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window GDN commit batch",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_gdn_commit_batch(context, encoder, rows,
                                                  buffers, encode_error);
        },
        error);
}

bool encode_window_gdn_block(Context& context, Encoder& encoder,
                             std::uint32_t rows,
                             const WindowGDNBlockBuffers& buffers,
                             std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window GDN block requires an ordered concurrent-group boundary";
        return false;
    }
    if (rows == 0u || rows > 8u) {
        error = "Window GDN block row count must be in 1..8";
        return false;
    }
    if (!q8_1_weight_format(buffers.qkv_format) ||
        !q8_1_weight_format(buffers.gate_format) ||
        !q8_1_weight_format(buffers.output_format)) {
        error = "Window GDN Q8_1 matvec stages require quantized weight formats";
        return false;
    }
    if (!validate_views(context, buffers, error)) return false;

    const std::uint32_t columns = compiled_columns(rows);
    const MatVecBuffers qkv_matvec{buffers.xq, buffers.qkv_weights,
                                   buffers.qkv, buffers.sink,
                                   buffers.active_rows};
    const MatVecBuffers gate_matvec{buffers.xq, buffers.gate_weights,
                                    buffers.z, buffers.sink,
                                    buffers.active_rows};
    const MatVecBuffers output_matvec{buffers.yq, buffers.output_weights,
                                      buffers.block_out, buffers.sink,
                                      buffers.active_rows};

    if (!encode_q8_1(context, encoder, rows, N_EMBD,
                     {buffers.mixed, buffers.xq}, error)) {
        return false;
    }
    if (!encode_matvec(context, encoder, buffers.qkv_format, N_EMBD,
                       GDN_CONV_CHANNELS, rows, columns, qkv_matvec, error)) {
        return false;
    }
    if (!encode_gdn_conv_l2(context, encoder, rows,
                            {buffers.history, buffers.qkv,
                             buffers.conv_weights, buffers.h},
                            error)) {
        return false;
    }
    if (!encode_gdn_ab(context, encoder, rows,
                       {buffers.mixed, buffers.alpha, buffers.beta_weights,
                        buffers.dt_bias, buffers.a, buffers.gate, buffers.beta},
                       error)) {
        return false;
    }
    if (!encode_matvec(context, encoder, buffers.gate_format, N_EMBD,
                       GDN_VALUE_DIM, rows, columns, gate_matvec, error)) {
        return false;
    }
    if (!encode_gdn_step_norm(
            context, encoder, rows, GDNStateMode::window,
            {buffers.state, buffers.h, buffers.gate, buffers.beta, buffers.z,
             buffers.gamma, buffers.y},
            error)) {
        return false;
    }
    if (!encode_q8_1(context, encoder, rows, GDN_VALUE_DIM,
                     {buffers.y, buffers.yq}, error)) {
        return false;
    }
    return encode_matvec(context, encoder, buffers.output_format,
                         GDN_VALUE_DIM, N_EMBD, rows, columns, output_matvec,
                         error);
}

bool encode_window_gdn_block(Context& context, CommandBuffer& command_buffer,
                             std::uint32_t rows,
                             const WindowGDNBlockBuffers& buffers,
                             std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window GDN block",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_gdn_block(context, encoder, rows, buffers,
                                           encode_error);
        },
        error);
}

}  // namespace strata::metal
