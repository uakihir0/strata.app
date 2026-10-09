// Ordered speculative Window GDN block encoding.
#include "strata/metal/window_gdn_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/window_params.hpp"
#include "strata/metal/window_scratch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

bool validate_window_gdn_ple_commit_views(
    Context& context, std::uint32_t rows,
    const WindowGdnPleCommitBuffers& buffers, std::string& error) {
    if (rows > kWindowScratchTokenRows) {
        error = "Window GDN/PLE commit row count must be in 0..8";
        return false;
    }
    if (rows == 0u) return true;
    if (buffers.gdn.states.size() != kSessionGdnLayers ||
        buffers.gdn.gamma.size() != kSessionGdnLayers) {
        error = "Window GDN/PLE commit requires one GDN state and gamma per layer";
        return false;
    }
    if (!buffers.ple.history || !context.buffers().owns(buffers.ple.history) ||
        !buffers.ple.ple_norm ||
        !context.buffers().owns(buffers.ple.ple_norm)) {
        error = "Window GDN/PLE commit PLE views are not owned by this Context";
        return false;
    }

    constexpr std::size_t kFloatBytes = sizeof(float);
    const std::size_t history_bytes =
        static_cast<std::size_t>(kSessionPleHistoryElements) * kFloatBytes;
    const std::size_t norm_bytes =
        static_cast<std::size_t>(rows) * kSessionEmbeddingWidth *
        kSessionHyperconnections * kFloatBytes;
    if (buffers.ple.history.length < history_bytes ||
        buffers.ple.ple_norm.length < norm_bytes) {
        error = "Window GDN/PLE commit has a PLE view shorter than its active extent";
        return false;
    }
    if (buffers.ple.history.offset % alignof(float) != 0u ||
        buffers.ple.ple_norm.offset % alignof(float) != 0u) {
        error = "Window GDN/PLE commit has a misaligned PLE view";
        return false;
    }
    if (views_overlap(buffers.ple.history, buffers.ple.ple_norm)) {
        error = "Window GDN/PLE commit PLE history and normalized rows overlap";
        return false;
    }

    const std::array gdn_bases{
        NamedView{buffers.gdn.qkv_replay, "qkv replay"},
        NamedView{buffers.gdn.h_replay, "h replay"},
        NamedView{buffers.gdn.gate_replay, "gate replay"},
        NamedView{buffers.gdn.beta_replay, "beta replay"},
        NamedView{buffers.gdn.z, "z scratch"},
        NamedView{buffers.gdn.output_scratch, "output scratch"},
    };
    for (const NamedView& gdn_view : gdn_bases) {
        if (!gdn_view.view || !context.buffers().owns(gdn_view.view)) {
            error = "Window GDN/PLE commit " + std::string(gdn_view.name) +
                    " buffer is not owned by this Context";
            return false;
        }
        if (views_overlap(buffers.ple.history, gdn_view.view) ||
            views_overlap(buffers.ple.ple_norm, gdn_view.view)) {
            error = "Window GDN/PLE commit PLE views overlap GDN " +
                    std::string(gdn_view.name);
            return false;
        }
    }
    for (std::size_t i = 0u; i < buffers.gdn.states.size(); ++i) {
        const GdnSessionStateViews& state = buffers.gdn.states[i];
        if (!state.conv_history || !context.buffers().owns(state.conv_history) ||
            !state.recurrent || !context.buffers().owns(state.recurrent) ||
            !buffers.gdn.gamma[i] ||
            !context.buffers().owns(buffers.gdn.gamma[i])) {
            error = "Window GDN/PLE commit has a GDN state or gamma outside this Context";
            return false;
        }
        for (const BufferRef& view : {state.conv_history, state.recurrent,
                                      buffers.gdn.gamma[i]}) {
            if (views_overlap(buffers.ple.history, view) ||
                views_overlap(buffers.ple.ple_norm, view)) {
                error = "Window GDN/PLE commit PLE views overlap GDN layer " +
                        std::to_string(i);
                return false;
            }
        }
    }
    return true;
}

struct CommitStageView {
    BufferRef view;
    std::string name;
    bool writes = false;
};

bool validate_window_accepted_prefix_cross_stage_views(
    Context& context, std::uint32_t rows,
    const WindowAcceptedPrefixStateCommitBuffers& buffers,
    std::string& error) {
    if (rows == 0u || rows > kWindowScratchTokenRows) {
        error = "Window accepted-prefix state commit row count must be in 1..8";
        return false;
    }
    if (buffers.qsa_indexers.size() != kSessionQsaLayers) {
        error = "Window accepted-prefix state commit requires exactly 12 QSA indexer layers";
        return false;
    }
    if (!validate_window_gdn_ple_commit_views(
            context, rows, buffers.gdn_ple, error)) {
        return false;
    }

    std::vector<CommitStageView> gdn_ple_views;
    gdn_ple_views.reserve(4u + kSessionGdnLayers * 3u);
    auto add_gdn_ple = [&](BufferRef view, std::string name,
                           bool writes) {
        gdn_ple_views.push_back(
            CommitStageView{view, std::move(name), writes});
    };
    const WindowGDNCommitBatchBuffers& gdn = buffers.gdn_ple.gdn;
    add_gdn_ple(gdn.qkv_replay, "GDN qkv replay", false);
    add_gdn_ple(gdn.h_replay, "GDN h replay", false);
    add_gdn_ple(gdn.gate_replay, "GDN gate replay", false);
    add_gdn_ple(gdn.beta_replay, "GDN beta replay", false);
    add_gdn_ple(buffers.gdn_ple.ple.ple_norm, "PLE normalized rows", false);
    add_gdn_ple(buffers.gdn_ple.ple.history, "PLE history", true);
    for (std::size_t layer = 0u; layer < gdn.states.size(); ++layer) {
        add_gdn_ple(gdn.states[layer].conv_history,
                    "GDN layer " + std::to_string(layer) +
                        " convolution history",
                    true);
        add_gdn_ple(gdn.states[layer].recurrent,
                    "GDN layer " + std::to_string(layer) +
                        " recurrent state",
                    true);
        add_gdn_ple(gdn.gamma[layer],
                    "GDN layer " + std::to_string(layer) + " gamma",
                    false);
    }

    std::vector<CommitStageView> qsa_views;
    qsa_views.reserve(static_cast<std::size_t>(kSessionQsaLayers) * 10u);
    auto add_qsa = [&](BufferRef view, std::uint32_t layer,
                       const char* role, bool writes) {
        if (!view || !context.buffers().owns(view)) {
            error = "Window accepted-prefix state commit QSA layer " +
                    std::to_string(layer) + " " + role +
                    " view is not owned by this Context";
            return false;
        }
        qsa_views.push_back(CommitStageView{
            view,
            "QSA layer " + std::to_string(layer) + " " + role,
            writes});
        return true;
    };
    for (std::uint32_t layer = 0u; layer < kSessionQsaLayers; ++layer) {
        const QsaIndexerAppendBuffers& qsa = buffers.qsa_indexers[layer];
        if (!add_qsa(qsa.raw_keys, layer, "raw keys", false) ||
            !add_qsa(qsa.steps, layer, "StepRows", false) ||
            !add_qsa(qsa.gamma, layer, "gamma", false) ||
            !add_qsa(qsa.rope, layer, "RoPE", false) ||
            !add_qsa(qsa.tail, layer, "tail", true) ||
            !add_qsa(qsa.dead, layer, "dead key", true) ||
            !add_qsa(qsa.pooled, layer, "pooled keys", true) ||
            !add_qsa(qsa.block_position, layer, "block position", true) ||
            !add_qsa(qsa.tail_snapshot, layer, "tail snapshot", false) ||
            !add_qsa(qsa.block_pos_snapshot, layer,
                     "block-position snapshot", false)) {
            return false;
        }
    }

    for (const CommitStageView& gdn_ple : gdn_ple_views) {
        for (const CommitStageView& qsa : qsa_views) {
            if ((!gdn_ple.writes && !qsa.writes) ||
                !views_overlap(gdn_ple.view, qsa.view)) {
                continue;
            }
            error = "Window accepted-prefix state commit " + gdn_ple.name +
                    " overlaps " + qsa.name;
            return false;
        }
    }
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

bool encode_window_gdn_ple_commit(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowGdnPleCommitBuffers& buffers, std::string& error) {
    error.clear();
    if (rows > kWindowScratchTokenRows) {
        error = "Window GDN/PLE commit row count must be in 0..8";
        return false;
    }
    if (rows == 0u) return true;
    if (encoder.concurrent_group_open()) {
        error =
            "Window GDN/PLE commit requires an ordered concurrent-group boundary";
        return false;
    }
    if (!validate_window_gdn_ple_commit_views(context, rows, buffers, error)) {
        return false;
    }
    if (!encode_window_gdn_commit_batch(context, encoder, rows, buffers.gdn,
                                        error)) {
        return false;
    }
    return encode_ple_history(context, encoder, rows, buffers.ple, error);
}

bool encode_window_gdn_ple_commit(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowGdnPleCommitBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window GDN/PLE accepted-prefix commit",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_gdn_ple_commit(context, encoder, rows,
                                                buffers, encode_error);
        },
        error);
}

bool encode_window_accepted_prefix_state_commit(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowAcceptedPrefixStateCommitBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "Window accepted-prefix state commit requires an ordered concurrent-group boundary";
        return false;
    }
    if (!validate_window_accepted_prefix_cross_stage_views(
            context, rows, buffers, error)) {
        return false;
    }
    if (!encode_window_gdn_ple_commit(context, encoder, rows,
                                      buffers.gdn_ple, error)) {
        return false;
    }
    return encode_qsa_indexer_commit_batch(
        context, encoder, rows, buffers.qsa_cell_capacity,
        buffers.qsa_rope_zero_cos, buffers.qsa_rope_zero_sin,
        buffers.qsa_indexers, error);
}

bool encode_window_accepted_prefix_state_commit(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowAcceptedPrefixStateCommitBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window accepted-prefix state commit",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_accepted_prefix_state_commit(
                context, encoder, rows, buffers, encode_error);
        },
        error);
}

namespace {

bool validate_write_through_params(Context& context, std::uint32_t rows,
                                   BufferRef params,
                                   const WindowGDNBlockBuffers& gdn,
                                   std::string& error) {
    if (!params || !context.buffers().owns(params)) {
        error = "Window GDN write-through parameter block is not owned by this Context";
        return false;
    }
    if (params.contents == nullptr || params.length < sizeof(WindowParams) ||
        reinterpret_cast<std::uintptr_t>(params.contents) %
                alignof(WindowParams) !=
            0u) {
        error = "Window GDN write-through parameter block is undersized or misaligned";
        return false;
    }
    WindowParams values{};
    std::memcpy(&values, params.contents, sizeof(values));
    if (values.T != rows || values.n_keep != rows ||
        values.nc != compiled_columns(rows)) {
        error = "Window GDN write-through parameters require T == n_keep == rows and matching nc";
        return false;
    }
    const std::array writes{
        NamedView{gdn.history, "history"},
        NamedView{gdn.state, "state"},
        NamedView{gdn.xq, "xq"},
        NamedView{gdn.qkv, "qkv"},
        NamedView{gdn.h, "h"},
        NamedView{gdn.gate, "gate"},
        NamedView{gdn.beta, "beta"},
        NamedView{gdn.z, "z"},
        NamedView{gdn.y, "y"},
        NamedView{gdn.yq, "yq"},
        NamedView{gdn.block_out, "block output"},
        NamedView{gdn.sink, "matvec sink"},
    };
    for (const auto& write : writes) {
        if (views_overlap(params, write.view)) {
            error = "Window GDN write-through parameter block overlaps " +
                    std::string(write.name);
            return false;
        }
    }
    return true;
}

bool encode_window_gdn_block_impl(Context& context, Encoder& encoder,
                                  std::uint32_t rows,
                                  const WindowGDNBlockBuffers& buffers,
                                  GDNStateMode state_mode,
                                  bool commit_history,
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
    if (state_mode != GDNStateMode::window &&
        state_mode != GDNStateMode::write_through) {
        error = "Window GDN block mode must be window or write-through";
        return false;
    }
    if (commit_history != (state_mode == GDNStateMode::write_through)) {
        error = "Window GDN write-through state and history modes must agree";
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
            context, encoder, rows, state_mode,
            {buffers.state, buffers.h, buffers.gate, buffers.beta, buffers.z,
             buffers.gamma, buffers.y},
            error)) {
        return false;
    }
    if (commit_history &&
        !encode_gdn_conv_commit(context, encoder, rows,
                                {buffers.history, buffers.qkv}, error)) {
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

}  // namespace

bool encode_window_gdn_block(Context& context, Encoder& encoder,
                             std::uint32_t rows,
                             const WindowGDNBlockBuffers& buffers,
                             std::string& error) {
    return encode_window_gdn_block_impl(
        context, encoder, rows, buffers, GDNStateMode::window, false, error);
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

bool encode_window_gdn_write_through_block(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const WindowGDNWriteThroughBuffers& buffers, std::string& error) {
    error.clear();
    if (rows == 0u || rows > 8u) {
        error = "Window GDN block row count must be in 1..8";
        return false;
    }
    if (!validate_write_through_params(context, rows, buffers.params,
                                       buffers.gdn, error)) {
        return false;
    }
    return encode_window_gdn_block_impl(
        context, encoder, rows, buffers.gdn,
        GDNStateMode::write_through, true, error);
}

bool encode_window_gdn_write_through_block(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const WindowGDNWriteThroughBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window GDN write-through block",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_gdn_write_through_block(
                context, encoder, rows, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
