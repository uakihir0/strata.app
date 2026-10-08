// Device-free tests for the exact T=8 Metal Window scratch layout.
#include "strata/metal/window_scratch.hpp"

#include "strata/metal/tiered.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::metal;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Geometry supported_geometry(std::uint32_t n_expert = 512u) {
    Geometry geometry;
    geometry.family = "qwen4exp";
    geometry.n_layers = kSessionModelLayers;
    geometry.n_expert = n_expert;
    geometry.n_qsa_layers = kSessionQsaLayers;
    geometry.n_kv_head = kSessionKvHeads;
    geometry.head_dim = kSessionHeadWidth;
    geometry.n_embd = kSessionEmbeddingWidth;
    geometry.vocab = kSessionVocabularySize;
    return geometry;
}

std::vector<WindowScratchRegion> regions_of(const WindowScratchLayout& layout) {
    return {
        layout.ple_input,
        layout.embedding,
        layout.residual,
        layout.mixed,
        layout.block_output,
        layout.inject,
        layout.inject2,
        layout.gr_lo,
        layout.gr_rs,
        layout.gr_xn,
        layout.xq,
        layout.gdn_qkv_replay,
        layout.gdn_h_replay,
        layout.gdn_gate_replay,
        layout.gdn_beta_replay,
        layout.gdn_z,
        layout.gdn_y,
        layout.ple_key,
        layout.ple_value,
        layout.ple_query,
        layout.ple_gate,
        layout.ple_gated,
        layout.ple_norm,
        layout.qsa_q_full,
        layout.qsa_q,
        layout.qsa_k,
        layout.qsa_v,
        layout.qsa_indexer_key_replay,
        layout.qsa_indexer_query,
        layout.qsa_scores,
        layout.qsa_ids,
        layout.qsa_attention,
        layout.qsa_attention_gated,
        layout.qsa_attention_scratch,
        layout.qsa_tail_snapshot,
        layout.qsa_block_position_snapshot,
        layout.router_logits,
        layout.router_ids,
        layout.router_weights,
        layout.moe_plans,
        layout.moe_gate,
        layout.moe_up,
        layout.moe_hidden,
        layout.moe_hidden_q,
        layout.moe_parts,
        layout.shared_gate,
        layout.shared_up,
        layout.shared_hidden,
        layout.shared_hidden_q,
        layout.shared_output,
        layout.shared_gate_logit,
        layout.head_normalized,
        layout.head_low_rank,
        layout.head_gate,
        layout.head_mixed,
        layout.head_q8_1,
        layout.head_logits,
        layout.sampler_history,
        layout.sampler_candidates,
        layout.sampler_selection,
        layout.sampler_output,
        layout.matvec_sink,
    };
}

void check_regions(const WindowScratchLayout& layout) {
    const std::vector<WindowScratchRegion> regions = regions_of(layout);
    std::size_t previous_end = 0u;
    for (const WindowScratchRegion region : regions) {
        require(region.bytes != 0u, "every scratch region must be nonempty");
        require(region.offset % 256u == 0u,
                "every scratch region must begin at a 256-byte boundary");
        require(region.offset >= previous_end,
                "scratch regions must be monotonic and non-overlapping");
        require(region.offset <= layout.allocation_bytes,
                "scratch region offset must fit inside the allocation");
        require(region.bytes <= layout.allocation_bytes - region.offset,
                "every scratch region must fit inside the allocation");
        const std::size_t expected_offset =
            (previous_end + 255u) / 256u * 256u;
        require(region.offset == expected_offset,
                "layout must contain only the documented alignment padding");
        previous_end = region.offset + region.bytes;
    }
    require(previous_end == layout.allocation_bytes,
            "allocation size must end at the final named scratch region");
}

void check_for_capacity(std::uint32_t capacity, std::size_t expected_bytes,
                        std::uint32_t n_expert = 512u) {
    WindowScratchLayout layout;
    std::string error;
    require(make_window_scratch_layout(supported_geometry(n_expert), capacity,
                                       layout, error),
            "make T=8 scratch layout: " + error);
    require(layout.allocation_bytes == expected_bytes,
            "fixed T=8 scratch allocation must match its byte-exact fixture");
    require(layout.cell_capacity == capacity &&
                layout.max_blocks == capacity / 4u + 2u &&
                layout.n_expert == n_expert,
            "QSA and router shape metadata must match the request");
    require(layout.attention_scratch_floats_per_row == 204'400u,
            "sparse-attention scratch must use 33 chunks and the upstream stride");
    require(layout.ple_input.bytes == 8u * 2'560u * sizeof(float) &&
                layout.embedding.bytes == layout.ple_input.bytes,
            "host PLE input and embedding must each hold eight 2560-value rows");
    require(layout.residual.bytes == 8u * 4u * 2'560u * sizeof(float),
            "residual storage must hold eight four-stream rows");
    require(layout.gdn_qkv_replay.bytes ==
                36u * 8u * 10'240u * sizeof(float) &&
                layout.gdn_h_replay.bytes == layout.gdn_qkv_replay.bytes &&
                layout.gdn_gate_replay.bytes ==
                    36u * 8u * 48u * sizeof(float) &&
                layout.gdn_beta_replay.bytes == layout.gdn_gate_replay.bytes,
            "commit replay storage must cover all 36 GDN layers and rows");
    require(layout.qsa_scores.bytes ==
                8u * (static_cast<std::size_t>(capacity / 4u) + 2u) *
                    sizeof(float) &&
                layout.qsa_ids.bytes == 8u * 2'051u * sizeof(std::int32_t),
            "QSA scores and selection IDs must cover the configured capacity");
    require(layout.qsa_attention_scratch.bytes ==
                8u * 204'400u * sizeof(float) &&
                layout.qsa_tail_snapshot.bytes ==
                    12u * 3u * 128u * sizeof(float) &&
                layout.qsa_block_position_snapshot.bytes ==
                    12u * sizeof(std::int32_t),
            "QSA attention and rollback snapshots must match the documented ABI");
    require(layout.moe_plans.bytes ==
                48u * sizeof(TieredMoePlan) &&
                layout.router_logits.bytes ==
                    8u * static_cast<std::size_t>(n_expert) * sizeof(float) &&
                layout.moe_gate.bytes ==
                    8u * 10u * 640u * sizeof(float) &&
                layout.moe_up.bytes == layout.moe_gate.bytes &&
                layout.moe_hidden.bytes == layout.moe_gate.bytes &&
                layout.moe_hidden_q.bytes == 8u * 10u * 20u * 36u,
            "MoE routing, plans and expert intermediates must match the shader ABI");
    require(layout.ple_norm.bytes == 8u * 10'240u * sizeof(float) &&
                layout.head_normalized.bytes ==
                    8u * 4u * 2'560u * sizeof(float) &&
                layout.head_low_rank.bytes == 8u * 4u * 320u * sizeof(float) &&
                layout.head_gate.bytes ==
                    8u * 4u * 2'560u * sizeof(float) &&
                layout.head_q8_1.bytes == 8u * 80u * 36u &&
                layout.head_logits.bytes ==
                    8u * 248'320u * sizeof(float) &&
                layout.sampler_history.bytes ==
                    8u * 4'096u * sizeof(std::int32_t) &&
                layout.sampler_candidates.bytes ==
                    8u * 64u * 64u * 2u * sizeof(std::int32_t) &&
                layout.sampler_selection.bytes ==
                    8u * 64u * 2u * sizeof(std::int32_t) &&
                layout.sampler_output.bytes ==
                    8u * sizeof(std::int32_t) &&
                layout.matvec_sink.bytes ==
                    3u * 248'320u * sizeof(float),
            "commit PLE rows, vocabulary logits, sampler buffers and sink must fit");
    check_regions(layout);
}

}  // namespace

int main() {
    try {
        check_for_capacity(32'768u, 48'360'704u, 256u);
        check_for_capacity(65'536u, 48'622'848u, 256u);
        check_for_capacity(kWindowScratchMaxCellCapacity, 50'195'712u, 256u);
        check_for_capacity(32'768u, 48'368'896u);
        check_for_capacity(65'536u, 48'631'040u);
        check_for_capacity(kWindowScratchMaxCellCapacity, 50'203'904u);

        WindowScratchLayout compact;
        WindowScratchLayout full;
        std::string error;
        require(make_window_scratch_layout(supported_geometry(256u), 65'536u,
                                           compact, error),
                "build the 256-expert Coder layout: " + error);
        require(make_window_scratch_layout(supported_geometry(512u), 65'536u,
                                           full, error),
                "build the 512-expert Swift layout: " + error);
        Geometry package_family = supported_geometry(512u);
        package_family.family = "swift";
        WindowScratchLayout family_layout;
        require(make_window_scratch_layout(package_family, 65'536u,
                                           family_layout, error) &&
                    family_layout.allocation_bytes == full.allocation_bytes,
                "supported package-family identity must preserve the Qwen4exp scratch ABI: " +
                    error);
        require(full.allocation_bytes >= compact.allocation_bytes &&
                    full.allocation_bytes - compact.allocation_bytes ==
                    8u * (512u - 256u) * sizeof(float),
                "only the router-logit table may grow with expert count");

        for (std::uint32_t capacity : {0u, 1u, 3u, 5u, 262'148u}) {
            WindowScratchLayout rejected;
            rejected.allocation_bytes = 123u;
            require(!make_window_scratch_layout(supported_geometry(), capacity,
                                                rejected, error) &&
                        rejected.allocation_bytes == 0u && !error.empty(),
                    "invalid QSA capacity must fail before publishing a layout");
        }
        Geometry wrong_family = supported_geometry();
        wrong_family.family = "other";
        Geometry wrong_layers = supported_geometry();
        wrong_layers.n_layers -= 1u;
        Geometry wrong_embedding = supported_geometry();
        wrong_embedding.n_embd -= 1u;
        for (const Geometry& geometry :
             {wrong_family, supported_geometry(0u), supported_geometry(257u),
              wrong_layers, wrong_embedding}) {
            WindowScratchLayout rejected;
            require(!make_window_scratch_layout(geometry, 65'536u, rejected,
                                                error) &&
                        rejected.allocation_bytes == 0u && !error.empty(),
                    "unsupported model geometry must fail closed");
        }

        std::cout << "metal_window_scratch_layout_test: all checks passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_window_scratch_layout_test: " << exception.what()
                  << '\n';
        return 1;
    }
}
