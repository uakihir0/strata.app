// Context-owned storage tests for the fixed T=8 Window scratch allocation.
#include "strata/metal/window_scratch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::metal;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Geometry supported_geometry() {
    Geometry geometry;
    geometry.family = "qwen4exp";
    geometry.n_layers = kSessionModelLayers;
    geometry.n_expert = 512u;
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

void check_region_view(BufferRef view, BufferRef origin_view,
                       WindowScratchRegion region) {
    std::uint64_t expected_gpu_address = 0u;
    require(detail::gpu_address_with_offset(origin_view.gpu_address,
                                            region.offset,
                                            expected_gpu_address),
            "region GPU address must not overflow");
    require(view && view.handle == origin_view.handle &&
                view.length == region.bytes &&
                view.offset == origin_view.offset + region.offset &&
                view.gpu_address == expected_gpu_address &&
                view.contents ==
                    static_cast<std::byte*>(origin_view.contents) + region.offset,
            "bounded scratch view must match its named allocation region");
}

}  // namespace

int main() {
    try {
        const Geometry geometry = supported_geometry();
        WindowScratchLayout expected_layout;
        std::string error;
        require(make_window_scratch_layout(geometry, 65'536u,
                                           expected_layout, error),
                "make expected Window scratch layout: " + error);

        ContextOptions options;
        options.pipelines = false;
        auto context = Context::open(options, error);
        require(context != nullptr, "open Metal Context: " + error);

        const std::size_t allocations_before =
            context->buffers().allocation_count();
        const std::uint64_t bytes_before =
            context->buffers().allocated_size();
        Geometry invalid_geometry = geometry;
        invalid_geometry.vocab -= 1u;
        require(WindowScratchStorage::open(*context, invalid_geometry,
                                           65'536u, error) == nullptr &&
                    context->buffers().allocation_count() == allocations_before,
                "unsupported geometry must fail before Metal allocation");
        require(WindowScratchStorage::open(*context, geometry, 5u, error) ==
                        nullptr &&
                    context->buffers().allocation_count() == allocations_before,
                "unsupported QSA capacity must fail before Metal allocation");

        const char* old_poison = std::getenv("STRATA_METAL_POISON");
        const bool had_old_poison = old_poison != nullptr;
        const std::string old_poison_value =
            had_old_poison ? old_poison : "";
        require(setenv("STRATA_METAL_POISON", "1", 1) == 0,
                "enable allocator poison for Window scratch");
        auto storage = WindowScratchStorage::open(
            *context, geometry, 65'536u, error);
        const int restore_result =
            had_old_poison
                ? setenv("STRATA_METAL_POISON", old_poison_value.c_str(), 1)
                : unsetenv("STRATA_METAL_POISON");
        require(restore_result == 0, "restore STRATA_METAL_POISON");
        require(storage != nullptr && storage->is_valid(),
                "open Context-owned Window scratch storage: " + error);
        require(context->buffers().allocation_count() == allocations_before + 1u &&
                    context->buffers().allocated_size() > bytes_before,
                "Window scratch must add exactly one Context-owned allocation");

        require(storage->layout().allocation_bytes ==
                        expected_layout.allocation_bytes &&
                    storage->layout().cell_capacity == 65'536u &&
                    storage->layout().n_expert == geometry.n_expert,
                "published Window scratch layout must match its checked geometry");
        require(storage->layout().head_q8_1.bytes ==
                    kWindowScratchTokenRows *
                        (kSessionEmbeddingWidth / 32u) * 36u,
                "head Q8_1 scratch must hold eight 2560-wide rows");
        BufferRef origin_view;
        require(storage->view(storage->layout().ple_input, origin_view, error) &&
                    origin_view.offset == 0u &&
                    context->buffers().owns(origin_view),
                "the first named region must expose a live bounded view: " + error);

        const std::vector<WindowScratchRegion> regions =
            regions_of(storage->layout());
        std::size_t previous_end = 0u;
        for (const WindowScratchRegion region : regions) {
            require(region.offset % 256u == 0u &&
                        region.offset >= previous_end &&
                        region.offset <= expected_layout.allocation_bytes &&
                        region.bytes <=
                            expected_layout.allocation_bytes - region.offset,
                    "every named region must be aligned, ordered and bounded");
            BufferRef view;
            require(storage->view(region, view, error),
                    "get bounded Window scratch region view: " + error);
            check_region_view(view, origin_view, region);
            require(context->buffers().owns(view),
                    "every scratch view must remain Context-owned");
            const auto* region_bytes =
                static_cast<const std::uint8_t*>(view.contents);
            require(std::all_of(
                        region_bytes, region_bytes + view.length,
                        [](std::uint8_t byte) { return byte == 0xffu; }),
                    "STRATA_METAL_POISON must fill each named region with 0xFF");
            previous_end = region.offset + region.bytes;
        }
        require(previous_end == expected_layout.allocation_bytes,
                "named regions must account for the complete allocation");

        BufferRef forged_view;
        require(!storage->view({0u, 256u}, forged_view, error) &&
                    !forged_view && !error.empty(),
                "Window scratch must reject a range that is not a named region");

        const std::size_t allocations_before_rollback =
            context->buffers().allocation_count();
        const std::uint64_t bytes_before_rollback =
            context->buffers().allocated_size();
        std::unique_ptr<WindowScratchStorage> provisional;
        BufferRef provisional_view;
        {
            auto outer = context->buffers().transaction();
            provisional = WindowScratchStorage::open(
                *context, geometry, 65'536u, error);
            require(provisional != nullptr && provisional->is_valid(),
                    "open provisional Window scratch storage: " + error);
            require(provisional->view(provisional->layout().head_logits,
                                      provisional_view, error),
                    "get provisional scratch view: " + error);
        }
        require(provisional != nullptr && !provisional->is_valid() &&
                    !provisional_view &&
                    context->buffers().allocation_count() ==
                        allocations_before_rollback &&
                    context->buffers().allocated_size() ==
                        bytes_before_rollback,
                "outer rollback must invalidate owner and every published view");

        ContextOptions limited_options;
        limited_options.pipelines = false;
        limited_options.max_buffer_length_cap =
            expected_layout.allocation_bytes - 1u;
        auto limited_context = Context::open(limited_options, error);
        require(limited_context != nullptr,
                "open limited-buffer Metal Context: " + error);
        require(WindowScratchStorage::open(
                    *limited_context, geometry, 65'536u, error) == nullptr &&
                    error.find("maxBufferLength") != std::string::npos &&
                    limited_context->buffers().allocation_count() == 0u &&
                    limited_context->buffers().allocated_size() == 0u,
                "Window scratch must honor maxBufferLength before allocating");

        std::cout << "metal_window_scratch_storage_test: all checks passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_window_scratch_storage_test: " << exception.what()
                  << '\n';
        return 1;
    }
}
