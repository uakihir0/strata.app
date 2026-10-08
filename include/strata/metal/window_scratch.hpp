// include/strata/metal/window_scratch.hpp - checked Metal Window scratch layout.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/geometry.hpp"
#include "strata/metal/session_state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::metal {

inline constexpr std::uint32_t kWindowScratchTokenRows = 8u;
inline constexpr std::uint32_t kWindowScratchExpertsPerRow = 10u;
inline constexpr std::uint32_t kWindowScratchFeedForwardWidth = 640u;
inline constexpr std::uint32_t kWindowScratchMaxCellCapacity = 262'144u;

struct WindowScratchRegion {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

/// Exact T=8 scratch geometry for the supported qwen4exp Window kernels.
/// Regions are disjoint and each begins at a 256-byte boundary.
struct WindowScratchLayout {
    std::uint32_t cell_capacity = 0;
    std::uint32_t max_blocks = 0;
    std::uint32_t n_expert = 0;
    std::size_t attention_scratch_floats_per_row = 0;
    std::size_t allocation_bytes = 0;

    WindowScratchRegion ple_input;
    WindowScratchRegion embedding;
    WindowScratchRegion residual;
    WindowScratchRegion mixed;
    WindowScratchRegion block_output;
    WindowScratchRegion inject;
    WindowScratchRegion inject2;
    WindowScratchRegion gr_lo;
    WindowScratchRegion gr_rs;
    WindowScratchRegion gr_xn;
    WindowScratchRegion xq;

    WindowScratchRegion gdn_qkv_replay;
    WindowScratchRegion gdn_h_replay;
    WindowScratchRegion gdn_gate_replay;
    WindowScratchRegion gdn_beta_replay;
    WindowScratchRegion gdn_z;
    WindowScratchRegion gdn_y;

    WindowScratchRegion ple_key;
    WindowScratchRegion ple_value;
    WindowScratchRegion ple_query;
    WindowScratchRegion ple_gate;
    WindowScratchRegion ple_gated;
    WindowScratchRegion ple_norm;

    WindowScratchRegion qsa_q_full;
    WindowScratchRegion qsa_q;
    WindowScratchRegion qsa_k;
    WindowScratchRegion qsa_v;
    WindowScratchRegion qsa_indexer_key_replay;
    WindowScratchRegion qsa_indexer_query;
    WindowScratchRegion qsa_scores;
    WindowScratchRegion qsa_ids;
    WindowScratchRegion qsa_attention;
    WindowScratchRegion qsa_attention_gated;
    WindowScratchRegion qsa_attention_scratch;
    WindowScratchRegion qsa_tail_snapshot;
    WindowScratchRegion qsa_block_position_snapshot;

    WindowScratchRegion router_logits;
    WindowScratchRegion router_ids;
    WindowScratchRegion router_weights;
    WindowScratchRegion moe_plans;
    WindowScratchRegion moe_gate;
    WindowScratchRegion moe_up;
    WindowScratchRegion moe_hidden;
    WindowScratchRegion moe_hidden_q;
    WindowScratchRegion moe_parts;
    WindowScratchRegion shared_gate;
    WindowScratchRegion shared_up;
    WindowScratchRegion shared_hidden;
    WindowScratchRegion shared_hidden_q;
    WindowScratchRegion shared_output;
    WindowScratchRegion shared_gate_logit;

    WindowScratchRegion head_mixed;
    WindowScratchRegion head_q8_1;
    WindowScratchRegion head_logits;
    WindowScratchRegion sampler_history;
    WindowScratchRegion sampler_candidates;
    WindowScratchRegion sampler_selection;
    WindowScratchRegion sampler_output;
    WindowScratchRegion matvec_sink;
};

/// Calculate the fixed-capacity T=8 Window scratch layout without a device or
/// allocation. `cell_capacity` is the rounded QSA/KV capacity in cells.
bool make_window_scratch_layout(const Geometry& geometry,
                                std::uint32_t cell_capacity,
                                WindowScratchLayout& layout,
                                std::string& error);

/// Own the fixed-capacity T=8 Window scratch allocation in a Context.
///
/// Context must outlive this owner and every command buffer that uses its
/// views. `view()` accepts only regions from the published layout and returns
/// a generation-checked bounded BufferRef. STRATA_METAL_POISON=1 is honored
/// by the shared allocator for unwritten-scratch diagnostics.
class WindowScratchStorage {
public:
    static std::unique_ptr<WindowScratchStorage> open(
        Context& context, const Geometry& geometry, std::uint32_t cell_capacity,
        std::string& error);
    ~WindowScratchStorage();
    WindowScratchStorage(const WindowScratchStorage&) = delete;
    WindowScratchStorage& operator=(const WindowScratchStorage&) = delete;

    bool is_valid() const noexcept;
    const WindowScratchLayout& layout() const noexcept;
    bool view(WindowScratchRegion region, BufferRef& result,
              std::string& error) const;

private:
    struct Impl;
    explicit WindowScratchStorage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
