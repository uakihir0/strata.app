// Context-owned Shared storage for the fixed-capacity Metal Window scratch.
#include "strata/metal/window_scratch.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <utility>

namespace strata::metal {
namespace {

constexpr std::size_t kRegionAlignment = 256u;

bool is_named_region(const WindowScratchLayout& layout,
                     WindowScratchRegion requested) {
    for (const WindowScratchRegion region : {
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
             layout.head_mixed,
             layout.head_logits,
             layout.sampler_history,
             layout.sampler_candidates,
             layout.sampler_selection,
             layout.sampler_output,
             layout.matvec_sink,
         }) {
        if (region.offset == requested.offset &&
            region.bytes == requested.bytes) {
            return true;
        }
    }
    return false;
}

}  // namespace

struct WindowScratchStorage::Impl {
    Buffers* buffers = nullptr;
    WindowScratchLayout layout;
    BufferRef allocation;
    std::uint64_t allocation_identity = 0u;
};

WindowScratchStorage::WindowScratchStorage(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

WindowScratchStorage::~WindowScratchStorage() = default;

std::unique_ptr<WindowScratchStorage> WindowScratchStorage::open(
    Context& context, const Geometry& geometry, std::uint32_t cell_capacity,
    std::string& error) {
    error.clear();
    auto candidate = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!candidate) {
        error = "cannot allocate Window scratch owner state";
        return nullptr;
    }
    if (!make_window_scratch_layout(geometry, cell_capacity,
                                    candidate->layout, error)) {
        return nullptr;
    }

    const DeviceFacts& facts = context.device_facts();
    if (facts.max_buffer_length == 0u ||
        candidate->layout.allocation_bytes > facts.max_buffer_length) {
        error = "Window scratch exceeds the Metal device's maxBufferLength";
        return nullptr;
    }

    Buffers& buffers = context.buffers();
    auto transaction = buffers.transaction();
    candidate->allocation = buffers.allocate(
        "T=8 Window scratch", candidate->layout.allocation_bytes, error);
    if (!candidate->allocation) return nullptr;
    candidate->allocation_identity =
        buffers.allocation_identity(candidate->allocation);
    if (candidate->allocation_identity == 0u ||
        !buffers.owns(candidate->allocation, candidate->allocation_identity) ||
        candidate->allocation.length != candidate->layout.allocation_bytes ||
        candidate->allocation.contents == nullptr) {
        error = "Metal did not return a valid owned Window scratch allocation";
        return nullptr;
    }
    if (!buffers.commit_residency(error)) return nullptr;

    candidate->buffers = &buffers;
    auto result = std::unique_ptr<WindowScratchStorage>(
        new (std::nothrow) WindowScratchStorage(std::move(candidate)));
    if (!result) {
        error = "cannot allocate Window scratch owner";
        return nullptr;
    }
    if (!result->is_valid()) {
        error = "Window scratch allocation became invalid before publication";
        return nullptr;
    }
    transaction.commit();
    error.clear();
    return result;
}

bool WindowScratchStorage::is_valid() const noexcept {
    return impl_ != nullptr && impl_->buffers != nullptr &&
           impl_->allocation_identity != 0u &&
           impl_->layout.allocation_bytes != 0u &&
           impl_->allocation.length == impl_->layout.allocation_bytes &&
           impl_->allocation.contents != nullptr &&
           impl_->buffers->owns(impl_->allocation,
                                impl_->allocation_identity);
}

const WindowScratchLayout& WindowScratchStorage::layout() const noexcept {
    return impl_->layout;
}

bool WindowScratchStorage::view(WindowScratchRegion region, BufferRef& result,
                                std::string& error) const {
    error.clear();
    result = {};
    if (!is_valid()) {
        error = "Window scratch allocation is unavailable";
        return false;
    }
    if (!is_named_region(impl_->layout, region) ||
        region.offset % kRegionAlignment != 0u ||
        region.bytes == 0u ||
        region.offset > impl_->allocation.length ||
        region.bytes > impl_->allocation.length - region.offset) {
        error = "Window scratch region is not a valid named allocation range";
        return false;
    }

    result = impl_->allocation.slice(region.offset, region.bytes);
    if (!result ||
        !impl_->buffers->owns(result, impl_->allocation_identity)) {
        result = {};
        error = "Window scratch region could not be represented as a live bounded view";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace strata::metal
