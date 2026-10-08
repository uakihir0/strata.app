// src/metal/engine/model_storage.cpp - select, validate and own model weight storage.
#include "strata/metal/model_storage.hpp"

#include "strata/metal/kv.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/mtp_runtime_storage.hpp"
#include "strata/metal/session_state.hpp"
#include "strata/metal/window_plan.hpp"
#include "strata/metal/window_scratch.hpp"

#include <limits>
#include <vector>

namespace strata::metal {
namespace {

bool validate_resident_order(const Geometry& geometry, const PlanResult& memory_plan,
                             bool require_all, std::string& error) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(geometry.n_layers) * geometry.n_expert;
    if (geometry.n_layers == 0 || geometry.n_expert == 0 ||
        memory_plan.resident_order.size() != memory_plan.experts_resident ||
        memory_plan.experts_resident > total ||
        (require_all && memory_plan.experts_resident != total)) {
        error = "memory plan resident count does not match its storage mode";
        return false;
    }

    std::vector<bool> seen(static_cast<std::size_t>(total), false);
    for (const auto& [layer, expert] : memory_plan.resident_order) {
        if (layer >= geometry.n_layers || expert >= geometry.n_expert) {
            error = "memory plan resident order contains an out-of-range expert";
            return false;
        }
        const std::size_t index =
            static_cast<std::size_t>(layer) * geometry.n_expert + expert;
        if (seen[index]) {
            error = "memory plan resident order contains a duplicate expert";
            return false;
        }
        seen[index] = true;
    }
    error.clear();
    return true;
}

bool add_bytes(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool multiply_bytes(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool validate_plan_geometry(const Geometry& geometry, const PlanResult& memory_plan,
                            bool full_mode, std::string& error) {
    const std::uint64_t total_experts =
        static_cast<std::uint64_t>(geometry.n_layers) * geometry.n_expert;
    std::uint64_t expert_bytes = 0;
    for (const LayerGeometry& layer : geometry.layers) {
        std::uint64_t layer_bytes = 0;
        if (!multiply_bytes(layer.blob_bytes, geometry.n_expert, layer_bytes) ||
            !add_bytes(expert_bytes, layer_bytes, expert_bytes)) {
            error = "model geometry expert byte count overflows";
            return false;
        }
    }

    std::uint64_t resident_bytes = 0;
    std::vector<std::uint64_t> resident_counts(geometry.n_layers, 0);
    for (const auto& [layer, expert] : memory_plan.resident_order) {
        (void)expert;
        ++resident_counts[layer];
    }
    for (std::size_t layer = 0; layer < resident_counts.size(); ++layer) {
        std::uint64_t layer_bytes = 0;
        if (!multiply_bytes(geometry.layers[layer].blob_bytes, resident_counts[layer],
                            layer_bytes) ||
            !add_bytes(resident_bytes, layer_bytes, resident_bytes)) {
            error = "model geometry resident byte count overflows";
            return false;
        }
    }
    const std::uint64_t expected_unread =
        full_mode ? geometry.wrapped_unread_bytes : 0;
    if (memory_plan.experts_total != total_experts ||
        memory_plan.experts_total_bytes != expert_bytes ||
        memory_plan.experts_resident_bytes != resident_bytes ||
        memory_plan.dense_bytes != geometry.nonexpert_bytes ||
        memory_plan.pack_bytes != geometry.dense_pack_bytes ||
        memory_plan.unread_bytes != expected_unread ||
        (full_mode && (memory_plan.slots != 0 || memory_plan.slot_bytes != 0))) {
        error = "memory plan byte counts do not match the inspected model geometry";
        return false;
    }

    std::uint64_t exact_fixed_subtotal = 0;
    if ((memory_plan.session_state_bytes == 0) !=
            (memory_plan.window_scratch_bytes == 0) ||
        !add_bytes(memory_plan.session_state_bytes,
                   memory_plan.window_scratch_bytes, exact_fixed_subtotal) ||
        exact_fixed_subtotal > memory_plan.fixed_bytes) {
        error = "exact session and Window scratch bytes exceed the fixed-memory allowance";
        return false;
    }

    std::uint64_t expected_total = 0;
    const std::uint64_t slot_bytes = full_mode ? 0 : memory_plan.slot_bytes;
    for (const std::uint64_t component : {
             memory_plan.dense_bytes, memory_plan.unread_bytes, memory_plan.pack_bytes,
             memory_plan.mtp_bytes, memory_plan.mtp_bind_bytes,
             memory_plan.mtp_params_bytes, memory_plan.fixed_bytes, memory_plan.kv_bytes,
             memory_plan.mtp_kv_bytes, memory_plan.prefill_bytes, slot_bytes,
             memory_plan.experts_resident_bytes}) {
        if (!add_bytes(expected_total, component, expected_total)) {
            error = "memory plan total byte count overflows";
            return false;
        }
    }
    const std::uint64_t budget = memory_plan.w_bytes > memory_plan.reserve_bytes
        ? memory_plan.w_bytes - memory_plan.reserve_bytes : 0;
    if (memory_plan.total_bytes != expected_total || expected_total > budget) {
        error = "memory plan total does not fit its reported working-set budget";
        return false;
    }
    if (full_mode && geometry.dense_pack_bytes != 0) {
        const std::uint64_t remainder =
            geometry.dense_pack_bytes % kWindowPageBytes;
        const std::uint64_t padding =
            remainder == 0 ? 0 : kWindowPageBytes - remainder;
        const std::uint64_t free_budget = budget - expected_total;
        if (padding > free_budget) {
            error = "full-mode dense-pack arena page padding exceeds the memory plan's free bytes";
            return false;
        }
    }
    error.clear();
    return true;
}

}  // namespace

ModelStorage::~ModelStorage() = default;

std::unique_ptr<ModelStorage> ModelStorage::open(Context& context,
                                                  const std::string& native_shard,
                                                 const Geometry& geometry,
                                                 const WindowSources& sources,
                                                 const PlanResult& memory_plan,
                                                 std::string& error,
                                                 const MtpRuntimeSources* mtp_sources) {
    error.clear();
    if (native_shard.empty() || !memory_plan.fits) {
        error = "model storage needs a GGUF shard and a fitting memory plan";
        return nullptr;
    }
    const bool plan_has_mtp =
        memory_plan.mtp_bytes != 0 || memory_plan.mtp_bind_bytes != 0 ||
        memory_plan.mtp_bind_capacity != 0 || memory_plan.mtp_params_bytes != 0 ||
        memory_plan.mtp_kv_bytes != 0;
    if (plan_has_mtp) {
        if (mtp_sources == nullptr || mtp_sources->catalog == nullptr) {
            error = "speculative model storage needs inspected MTP runtime sources";
            return nullptr;
        }
        if (!validate_mtp_memory_plan(geometry, memory_plan, error)) return nullptr;
    }

    const bool full_mode = memory_plan.mode == "full";
    if (!full_mode && memory_plan.mode != "tiered") {
        error = "memory plan selected an unknown model storage mode";
        return nullptr;
    }
    if (!validate_geometry(geometry, error)) return nullptr;
    if (!validate_resident_order(geometry, memory_plan, full_mode, error)) return nullptr;
    if (!validate_plan_geometry(geometry, memory_plan, full_mode, error)) return nullptr;

    const DeviceFacts facts = context.device_facts();
    QsaKvLayout kv_layout;
    if (!validate_main_qsa_kv_plan(memory_plan, facts.max_buffer_length,
                                   kv_layout, error)) return nullptr;
    WindowScratchLayout scratch_layout;
    bool open_window_scratch = false;
    if (is_qwen4exp_catalog_family(geometry.family)) {
        SessionStateLayout session_layout;
        if (!is_qwen4exp_runtime_geometry(geometry) ||
            !make_session_state_layout(geometry, session_layout, error) ||
            !make_window_scratch_layout(geometry, kv_layout.cell_capacity,
                                        scratch_layout, error)) {
            if (error.empty()) {
                error = "model geometry does not match the supported qwen4exp runtime";
            }
            return nullptr;
        }
        if (memory_plan.session_state_bytes != session_layout.allocation_bytes ||
            memory_plan.window_scratch_bytes != scratch_layout.allocation_bytes) {
            error = "exact session and Window scratch bytes do not match the memory plan";
            return nullptr;
        }
        std::uint64_t exact_fixed_subtotal = 0;
        if (!add_bytes(session_layout.allocation_bytes,
                       scratch_layout.allocation_bytes, exact_fixed_subtotal) ||
            exact_fixed_subtotal > memory_plan.fixed_bytes) {
            error = "exact session and Window scratch bytes exceed the fixed-memory allowance";
            return nullptr;
        }
        if (facts.max_buffer_length == 0 ||
            scratch_layout.allocation_bytes > facts.max_buffer_length) {
            error = "Window scratch exceeds the Metal device's maxBufferLength";
            return nullptr;
        }
        open_window_scratch = true;
    } else if (memory_plan.session_state_bytes != 0 ||
               memory_plan.window_scratch_bytes != 0) {
        error = "fixed runtime-state plan fields require a supported qwen4exp family";
        return nullptr;
    }
    if (facts.working_set_bytes == 0 || memory_plan.w_bytes == 0 ||
        memory_plan.w_bytes > facts.working_set_bytes ||
        (facts.physical_memory_bytes != 0 &&
         memory_plan.ram_bytes > facts.physical_memory_bytes) ||
        (memory_plan.available_bytes > 0 && facts.physical_memory_bytes != 0 &&
         static_cast<std::uint64_t>(memory_plan.available_bytes) > facts.physical_memory_bytes)) {
        error = "memory plan exceeds the live Metal device or physical-memory facts";
        return nullptr;
    }

    auto candidate = std::unique_ptr<ModelStorage>(new ModelStorage());
    candidate->resident_experts_ = memory_plan.experts_resident;
    auto buffer_transaction = context.buffers().transaction();
    candidate->kv_store_ =
        KvStore::open(context, memory_plan.context, memory_plan.kv, error);
    if (!candidate->kv_store_) return nullptr;
    if (plan_has_mtp) {
        candidate->mtp_runtime_ =
            MtpRuntimeStorage::open_with_main_kv(
                context, *candidate->kv_store_, geometry, *mtp_sources,
                memory_plan, error);
        if (!candidate->mtp_runtime_) return nullptr;
    }
    if (open_window_scratch) {
        candidate->window_scratch_ =
            WindowScratchStorage::open(context, geometry,
                                       kv_layout.cell_capacity, error);
        if (!candidate->window_scratch_) return nullptr;
    }
    if (full_mode) {
        WindowPlan window_plan;
        if (facts.max_buffer_length == 0 ||
            !plan_windows(sources, facts.max_buffer_length, window_plan, error)) {
            if (error.empty()) error = "Metal device reported an invalid maxBufferLength";
            return nullptr;
        }
        for (const FileWindowPlan& file : window_plan.files) {
            if (file.windows.size() >
                std::numeric_limits<std::size_t>::max() - candidate->window_count_) {
                error = "full-mode window count overflows";
                return nullptr;
            }
            candidate->window_count_ += file.windows.size();
        }
        candidate->full_ = FullModeStorage::open(context, native_shard, geometry, sources,
                                                  window_plan, error);
        if (!candidate->full_) return nullptr;
        if (!candidate->full_->warm_up(memory_plan.resident_order, error)) return nullptr;
        candidate->full_mode_warmed_ = true;
        buffer_transaction.commit();
        return candidate;
    }

    candidate->tiered_ = TieredStorage::open(context, native_shard, geometry, sources,
                                              memory_plan, error);
    if (!candidate->tiered_) return nullptr;
    buffer_transaction.commit();
    return candidate;
}

const char* ModelStorage::mode() const noexcept {
    if (full_) return "full";
    if (tiered_) return "tiered";
    return "unopened";
}

std::uint64_t ModelStorage::resident_experts() const noexcept {
    return resident_experts_;
}

std::size_t ModelStorage::window_count() const noexcept {
    return window_count_;
}

bool ModelStorage::full_mode_warmed() const noexcept {
    return full_mode_warmed_;
}

const KvStore* ModelStorage::kv_store() const noexcept {
    return kv_store_.get();
}

const MtpRuntimeStorage* ModelStorage::mtp_runtime() const noexcept {
    return mtp_runtime_.get();
}

const WindowScratchStorage* ModelStorage::window_scratch() const noexcept {
    return window_scratch_.get();
}

const FullModeStorage* ModelStorage::full_mode_storage() const noexcept {
    return full_.get();
}

strata::kernels::PleTableReader* ModelStorage::ple_reader() noexcept {
    return full_ != nullptr ? full_->ple_reader() : nullptr;
}

}  // namespace strata::metal
