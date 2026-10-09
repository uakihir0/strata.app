// src/metal/engine/mtp_runtime_storage.cpp - all planned MTP allocations under one transaction.
#include "strata/metal/mtp_runtime_storage.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/kv.hpp"

#include <cstring>
#include <limits>
#include <utility>

#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

namespace strata::metal {

struct MtpRuntimeStorage::Impl {
    Buffers* buffers = nullptr;
    std::unique_ptr<MtpStorage> weights;
    std::unique_ptr<MtpKvStore> kv;
    MtpBindLayout bind_layout;
    BufferRef bind_scratch;
    std::uint64_t bind_scratch_identity = 0;
    MtpParamsLayout params_layout;
    BufferRef params;
    std::uint64_t params_identity = 0;
    BufferRef shared_rope;
    bool shared_rope_bound = false;
};

MtpRuntimeStorage::MtpRuntimeStorage(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MtpRuntimeStorage::~MtpRuntimeStorage() = default;

std::unique_ptr<MtpRuntimeStorage> MtpRuntimeStorage::open(
    Context& context, const Geometry& geometry,
    const MtpRuntimeSources& sources,
    const PlanResult& memory_plan, std::string& error) {
    return open_impl(context, geometry, sources, memory_plan, nullptr, false, error);
}

std::unique_ptr<MtpRuntimeStorage> MtpRuntimeStorage::open_with_main_kv(
    Context& context, const KvStore& main_kv,
    const Geometry& geometry, const MtpRuntimeSources& sources,
    const PlanResult& memory_plan, std::string& error) {
    return open_impl(context, geometry, sources, memory_plan, &main_kv, false, error);
}

#if defined(STRATA_METAL_TEST_LOCK_ACCESS)
std::unique_ptr<MtpRuntimeStorage> MtpRuntimeStorage::open_for_test(
    Context& context, const Geometry& geometry,
    const MtpRuntimeSources& sources,
    const PlanResult& memory_plan, bool fail_after_weights,
    std::string& error) {
    return open_impl(context, geometry, sources, memory_plan, nullptr,
                     fail_after_weights, error);
}
#endif

std::unique_ptr<MtpRuntimeStorage> MtpRuntimeStorage::open_impl(
    Context& context, const Geometry& geometry,
    const MtpRuntimeSources& sources,
    const PlanResult& memory_plan, const KvStore* main_kv,
    bool fail_after_weights,
    std::string& error) {
    error.clear();
    if (!validate_mtp_memory_plan(geometry, memory_plan, error)) return nullptr;
    if (memory_plan.context == 0) {
        error = "MTP runtime storage needs a positive context capacity";
        return nullptr;
    }

    const DeviceFacts facts = context.device_facts();
    if (facts.max_buffer_length == 0 ||
        memory_plan.mtp_bind_bytes > facts.max_buffer_length ||
        memory_plan.mtp_bind_bytes > std::numeric_limits<std::size_t>::max() ||
        memory_plan.mtp_params_bytes > facts.max_buffer_length ||
        memory_plan.mtp_params_bytes > std::numeric_limits<std::size_t>::max()) {
        error = "MTP bind scratch or parameter blocks exceed Metal's buffer or host size limit";
        return nullptr;
    }
    BufferRef shared_rope;
    bool shared_rope_bound = false;
    if (main_kv != nullptr) {
        QsaKvLayout expected_main_layout;
        if (!validate_main_qsa_kv_plan(memory_plan, facts.max_buffer_length,
                                       expected_main_layout, error)) {
            return nullptr;
        }
        const QsaKvLayout& actual_main_layout = main_kv->layout();
        const BufferRef main_allocation = main_kv->allocation();
        shared_rope = main_kv->rope();
        if (actual_main_layout.requested_cells != memory_plan.context ||
            actual_main_layout.format != memory_plan.kv ||
            actual_main_layout.allocation_bytes != expected_main_layout.allocation_bytes ||
            !main_allocation || !context.buffers().owns(main_allocation) ||
            main_allocation.length != expected_main_layout.allocation_bytes ||
            main_allocation.contents == nullptr ||
            !shared_rope || !context.buffers().owns(shared_rope)) {
            error = "MTP runtime storage needs the matching main QSA KV and RoPE store";
            return nullptr;
        }
        std::uint64_t expected_rope_gpu_address = 0;
        const std::uint64_t main_allocation_identity =
            context.buffers().allocation_identity(main_allocation);
        if (!detail::gpu_address_with_offset(
                main_allocation.gpu_address, expected_main_layout.rope.offset,
                expected_rope_gpu_address) ||
            shared_rope.handle != main_allocation.handle ||
            shared_rope.length != expected_main_layout.rope.bytes ||
            shared_rope.offset != expected_main_layout.rope.offset ||
            shared_rope.gpu_address != expected_rope_gpu_address ||
            shared_rope.contents !=
                static_cast<std::byte*>(main_allocation.contents) +
                    expected_main_layout.rope.offset ||
            main_allocation_identity == 0 ||
            context.buffers().allocation_identity(shared_rope) !=
                main_allocation_identity) {
            error = "MTP runtime storage needs the matching main QSA KV and RoPE store";
            return nullptr;
        }
        shared_rope_bound = true;
    }
    MtpParamsLayout params_layout;
    if (!make_mtp_params_layout(memory_plan.prefill_chunk, params_layout, error) ||
        params_layout.allocation_bytes != memory_plan.mtp_params_bytes) {
        if (error.empty()) error = "MTP parameter layout does not match the memory plan";
        return nullptr;
    }
    if (sources.catalog == nullptr) {
        error = "MTP runtime storage needs an inspected MTP catalog";
        return nullptr;
    }

    MtpRuntimeLayout planned_layout;
    if (!build_mtp_runtime_layout(sources, facts.max_buffer_length,
                                  planned_layout, error)) {
        return nullptr;
    }
    if (planned_layout.mapped_file_bytes != memory_plan.mtp_bytes) {
        error = "MTP runtime file bytes do not match the memory plan";
        return nullptr;
    }
    const auto& catalog = *sources.catalog;
    if (catalog.draft_vocab_ids.size() >
            std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t) ||
        catalog.draft_vocab_ids.size() > kMtpFullVocabularySize ||
        catalog.draft_vocab_file_size !=
            static_cast<std::uint64_t>(catalog.draft_vocab_ids.size()) *
                sizeof(std::uint32_t) ||
        catalog.draft_vocab_file_size > memory_plan.mtp_bind_bytes) {
        error = "MTP draft vocabulary does not fit the planned bind arena";
        return nullptr;
    }
    MtpBindLayout baseline_bind_layout;
    if (!make_mtp_bind_layout(
            kMtpDefaultBindCapacity,
            static_cast<std::uint32_t>(catalog.draft_vocab_ids.size()),
            baseline_bind_layout, error)) {
        error = "MTP bind geometry cannot be derived: " + error;
        return nullptr;
    }
    if (geometry.mtp_bind_bytes != baseline_bind_layout.allocation_bytes) {
        error = "MTP bind geometry does not match its vocabulary and arena layout";
        return nullptr;
    }
    MtpBindLayout planned_bind_layout;
    if (!make_mtp_bind_layout(
            memory_plan.mtp_bind_capacity,
            static_cast<std::uint32_t>(catalog.draft_vocab_ids.size()),
            planned_bind_layout, error)) {
        error = "MTP bind plan cannot be carved: " + error;
        return nullptr;
    }
    if (planned_bind_layout.allocation_bytes != memory_plan.mtp_bind_bytes) {
        error = "MTP bind layout does not match the planned arena byte count";
        return nullptr;
    }

    auto transaction = context.buffers().transaction();
    auto candidate = std::make_unique<Impl>();
    candidate->buffers = &context.buffers();
    candidate->shared_rope = shared_rope;
    candidate->shared_rope_bound = shared_rope_bound;
    candidate->bind_layout = std::move(planned_bind_layout);
    candidate->params_layout = params_layout;
    candidate->weights = MtpStorage::open(context, sources, error);
    if (!candidate->weights) return nullptr;
    if (fail_after_weights) {
        error = "injected MTP runtime failure after weights open";
        return nullptr;
    }
    candidate->kv = MtpKvStore::open(context, memory_plan.context,
                                     memory_plan.kv, error);
    if (!candidate->kv) return nullptr;
    candidate->bind_scratch = candidate->buffers->allocate(
        "MTP bind scratch", static_cast<std::size_t>(memory_plan.mtp_bind_bytes), error);
    if (!candidate->bind_scratch) return nullptr;
    candidate->bind_scratch_identity =
        candidate->buffers->allocation_identity(candidate->bind_scratch);
    if (candidate->bind_scratch_identity == 0 ||
        !candidate->buffers->owns(candidate->bind_scratch,
                                  candidate->bind_scratch_identity) ||
        candidate->bind_scratch.length != memory_plan.mtp_bind_bytes ||
        candidate->bind_scratch.contents == nullptr) {
        error = "Metal did not return a valid owned MTP bind arena";
        return nullptr;
    }
    const MtpBindRegion ids_region =
        candidate->bind_layout.region(MtpBindRegionId::mtp_ids);
    const std::uint64_t expected_ids_bytes =
        static_cast<std::uint64_t>(memory_plan.mtp_bind_capacity) *
        sizeof(std::int32_t);
    if (ids_region.bytes != expected_ids_bytes ||
        ids_region.offset > std::numeric_limits<std::size_t>::max() ||
        ids_region.bytes > std::numeric_limits<std::size_t>::max()) {
        error = "MTP window ID region does not match the planned bind capacity";
        return nullptr;
    }
    BufferRef window_ids = candidate->bind_scratch.slice(
        static_cast<std::size_t>(ids_region.offset),
        static_cast<std::size_t>(ids_region.bytes));
    if (!window_ids || window_ids.length != expected_ids_bytes ||
        window_ids.contents == nullptr ||
        !candidate->buffers->owns(window_ids,
                                  candidate->bind_scratch_identity)) {
        error = "MTP window ID region is not a valid owned subview";
        return nullptr;
    }
    auto* initial_ids = static_cast<std::int32_t*>(window_ids.contents);
    for (std::uint32_t index = 0; index < memory_plan.mtp_bind_capacity; ++index) {
        initial_ids[index] = static_cast<std::int32_t>(index);
    }
    const MtpBindRegion draft_vocab_region =
        candidate->bind_layout.region(MtpBindRegionId::draft_vocab_ids);
    if (draft_vocab_region.bytes != catalog.draft_vocab_file_size ||
        draft_vocab_region.offset > std::numeric_limits<std::size_t>::max() ||
        draft_vocab_region.bytes > std::numeric_limits<std::size_t>::max()) {
        error = "MTP device vocabulary region does not match the inspected IDs";
        return nullptr;
    }
    if (draft_vocab_region.bytes != 0) {
        BufferRef device_vocab_ids = candidate->bind_scratch.slice(
            static_cast<std::size_t>(draft_vocab_region.offset),
            static_cast<std::size_t>(draft_vocab_region.bytes));
        if (!device_vocab_ids ||
            device_vocab_ids.length != catalog.draft_vocab_file_size ||
            device_vocab_ids.contents == nullptr ||
            !candidate->buffers->owns(device_vocab_ids,
                                      candidate->bind_scratch_identity)) {
            error = "MTP device vocabulary region is not a valid owned subview";
            return nullptr;
        }
        std::memcpy(device_vocab_ids.contents, catalog.draft_vocab_ids.data(),
                    static_cast<std::size_t>(draft_vocab_region.bytes));
    }
    candidate->params = candidate->buffers->allocate(
        "MTP parameter blocks",
        static_cast<std::size_t>(candidate->params_layout.allocation_bytes), error);
    if (!candidate->params) return nullptr;
    candidate->params_identity =
        candidate->buffers->allocation_identity(candidate->params);
    if (candidate->params_identity == 0 ||
        !candidate->buffers->owns(candidate->params, candidate->params_identity) ||
        candidate->params.length != candidate->params_layout.allocation_bytes ||
        candidate->params.contents == nullptr) {
        error = "Metal did not return a valid owned MTP parameter array";
        return nullptr;
    }
    std::memset(candidate->params.contents, 0, candidate->params.length);
    if (!candidate->buffers->commit_residency(error)) return nullptr;

    auto result = std::unique_ptr<MtpRuntimeStorage>(
        new MtpRuntimeStorage(std::move(candidate)));
    if (!result->is_valid()) {
        error = "MTP runtime allocations became invalid before publication";
        return nullptr;
    }
    transaction.commit();
    error.clear();
    return result;
}

bool MtpRuntimeStorage::is_valid() const noexcept {
    if (impl_ == nullptr || impl_->buffers == nullptr || impl_->weights == nullptr ||
        impl_->kv == nullptr ||
        !impl_->buffers->owns(impl_->bind_scratch, impl_->bind_scratch_identity) ||
        !impl_->buffers->owns(impl_->params, impl_->params_identity) ||
        impl_->params.length != impl_->params_layout.allocation_bytes ||
        (impl_->shared_rope_bound &&
         !impl_->buffers->owns(impl_->shared_rope)) ||
        !impl_->kv->allocation()) {
        return false;
    }
    const MtpRuntimeLayout& runtime_layout = impl_->weights->layout();
    if (runtime_layout.expert_windows.empty() || runtime_layout.dense_windows.empty()) {
        return false;
    }
    for (std::size_t index = 0; index < runtime_layout.expert_windows.size(); ++index) {
        if (!impl_->weights->expert_window(index)) return false;
    }
    for (std::size_t index = 0; index < runtime_layout.dense_windows.size(); ++index) {
        if (!impl_->weights->dense_window(index)) return false;
    }
    return static_cast<bool>(impl_->kv->allocation());
}

const MtpRuntimeLayout& MtpRuntimeStorage::layout() const noexcept {
    return impl_->weights->layout();
}

const MtpKvLayout& MtpRuntimeStorage::kv_layout() const noexcept {
    return impl_->kv->layout();
}

const MtpBindLayout& MtpRuntimeStorage::bind_layout() const noexcept {
    return impl_->bind_layout;
}

const MtpParamsLayout& MtpRuntimeStorage::params_layout() const noexcept {
    return impl_->params_layout;
}

BufferRef MtpRuntimeStorage::mtp_params_slot(std::uint32_t slot) const noexcept {
    if (impl_ == nullptr || impl_->buffers == nullptr ||
        !impl_->buffers->owns(impl_->params, impl_->params_identity)) {
        return {};
    }
    std::uint64_t offset = 0;
    if (!impl_->params_layout.slot_offset(slot, offset) ||
        offset > std::numeric_limits<std::size_t>::max()) {
        return {};
    }
    BufferRef view = impl_->params.slice(static_cast<std::size_t>(offset),
                                         kMtpParamsSlotBytes);
    if (!view || view.length != kMtpParamsSlotBytes ||
        !impl_->buffers->owns(view, impl_->params_identity)) {
        return {};
    }
    return view;
}

const std::vector<std::uint32_t>& MtpRuntimeStorage::draft_vocab_ids() const noexcept {
    return impl_->weights->draft_vocab_ids();
}

BufferRef MtpRuntimeStorage::expert(std::uint32_t expert_index) const noexcept {
    return impl_->weights->expert(expert_index);
}

BufferRef MtpRuntimeStorage::dense_weight(const std::string& tensor_name) const noexcept {
    return impl_->weights->dense_weight(tensor_name);
}

BufferRef MtpRuntimeStorage::bind_region(MtpBindRegionId id) const noexcept {
    if (impl_ == nullptr || impl_->buffers == nullptr ||
        !impl_->buffers->owns(impl_->bind_scratch,
                              impl_->bind_scratch_identity)) {
        return {};
    }
    const MtpBindRegion region = impl_->bind_layout.region(id);
    if (region.bytes == 0 ||
        region.offset > std::numeric_limits<std::size_t>::max() ||
        region.bytes > std::numeric_limits<std::size_t>::max()) {
        return {};
    }
    return impl_->bind_scratch.slice(static_cast<std::size_t>(region.offset),
                                     static_cast<std::size_t>(region.bytes));
}

bool MtpRuntimeStorage::bind_input_projection(
    Context& context, const MtpInputProjectionSources& sources,
    MtpInputProjectionBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP input projection needs live runtime storage from this Context";
        return false;
    }
    if (!context.buffers().owns(sources.token_embedding_table) ||
        !sources.residual || !context.buffers().owns(sources.residual)) {
        error = "MTP input projection sources must belong to the runtime Context";
        return false;
    }

    const BufferRef params_slot = mtp_params_slot(sources.params_slot);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP input projection parameter slot is unavailable";
        return false;
    }
    MtpParams params{};
    std::memcpy(&params, params_slot.contents, sizeof(params));
    if (params.T == 0u || params.T > 8u) {
        error = "MTP input projection parameter slot T must be in 1..8";
        return false;
    }

    switch (sources.token_source) {
        case MtpInputTokenSource::params_tokens:
            if (sources.selected_token) {
                error = "MTP params-token binding cannot include selected_token";
                return false;
            }
            break;
        case MtpInputTokenSource::selected_token:
            if (params.T != 1u) {
                error = "MTP selected-token binding requires T=1";
                return false;
            }
            if (!sources.selected_token ||
                !context.buffers().owns(sources.selected_token)) {
                error = "MTP selected token must belong to the runtime Context";
                return false;
            }
            break;
        default:
            error = "MTP input projection token source is unsupported";
            return false;
    }

    const ReadOnlyBufferRef readonly_params =
        context.buffers().read_only(params_slot);
    const ReadOnlyBufferRef embedding_norm_gamma =
        context.buffers().read_only(
            dense_weight("pre_fc_norm_embedding.weight"));
    const ReadOnlyBufferRef fc_embedding_weights =
        context.buffers().read_only(dense_weight("fc_embedding.weight"));
    const ReadOnlyBufferRef hidden_norm_gamma =
        context.buffers().read_only(dense_weight("pre_fc_norm_hidden.weight"));
    const ReadOnlyBufferRef fc_hidden_weights =
        context.buffers().read_only(dense_weight("fc_hidden.weight"));
    BufferRef embedding = bind_region(MtpBindRegionId::emb);
    BufferRef embedding_norm = bind_region(MtpBindRegionId::en);
    BufferRef xq = bind_region(MtpBindRegionId::xq);
    BufferRef e2 = bind_region(MtpBindRegionId::e2);
    BufferRef hidden_norm = bind_region(MtpBindRegionId::hn);
    BufferRef h2 = bind_region(MtpBindRegionId::h2);
    BufferRef matvec_sink;

    constexpr std::size_t kEmbeddingRowBytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    const std::size_t active_embedding_bytes =
        static_cast<std::size_t>(params.T) * kEmbeddingRowBytes;
    if (params.T < 8u) {
        if (!e2 || active_embedding_bytes >= e2.length) {
            error = "MTP input projection e2 tail is too small for mat-vec padding";
            return false;
        }
        matvec_sink = e2.slice(active_embedding_bytes,
                               e2.length - active_embedding_bytes);
    } else {
        const BufferRef bo = bind_region(MtpBindRegionId::bo);
        matvec_sink = bo.slice(0u, sizeof(float));
    }

    if (!readonly_params || !embedding_norm_gamma || !fc_embedding_weights ||
        !hidden_norm_gamma || !fc_hidden_weights || !embedding ||
        !embedding_norm || !xq || !e2 || !hidden_norm || !h2 ||
        !matvec_sink) {
        error = "MTP input projection runtime weights or bind regions are unavailable";
        return false;
    }
    buffers = {
        readonly_params,
        sources.token_source,
        sources.selected_token,
        sources.embedding_format,
        sources.vocabulary_size,
        sources.token_embedding_table,
        embedding_norm_gamma,
        fc_embedding_weights,
        sources.residual,
        hidden_norm_gamma,
        fc_hidden_weights,
        embedding,
        embedding_norm,
        xq,
        e2,
        hidden_norm,
        h2,
        matvec_sink,
    };
    error.clear();
    return true;
}

BufferRef MtpRuntimeStorage::shared_rope() const noexcept {
    if (impl_ == nullptr || impl_->buffers == nullptr ||
        !impl_->shared_rope_bound ||
        !is_valid()) {
        return {};
    }
    return impl_->shared_rope;
}

bool MtpRuntimeStorage::kv_views(MtpKvViews& views, std::string& error) const {
    if (impl_ == nullptr || impl_->kv == nullptr) {
        views = {};
        error = "MTP runtime storage is unavailable";
        return false;
    }
    return impl_->kv->views(views, error);
}

}  // namespace strata::metal
