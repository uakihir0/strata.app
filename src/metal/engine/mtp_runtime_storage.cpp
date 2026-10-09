// src/metal/engine/mtp_runtime_storage.cpp - all planned MTP allocations under one transaction.
#include "strata/metal/mtp_runtime_storage.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/encoder.hpp"

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
    MtpDraftVocabularyMap draft_vocabulary;
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
        const ReadOnlyBufferRef read_only_ids =
            candidate->buffers->read_only(device_vocab_ids);
        if (!read_only_ids ||
            !make_mtp_draft_vocabulary_map(
                context, read_only_ids, kMtpFullVocabularySize,
                candidate->draft_vocabulary, error)) {
            if (error.empty()) {
                error = "MTP draft vocabulary could not be validated in its bind arena";
            }
            return nullptr;
        }
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
        id == MtpBindRegionId::draft_vocab_ids ||
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

bool MtpRuntimeStorage::bind_add_streams(
    Context& context, std::uint32_t params_slot_index,
    MtpAddStreamsBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP add-streams binding needs live runtime storage from this Context";
        return false;
    }

    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP add-streams parameter slot is unavailable";
        return false;
    }
    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const ReadOnlyBufferRef h2 =
        context.buffers().read_only(bind_region(MtpBindRegionId::h2));
    const ReadOnlyBufferRef e2 =
        context.buffers().read_only(bind_region(MtpBindRegionId::e2));
    const BufferRef output = bind_region(MtpBindRegionId::residual);
    if (!params || !h2 || !e2 || !output) {
        error = "MTP add-streams runtime parameter or scratch view is unavailable";
        return false;
    }
    buffers = {params, h2, e2, output};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_attention_read(
    Context& context, std::uint32_t params_slot_index,
    MtpAttentionReadBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP attention-read binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP attention-read parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const BufferRef norm =
        dense_weight("attn_hyper_connection.hc_norm.weight");
    const BufferRef hc_down =
        dense_weight("attn_hyper_connection.input_mix_weight_down.weight");
    const BufferRef hc_inject =
        dense_weight("attn_hyper_connection.block_inject_weight.weight");
    const BufferRef hc_up =
        dense_weight("attn_hyper_connection.input_mix_weight_up.weight");
    const BufferRef residual = bind_region(MtpBindRegionId::residual);
    const BufferRef dead_h2_scratch = bind_region(MtpBindRegionId::h2);
    const BufferRef normalized = bind_region(MtpBindRegionId::xn);
    const BufferRef rs = bind_region(MtpBindRegionId::rs);
    const BufferRef lo = bind_region(MtpBindRegionId::lo);
    const BufferRef inject_out = bind_region(MtpBindRegionId::inj);
    const BufferRef mixed = bind_region(MtpBindRegionId::mixed);
    if (!params || !norm || !hc_down || !hc_inject || !hc_up ||
        !residual || !dead_h2_scratch || !normalized || !rs || !lo ||
        !inject_out || !mixed) {
        error = "MTP attention-read runtime weights or bind regions are unavailable";
        return false;
    }
    buffers = {params, residual, dead_h2_scratch, norm, normalized, rs,
               hc_down, hc_inject, lo, inject_out, hc_up, mixed};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_attention_kv_projection(
    Context& context, std::uint32_t params_slot_index,
    MtpAttentionKvProjectionBuffers& buffers,
    std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP K/V projection binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP K/V projection parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const ReadOnlyBufferRef key_weights = context.buffers().read_only(
        dense_weight("self_attn.k_proj.weight"));
    const ReadOnlyBufferRef value_weights = context.buffers().read_only(
        dense_weight("self_attn.v_proj.weight"));
    const BufferRef mixed = bind_region(MtpBindRegionId::mixed);
    const BufferRef xq = bind_region(MtpBindRegionId::xq);
    const BufferRef kcur = bind_region(MtpBindRegionId::kcur);
    const BufferRef vcur = bind_region(MtpBindRegionId::vcur);
    const BufferRef sink = bind_region(MtpBindRegionId::bo);
    if (!params || !key_weights || !value_weights || !mixed || !xq ||
        !kcur || !vcur || !sink) {
        error = "MTP K/V projection runtime weights or bind regions are unavailable";
        return false;
    }
    buffers = {params, mixed, xq, key_weights, value_weights,
               kcur, vcur, sink};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_attention_kv_append(
    Context& context, std::uint32_t params_slot_index,
    const MtpAttentionRuntimeMode& mode,
    MtpAttentionKvAppendBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP K/V append binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params = mtp_params_slot(params_slot_index);
    if (!params || params.length < sizeof(MtpParams) ||
        params.contents == nullptr) {
        error = "MTP K/V append parameter slot is unavailable";
        return false;
    }

    MtpKvViews views{};
    if (!kv_views(views, error)) {
        return false;
    }
    const BufferRef kcur = bind_region(MtpBindRegionId::kcur);
    const BufferRef vcur = bind_region(MtpBindRegionId::vcur);
    const BufferRef k_norm = dense_weight("self_attn.k_norm.weight");
    const BufferRef rope = shared_rope();
    if (!kcur || !vcur || !k_norm || !rope) {
        error = "MTP K/V append runtime weights, cache, RoPE or bind regions are unavailable";
        return false;
    }
    buffers = {params, kcur, vcur, k_norm, rope, kv_layout(), views,
               mode.kv_rot()};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_attention_query_projection(
    Context& context, std::uint32_t params_slot_index,
    MtpAttentionQueryProjectionBuffers& buffers,
    std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP query-projection binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP query-projection parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const BufferRef xq = bind_region(MtpBindRegionId::xq);
    const ReadOnlyBufferRef q_weights = context.buffers().read_only(
        dense_weight("self_attn.q_proj.weight"));
    const BufferRef q_full = bind_region(MtpBindRegionId::q_full);
    const BufferRef bo = bind_region(MtpBindRegionId::bo);
    const BufferRef sink = bo.slice(0u, sizeof(float));
    if (!params || !xq || !q_weights || !q_full || !sink) {
        error = "MTP query-projection runtime weights or bind regions are unavailable";
        return false;
    }
    buffers = {params, xq, q_weights, q_full, sink};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_attention_query_prefix(
    Context& context, std::uint32_t params_slot_index,
    const MtpAttentionRuntimeMode& mode,
    MtpAttentionQueryPrefixBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};

    MtpAttentionQueryPrefixBuffers candidate;
    if (!bind_attention_kv_append(context, params_slot_index, mode,
                                  candidate.kv_append_, error) ||
        !bind_attention_query_projection(
            context, params_slot_index, candidate.query_projection_, error)) {
        return false;
    }

    const BufferRef params = mtp_params_slot(params_slot_index);
    if (!params || params.length < sizeof(MtpParams) ||
        params.contents == nullptr) {
        error = "MTP query-prefix parameter slot is unavailable";
        return false;
    }

    const BufferRef q_full = bind_region(MtpBindRegionId::q_full);
    const BufferRef q_norm = dense_weight("self_attn.q_norm.weight");
    const BufferRef rope = shared_rope();
    const BufferRef qcur = bind_region(MtpBindRegionId::qcur);
    if (!q_full || !q_norm || !rope || !qcur) {
        error = "MTP query-prefix runtime weights or bind regions are unavailable";
        return false;
    }

    candidate.query_norm_ = {
        params, q_full, q_norm, rope, qcur, candidate.kv_append_.kv_rot};
    buffers = std::move(candidate);
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_attention_pass(
    Context& context, std::uint32_t params_slot_index,
    const MtpAttentionRuntimeMode& mode, std::uint32_t window,
    MtpAttentionPassBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};

    MtpAttentionPassBuffers candidate;
    if (!bind_attention_query_prefix(context, params_slot_index, mode,
                                     candidate.query_prefix_, error)) {
        return false;
    }
    const BufferRef params = mtp_params_slot(params_slot_index);
    if (!params || params.length < sizeof(MtpParams) ||
        params.contents == nullptr) {
        error = "MTP attention parameter slot is unavailable";
        return false;
    }
    if (window > impl_->bind_layout.capacity) {
        error = "MTP attention window exceeds the bind ID capacity";
        return false;
    }

    const BufferRef mtp_ids = bind_region(MtpBindRegionId::mtp_ids);
    const BufferRef attn = bind_region(MtpBindRegionId::attn);
    const BufferRef attn32 = bind_region(MtpBindRegionId::attn32);
    const BufferRef att_scratch =
        bind_region(MtpBindRegionId::att_scratch);
    const BufferRef xq = bind_region(MtpBindRegionId::xq);
    const BufferRef bo = bind_region(MtpBindRegionId::bo);
    const BufferRef sink =
        bind_region(MtpBindRegionId::y).slice(0u, sizeof(float));
    const BufferRef qcur = bind_region(MtpBindRegionId::qcur);
    const BufferRef q_full = bind_region(MtpBindRegionId::q_full);
    const BufferRef o_weights = dense_weight("self_attn.o_proj.weight");
    MtpKvViews views{};
    if (!kv_views(views, error) || !mtp_ids || !attn || !attn32 ||
        !att_scratch || !xq || !bo || !sink || !qcur || !q_full ||
        !o_weights) {
        if (error.empty()) {
            error = "MTP attention runtime weights or bind regions are unavailable";
        }
        return false;
    }

    candidate.window_ids_ = {
        context.buffers().read_only(params), mtp_ids, window};
    candidate.attention_ = {
        params, qcur, mtp_ids, kv_layout(), views, att_scratch, attn,
        impl_->bind_layout.capacity,
        candidate.query_prefix_.kv_append_.kv_rot};
    candidate.output_projection_ = {
        context.buffers().read_only(params), attn, q_full, attn32, xq,
        context.buffers().read_only(o_weights), bo, sink};
    buffers = std::move(candidate);
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_feed_forward_read(
    Context& context, std::uint32_t params_slot_index,
    MtpFeedForwardReadBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP feed-forward binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP feed-forward parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const BufferRef residual = bind_region(MtpBindRegionId::residual);
    const BufferRef bo = bind_region(MtpBindRegionId::bo);
    const BufferRef inj = bind_region(MtpBindRegionId::inj);
    const BufferRef norm =
        dense_weight("mlp_hyper_connection.hc_norm.weight");
    const BufferRef normalized = bind_region(MtpBindRegionId::xn);
    const BufferRef rs = bind_region(MtpBindRegionId::rs);
    const BufferRef hc_down =
        dense_weight("mlp_hyper_connection.input_mix_weight_down.weight");
    const BufferRef hc_inject =
        dense_weight("mlp_hyper_connection.block_inject_weight.weight");
    const BufferRef lo = bind_region(MtpBindRegionId::lo);
    const BufferRef inject_out = bind_region(MtpBindRegionId::inj2);
    const BufferRef hc_up =
        dense_weight("mlp_hyper_connection.input_mix_weight_up.weight");
    const BufferRef mixed = bind_region(MtpBindRegionId::mixed);
    if (!params || !residual || !bo || !inj || !norm || !normalized || !rs ||
        !hc_down || !hc_inject || !lo || !inject_out || !hc_up || !mixed) {
        error = "MTP feed-forward runtime weights or bind regions are unavailable";
        return false;
    }
    buffers = {params, residual, bo, inj, norm, normalized, rs, hc_down,
               hc_inject, lo, inject_out, hc_up, mixed};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_router(Context& context,
                                    std::uint32_t params_slot_index,
                                    MtpRouterBuffers& buffers,
                                    std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP router binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP router parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const ReadOnlyBufferRef mixed = context.buffers().read_only(
        bind_region(MtpBindRegionId::mixed));
    const ReadOnlyBufferRef router_weights = context.buffers().read_only(
        dense_weight("mlp.gate.weight"));
    const BufferRef logits_r = bind_region(MtpBindRegionId::logits_r);
    const BufferRef ids = bind_region(MtpBindRegionId::ids);
    const BufferRef weights = bind_region(MtpBindRegionId::wts);
    const BufferRef matvec_sink =
        bind_region(MtpBindRegionId::y).slice(0u, sizeof(float));
    if (!params || !mixed || !router_weights || !logits_r || !ids ||
        !weights || !matvec_sink) {
        error = "MTP router runtime weights or bind regions are unavailable";
        return false;
    }
    buffers = {params, mixed, router_weights, logits_r, ids, weights,
               matvec_sink};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_routed_experts(
    Context& context, MtpRoutedExpertsBuffers& buffers,
    std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers() ||
        impl_->weights == nullptr) {
        error = "MTP routed-expert binding needs live runtime storage from this Context";
        return false;
    }

    const ReadOnlyBufferRef mixed = context.buffers().read_only(
        bind_region(MtpBindRegionId::mixed));
    const ReadOnlyBufferRef ids = context.buffers().read_only(
        bind_region(MtpBindRegionId::ids));
    const BufferRef mq = bind_region(MtpBindRegionId::mq);
    const BufferRef ms = bind_region(MtpBindRegionId::ms);
    const BufferRef gate = bind_region(MtpBindRegionId::eg);
    const BufferRef up = bind_region(MtpBindRegionId::eu);
    const BufferRef eh = bind_region(MtpBindRegionId::eh);
    const BufferRef hq = bind_region(MtpBindRegionId::hq);
    const BufferRef hs = bind_region(MtpBindRegionId::hs);
    const BufferRef parts = bind_region(MtpBindRegionId::parts);
    if (!mixed || !ids || !mq || !ms || !gate || !up || !eh || !hq ||
        !hs || !parts) {
        error = "MTP routed-expert scratch regions are unavailable";
        return false;
    }

    const auto& ranges = impl_->weights->layout().expert_windows;
    if (ranges.empty()) {
        error = "MTP routed-expert storage has no expert windows";
        return false;
    }
    MtpRoutedExpertsBuffers candidate;
    candidate.mixed = mixed;
    candidate.ids = ids;
    candidate.mq = mq;
    candidate.ms = ms;
    candidate.gate = gate;
    candidate.up = up;
    candidate.eh = eh;
    candidate.hq = hq;
    candidate.hs = hs;
    candidate.parts = parts;
    candidate.expert_windows.reserve(ranges.size());

    std::uint64_t next_expert = 0;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const FileWindow& range = ranges[index];
        if (range.end <= range.start ||
            range.start % kMtpExpertBlobBytes != 0 ||
            (range.end - range.start) % kMtpExpertBlobBytes != 0 ||
            range.start / kMtpExpertBlobBytes != next_expert) {
            error = "MTP routed-expert windows do not form complete ordered expert ranges";
            return false;
        }
        const std::uint64_t count =
            (range.end - range.start) / kMtpExpertBlobBytes;
        if (next_expert >= kMtpExpertCount || count == 0 ||
            count > kMtpExpertCount - next_expert ||
            next_expert > std::numeric_limits<std::uint32_t>::max() ||
            count > std::numeric_limits<std::uint32_t>::max()) {
            error = "MTP routed-expert window range exceeds the expert table";
            return false;
        }
        const BufferRef window = impl_->weights->expert_window(index);
        const ReadOnlyBufferRef read_only =
            context.buffers().read_only(window);
        if (!window || window.length != range.end - range.start ||
            !read_only) {
            error = "MTP routed-expert storage window is unavailable";
            return false;
        }
        candidate.expert_windows.push_back(
            {read_only, static_cast<std::uint32_t>(next_expert),
             static_cast<std::uint32_t>(count)});
        next_expert += count;
    }
    if (next_expert != kMtpExpertCount) {
        error = "MTP routed-expert windows do not cover all 512 experts";
        return false;
    }

    buffers = std::move(candidate);
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_shared_ffn_write(
    Context& context, std::uint32_t params_slot_index,
    MtpSharedFFNWriteBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP shared FFN binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP shared FFN parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const ReadOnlyBufferRef gate_weights = context.buffers().read_only(
        dense_weight("mlp.shared_expert.gate_proj.weight"));
    const ReadOnlyBufferRef up_weights = context.buffers().read_only(
        dense_weight("mlp.shared_expert.up_proj.weight"));
    const ReadOnlyBufferRef down_weights = context.buffers().read_only(
        dense_weight("mlp.shared_expert.down_proj.weight"));
    const ReadOnlyBufferRef shared_gate_weights = context.buffers().read_only(
        dense_weight("mlp.shared_expert_gate.weight"));
    const ReadOnlyBufferRef routed_parts = context.buffers().read_only(
        bind_region(MtpBindRegionId::parts));
    const ReadOnlyBufferRef route_weights = context.buffers().read_only(
        bind_region(MtpBindRegionId::wts));
    const BufferRef mixed = bind_region(MtpBindRegionId::mixed);
    const BufferRef mixed_q8_1 = bind_region(MtpBindRegionId::xq);
    const BufferRef gate = bind_region(MtpBindRegionId::sg);
    const BufferRef up = bind_region(MtpBindRegionId::su);
    const BufferRef hidden = bind_region(MtpBindRegionId::sh);
    const BufferRef down_activations = bind_region(MtpBindRegionId::shq);
    const BufferRef shared_output = bind_region(MtpBindRegionId::shx);
    const BufferRef shared_gate_logits = bind_region(MtpBindRegionId::gl);
    const BufferRef output = bind_region(MtpBindRegionId::y);
    const BufferRef gate_sink =
        bind_region(MtpBindRegionId::sample).slice(0u, sizeof(float));
    const BufferRef up_sink =
        bind_region(MtpBindRegionId::bo).slice(0u, sizeof(float));
    const BufferRef down_sink =
        bind_region(MtpBindRegionId::q_full).slice(0u, sizeof(float));
    const BufferRef shared_gate_sink =
        bind_region(MtpBindRegionId::attn32).slice(0u, sizeof(float));
    const BufferRef residual = bind_region(MtpBindRegionId::residual);
    const BufferRef inject = bind_region(MtpBindRegionId::inj2);
    if (!params || !gate_weights || !up_weights || !down_weights ||
        !shared_gate_weights || !routed_parts || !route_weights ||
        !mixed || !mixed_q8_1 || !gate || !up || !hidden ||
        !down_activations || !shared_output || !shared_gate_logits ||
        !output || !gate_sink || !up_sink || !down_sink ||
        !shared_gate_sink || !residual || !inject) {
        error = "MTP shared FFN runtime weights or bind regions are unavailable";
        return false;
    }

    buffers = {
        params,
        mixed,
        mixed_q8_1,
        {
            gate_weights,
            up_weights,
            down_weights,
            shared_gate_weights,
            routed_parts,
            route_weights,
            gate,
            up,
            hidden,
            down_activations,
            shared_output,
            shared_gate_logits,
            output,
            gate_sink,
            up_sink,
            down_sink,
            shared_gate_sink,
            residual,
            inject,
        },
    };
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_final_mixer(
    Context& context, std::uint32_t params_slot_index,
    MtpFinalMixerBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP final-mixer binding needs live runtime storage from this Context";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP final-mixer parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const BufferRef residual = bind_region(MtpBindRegionId::residual);
    const ReadOnlyBufferRef norm_gamma = context.buffers().read_only(
        dense_weight("hyper_connection_mixer.hc_norm.weight"));
    const ReadOnlyBufferRef down_weights = context.buffers().read_only(
        dense_weight("hyper_connection_mixer.input_mix_weight_down.weight"));
    const ReadOnlyBufferRef up_weights = context.buffers().read_only(
        dense_weight("hyper_connection_mixer.input_mix_weight_up.weight"));
    const BufferRef normalized = bind_region(MtpBindRegionId::xn);
    const BufferRef low_rank = bind_region(MtpBindRegionId::lo);
    const BufferRef gate = bind_region(MtpBindRegionId::gate);
    const BufferRef mixed = bind_region(MtpBindRegionId::sample);
    const BufferRef matvec_sink =
        bind_region(MtpBindRegionId::y).slice(0u, sizeof(float));
    if (!params || !residual || !norm_gamma || !down_weights ||
        !up_weights || !normalized || !low_rank || !gate || !mixed ||
        !matvec_sink) {
        error = "MTP final-mixer runtime weights or bind regions are unavailable";
        return false;
    }
    buffers = {params, residual, norm_gamma, down_weights, up_weights,
               normalized, low_rank, gate, mixed, matvec_sink};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_draft_head(
    Context& context, std::uint32_t params_slot_index,
    ReadOnlyBufferRef output_weights, std::uint32_t full_head_vocab_size,
    MtpDraftHeadBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP draft-head binding needs live runtime storage from this Context";
        return false;
    }
    if (!output_weights || !context.buffers().owns(output_weights)) {
        error = "MTP draft-head weights must be a live read-only view from this Context";
        return false;
    }
    if (full_head_vocab_size != kMtpFullVocabularySize) {
        error = "MTP draft-head binding requires the inspected 248320-row main head";
        return false;
    }
    if (impl_->draft_vocabulary &&
        impl_->draft_vocabulary.full_head_size() != full_head_vocab_size) {
        error = "MTP draft vocabulary does not match the main head row count";
        return false;
    }
    const BufferRef params_slot = mtp_params_slot(params_slot_index);
    if (!params_slot || params_slot.length < sizeof(MtpParams) ||
        params_slot.contents == nullptr) {
        error = "MTP draft-head parameter slot is unavailable";
        return false;
    }

    const ReadOnlyBufferRef params =
        context.buffers().read_only(params_slot);
    const BufferRef sample = bind_region(MtpBindRegionId::sample);
    const BufferRef activations_q8_1 = bind_region(MtpBindRegionId::xq);
    const BufferRef logits = bind_region(MtpBindRegionId::hl);
    const BufferRef matvec_sink =
        bind_region(MtpBindRegionId::y).slice(0u, sizeof(float));
    if (!params || !sample || !activations_q8_1 || !logits ||
        !matvec_sink) {
        error = "MTP draft-head runtime or bind-arena views are unavailable";
        return false;
    }
    buffers = {params, sample, output_weights, impl_->draft_vocabulary,
               activations_q8_1, logits, matvec_sink};
    error.clear();
    return true;
}

bool MtpRuntimeStorage::bind_draft_sampling(
    Context& context, std::uint32_t params_slot_index,
    const MtpDraftVocabularyMap& draft_vocabulary,
    MtpDraftSamplingBuffers& buffers, std::string& error) const {
    error.clear();
    buffers = {};
    if (!is_valid() || impl_->buffers != &context.buffers()) {
        error = "MTP draft-sampling binding needs live runtime storage from this Context";
        return false;
    }

    const ReadOnlyBufferRef expected_map = impl_->draft_vocabulary.rows();
    const ReadOnlyBufferRef supplied_map = draft_vocabulary.rows();
    const bool same_map =
        static_cast<bool>(impl_->draft_vocabulary) ==
            static_cast<bool>(draft_vocabulary) &&
        (!impl_->draft_vocabulary ||
         (draft_vocabulary.full_head_size() == kMtpFullVocabularySize &&
          draft_vocabulary.row_count() ==
              impl_->draft_vocabulary.row_count() &&
          context.buffers().owns(supplied_map) &&
          expected_map.offset() == supplied_map.offset() &&
          expected_map.length() == supplied_map.length() &&
          expected_map.contents() == supplied_map.contents() &&
          expected_map.gpu_address() == supplied_map.gpu_address()));
    if (!same_map) {
        error = "MTP draft-sampling vocabulary must reuse this runtime's validated draft-head map";
        return false;
    }

    const BufferRef params = mtp_params_slot(params_slot_index);
    if (!params || params.length < sizeof(MtpParams) ||
        params.contents == nullptr) {
        error = "MTP draft-sampling parameter slot is unavailable";
        return false;
    }
    const BufferRef logits = bind_region(MtpBindRegionId::hl);
    const BufferRef output_ids = bind_region(MtpBindRegionId::out_ids);
    const BufferRef probabilities = bind_region(MtpBindRegionId::probs);
    if (!logits || !output_ids || !probabilities) {
        error = "MTP draft-sampling bind-arena views are unavailable";
        return false;
    }
    buffers = {params, logits, draft_vocabulary, output_ids, probabilities};
    error.clear();
    return true;
}

bool encode_mtp_attention_query_prefix(
    Context& context, Encoder& encoder,
    const MtpAttentionQueryPrefixBuffers& buffers, std::string& error) {
    error.clear();
    const MtpAttentionKvAppendBuffers& kv = buffers.kv_append_;
    const MtpAttentionQueryProjectionBuffers& projection =
        buffers.query_projection_;
    const MtpAttentionQueryNormBuffers& norm = buffers.query_norm_;
    const auto same_params = [](BufferRef writable,
                                ReadOnlyBufferRef read_only) {
        return writable && read_only &&
               writable.contents == read_only.contents() &&
               writable.gpu_address == read_only.gpu_address() &&
               writable.length == read_only.length() &&
               writable.offset == read_only.offset();
    };
    const bool shared_params =
        kv.params && norm.params &&
        kv.params.handle == norm.params.handle &&
        kv.params.contents == norm.params.contents &&
        kv.params.gpu_address == norm.params.gpu_address &&
        kv.params.length == norm.params.length &&
        kv.params.offset == norm.params.offset &&
        same_params(kv.params, projection.params);
    if (!shared_params) {
        error = "MTP query-prefix stages 9–11 must share one parameter view";
        return false;
    }
    if (kv.kv_rot != norm.kv_rot) {
        error = "MTP query-prefix stages 9 and 11 must share one rotation mode";
        return false;
    }
    if (encoder.concurrent_group_open()) {
        error = "MTP query-prefix requires an ordered concurrent-group boundary";
        return false;
    }

    if (!encode_mtp_attention_kv_append(context, encoder, kv, error) ||
        !encode_mtp_attention_query_projection(context, encoder, projection,
                                               error) ||
        !encode_mtp_attention_query_norm(context, encoder, norm, error)) {
        return false;
    }
    error.clear();
    return true;
}

bool encode_mtp_attention_query_prefix(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionQueryPrefixBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention query prefix",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_query_prefix(
                context, encoder, buffers, encode_error);
        },
        error);
}

bool encode_mtp_attention_pass(
    Context& context, Encoder& encoder,
    const MtpAttentionPassBuffers& buffers, std::string& error) {
    error.clear();
    const MtpAttentionQueryPrefixBuffers& query_prefix =
        buffers.query_prefix_;
    const MtpAttentionKvAppendBuffers& kv = query_prefix.kv_append();
    const MtpAttentionQueryProjectionBuffers& query_projection =
        query_prefix.query_projection();
    const MtpAttentionQueryNormBuffers& query_norm =
        query_prefix.query_norm();
    const MtpWindowIdsBuffers& window_ids = buffers.window_ids_;
    const MtpAttentionBuffers& attention = buffers.attention_;
    const MtpAttentionOutputProjectionBuffers& output =
        buffers.output_projection_;

    const auto same_writable_view = [](BufferRef left, BufferRef right) {
        return left && right && left.handle == right.handle &&
               left.contents == right.contents &&
               left.gpu_address == right.gpu_address &&
               left.length == right.length && left.offset == right.offset;
    };
    const auto same_read_only_view = [](BufferRef writable,
                                        ReadOnlyBufferRef read_only) {
        return writable && read_only &&
               writable.contents == read_only.contents() &&
               writable.gpu_address == read_only.gpu_address() &&
               writable.length == read_only.length() &&
               writable.offset == read_only.offset();
    };
    const auto same_region = [](KvRegion left, KvRegion right) {
        return left.offset == right.offset && left.bytes == right.bytes;
    };
    const auto same_kv_layout = [&](const MtpKvLayout& left,
                                    const MtpKvLayout& right) {
        return left.requested_format == right.requested_format &&
               left.storage_format == right.storage_format &&
               left.requested_cells == right.requested_cells &&
               left.cell_capacity == right.cell_capacity &&
               left.kv_rows == right.kv_rows &&
               left.allocation_bytes == right.allocation_bytes &&
               same_region(left.k, right.k) &&
               same_region(left.k_scale, right.k_scale) &&
               same_region(left.v, right.v) &&
               same_region(left.v_scale, right.v_scale);
    };
    const auto same_kv_views = [&](const MtpKvViews& left,
                                   const MtpKvViews& right) {
        return same_writable_view(left.k, right.k) &&
               same_writable_view(left.k_scale, right.k_scale) &&
               same_writable_view(left.v, right.v) &&
               same_writable_view(left.v_scale, right.v_scale);
    };
    const bool shared_params =
        kv.params && query_norm.params &&
        kv.params.handle == query_norm.params.handle &&
        kv.params.contents == query_norm.params.contents &&
        kv.params.gpu_address == query_norm.params.gpu_address &&
        kv.params.length == query_norm.params.length &&
        kv.params.offset == query_norm.params.offset &&
        same_read_only_view(kv.params, query_projection.params) &&
        same_read_only_view(kv.params, window_ids.params) &&
        same_writable_view(kv.params, attention.params) &&
        same_read_only_view(kv.params, output.params);
    if (!shared_params) {
        error = "MTP attention stages 9–14 must share one parameter view";
        return false;
    }
    const bool shared_rotation =
        kv.kv_rot == query_norm.kv_rot &&
        kv.kv_rot == attention.kv_rot;
    if (!shared_rotation) {
        error = "MTP attention stages 9, 11 and 13 must share one rotation mode";
        return false;
    }
    if (!same_writable_view(query_projection.q_full, query_norm.q_full) ||
        !same_writable_view(query_projection.q_full, output.q_full) ||
        !same_writable_view(query_norm.qcur, attention.qcur) ||
        !same_writable_view(window_ids.mtp_ids, attention.mtp_ids) ||
        !same_writable_view(attention.attn, output.attn) ||
        !same_writable_view(query_projection.xq, output.xq) ||
        !same_writable_view(query_projection.sink,
                            output.bo.slice(0u, sizeof(float))) ||
        !same_writable_view(kv.rope, query_norm.rope) ||
        !same_kv_layout(kv.kv_layout, attention.kv_layout) ||
        !same_kv_views(kv.kv_views, attention.kv_views) ||
        attention.mtp_ids.length % sizeof(std::int32_t) != 0u ||
        attention.id_row_stride !=
            attention.mtp_ids.length / sizeof(std::int32_t)) {
        error = "MTP attention stages 9–14 runtime views do not form one ordered pass";
        return false;
    }
    if (encoder.concurrent_group_open()) {
        error = "MTP attention pass requires an ordered concurrent-group boundary";
        return false;
    }

    if (!encode_mtp_attention_kv_append(context, encoder, kv, error) ||
        !encode_mtp_attention_query_projection(
            context, encoder, query_projection, error) ||
        !encode_mtp_attention_query_norm(context, encoder, query_norm,
                                         error)) {
        return false;
    }
    if (!encode_mtp_window_ids(context, encoder, window_ids, error)) {
        return false;
    }
    if (!encode_mtp_attention(context, encoder, attention, error) ||
        !encode_mtp_attention_output_projection(context, encoder, output,
                                                error)) {
        return false;
    }
    error.clear();
    return true;
}

bool encode_mtp_attention_pass(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionPassBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention pass stages 9–14",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_pass(context, encoder, buffers,
                                             encode_error);
        },
        error);
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
