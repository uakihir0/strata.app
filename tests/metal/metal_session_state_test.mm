// tests/metal/metal_session_state_test.mm - per-sequence GDN and PLE state storage.
#include "strata/metal/session_state.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace strata::metal;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_zero(BufferRef view, const std::string& label) {
    const auto* bytes = static_cast<const std::uint8_t*>(view.contents);
    require(view && std::all_of(bytes, bytes + view.length,
                                [](std::uint8_t byte) { return byte == 0; }),
            label + " must be zero initialized");
}

void require_byte_pattern(BufferRef view, std::uint8_t expected,
                          const std::string& label) {
    const auto* bytes = static_cast<const std::uint8_t*>(view.contents);
    require(view && std::all_of(bytes, bytes + view.length,
                                [expected](std::uint8_t byte) {
                                    return byte == expected;
                                }),
            label + " must preserve its byte pattern");
}

void check_region(BufferRef view, BufferRef allocation,
                  SessionStateRegion region, const std::string& label) {
    require(view && view.handle == allocation.handle &&
                view.length == region.bytes &&
                view.offset == allocation.offset + region.offset &&
                view.gpu_address == allocation.gpu_address + region.offset &&
                view.contents ==
                    static_cast<std::byte*>(allocation.contents) + region.offset,
            label + " does not select the planned allocation region");
}

}  // namespace

int main() {
    try {
        Geometry geometry;
        geometry.family = "qwen4exp";
        geometry.n_layers = kSessionModelLayers;
        geometry.n_qsa_layers = kSessionQsaLayers;
        geometry.n_kv_head = kSessionKvHeads;
        geometry.head_dim = kSessionHeadWidth;
        geometry.n_embd = kSessionEmbeddingWidth;
        geometry.vocab = kSessionVocabularySize;

        SessionStateLayout layout;
        std::string error;
        require(make_session_state_layout(geometry, layout, error),
                "make fixed session-state layout: " + error);
        Geometry qwen_family_geometry = geometry;
        qwen_family_geometry.family = "qwen";
        require(make_session_state_layout(qwen_family_geometry, layout, error),
                "package-family identity must preserve the qwen4exp state ABI: " + error);
        require(layout.gdn.size() == kSessionGdnLayers,
                "layout must contain one state region per GDN layer");
        require(layout.gdn.front().recurrent.bytes ==
                    static_cast<std::size_t>(kSessionGdnStateElements) * sizeof(float) &&
                    layout.gdn.front().conv_history.bytes ==
                    static_cast<std::size_t>(kSessionGdnConvHistoryElements) * sizeof(float),
                "GDN state regions must match the shader geometry");
        require(layout.ple_history.bytes ==
                    static_cast<std::size_t>(kSessionPleHistoryElements) * sizeof(float),
                "PLE history must contain nine row-fastest history rows");
        require(layout.allocation_bytes == 118'038'528u,
                "session allocation must exactly account for GDN and PLE state");

        geometry.n_qsa_layers = 11;
        require(!make_session_state_layout(geometry, layout, error) &&
                    layout.allocation_bytes == 0 &&
                    error.find("48 layers, 12 QSA layers") != std::string::npos,
                "unsupported model geometry must fail before allocation");
        geometry.n_qsa_layers = kSessionQsaLayers;
        geometry.n_embd = kSessionEmbeddingWidth - 1u;
        require(!make_session_state_layout(geometry, layout, error) &&
                    layout.allocation_bytes == 0 &&
                    error.find("qwen4exp geometry") != std::string::npos,
                "unsupported embedding width must fail before allocation");
        geometry.n_embd = kSessionEmbeddingWidth;
        for (const auto& invalid_geometry :
             {Geometry{.family = "other"},
              Geometry{.family = "qwen4exp", .n_layers = kSessionModelLayers,
                       .n_qsa_layers = kSessionQsaLayers, .n_kv_head = 1,
                       .head_dim = kSessionHeadWidth,
                       .n_embd = kSessionEmbeddingWidth,
                       .vocab = kSessionVocabularySize},
              Geometry{.family = "qwen4exp", .n_layers = kSessionModelLayers,
                       .n_qsa_layers = kSessionQsaLayers, .n_kv_head = kSessionKvHeads,
                       .head_dim = 128, .n_embd = kSessionEmbeddingWidth,
                       .vocab = kSessionVocabularySize},
              Geometry{.family = "qwen4exp", .n_layers = kSessionModelLayers,
                       .n_qsa_layers = kSessionQsaLayers, .n_kv_head = kSessionKvHeads,
                       .head_dim = kSessionHeadWidth,
                       .n_embd = kSessionEmbeddingWidth, .vocab = 1}}) {
            require(!make_session_state_layout(invalid_geometry, layout, error) &&
                        layout.allocation_bytes == 0,
                    "unsupported qwen4exp identity and dimensions must fail closed");
        }
        ContextOptions options;
        options.pipelines = false;
        auto context = Context::open(options, error);
        require(context != nullptr, "open Metal context: " + error);

        const std::size_t allocations_before = context->buffers().allocation_count();
        Geometry invalid_geometry = geometry;
        invalid_geometry.vocab -= 1u;
        require(SessionStateStorage::open(*context, invalid_geometry, error) == nullptr &&
                    context->buffers().allocation_count() == allocations_before,
                "unsupported geometry must be rejected before Metal allocation");
        auto state = SessionStateStorage::open(*context, geometry, error);
        require(state != nullptr, "open session-state storage: " + error);
        require(state->ple_token_id_suffix().empty(),
                "session token history must start empty");
        const std::array<std::uint32_t, 1> baseline_token{88u};
        const std::array<std::uint32_t, 1> uninitialized_token{77u};
        WindowPleRows uninitialized_rows;
        require(build_window_ple_rows(baseline_token, {}, geometry,
                                      uninitialized_rows, error),
                "prepare a nonempty output before uninitialized-row rejection: " +
                    error);
        const std::array<std::uint32_t, kWindowPleRowsPerToken>
            baseline_rows = [&] {
                std::array<std::uint32_t, kWindowPleRowsPerToken> copy{};
                std::copy(uninitialized_rows.all_rows().begin(),
                          uninitialized_rows.all_rows().end(), copy.begin());
                return copy;
            }();
        require(!state->build_ple_rows(uninitialized_token, geometry,
                                       uninitialized_rows, error) &&
                    error.find("must be initialized") != std::string::npos &&
                    uninitialized_rows.token_count() == 1u &&
                    std::equal(uninitialized_rows.all_rows().begin(),
                               uninitialized_rows.all_rows().end(),
                               baseline_rows.begin(), baseline_rows.end()),
                "Window rows must preserve existing output before history initialization");
        require(!state->commit_accepted_ple_tokens(uninitialized_token, 1u,
                                                   geometry, error) &&
                    error.find("must be initialized") != std::string::npos &&
                    state->ple_token_id_suffix().empty(),
                "accepted-prefix updates must fail before history initialization");
        require(context->buffers().allocation_count() == allocations_before + 1,
                "session state must use one Context-owned allocation");
        const BufferRef allocation = state->allocation();
        require(context->buffers().owns(allocation) &&
                    allocation.length == state->layout().allocation_bytes,
                "session allocation must be live and exact-sized");
        require_zero(allocation, "complete session state on open");

        std::size_t previous_end = 0;
        for (std::uint32_t ordinal = 0; ordinal < kSessionGdnLayers; ++ordinal) {
            GdnSessionStateViews views;
            require(state->gdn_layer(ordinal, views, error),
                    "get GDN session-state views: " + error);
            check_region(views.recurrent, allocation,
                         state->layout().gdn[ordinal].recurrent, "GDN recurrence");
            check_region(views.conv_history, allocation,
                         state->layout().gdn[ordinal].conv_history, "GDN conv history");
            require(views.recurrent.gpu_address % 256u == 0 &&
                        views.conv_history.gpu_address % 256u == 0,
                    "GDN regions must meet Metal's 256-byte alignment contract");
            require(state->layout().gdn[ordinal].recurrent.offset >= previous_end,
                    "GDN regions must be monotonically laid out without overlap");
            previous_end = state->layout().gdn[ordinal].conv_history.offset +
                           state->layout().gdn[ordinal].conv_history.bytes;
        }

        GdnSessionStateViews invalid_views;
        require(!state->gdn_layer(kSessionGdnLayers, invalid_views, error) &&
                    !invalid_views.recurrent && !invalid_views.conv_history &&
                    error.find("out of range") != std::string::npos,
                "out-of-range GDN state lookup must fail closed");
        const BufferRef ple = state->ple_history();
        check_region(ple, allocation, state->layout().ple_history, "PLE history");
        require_zero(ple, "PLE history");

        GdnSessionStateViews first_layer;
        require(state->gdn_layer(0, first_layer, error), "reopen first GDN state");
        std::memset(first_layer.recurrent.contents, 0xa5, first_layer.recurrent.length);
        std::memset(first_layer.conv_history.contents, 0xa5,
                    first_layer.conv_history.length);
        std::memset(ple.contents, 0xa5, ple.length);
        const std::array<std::uint32_t, 2> first_tokens{101u, 102u};
        require(state->initialize_ple_token_ids(first_tokens, geometry, error),
                "seed session-owned PLE token history: " + error);
        const std::array<std::uint32_t, 1> duplicate_seed{103u};
        require(!state->initialize_ple_token_ids(duplicate_seed, geometry,
                                                 error) &&
                    error.find("already initialized") != std::string::npos,
                "re-seeding must not clear or replace live sequence history");
        const auto seeded_suffix = state->ple_token_id_suffix();
        require(std::equal(seeded_suffix.begin(), seeded_suffix.end(),
                           first_tokens.begin(), first_tokens.end()),
                "rejected re-seeding must preserve the existing token suffix");
        const std::array<std::uint32_t, 2> row_tokens{103u, 104u};
        WindowPleRows session_rows;
        WindowPleRows direct_rows;
        require(state->build_ple_rows(row_tokens, geometry, session_rows,
                                      error),
                "build rows from session-owned PLE token IDs: " + error);
        const auto applied_suffix = state->ple_token_id_suffix();
        require(build_window_ple_rows(row_tokens, applied_suffix, geometry,
                                      direct_rows, error),
                "build direct rows from the same applied suffix: " + error);
        require(session_rows.token_count() == direct_rows.token_count() &&
                    std::equal(session_rows.all_rows().begin(),
                               session_rows.all_rows().end(),
                               direct_rows.all_rows().begin(),
                               direct_rows.all_rows().end()),
                "session-owned row builder must match direct PLE construction");
        require(std::equal(applied_suffix.begin(), applied_suffix.end(),
                           first_tokens.begin(), first_tokens.end()),
                "building PLE rows must leave session token IDs unchanged");
        require(state->reset(*context, error), "reset session state: " + error);
        require_zero(allocation, "complete session state after reset");
        require_zero(first_layer.recurrent, "reset GDN recurrence");
        require_zero(first_layer.conv_history, "reset GDN conv history");
        require_zero(ple, "reset PLE history");
        require(state->ple_token_id_suffix().empty(),
                "successful session reset clears host PLE token history");

        const std::array<std::uint32_t, 2> pending_tokens{201u, 202u};
        require(state->initialize_ple_token_ids(pending_tokens, geometry, error),
                "seed token history before a refused reset: " + error);
        std::memset(allocation.contents, 0x5a, allocation.length);
        auto pending_use = context->make_command_buffer("pending session-state use", error);
        require(pending_use != nullptr,
                "create pending-use command buffer: " + error);
        require(state->track_use(*pending_use, error),
                "track pending session-state use: " + error);
        require(!state->reset(*context, error) &&
                    error.find("command buffers are pending") != std::string::npos,
                "reset must refuse while a tracked command buffer is pending");
        const auto retained_pending_tokens = state->ple_token_id_suffix();
        require(std::equal(retained_pending_tokens.begin(),
                           retained_pending_tokens.end(),
                           pending_tokens.begin(), pending_tokens.end()),
                "refused reset must preserve host PLE token history");
        require_byte_pattern(allocation, 0x5au,
                             "GPU state after a pending-use reset refusal");
        pending_use->discard();
        const std::array<std::uint32_t, 1> refused_reseed{203u};
        require(!state->initialize_ple_token_ids(refused_reseed, geometry,
                                                 error) &&
                    error.find("already initialized") != std::string::npos,
                "pending-use reset refusal must preserve the one-time seed gate");
        require(state->reset(*context, error),
                "reset after tracked command buffer discard: " + error);
        require_zero(allocation,
                     "GPU state after successful reset following discard");
        require(state->ple_token_id_suffix().empty(),
                "successful reset after discard clears host token history");

        auto rejected_use = context->make_command_buffer("rejected session-state use", error);
        require(rejected_use != nullptr,
                "create rejected-use command buffer: " + error);
        rejected_use->discard();
        require(!state->track_use(*rejected_use, error) &&
                    error.find("cannot register a completion handler") != std::string::npos,
                "discarded command-buffer registration must fail");
        require(state->reset(*context, error),
                "failed completion registration must release its tracked-use count: " + error);

        auto completed_use = context->make_command_buffer("completed session-state use", error);
        require(completed_use != nullptr,
                "create completed-use command buffer: " + error);
        require(state->track_use(*completed_use, error),
                "track completed session-state use: " + error);
        require(completed_use->commit(error),
                "commit completed-use command buffer: " + error);
        require(completed_use->wait_until_completed(error),
                "wait for completed-use command buffer: " + error);
        require(state->reset(*context, error),
                "reset after tracked command buffer completion: " + error);

        const std::array<std::uint32_t, 2> foreign_reset_tokens{301u, 302u};
        require(state->initialize_ple_token_ids(foreign_reset_tokens, geometry,
                                                error),
                "seed token history before foreign-context reset: " + error);
        std::memset(allocation.contents, 0xa6, allocation.length);
        auto other_context = Context::open(options, error);
        require(other_context != nullptr, "open second Metal context: " + error);
        require(!state->reset(*other_context, error) &&
                    error.find("different Metal Context") != std::string::npos,
                "reset must reject a foreign Context");
        const auto retained_tokens = state->ple_token_id_suffix();
        require(std::equal(retained_tokens.begin(), retained_tokens.end(),
                           foreign_reset_tokens.begin(),
                           foreign_reset_tokens.end()),
                "foreign-context reset must preserve host token history");
        require_byte_pattern(allocation, 0xa6u,
                             "GPU state after a foreign-context reset refusal");
        const std::array<std::uint32_t, 1> foreign_refused_reseed{303u};
        require(!state->initialize_ple_token_ids(foreign_refused_reseed,
                                                 geometry, error) &&
                    error.find("already initialized") != std::string::npos,
                "foreign-context reset refusal must preserve the one-time seed gate");

        std::unique_ptr<SessionStateStorage> rolled_back_state;
        BufferRef rolled_back_allocation;
        {
            auto outer_transaction = context->buffers().transaction();
            rolled_back_state = SessionStateStorage::open(*context, geometry, error);
            require(rolled_back_state != nullptr,
                    "open state inside an outer allocation transaction: " + error);
            rolled_back_allocation = rolled_back_state->allocation();
            require(context->buffers().owns(rolled_back_allocation),
                    "outer transaction must see the provisional state allocation");
        }
        require(!context->buffers().owns(rolled_back_allocation) &&
                    !rolled_back_state->allocation(),
                "outer rollback must invalidate the provisional state allocation");
        GdnSessionStateViews rolled_back_views;
        require(!rolled_back_state->gdn_layer(0u, rolled_back_views, error) &&
                    !rolled_back_views.recurrent &&
                    !rolled_back_state->ple_history(),
                "outer rollback must invalidate every published state view");
        require(!rolled_back_state->reset(*context, error) &&
                    error.find("no longer valid") != std::string::npos,
                "reset must fail closed after outer transaction rollback");

        std::cout << "metal_session_state_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_session_state_test: " << exception.what() << '\n';
        return 1;
    }
}
