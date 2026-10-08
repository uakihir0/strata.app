// src/metal/engine/session_state.cpp - checked per-sequence recurrent state storage.
#include "strata/metal/session_state.hpp"

#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

namespace strata::metal {
namespace {

bool region_view(BufferRef allocation, SessionStateRegion region,
                 BufferRef& view) {
    view = {};
    if (region.bytes == 0 || !allocation ||
        region.offset > allocation.length ||
        region.bytes > allocation.length - region.offset) {
        return false;
    }
    view = allocation.slice(region.offset, region.bytes);
    return static_cast<bool>(view);
}

}  // namespace

struct SessionStateStorage::UseTracker {
    std::mutex mutex;
    std::size_t outstanding = 0;
};

SessionStateStorage::SessionStateStorage(Context& context, SessionStateLayout layout,
                                         BufferRef allocation)
    : context_(&context),
      layout_(std::move(layout)),
      allocation_(allocation),
      use_tracker_(std::make_shared<UseTracker>()) {}

SessionStateStorage::~SessionStateStorage() = default;

std::unique_ptr<SessionStateStorage> SessionStateStorage::open(
    Context& context, const Geometry& geometry, std::string& error) {
    error.clear();
    SessionStateLayout layout;
    if (!make_session_state_layout(geometry, layout, error)) return nullptr;

    const DeviceFacts& facts = context.device_facts();
    if (facts.max_buffer_length == 0 ||
        layout.allocation_bytes > facts.max_buffer_length) {
        error = "session state exceeds the Metal device's maxBufferLength";
        return nullptr;
    }

    auto transaction = context.buffers().transaction();
    BufferRef allocation = context.buffers().allocate(
        "sequence GDN and PLE state", layout.allocation_bytes, error);
    if (!allocation) return nullptr;
    if (!context.buffers().owns(allocation) ||
        allocation.length != layout.allocation_bytes ||
        allocation.contents == nullptr) {
        error = "Metal did not return a valid owned session-state allocation";
        return nullptr;
    }

    std::memset(allocation.contents, 0, layout.allocation_bytes);
    if (!context.buffers().commit_residency(error)) return nullptr;

    std::unique_ptr<SessionStateStorage> owner;
    try {
        owner.reset(new SessionStateStorage(context, std::move(layout), allocation));
    } catch (const std::bad_alloc&) {
        error = "cannot allocate session-state owner";
        return nullptr;
    }
    transaction.commit();
    error.clear();
    return owner;
}

const SessionStateLayout& SessionStateStorage::layout() const noexcept {
    return layout_;
}

BufferRef SessionStateStorage::allocation() const noexcept {
    return allocation_;
}

bool SessionStateStorage::gdn_layer(std::uint32_t ordinal,
                                    GdnSessionStateViews& views,
                                    std::string& error) const {
    error.clear();
    views = {};
    if (ordinal >= layout_.gdn.size()) {
        error = "GDN session-state layer ordinal is out of range";
        return false;
    }
    const GdnSessionStateLayout& regions = layout_.gdn[ordinal];
    if (!region_view(allocation_, regions.recurrent, views.recurrent) ||
        !region_view(allocation_, regions.conv_history, views.conv_history)) {
        views = {};
        error = "GDN session-state view is no longer valid";
        return false;
    }
    return true;
}

BufferRef SessionStateStorage::ple_history() const noexcept {
    BufferRef view;
    if (!region_view(allocation_, layout_.ple_history, view)) return {};
    return view;
}

bool SessionStateStorage::initialize_ple_token_ids(
    std::span<const std::uint32_t> applied_prefix, const Geometry& geometry,
    std::string& error) {
    if (ple_token_ids_initialized_) {
        error = "PLE token history is already initialized; reset the session before reseeding";
        return false;
    }
    if (!ple_token_ids_.replace_committed_history(applied_prefix, geometry,
                                                 error)) {
        return false;
    }
    ple_token_ids_initialized_ = true;
    error.clear();
    return true;
}

std::span<const std::uint32_t>
SessionStateStorage::ple_token_id_suffix() const noexcept {
    return ple_token_ids_.committed_suffix();
}

bool SessionStateStorage::build_ple_rows(
    std::span<const std::uint32_t> tokens, const Geometry& geometry,
    WindowPleRows& output, std::string& error) const {
    if (!ple_token_ids_initialized_) {
        error = "PLE token history must be initialized before building Window rows";
        return false;
    }
    return ple_token_ids_.build_rows(tokens, geometry, output, error);
}

bool SessionStateStorage::commit_accepted_ple_tokens(
    std::span<const std::uint32_t> window_tokens, std::size_t accepted_count,
    const Geometry& geometry, std::string& error) {
    if (!ple_token_ids_initialized_) {
        error = "PLE token history must be initialized before committing Window tokens";
        return false;
    }
    return ple_token_ids_.commit_accepted_prefix(window_tokens, accepted_count,
                                                 geometry, error);
}

bool SessionStateStorage::track_use(CommandBuffer& command_buffer,
                                    std::string& error) {
    error.clear();
    if (context_ == nullptr || !command_buffer.belongs_to(*context_)) {
        error = "session state and command buffer belong to different Metal Contexts";
        return false;
    }
    if (!context_->buffers().owns(allocation_)) {
        error = "session-state allocation is no longer valid";
        return false;
    }

    const std::shared_ptr<UseTracker> tracker = use_tracker_;
    {
        std::lock_guard<std::mutex> lock(tracker->mutex);
        if (tracker->outstanding == std::numeric_limits<std::size_t>::max()) {
            error = "session-state command-buffer use count is exhausted";
            return false;
        }
        ++tracker->outstanding;
    }

    struct RegistrationRollback {
        std::shared_ptr<UseTracker> tracker;
        bool active = true;

        ~RegistrationRollback() {
            if (!active) return;
            std::lock_guard<std::mutex> lock(tracker->mutex);
            if (tracker->outstanding != 0) --tracker->outstanding;
        }
    } rollback{tracker};

    try {
        if (!command_buffer.add_completion_handler(
                [tracker](bool) {
                    std::lock_guard<std::mutex> lock(tracker->mutex);
                    if (tracker->outstanding != 0) --tracker->outstanding;
                },
                error)) {
            return false;
        }
    } catch (const std::exception& exception) {
        error = "cannot register session-state completion handler: ";
        error += exception.what();
        return false;
    } catch (...) {
        error = "cannot register session-state completion handler";
        return false;
    }

    rollback.active = false;
    error.clear();
    return true;
}

bool SessionStateStorage::reset(Context& context, std::string& error) {
    error.clear();
    if (&context != context_) {
        error = "session state belongs to a different Metal Context";
        return false;
    }
    std::lock_guard<std::mutex> use_lock(use_tracker_->mutex);
    if (use_tracker_->outstanding != 0) {
        error = "cannot reset session state while tracked command buffers are pending";
        return false;
    }
    if (!context.buffers().owns(allocation_) ||
        allocation_.length != layout_.allocation_bytes ||
        allocation_.contents == nullptr) {
        error = "session-state allocation is no longer valid";
        return false;
    }

    std::memset(allocation_.contents, 0, layout_.allocation_bytes);
    ple_token_ids_.reset();
    ple_token_ids_initialized_ = false;
    error.clear();
    return true;
}

}  // namespace strata::metal
