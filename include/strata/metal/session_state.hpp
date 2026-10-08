// include/strata/metal/session_state.hpp - per-session recurrent Metal state.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/geometry.hpp"
#include "strata/metal/window_ple_rows.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace strata::metal {

inline constexpr std::uint32_t kSessionModelLayers = 48;
inline constexpr std::uint32_t kSessionQsaLayers = 12;
inline constexpr std::uint32_t kSessionKvHeads = 2;
inline constexpr std::uint32_t kSessionHeadWidth = 256;
inline constexpr std::uint32_t kSessionGdnLayers =
    kSessionModelLayers - kSessionQsaLayers;
inline constexpr std::uint32_t kSessionEmbeddingWidth = 2'560;
inline constexpr std::uint32_t kSessionHyperconnections = 4;
inline constexpr std::uint32_t kSessionVocabularySize = 248'320;
inline constexpr std::uint32_t kSessionGdnStateElements = 48u * 128u * 128u;
inline constexpr std::uint32_t kSessionGdnConvHistoryElements =
    kSessionEmbeddingWidth * kSessionHyperconnections * 3u;
inline constexpr std::uint32_t kSessionPleHistoryElements =
    9u * kSessionEmbeddingWidth * kSessionHyperconnections;

struct SessionStateRegion {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

struct GdnSessionStateLayout {
    SessionStateRegion recurrent;
    SessionStateRegion conv_history;
};

/// Byte layout for the supported 48-layer model's mutable recurrent state.
/// QSA KV/indexer storage remains owned by KvStore.
struct SessionStateLayout {
    std::array<GdnSessionStateLayout, kSessionGdnLayers> gdn{};
    SessionStateRegion ple_history;
    std::size_t allocation_bytes = 0;
};

/// Build the checked layout without a Metal device or allocation.
bool make_session_state_layout(const Geometry& geometry, SessionStateLayout& layout,
                               std::string& error);

struct GdnSessionStateViews {
    BufferRef recurrent;
    BufferRef conv_history;
};

/// Own one zero-initialized Shared allocation for a single live sequence's
/// GDN recurrence, GDN convolution history and PLE history.
///
/// Context must outlive this object. Call `track_use()` before encoding any
/// command buffer that reads or writes one of its views. `reset()` fails while
/// a tracked command buffer is pending and clears Shared memory only after
/// every tracked command buffer completes or is discarded.
class SessionStateStorage {
public:
    static std::unique_ptr<SessionStateStorage> open(Context& context,
                                                     const Geometry& geometry,
                                                     std::string& error);
    ~SessionStateStorage();
    SessionStateStorage(const SessionStateStorage&) = delete;
    SessionStateStorage& operator=(const SessionStateStorage&) = delete;

    const SessionStateLayout& layout() const noexcept;
    BufferRef allocation() const noexcept;
    bool gdn_layer(std::uint32_t ordinal, GdnSessionStateViews& views,
                   std::string& error) const;
    BufferRef ple_history() const noexcept;
    /// Serialize token-history calls with this sequence's Window, commit and
    /// reset lifecycle. The returned suffix view expires on its next mutation.
    /// Initialize once per reset from IDs already applied to model state.
    /// Row building and accepted-prefix updates fail until this succeeds.
    bool initialize_ple_token_ids(
        std::span<const std::uint32_t> applied_prefix,
        const Geometry& geometry, std::string& error);
    std::span<const std::uint32_t> ple_token_id_suffix() const noexcept;
    /// Build without mutation; serialize with this sequence's Window lifecycle.
    bool build_ple_rows(std::span<const std::uint32_t> tokens,
                        const Geometry& geometry, WindowPleRows& output,
                        std::string& error) const;
    /// Call only after the corresponding GPU state commit succeeds.
    bool commit_accepted_ple_tokens(
        std::span<const std::uint32_t> window_tokens,
        std::size_t accepted_count, const Geometry& geometry,
        std::string& error);
    bool track_use(CommandBuffer& command_buffer, std::string& error);
    /// Reset GPU and host state together; this is the only reset path for
    /// token IDs owned by this session.
    bool reset(Context& context, std::string& error);

private:
    struct UseTracker;
    SessionStateStorage(Context& context, SessionStateLayout layout,
                        BufferRef allocation);

    Context* context_ = nullptr;
    SessionStateLayout layout_;
    BufferRef allocation_;
    WindowPleTokenHistory ple_token_ids_;
    bool ple_token_ids_initialized_ = false;
    std::shared_ptr<UseTracker> use_tracker_;
};

}  // namespace strata::metal
