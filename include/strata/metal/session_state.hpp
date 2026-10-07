// include/strata/metal/session_state.hpp - per-session recurrent Metal state.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
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
    bool track_use(CommandBuffer& command_buffer, std::string& error);
    bool reset(Context& context, std::string& error);

private:
    struct UseTracker;
    SessionStateStorage(Context& context, SessionStateLayout layout,
                        BufferRef allocation);

    Context* context_ = nullptr;
    SessionStateLayout layout_;
    BufferRef allocation_;
    std::shared_ptr<UseTracker> use_tracker_;
};

}  // namespace strata::metal
