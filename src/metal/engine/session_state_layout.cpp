// Device-free checked layout arithmetic for per-session GDN and PLE state.
#include "strata/metal/session_state.hpp"

#include <limits>
#include <utility>

namespace strata::metal {
namespace {

constexpr std::size_t kSessionAlignment = 256;

bool append_region(std::size_t bytes, std::size_t& cursor,
                   SessionStateRegion& region) {
    if (bytes == 0) return false;
    const std::size_t remainder = cursor % kSessionAlignment;
    if (remainder != 0) {
        const std::size_t padding = kSessionAlignment - remainder;
        if (padding > std::numeric_limits<std::size_t>::max() - cursor) return false;
        cursor += padding;
    }
    if (bytes > std::numeric_limits<std::size_t>::max() - cursor) return false;
    region = {cursor, bytes};
    cursor += bytes;
    return true;
}

}  // namespace

bool make_session_state_layout(const Geometry& geometry, SessionStateLayout& layout,
                               std::string& error) {
    layout = {};
    error.clear();
    if (!is_qwen4exp_runtime_geometry(geometry)) {
        error = "session state requires qwen4exp geometry with 48 layers, 12 QSA layers, "
                "2 KV heads, 256-wide heads, 2560 embedding width and vocabulary 248320";
        return false;
    }

    constexpr std::size_t recurrent_bytes =
        static_cast<std::size_t>(kSessionGdnStateElements) * sizeof(float);
    constexpr std::size_t conv_history_bytes =
        static_cast<std::size_t>(kSessionGdnConvHistoryElements) * sizeof(float);
    constexpr std::size_t ple_history_bytes =
        static_cast<std::size_t>(kSessionPleHistoryElements) * sizeof(float);

    SessionStateLayout candidate;
    std::size_t cursor = 0;
    for (GdnSessionStateLayout& layer : candidate.gdn) {
        if (!append_region(recurrent_bytes, cursor, layer.recurrent) ||
            !append_region(conv_history_bytes, cursor, layer.conv_history)) {
            error = "GDN session-state layout overflows";
            return false;
        }
    }
    if (!append_region(ple_history_bytes, cursor, candidate.ple_history)) {
        error = "PLE session-state layout overflows";
        return false;
    }
    candidate.allocation_bytes = cursor;
    layout = std::move(candidate);
    error.clear();
    return true;
}

}  // namespace strata::metal
