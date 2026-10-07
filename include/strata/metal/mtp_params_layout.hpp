// include/strata/metal/mtp_params_layout.hpp - device-free MTP parameter slots.
#pragma once

#include <cstdint>
#include <string>

namespace strata::metal {

inline constexpr std::uint32_t kMtpParamsSlotBytes = 256;
inline constexpr std::uint32_t kMtpParamsFixedSlots = 8;
inline constexpr std::uint32_t kMtpParamsPrefillGroupRows = 8;

struct MtpParamsLayout {
    std::uint32_t prefill_group_slots = 0;
    std::uint32_t slot_count = 0;
    std::uint64_t allocation_bytes = 0;

    bool slot_offset(std::uint32_t slot, std::uint64_t& offset) const noexcept;
};

/// Build the dedicated 256-byte MTP parameter array for a prefill chunk.
/// The first eight slots serve K and F0-F6; each following slot serves one
/// group of at most eight rows in that chunk. A zero chunk still reserves one
/// group for MTP catch-up after short-read decode windows.
bool make_mtp_params_layout(std::uint32_t prefill_chunk,
                            MtpParamsLayout& layout, std::string& error);

}  // namespace strata::metal
