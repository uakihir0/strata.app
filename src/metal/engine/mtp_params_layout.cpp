// src/metal/engine/mtp_params_layout.cpp - checked MTP parameter slot sizing.
#include "strata/metal/mtp_params_layout.hpp"
#include "../kernels/common/types.h"

#include <limits>

namespace strata::metal {

static_assert(sizeof(MtpParams) <= kMtpParamsSlotBytes,
              "MtpParams must fit in its documented 256-byte slot");
static_assert(alignof(MtpParams) <= kMtpParamsSlotBytes,
              "MtpParams alignment must fit in its documented slot");

bool MtpParamsLayout::slot_offset(std::uint32_t slot,
                                  std::uint64_t& offset) const noexcept {
    offset = 0;
    if (slot >= slot_count) return false;
    offset = static_cast<std::uint64_t>(slot) * kMtpParamsSlotBytes;
    return true;
}

bool make_mtp_params_layout(std::uint32_t prefill_chunk,
                            MtpParamsLayout& layout, std::string& error) {
    layout = {};
    error.clear();

    const std::uint64_t requested_group_slots =
        static_cast<std::uint64_t>(prefill_chunk) /
            kMtpParamsPrefillGroupRows +
        (prefill_chunk % kMtpParamsPrefillGroupRows != 0);
    const std::uint64_t group_slots =
        requested_group_slots == 0 ? 1 : requested_group_slots;
    const std::uint64_t slots = kMtpParamsFixedSlots + group_slots;
    if (slots > std::numeric_limits<std::uint32_t>::max() ||
        slots > std::numeric_limits<std::uint64_t>::max() /
                    kMtpParamsSlotBytes) {
        error = "MTP parameter slot count overflows";
        return false;
    }

    layout.prefill_group_slots = static_cast<std::uint32_t>(group_slots);
    layout.slot_count = static_cast<std::uint32_t>(slots);
    layout.allocation_bytes = slots * kMtpParamsSlotBytes;
    return true;
}

}  // namespace strata::metal
