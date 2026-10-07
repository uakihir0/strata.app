#pragma once

#include "../../../src/metal/kernels/common/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace twin {

static_assert(sizeof(block_iq4_xs) == 136u);
static_assert(offsetof(block_iq4_xs, d) == 0u);
static_assert(offsetof(block_iq4_xs, scales_h) == 2u);
static_assert(offsetof(block_iq4_xs, scales_l) == 4u);
static_assert(offsetof(block_iq4_xs, qs) == 8u);

// Literal raw fields for a single IQ4_XS block. Part 1 has scale code 47;
// qs[25].low is codebook value 113 and maps to activation position 41.
inline constexpr std::uint16_t kIQ4XSMoeD = 0x3c00u;
inline constexpr std::uint16_t kIQ4XSMoeScalesH = 0x0008u;
inline constexpr std::array<std::uint8_t, 4> kIQ4XSMoeScalesL = {
    0xf0u, 0x00u, 0x00u, 0x00u};
inline constexpr std::array<std::uint8_t, 128> kIQ4XSMoeQs = [] {
    std::array<std::uint8_t, 128> bytes{};
    bytes[25] = 0x0fu;
    return bytes;
}();

inline constexpr std::uint32_t kIQ4XSMoeOneHotPosition = 41u;
inline constexpr float kIQ4XSMoeOneHotAnswer = 215265.0f;

inline block_iq4_xs iq4_xs_moe_fixture() {
    block_iq4_xs weight{};
    weight.d = kIQ4XSMoeD;
    weight.scales_h = kIQ4XSMoeScalesH;
    std::copy(kIQ4XSMoeScalesL.begin(), kIQ4XSMoeScalesL.end(),
              weight.scales_l);
    std::copy(kIQ4XSMoeQs.begin(), kIQ4XSMoeQs.end(), weight.qs);
    return weight;
}

}  // namespace twin
