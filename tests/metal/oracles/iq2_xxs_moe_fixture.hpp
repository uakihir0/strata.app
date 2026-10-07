#pragma once

#include "../../../src/metal/kernels/common/types.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace twin {

static_assert(sizeof(block_iq2_xxs) == 66u);
static_assert(offsetof(block_iq2_xxs, qs) == 2u);

// Literal block and one-hot answers shared with the standalone IQ2_XXS
// decoder fixture in metal_mv_test.mm.
inline constexpr std::array<std::uint16_t, 32> kIQ2XXSMoePackedWords = {
    0x1100u, 0xff3fu, 0x7f81u, 0x0555u, 0x401fu, 0xfe80u, 0x1212u, 0x1ff2u,
    0x2a02u, 0xc863u, 0x8080u, 0x3fc0u, 0x5305u, 0xc179u, 0xe535u, 0x52b8u,
    0x4b0bu, 0xde95u, 0xc07fu, 0x7b4cu, 0x5b1du, 0xf0aau, 0x64a6u, 0x907bu,
    0x3703u, 0xcd6fu, 0x1111u, 0xbff1u, 0x4909u, 0xee9du, 0x6b2du, 0xffdau,
};

inline constexpr std::array<std::uint32_t, 8> kIQ2XXSMoeOneHotPositions = {
    0u, 41u, 82u, 123u, 132u, 173u, 214u, 255u,
};

inline constexpr std::array<float, 8> kIQ2XXSMoeOneHotAnswers = {
    -127.0f, 1190.0f, 889.0f, 4365.0f,
    -1905.0f, 7540.0f, -15700.0f, 21161.0f,
};

}  // namespace twin
