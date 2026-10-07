#pragma once

#include <array>
#include <cstdint>

namespace twin {

inline constexpr std::array<std::uint8_t, 8> kIQ3XXSMoeIndices = {
    0u, 1u, 5u, 200u, 64u, 17u, 5u, 255u,
};

inline constexpr std::array<std::uint8_t, 4> kIQ3XXSMoeAux = {
    0x80u, 0x00u, 0xb0u, 0xfau,
};

inline constexpr std::array<std::uint32_t, 32> kIQ3XXSMoeOneHotPositions = {
    224u, 225u, 226u, 227u, 228u, 229u, 230u, 231u,
    232u, 233u, 234u, 235u, 236u, 237u, 238u, 239u,
    240u, 241u, 242u, 243u, 244u, 245u, 246u, 247u,
    248u, 249u, 250u, 251u, 252u, 253u, 254u, 255u,
};

inline constexpr std::array<float, 32> kIQ3XXSMoeOneHotAnswers = {
    3937.0f, 3937.0f, 3937.0f, 3937.0f,
    19685.0f, 3937.0f, 3937.0f, 3937.0f,
    -61023.0f, 11811.0f, 3937.0f, 3937.0f,
    3937.0f, 61023.0f, 3937.0f, -43307.0f,
    3937.0f, 27559.0f, 3937.0f, 11811.0f,
    43307.0f, 19685.0f, -11811.0f, -3937.0f,
    -61023.0f, 11811.0f, -3937.0f, 3937.0f,
    -3937.0f, 27559.0f, -51181.0f, 61023.0f,
};

}  // namespace twin
