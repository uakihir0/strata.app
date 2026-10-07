#pragma once

#include <array>
#include <cstdint>

namespace twin {

inline constexpr std::array<std::uint16_t, 32> kIQ2XSMoePackedWords = {
    0x0000u, 0x0211u, 0x7e3fu, 0xffffu, 0x221fu, 0x5440u, 0xaa80u,
    0xd5feu, 0x0402u, 0x188eu, 0x612bu, 0x9590u, 0x4605u, 0x52b7u,
    0xb741u, 0xf1c1u, 0x060bu, 0x3313u, 0x6f95u, 0xdff6u, 0x101du,
    0x445bu, 0xc7aau, 0xfbf0u, 0x2603u, 0x88ffu, 0xcd6fu, 0xfdcdu,
    0x3c09u, 0x5c49u, 0xb39du, 0xcaeeu,
};

inline constexpr std::array<std::uint8_t, 8> kIQ2XSMoeScales = {
    0x10u, 0x32u, 0x54u, 0x76u, 0x98u, 0xbau, 0xdcu, 0xfeu,
};

inline constexpr std::array<std::uint32_t, 8> kIQ2XSMoeOneHotPositions = {
    0u, 41u, 82u, 123u, 132u, 173u, 214u, 255u,
};

inline constexpr std::array<float, 8> kIQ2XSMoeOneHotAnswers = {
    127.0f, -635.0f, 4365.0f, -1905.0f,
    2159.0f, -8334.0f, -10715.0f, 3937.0f,
};

}  // namespace twin
