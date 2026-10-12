// GPU twin for isolated Prefill MoE Q2_0/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S gate/up tiles (docs/mac/04 §6.6).
#include "metallib_sha256.inc"

#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_moe_expert_dispatch.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"
#include "strata/metal/prefill.hpp"
#include "../../src/metal/kernels/common/types.h"
#include "oracles/iq4_xs_moe_fixture.hpp"
#include "oracles/iq2_xxs_moe_fixture.hpp"
#include "oracles/iq2_xxs_mv.hpp"
#include "oracles/iq2_xs_moe_fixture.hpp"
#include "oracles/iq2_xs_mv.hpp"
#include "oracles/iq2_s_moe_fixture.hpp"
#include "oracles/iq2_s_mv.hpp"
#include "twin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kHidden = 640u;
constexpr std::uint32_t kExperts = 256u;
constexpr std::uint32_t kSecondExpert = 3u;
constexpr std::uint32_t kLayer = 2u;
constexpr std::size_t kQ2BlocksPerRow = 40u;
constexpr std::size_t kQ2BlockBytes = 18u;
constexpr std::size_t kIQ4XSBlocksPerRow = 10u;
constexpr std::size_t kIQ4XSBlockBytes = 136u;
constexpr std::size_t kIQ2XXSBlocksPerRow = 10u;
constexpr std::size_t kIQ2XXSBlockBytes = 66u;
constexpr std::size_t kIQ2XSBlocksPerRow = 10u;
constexpr std::size_t kIQ2XSBlockBytes = 74u;
constexpr std::size_t kIQ2SBlocksPerRow = 10u;
constexpr std::size_t kIQ2SBlockBytes = 82u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint8_t kGuardPoison = 0xffu;

struct PfPlan {
    std::uint32_t n_groups = 0u;
    std::uint32_t n_entries = 0u;
    std::uint32_t n_miss = 0u;
    std::uint32_t skip = 0u;
    std::uint32_t n_tiles = 0u;
    std::uint32_t bad = 0u;
    std::uint32_t pad[2]{};
    std::int32_t grp_e[512]{};
    std::int32_t grp_res[512]{};
    std::int32_t grp_start[513]{};
};
static_assert(sizeof(PfPlan) == kPrefillMoePlanBytes);

struct Tile {
    std::uint32_t group;
    std::uint32_t first_entry;
};
static_assert(sizeof(Tile) == 8u);

enum class WeightFormat { q2_0, iq4_xs, iq2_xxs, iq2_xs, iq2_s };

std::string_view weight_format_name(WeightFormat format) {
    switch (format) {
        case WeightFormat::q2_0: return "Q2_0";
        case WeightFormat::iq4_xs: return "IQ4_XS";
        case WeightFormat::iq2_xxs: return "IQ2_XXS";
        case WeightFormat::iq2_xs: return "IQ2_XS";
        case WeightFormat::iq2_s: return "IQ2_S";
    }
    return "unknown";
}

std::size_t weight_format_expert_bytes(WeightFormat format) {
    switch (format) {
        case WeightFormat::q2_0:
            return kHidden * kQ2BlocksPerRow * kQ2BlockBytes;
        case WeightFormat::iq4_xs:
            return kHidden * kIQ4XSBlocksPerRow * kIQ4XSBlockBytes;
        case WeightFormat::iq2_xxs:
            return kHidden * kIQ2XXSBlocksPerRow * kIQ2XXSBlockBytes;
        case WeightFormat::iq2_xs:
            return kHidden * kIQ2XSBlocksPerRow * kIQ2XSBlockBytes;
        case WeightFormat::iq2_s:
            return kHidden * kIQ2SBlocksPerRow * kIQ2SBlockBytes;
    }
    return 0u;
}

struct PrefillMoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(PrefillMoeExpertAddress) == 24u);

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

template <class T>
std::span<T> typed(twin::GuardedBuffer& buffer) {
    return {reinterpret_cast<T*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(T)};
}

template <class T>
std::span<const T> typed(const twin::GuardedBuffer& buffer) {
    return {reinterpret_cast<const T*>(buffer.payload().data()),
            buffer.payload().size() / sizeof(T)};
}

bool guards_intact(const twin::GuardedBuffer& buffer) {
    const auto allocation = buffer.allocation();
    const auto payload = buffer.payload();
    const std::size_t prefix = twin::GuardedBuffer::kGuardBytes;
    const std::size_t suffix = prefix + payload.size();
    return allocation.size() >= suffix &&
           std::all_of(allocation.begin(), allocation.begin() + prefix,
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       }) &&
           std::all_of(allocation.begin() + suffix, allocation.end(),
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       });
}

bool tail_is_poison(const twin::GuardedBuffer& buffer,
                    std::size_t active_bytes) {
    const auto payload = buffer.payload();
    return active_bytes <= payload.size() &&
           std::all_of(payload.begin() + active_bytes, payload.end(),
                       [](std::uint8_t byte) { return byte == kPoison; });
}

std::uint16_t half(float value) {
    return strata::kernels::f16_from_f32(value);
}

float f32(std::uint16_t value) {
    return strata::kernels::f32_from_f16(value);
}

std::uint16_t output_half(double value) {
    const double saturated = std::clamp(
        value, -static_cast<double>(65504.0f),
        static_cast<double>(65504.0f));
    return half(static_cast<float>(saturated));
}

std::uint16_t ordered_half(std::uint16_t bits) {
    return (bits & 0x8000u) != 0u
               ? static_cast<std::uint16_t>(~bits)
               : static_cast<std::uint16_t>(bits | 0x8000u);
}

std::uint16_t half_ulp_distance(std::uint16_t left, std::uint16_t right) {
    const std::uint16_t a = ordered_half(left);
    const std::uint16_t b = ordered_half(right);
    return a > b ? static_cast<std::uint16_t>(a - b)
                 : static_cast<std::uint16_t>(b - a);
}

void store_code(block_q2_0& block, std::size_t k, std::uint8_t code) {
    const std::size_t byte = k / 4u;
    const std::size_t shift = 2u * (k % 4u);
    block.qs[byte] = static_cast<std::uint8_t>(
        (block.qs[byte] & static_cast<std::uint8_t>(~(3u << shift))) |
        static_cast<std::uint8_t>(code << shift));
}

void store_iq4_xs_code(block_iq4_xs& block, std::size_t k,
                       std::uint8_t code) {
    const std::size_t group = k / 32u;
    const std::size_t within_group = k % 32u;
    const std::size_t byte =
        16u * group + (within_group < 16u ? within_group
                                          : within_group - 16u);
    const std::size_t shift = within_group < 16u ? 0u : 4u;
    block.qs[byte] = static_cast<std::uint8_t>(
        (block.qs[byte] & static_cast<std::uint8_t>(~(0x0fu << shift))) |
        static_cast<std::uint8_t>(code << shift));
}

void fill_q2_weights(twin::GuardedBuffer& buffer, std::uint32_t role,
                     std::uint32_t expert) {
    auto blocks = typed<block_q2_0>(buffer);
    for (std::uint32_t row = 0u; row < kHidden; ++row) {
        for (std::size_t block_index = 0u;
             block_index < kQ2BlocksPerRow; ++block_index) {
            block_q2_0& block =
                blocks[static_cast<std::size_t>(row) * kQ2BlocksPerRow +
                       block_index];
            block.d = 0x2800u;  // F16 0.03125.
            std::fill(std::begin(block.qs), std::end(block.qs), 0u);
            for (std::size_t k = 0u; k < 64u; ++k) {
                std::uint8_t code = 0u;
                if (expert == 0u && role == 0u) {
                    code = 0u;  // Negative gate weights.
                } else if (expert == 0u && role == 1u) {
                    code = 3u;  // Positive up weights.
                } else {
                    code = static_cast<std::uint8_t>(
                        (row + block_index + k + role + expert) % 4u);
                }
                store_code(block, k, code);
            }
        }
    }
}

void fill_iq4_xs_weights(twin::GuardedBuffer& buffer, std::uint32_t role,
                         std::uint32_t expert) {
    auto blocks = typed<block_iq4_xs>(buffer);
    for (std::uint32_t row = 0u; row < kHidden; ++row) {
        for (std::size_t block_index = 0u;
             block_index < kIQ4XSBlocksPerRow; ++block_index) {
            block_iq4_xs& block =
                blocks[static_cast<std::size_t>(row) * kIQ4XSBlocksPerRow +
                       block_index];
            block.d = 0x1400u;  // F16 2^-10.
            std::fill(std::begin(block.qs), std::end(block.qs), 0u);
            std::fill(std::begin(block.scales_l),
                      std::end(block.scales_l), 0u);
            block.scales_h = 0u;
            for (std::size_t group = 0u; group < 8u; ++group) {
                std::uint32_t scale_code = 0u;
                if (expert == 0u) {
                    scale_code =
                        33u + (row * 3u + block_index * 5u + group * 7u +
                               role * 11u) %
                                  31u;
                } else {
                    scale_code =
                        (row * 13u + block_index * 17u + group * 19u +
                         role * 23u + expert * 29u) &
                        0x3fu;
                }
                const std::uint32_t low_shift =
                    4u * static_cast<std::uint32_t>(group % 2u);
                block.scales_l[group / 2u] |=
                    static_cast<std::uint8_t>((scale_code & 0x0fu)
                                              << low_shift);
                block.scales_h |= static_cast<std::uint16_t>(
                    ((scale_code >> 4u) & 0x03u) << (2u * group));
            }
            for (std::size_t k = 0u; k < 256u; ++k) {
                std::uint8_t code = 0u;
                if (expert == 0u && role == 0u) {
                    code = 0u;  // Negative gate weights.
                } else if (expert == 0u && role == 1u) {
                    code = 15u;  // Positive up weights.
                } else {
                    code = static_cast<std::uint8_t>(
                        (row * 7u + block_index * 11u + (k % 16u) * 13u +
                         (k / 16u) * 3u + role * 3u + expert * 5u) %
                        16u);
                }
                store_iq4_xs_code(block, k, code);
            }
        }
    }
}

void fill_iq2_xxs_weights(twin::GuardedBuffer& buffer, std::uint32_t role,
                          std::uint32_t expert) {
    auto blocks = typed<block_iq2_xxs>(buffer);
    for (std::uint32_t row = 0u; row < kHidden; ++row) {
        for (std::size_t block_index = 0u;
             block_index < kIQ2XXSBlocksPerRow; ++block_index) {
            block_iq2_xxs& block =
                blocks[static_cast<std::size_t>(row) * kIQ2XXSBlocksPerRow +
                       block_index];
            if (expert == 0u && row == 0u) {
                block.d = 0u;
                std::fill(std::begin(block.qs), std::end(block.qs), 0u);
                if (block_index == 0u) {
                    block.d = 0x2a00u;  // F16 0.046875.
                    std::copy(twin::kIQ2XXSMoePackedWords.begin(),
                              twin::kIQ2XXSMoePackedWords.end(), block.qs);
                }
                continue;
            }
            block.d = 0x2a00u;  // F16 0.046875.
            for (std::uint32_t part = 0u; part < 8u; ++part) {
                std::uint32_t indices = 0u;
                std::uint32_t auxiliary = 0u;
                for (std::uint32_t code_slot = 0u; code_slot < 4u;
                     ++code_slot) {
                    const std::uint32_t code =
                        (row * 97u + block_index * 23u + part * 41u +
                         code_slot * 67u + role * 31u + expert * 19u) &
                        0xffu;
                    indices |= code << (8u * code_slot);
                    const std::uint32_t signs =
                        (row * 13u + block_index * 17u + part * 29u +
                         code_slot * 37u + role * 43u + expert * 53u) &
                        0x7fu;
                    auxiliary |= signs << (7u * code_slot);
                }
                const std::uint32_t scale_code =
                    (row * 7u + block_index * 11u + part * 13u +
                     role * 5u + expert * 3u) &
                    0x0fu;
                auxiliary |= scale_code << 28u;
                block.qs[4u * part] =
                    static_cast<std::uint16_t>(indices & 0xffffu);
                block.qs[4u * part + 1u] =
                    static_cast<std::uint16_t>(indices >> 16u);
                block.qs[4u * part + 2u] =
                    static_cast<std::uint16_t>(auxiliary & 0xffffu);
                block.qs[4u * part + 3u] =
                    static_cast<std::uint16_t>(auxiliary >> 16u);
            }
        }
    }
}

void fill_iq2_xs_weights(twin::GuardedBuffer& buffer, std::uint32_t role,
                         std::uint32_t expert) {
    auto blocks = typed<block_iq2_xs>(buffer);
    for (std::uint32_t row = 0u; row < kHidden; ++row) {
        for (std::size_t block_index = 0u;
             block_index < kIQ2XSBlocksPerRow; ++block_index) {
            block_iq2_xs& block =
                blocks[static_cast<std::size_t>(row) * kIQ2XSBlocksPerRow +
                       block_index];
            if (expert == 0u && row == 0u) {
                block.d = 0u;
                std::fill(std::begin(block.qs), std::end(block.qs), 0u);
                std::fill(std::begin(block.scales), std::end(block.scales), 0u);
                if (block_index == 0u) {
                    block.d = 0x2a00u;  // F16 0.046875.
                    std::copy(twin::kIQ2XSMoePackedWords.begin(),
                              twin::kIQ2XSMoePackedWords.end(), block.qs);
                    std::copy(twin::kIQ2XSMoeScales.begin(),
                              twin::kIQ2XSMoeScales.end(), block.scales);
                }
                continue;
            }
            block.d = 0x2a00u;  // F16 0.046875.
            for (std::uint32_t part = 0u; part < 8u; ++part) {
                const std::uint32_t low_scale =
                    (row * 7u + block_index * 3u + part * 5u + role * 11u +
                     expert * 13u) &
                    0x0fu;
                const std::uint32_t high_scale =
                    (row * 3u + block_index * 7u + part * 9u + role * 5u +
                     expert * 17u) &
                    0x0fu;
                block.scales[part] = static_cast<std::uint8_t>(
                    low_scale | (high_scale << 4u));
                for (std::uint32_t code_slot = 0u; code_slot < 4u;
                     ++code_slot) {
                    const std::uint32_t ordinal =
                        (((row * kIQ2XSBlocksPerRow + block_index) * 8u +
                          part) *
                         4u) +
                        code_slot;
                    const std::uint32_t grid =
                        (ordinal * 37u + role * 31u + expert * 19u) & 0x1ffu;
                    const std::uint32_t sign =
                        (ordinal * 29u + role * 43u + expert * 53u) & 0x7fu;
                    block.qs[4u * part + code_slot] =
                        static_cast<std::uint16_t>(grid | (sign << 9u));
                }
            }
        }
    }
}

void fill_iq2_s_weights(twin::GuardedBuffer& buffer, std::uint32_t role,
                        std::uint32_t expert) {
    auto blocks = typed<block_iq2_s>(buffer);
    for (std::uint32_t row = 0u; row < kHidden; ++row) {
        for (std::size_t block_index = 0u;
             block_index < kIQ2SBlocksPerRow; ++block_index) {
            block_iq2_s& block =
                blocks[static_cast<std::size_t>(row) * kIQ2SBlocksPerRow +
                       block_index];
            if (expert == 0u && row == 0u) {
                block.d = 0u;
                std::fill(std::begin(block.qs), std::end(block.qs), 0u);
                std::fill(std::begin(block.qh), std::end(block.qh), 0u);
                std::fill(std::begin(block.scales), std::end(block.scales), 0u);
                if (block_index == 0u) {
                    block.d = 0x2a00u;  // F16 0.046875.
                    std::copy(twin::kIQ2SMoePackedBytes.begin(),
                              twin::kIQ2SMoePackedBytes.end(), block.qs);
                    std::copy(twin::kIQ2SMoeHighCodes.begin(),
                              twin::kIQ2SMoeHighCodes.end(), block.qh);
                    std::copy(twin::kIQ2SMoeScales.begin(),
                              twin::kIQ2SMoeScales.end(), block.scales);
                }
                continue;
            }
            block.d = 0x2a00u;  // F16 0.046875.
            for (std::uint32_t part = 0u; part < 8u; ++part) {
                const std::uint32_t low_scale =
                    (row * 7u + block_index * 3u + part * 5u + role * 11u +
                     expert * 13u) &
                    0x0fu;
                const std::uint32_t high_scale =
                    (row * 3u + block_index * 7u + part * 9u + role * 5u +
                     expert * 17u) &
                    0x0fu;
                block.scales[part] = static_cast<std::uint8_t>(
                    low_scale | (high_scale << 4u));
                std::uint32_t high_codes = 0u;
                for (std::uint32_t code_slot = 0u; code_slot < 4u;
                     ++code_slot) {
                    const std::uint32_t ordinal =
                        (((row * kIQ2SBlocksPerRow + block_index) * 8u +
                          part) *
                         4u) +
                        code_slot;
                    const std::uint32_t grid =
                        (ordinal * 53u + role * 31u + expert * 19u) & 0x3ffu;
                    const std::uint32_t sign =
                        (ordinal * 29u + role * 43u + expert * 53u) & 0xffu;
                    block.qs[4u * part + code_slot] =
                        static_cast<std::uint8_t>(grid & 0xffu);
                    block.qs[32u + 4u * part + code_slot] =
                        static_cast<std::uint8_t>(sign);
                    high_codes |= (grid >> 8u) << (2u * code_slot);
                }
                block.qh[part] = static_cast<std::uint8_t>(high_codes);
            }
        }
    }
}

void fill_weights(twin::GuardedBuffer& buffer, std::uint32_t role,
                  std::uint32_t expert, WeightFormat format) {
    if (format == WeightFormat::q2_0) {
        fill_q2_weights(buffer, role, expert);
    } else if (format == WeightFormat::iq4_xs) {
        fill_iq4_xs_weights(buffer, role, expert);
    } else if (format == WeightFormat::iq2_xxs) {
        fill_iq2_xxs_weights(buffer, role, expert);
    } else if (format == WeightFormat::iq2_xs) {
        fill_iq2_xs_weights(buffer, role, expert);
    } else {
        fill_iq2_s_weights(buffer, role, expert);
    }
}

bool iq4_xs_fixture_covers_codes_and_scales(
    const std::array<const twin::GuardedBuffer*, 4u>& weight_buffers) {
    std::array<bool, 16u> observed_codes{};
    std::array<bool, 64u> observed_scales{};
    for (const twin::GuardedBuffer* weight_buffer : weight_buffers) {
        for (const block_iq4_xs& block : typed<block_iq4_xs>(*weight_buffer)) {
            for (const std::uint8_t packed : block.qs) {
                observed_codes[packed & 0x0fu] = true;
                observed_codes[packed >> 4u] = true;
            }
            for (std::size_t group = 0u; group < 8u; ++group) {
                const std::uint32_t low =
                    (static_cast<std::uint32_t>(block.scales_l[group / 2u]) >>
                     (4u * (group % 2u))) &
                    0x0fu;
                const std::uint32_t high =
                    (static_cast<std::uint32_t>(block.scales_h) >>
                     (2u * group)) &
                    0x03u;
                observed_scales[low | (high << 4u)] = true;
            }
        }
    }
    return std::all_of(observed_codes.begin(), observed_codes.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_scales.begin(), observed_scales.end(),
                       [](bool observed) { return observed; });
}

bool iq2_xxs_fixture_covers_codes_scales_and_signs(
    const std::array<const twin::GuardedBuffer*, 4u>& weight_buffers) {
    std::array<bool, 256u> observed_codes{};
    std::array<bool, 16u> observed_scales{};
    std::array<bool, 128u> observed_signs{};
    for (const twin::GuardedBuffer* weight_buffer : weight_buffers) {
        for (const block_iq2_xxs& block :
             typed<block_iq2_xxs>(*weight_buffer)) {
            for (std::uint32_t part = 0u; part < 8u; ++part) {
                const std::uint32_t word = 4u * part;
                const std::uint32_t indices =
                    static_cast<std::uint32_t>(block.qs[word]) |
                    (static_cast<std::uint32_t>(block.qs[word + 1u]) << 16u);
                const std::uint32_t auxiliary =
                    static_cast<std::uint32_t>(block.qs[word + 2u]) |
                    (static_cast<std::uint32_t>(block.qs[word + 3u]) << 16u);
                observed_scales[auxiliary >> 28u] = true;
                for (std::uint32_t code_slot = 0u; code_slot < 4u;
                     ++code_slot) {
                    observed_codes[(indices >> (8u * code_slot)) & 0xffu] =
                        true;
                    observed_signs[(auxiliary >> (7u * code_slot)) & 0x7fu] =
                        true;
                }
            }
        }
    }
    return std::all_of(observed_codes.begin(), observed_codes.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_scales.begin(), observed_scales.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_signs.begin(), observed_signs.end(),
                       [](bool observed) { return observed; });
}

bool iq2_xs_fixture_covers_codes_scales_and_signs(
    const std::array<const twin::GuardedBuffer*, 4u>& weight_buffers) {
    std::array<bool, 512u> observed_grids{};
    std::array<bool, 16u> observed_low_scales{};
    std::array<bool, 16u> observed_high_scales{};
    std::array<bool, 128u> observed_signs{};
    for (const twin::GuardedBuffer* weight_buffer : weight_buffers) {
        for (const block_iq2_xs& block :
             typed<block_iq2_xs>(*weight_buffer)) {
            for (std::uint32_t part = 0u; part < 8u; ++part) {
                const std::uint32_t scale_pair = block.scales[part];
                observed_low_scales[scale_pair & 0x0fu] = true;
                observed_high_scales[scale_pair >> 4u] = true;
                for (std::uint32_t code_slot = 0u; code_slot < 4u;
                     ++code_slot) {
                    const std::uint32_t codeword =
                        block.qs[4u * part + code_slot];
                    observed_grids[codeword & 0x1ffu] = true;
                    observed_signs[codeword >> 9u] = true;
                }
            }
        }
    }
    return std::all_of(observed_grids.begin(), observed_grids.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_low_scales.begin(), observed_low_scales.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_high_scales.begin(),
                       observed_high_scales.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_signs.begin(), observed_signs.end(),
                       [](bool observed) { return observed; });
}

bool iq2_s_fixture_covers_codes_scales_and_signs(
    const std::array<const twin::GuardedBuffer*, 4u>& weight_buffers) {
    std::array<bool, 1024u> observed_grids{};
    std::array<bool, 16u> observed_low_scales{};
    std::array<bool, 16u> observed_high_scales{};
    std::array<bool, 256u> observed_sign_masks{};
    for (const twin::GuardedBuffer* weight_buffer : weight_buffers) {
        for (const block_iq2_s& block : typed<block_iq2_s>(*weight_buffer)) {
            for (std::uint32_t part = 0u; part < 8u; ++part) {
                const std::uint32_t scale_pair = block.scales[part];
                observed_low_scales[scale_pair & 0x0fu] = true;
                observed_high_scales[scale_pair >> 4u] = true;
                for (std::uint32_t code_slot = 0u; code_slot < 4u;
                     ++code_slot) {
                    const std::uint32_t high =
                        (block.qh[part] >> (2u * code_slot)) & 0x03u;
                    const std::uint32_t grid =
                        (high << 8u) | block.qs[4u * part + code_slot];
                    observed_grids[grid] = true;
                    observed_sign_masks[
                        block.qs[32u + 4u * part + code_slot]] = true;
                }
            }
        }
    }
    return std::all_of(observed_grids.begin(), observed_grids.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_low_scales.begin(), observed_low_scales.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_high_scales.begin(),
                       observed_high_scales.end(),
                       [](bool observed) { return observed; }) &&
           std::all_of(observed_sign_masks.begin(), observed_sign_masks.end(),
                       [](bool observed) { return observed; });
}

float q2_value(std::span<const block_q2_0> blocks, std::size_t row,
               std::size_t k) {
    const block_q2_0& block =
        blocks[row * kQ2BlocksPerRow + k / 64u];
    const std::uint8_t packed = block.qs[(k % 64u) / 4u];
    const int code =
        static_cast<int>((packed >> (2u * (k % 4u))) & 3u) - 1;
    return f32(block.d) * static_cast<float>(code);
}

double dot_oracle(std::span<const std::uint16_t> x,
                  const twin::GuardedBuffer& weights, WeightFormat format,
                  std::size_t row, bool round_weights = true) {
    double result = 0.0;
    if (format == WeightFormat::q2_0) {
        const auto blocks = typed<block_q2_0>(weights);
        for (std::size_t k = 0u; k < kWidth; ++k) {
            result += static_cast<double>(f32(x[k])) *
                      static_cast<double>(q2_value(blocks, row, k));
        }
    } else if (format == WeightFormat::iq4_xs) {
        const auto blocks = typed<block_iq4_xs>(weights);
        std::array<float, 256u> decoded{};
        for (std::size_t block_index = 0u;
             block_index < kIQ4XSBlocksPerRow; ++block_index) {
            const block_iq4_xs& block =
                blocks[row * kIQ4XSBlocksPerRow + block_index];
            strata::dequantize_iq4_xs(
                reinterpret_cast<const std::uint8_t*>(&block),
                decoded.data());
            const std::size_t first_k = block_index * decoded.size();
            for (std::size_t offset = 0u; offset < decoded.size(); ++offset) {
                const float weight =
                    round_weights ? f32(half(decoded[offset]))
                                  : decoded[offset];
                result += static_cast<double>(f32(x[first_k + offset])) *
                          static_cast<double>(weight);
            }
        }
    } else if (format == WeightFormat::iq2_xxs) {
        const auto blocks = typed<block_iq2_xxs>(weights);
        std::array<float, 256u> decoded{};
        for (std::size_t block_index = 0u;
             block_index < kIQ2XXSBlocksPerRow; ++block_index) {
            const block_iq2_xxs& block =
                blocks[row * kIQ2XXSBlocksPerRow + block_index];
            twin::iq2_xxs_dequantize_reference(block, decoded);
            const std::size_t first_k = block_index * decoded.size();
            for (std::size_t offset = 0u; offset < decoded.size(); ++offset) {
                const float weight =
                    round_weights ? f32(half(decoded[offset]))
                                  : decoded[offset];
                result += static_cast<double>(f32(x[first_k + offset])) *
                          static_cast<double>(weight);
            }
        }
    } else if (format == WeightFormat::iq2_xs) {
        const auto blocks = typed<block_iq2_xs>(weights);
        std::array<float, 256u> decoded{};
        for (std::size_t block_index = 0u;
             block_index < kIQ2XSBlocksPerRow; ++block_index) {
            const block_iq2_xs& block =
                blocks[row * kIQ2XSBlocksPerRow + block_index];
            twin::iq2_xs_dequantize_reference(block, decoded);
            const std::size_t first_k = block_index * decoded.size();
            for (std::size_t offset = 0u; offset < decoded.size(); ++offset) {
                const float weight =
                    round_weights ? f32(half(decoded[offset]))
                                  : decoded[offset];
                result += static_cast<double>(f32(x[first_k + offset])) *
                          static_cast<double>(weight);
            }
        }
    } else {
        const auto blocks = typed<block_iq2_s>(weights);
        std::array<float, 256u> decoded{};
        for (std::size_t block_index = 0u;
             block_index < kIQ2SBlocksPerRow; ++block_index) {
            const block_iq2_s& block =
                blocks[row * kIQ2SBlocksPerRow + block_index];
            twin::iq2_s_dequantize_reference(block, decoded);
            const std::size_t first_k = block_index * decoded.size();
            for (std::size_t offset = 0u; offset < decoded.size(); ++offset) {
                const float weight =
                    round_weights ? f32(half(decoded[offset]))
                                  : decoded[offset];
                result += static_cast<double>(f32(x[first_k + offset])) *
                          static_cast<double>(weight);
            }
        }
    }
    return result;
}

bool test_iq4_xs_literal_decode_anchor() {
    const block_iq4_xs block = twin::iq4_xs_moe_fixture();
    std::array<float, 256u> decoded{};
    strata::dequantize_iq4_xs(
        reinterpret_cast<const std::uint8_t*>(&block), decoded.data());
    const bool matches =
        decoded[twin::kIQ4XSMoeOneHotPosition] == 1695.0f &&
        decoded[57u] == -1905.0f;
    check(matches,
          "canonical IQ4_XS decoder honors the literal low/high nibble anchor",
          "d=1, group-1 scale=15, qs[25]=0x0f, K[15]=113");
    return matches;
}

bool test_iq2_xxs_literal_decode_anchor() {
    block_iq2_xxs block{};
    block.d = 0x3c00u;  // F16 1.
    block.qs[2] = 0x0001u;  // Sign index 1: components 0 and 7 are negative.
    block.qs[3] = 0x1000u;  // Part scale code 1.
    std::array<float, 256u> decoded{};
    twin::iq2_xxs_dequantize_reference(block, decoded);
    const bool matches =
        decoded[0] == -3.0f && decoded[1] == 3.0f &&
        decoded[7] == -3.0f;
    check(matches,
          "canonical IQ2_XXS decoder honors literal grid, scale, and sign bits",
          "grid index 0, scale code 1, sign index 1");
    return matches;
}

bool test_iq2_xs_literal_decode_anchor() {
    block_iq2_xs block{};
    block.d = 0x3c00u;  // F16 1.
    block.qs[1] = 0x0200u;  // Grid 0, sign index 1.
    block.scales[0] = 0x10u;  // Low scale 0, high scale 1.
    std::array<float, 256u> decoded{};
    twin::iq2_xs_dequantize_reference(block, decoded);
    const bool matches =
        decoded[0] == 1.0f && decoded[8] == -1.0f &&
        decoded[15] == -1.0f && decoded[16] == 3.0f;
    check(matches,
          "canonical IQ2_XS decoder honors literal grid, sign, and scale nibbles",
          "grid 0, sign index 1, scale pair 0x10");
    return matches;
}

bool test_iq2_s_literal_decode_anchor() {
    block_iq2_s block{};
    block.d = 0x3c00u;  // F16 1.
    block.qh[0] = 0x01u;  // First grid index has high bits 1.
    block.qs[32] = 0x81u;  // Sign bits invert components 0 and 7.
    block.scales[0] = 0x10u;  // Low scale 0, high scale 1.
    std::array<float, 256u> decoded{};
    twin::iq2_s_dequantize_reference(block, decoded);
    const bool matches =
        decoded[0] == -3.125f && decoded[7] == -1.0f &&
        decoded[8] == 1.0f && decoded[16] == 3.0f;
    check(matches,
          "canonical IQ2_S decoder honors literal grid-high, sign, and scale fields",
          "grid 0x100, sign mask 0x81, scale pair 0x10");
    return matches;
}

std::size_t round_up_div(std::size_t value, std::size_t divisor) {
    return value / divisor +
           static_cast<std::size_t>(value % divisor != 0u);
}

bool encode_expert(WeightFormat format, Context& context, Encoder& encoder,
                   std::uint32_t rows, std::uint32_t experts,
                   std::uint32_t layer,
                   const PrefillMoeExpertTileBuffers& buffers,
                   std::string& error) {
    if (format == WeightFormat::q2_0) {
        return encode_prefill_moe_gu_q2_0(
            context, encoder, rows, experts, layer, buffers, error);
    }
    if (format == WeightFormat::iq4_xs) {
        return encode_prefill_moe_gu_iq4_xs(
            context, encoder, rows, experts, layer, buffers, error);
    }
    if (format == WeightFormat::iq2_xxs) {
        return encode_prefill_moe_gu_iq2_xxs(
            context, encoder, rows, experts, layer, buffers, error);
    }
    if (format == WeightFormat::iq2_xs) {
        return encode_prefill_moe_gu_iq2_xs(
            context, encoder, rows, experts, layer, buffers, error);
    }
    return encode_prefill_moe_gu_iq2_s(
        context, encoder, rows, experts, layer, buffers, error);
}

bool encode_expert(WeightFormat format, Context& context,
                   CommandBuffer& command_buffer, std::uint32_t rows,
                   std::uint32_t experts, std::uint32_t layer,
                   const PrefillMoeExpertTileBuffers& buffers,
                   std::string& error) {
    if (format == WeightFormat::q2_0) {
        return encode_prefill_moe_gu_q2_0(
            context, command_buffer, rows, experts, layer, buffers, error);
    }
    if (format == WeightFormat::iq4_xs) {
        return encode_prefill_moe_gu_iq4_xs(
            context, command_buffer, rows, experts, layer, buffers, error);
    }
    if (format == WeightFormat::iq2_xxs) {
        return encode_prefill_moe_gu_iq2_xxs(
            context, command_buffer, rows, experts, layer, buffers, error);
    }
    if (format == WeightFormat::iq2_xs) {
        return encode_prefill_moe_gu_iq2_xs(
            context, command_buffer, rows, experts, layer, buffers, error);
    }
    return encode_prefill_moe_gu_iq2_s(
        context, command_buffer, rows, experts, layer, buffers, error);
}

bool run_case(Context& context, WeightFormat format, std::uint32_t rows,
              bool caller_owned,
              bool missing_second_expert = false,
              bool missing_first_expert = false) {
    const std::string format_name(weight_format_name(format));
    const std::string label =
        "Prefill " + format_name + " expert tile T=" + std::to_string(rows) +
        (caller_owned ? " caller-owned" : " standalone") +
        (missing_second_expert || missing_first_expert ? " missing-address"
                                                       : "");
    const std::size_t entries = static_cast<std::size_t>(rows) * 10u;
    const std::size_t route_bytes = entries * sizeof(std::int32_t);
    const std::size_t mixed_bytes =
        static_cast<std::size_t>(rows) * kWidth * sizeof(std::uint16_t);
    const std::size_t gathered_bytes =
        entries * kWidth * sizeof(std::uint16_t);
    const std::size_t hidden_bytes =
        entries * kHidden * sizeof(std::uint16_t);
    const std::size_t histogram_bytes =
        round_up_div(entries, 256u) * 512u * sizeof(std::uint32_t);
    const std::size_t grouping_tile_capacity =
        round_up_div(entries, 64u) + 512u;
    const std::size_t plan_table_bytes =
        (static_cast<std::size_t>(kLayer) + 1u) *
        kPrefillMoePlanStrideBytes;
    const std::size_t address_table_bytes =
        (static_cast<std::size_t>(kLayer) + 1u) * kExperts *
        sizeof(PrefillMoeExpertAddress);
    const std::size_t weight_bytes = weight_format_expert_bytes(format);

    twin::GuardedBuffer ids, resident, mixed, histogram, plans, tiles;
    twin::GuardedBuffer source, destination, gathered, addresses;
    twin::GuardedBuffer address_snapshot;
    twin::GuardedBuffer gate0, up0, gate3, up3, hidden, status;
    std::string error;
    const bool allocated =
        ids.allocate(context, label + " IDs", route_bytes, error) &&
        resident.allocate(context, label + " residency",
                          (static_cast<std::size_t>(kLayer) + 1u) * kExperts *
                              sizeof(std::int32_t),
                          error) &&
        mixed.allocate(context, label + " mixed input", mixed_bytes, error) &&
        histogram.allocate(context, label + " histogram", histogram_bytes,
                           error) &&
        plans.allocate(context, label + " PfPlan table", plan_table_bytes,
                       error) &&
        tiles.allocate(context, label + " tile list",
                       grouping_tile_capacity * sizeof(Tile), error) &&
        source.allocate(context, label + " source rows", route_bytes, error) &&
        destination.allocate(context, label + " destination rows",
                             route_bytes, error) &&
        gathered.allocate(context, label + " gathered input",
                          gathered_bytes, error) &&
        addresses.allocate(context, label + " expert addresses",
                           address_table_bytes, error) &&
        address_snapshot.allocate(
            context, label + " address snapshot",
            kExperts * sizeof(PrefillMoeExpertAddress), error) &&
        gate0.allocate(context, label + " expert 0 gate " + format_name,
                       weight_bytes,
                       error) &&
        up0.allocate(context, label + " expert 0 up " + format_name,
                     weight_bytes,
                     error) &&
        gate3.allocate(context, label + " expert 3 gate " + format_name,
                       weight_bytes,
                       error) &&
        up3.allocate(context, label + " expert 3 up " + format_name,
                     weight_bytes,
                     error) &&
        hidden.allocate(context, label + " hidden output",
                        hidden_bytes + 64u, error) &&
        status.allocate(context, label + " status output", 64u, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    for (twin::GuardedBuffer* buffer :
         {&histogram, &plans, &tiles, &source, &destination, &gathered,
          &hidden, &status}) {
        std::fill(buffer->payload().begin(), buffer->payload().end(), kPoison);
    }
    std::fill(typed<std::int32_t>(ids).begin(),
              typed<std::int32_t>(ids).end(), 0);
    std::fill(typed<std::int32_t>(resident).begin(),
              typed<std::int32_t>(resident).end(), 0);
    std::fill(addresses.payload().begin(), addresses.payload().end(), 0u);
    auto input = typed<std::uint16_t>(mixed);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            const float magnitude =
                0.005f +
                static_cast<float>((row * 19u + column * 7u) % 101u) *
                    (0.010f / 100.0f);
            const float value = ((row + column) & 1u) == 0u
                                    ? magnitude
                                    : -magnitude;
            input[static_cast<std::size_t>(row) * kWidth + column] =
                half(value);
        }
    }
    fill_weights(gate0, 0u, 0u, format);
    fill_weights(up0, 1u, 0u, format);
    fill_weights(gate3, 0u, kSecondExpert, format);
    fill_weights(up3, 1u, kSecondExpert, format);
    auto route_ids = typed<std::int32_t>(ids);
    for (std::size_t entry = 0u; entry < entries; ++entry) {
        const bool first_expert =
            entries > 64u ? entry < 65u : entry % 2u == 0u;
        route_ids[entry] =
            first_expert ? 0 : static_cast<std::int32_t>(kSecondExpert);
    }
    std::vector<std::size_t> grouped_to_route;
    grouped_to_route.reserve(entries);
    for (const std::int32_t expert_id :
         {0, static_cast<std::int32_t>(kSecondExpert)}) {
        for (std::size_t entry = 0u; entry < entries; ++entry) {
            if (route_ids[entry] == expert_id) {
                grouped_to_route.push_back(entry);
            }
        }
    }
    auto address_table = typed<PrefillMoeExpertAddress>(addresses);
    for (std::uint32_t layer = 0u; layer <= kLayer; ++layer) {
        auto& expert0 = address_table[
            static_cast<std::size_t>(layer) * kExperts];
        auto& expert3 = address_table[
            static_cast<std::size_t>(layer) * kExperts + kSecondExpert];
        if (!missing_first_expert) {
            expert0.role[0] = gate0.view().gpu_address;
            expert0.role[1] = up0.view().gpu_address;
        }
        if (!missing_second_expert) {
            expert3.role[0] = gate3.view().gpu_address;
            expert3.role[1] = up3.view().gpu_address;
        }
    }

    const auto input_before =
        std::vector<std::uint8_t>(mixed.payload().begin(),
                                  mixed.payload().end());
    const auto gate0_before =
        std::vector<std::uint8_t>(gate0.payload().begin(),
                                  gate0.payload().end());
    const auto up0_before =
        std::vector<std::uint8_t>(up0.payload().begin(), up0.payload().end());
    const auto gate3_before =
        std::vector<std::uint8_t>(gate3.payload().begin(),
                                  gate3.payload().end());
    const auto up3_before =
        std::vector<std::uint8_t>(up3.payload().begin(), up3.payload().end());
    const auto ids_before =
        std::vector<std::uint8_t>(ids.payload().begin(), ids.payload().end());
    for (twin::GuardedBuffer* buffer :
         {&histogram, &plans, &tiles, &source, &destination, &gathered,
          &hidden, &status}) {
        buffer->capture_before_dispatch();
    }

    auto plan_slot = plans.view().slice(
        static_cast<std::size_t>(kLayer) * kPrefillMoePlanStrideBytes,
        kPrefillMoePlanStrideBytes);
    const PrefillMoeGroupingBuffers grouping{
        context.buffers().read_only(ids.view()),
        context.buffers().read_only(resident.view()),
        context.buffers().read_only(mixed.view()),
        histogram.view(),
        plan_slot,
        tiles.view(),
        source.view(),
        destination.view(),
        gathered.view(),
    };
    const std::array<PrefillMoeWeightStorageView, 4u> weight_storage{{
        {context.buffers().read_only(gate0.view())},
        {context.buffers().read_only(up0.view())},
        {context.buffers().read_only(gate3.view())},
        {context.buffers().read_only(up3.view())},
    }};
    const PrefillMoeExpertTileBuffers expert{
        context.buffers().read_only(addresses.view()),
        address_snapshot.view(),
        weight_storage,
        nullptr,
        context.buffers().read_only(gathered.view()),
        context.buffers().read_only(plan_slot),
        context.buffers().read_only(tiles.view()),
        hidden.view(),
        status.view(),
    };

    auto malformed_expert = expert;
    malformed_expert.addresses =
        context.buffers().read_only(addresses.view().slice(0u, 100u));
    const std::array rollback_buffers{
        &histogram, &plans, &tiles, &source, &destination, &gathered, &hidden,
        &status};
    std::vector<std::vector<std::uint8_t>> rollback_before;
    rollback_before.reserve(rollback_buffers.size());
    for (const twin::GuardedBuffer* buffer : rollback_buffers) {
        const auto payload = buffer->payload();
        rollback_before.emplace_back(payload.begin(), payload.end());
    }
    auto rollback_unchanged = [&] {
        for (std::size_t i = 0u; i < rollback_buffers.size(); ++i) {
            const auto payload = rollback_buffers[i]->payload();
            if (!std::equal(rollback_before[i].begin(),
                            rollback_before[i].end(), payload.begin())) {
                return false;
            }
        }
        return true;
    };

    bool rollback_rejected = false;
    if (caller_owned) {
        auto rollback_command =
            context.make_command_buffer(label + " rollback", error);
        check(rollback_command != nullptr,
              label + " rollback command buffer", error);
        if (!rollback_command) return false;
        rollback_rejected = !Encoder::encode(
            context, *rollback_command, label + " rollback",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_moe_grouping(
                           context, encoder, rows, kExperts, false, kLayer,
                           grouping, encode_error) &&
                       encode_expert(format, context, encoder, rows, kExperts, kLayer,
                           malformed_expert, encode_error);
            },
            error);
    } else {
        auto rollback_command =
            context.make_command_buffer(label + " rollback", error);
        check(rollback_command != nullptr,
              label + " rollback command buffer", error);
        if (!rollback_command) return false;
        const bool grouping_encoded = encode_prefill_moe_grouping(
            context, *rollback_command, rows, kExperts, false, kLayer,
            grouping, error);
        rollback_rejected =
            grouping_encoded &&
            !encode_expert(format, context, *rollback_command, rows, kExperts, kLayer,
                malformed_expert, error) &&
            rollback_command->unsafe_native_handle() == nullptr;
    }
    const bool rollback_ok =
        rollback_rejected && error.find("undersized") != std::string::npos &&
        rollback_unchanged();
    check(rollback_ok,
          label + " rolls back grouping when expert preflight fails", error);
    auto late_failure_command =
        context.make_command_buffer(label + " late status rollback", error);
    check(late_failure_command != nullptr,
          label + " late status rollback command buffer", error);
    if (!late_failure_command) return false;
    const bool late_failure_rejected = !Encoder::encode(
        context, *late_failure_command, label + " late status rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_moe_grouping(
                    context, encoder, rows, kExperts, false, kLayer,
                    grouping, encode_error) ||
                !encode_expert(format, context, encoder, rows, kExperts, kLayer, expert,
                    encode_error)) {
                return false;
            }
            encode_error = "injected failure after the expert dispatch";
            return false;
        },
        error);
    const bool late_rollback_ok =
        late_failure_rejected &&
        error.find("injected failure after the expert dispatch") !=
            std::string::npos &&
        late_failure_command->unsafe_native_handle() == nullptr &&
        rollback_unchanged();
    check(late_rollback_ok,
          label + " rolls back the queued GPU status clear after late failure",
          error);

    auto encode_grouping_and_expert =
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_moe_grouping(
                       context, encoder, rows, kExperts, false, kLayer,
                       grouping, encode_error) &&
                   encode_expert(format, context, encoder, rows, kExperts, kLayer, expert,
                       encode_error);
        };

    bool encoded = false;
    if (caller_owned) {
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) return false;
        encoded = Encoder::encode(context, *command, label,
                                  encode_grouping_and_expert, error);
        check(encoded, label + " encode composed pipeline", error);
        if (encoded) {
            const bool completed =
                command->commit(error) && command->wait_until_completed(error);
            check(completed, label + " GPU completion", error);
            encoded = completed;
        }
    } else {
        auto grouping_command =
            context.make_command_buffer(label + " grouping", error);
        check(grouping_command != nullptr,
              label + " grouping command buffer", error);
        if (!grouping_command) return false;
        encoded = encode_prefill_moe_grouping(
            context, *grouping_command, rows, kExperts, false, kLayer,
            grouping, error);
        check(encoded, label + " standalone grouping encode", error);
        if (!encoded) return false;
        encoded = grouping_command->commit(error) &&
                  grouping_command->wait_until_completed(error);
        check(encoded, label + " grouping GPU completion", error);
        if (!encoded) return false;

        auto expert_command =
            context.make_command_buffer(label + " expert", error);
        check(expert_command != nullptr, label + " expert command buffer",
              error);
        if (!expert_command) return false;
        encoded = encode_expert(format, context, *expert_command, rows, kExperts, kLayer, expert, error);
        check(encoded, label + " standalone expert encode", error);
        if (!encoded) return false;
        encoded = expert_command->commit(error) &&
                  expert_command->wait_until_completed(error);
        check(encoded, label + " expert GPU completion", error);
    }
    if (!encoded) {
        check(false, label + " dispatch", error);
        return false;
    }

    const auto* plan = reinterpret_cast<const PfPlan*>(plan_slot.contents);
    const std::size_t first_group_entries = static_cast<std::size_t>(
        std::count(route_ids.begin(), route_ids.end(), 0));
    const bool plan_valid =
        plan->bad == 0u && plan->n_entries == entries &&
        plan->n_groups == 2u && plan->grp_e[0] == 0 &&
        plan->grp_e[1] == static_cast<std::int32_t>(kSecondExpert) &&
        plan->grp_start[0] == 0 &&
        plan->grp_start[1] ==
            static_cast<std::int32_t>(first_group_entries) &&
        plan->grp_start[2] == static_cast<std::int32_t>(entries) &&
        plan->n_tiles ==
            round_up_div(first_group_entries, 64u) +
                round_up_div(entries - first_group_entries, 64u);
    std::vector<Tile> expected_tiles;
    for (std::uint32_t group = 0u; group < 2u; ++group) {
        const std::size_t group_start =
            group == 0u ? 0u : first_group_entries;
        const std::size_t group_end =
            group == 0u ? first_group_entries : entries;
        for (std::size_t entry = group_start; entry < group_end;
             entry += 64u) {
            expected_tiles.push_back(
                {group, static_cast<std::uint32_t>(entry)});
        }
    }
    const auto tile_values = typed<Tile>(tiles);
    const bool tile_ranges_valid =
        plan_valid &&
        std::equal(expected_tiles.begin(), expected_tiles.end(),
                   tile_values.begin(),
                   [](const Tile& expected, const Tile& actual_tile) {
                       return expected.group == actual_tile.group &&
                              expected.first_entry == actual_tile.first_entry;
                   });
    check(tile_ranges_valid,
          label + " grouping supplies two independently addressed experts");

    const std::array<const twin::GuardedBuffer*, 4u> role_weights{
        &gate0, &up0, &gate3, &up3};
    if (format == WeightFormat::iq4_xs && rows == 1u && caller_owned &&
        !missing_second_expert && !missing_first_expert) {
        check(iq4_xs_fixture_covers_codes_and_scales(role_weights),
              label + " fixture covers all nonlinear codes and six-bit scales");
    }
    if (format == WeightFormat::iq2_xxs && rows == 1u && caller_owned &&
        !missing_second_expert && !missing_first_expert) {
        check(iq2_xxs_fixture_covers_codes_scales_and_signs(role_weights),
              label + " fixture covers every code, scale, and sign index");
    }
    if (format == WeightFormat::iq2_xs && rows == 1u && caller_owned &&
        !missing_second_expert && !missing_first_expert) {
        check(iq2_xs_fixture_covers_codes_scales_and_signs(role_weights),
              label + " fixture covers every grid, scale, and sign index");
    }
    if (format == WeightFormat::iq2_s && rows == 1u && caller_owned &&
        !missing_second_expert && !missing_first_expert) {
        check(iq2_s_fixture_covers_codes_scales_and_signs(role_weights),
              label + " fixture covers every grid, scale, and sign mask");
    }
    const auto actual = typed<std::uint16_t>(hidden);
    bool oracle_matches = true;
    bool gather_matches = true;
    std::uint16_t maximum_distance = 0u;
    std::vector<std::uint16_t> row_expected(
        static_cast<std::size_t>(rows) * 2u * kHidden);
    const bool check_weight_rounding =
        (format == WeightFormat::iq4_xs ||
         format == WeightFormat::iq2_xxs ||
         format == WeightFormat::iq2_xs ||
         format == WeightFormat::iq2_s) &&
        rows == 1u && caller_owned && !missing_second_expert &&
        !missing_first_expert;
    bool weight_rounding_changes_output = !check_weight_rounding;
    const auto input_view = typed<std::uint16_t>(mixed);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const auto x = std::span<const std::uint16_t>(
            input_view.data() + static_cast<std::size_t>(row) * kWidth,
            kWidth);
        for (std::size_t column = 0u; column < kHidden; ++column) {
            for (std::size_t expert_slot = 0u; expert_slot < 2u;
                 ++expert_slot) {
                const auto& gate_weights = *role_weights[expert_slot * 2u];
                const auto& up_weights = *role_weights[expert_slot * 2u + 1u];
                const double gate_sum =
                    dot_oracle(x, gate_weights, format, column);
                const double up_sum =
                    dot_oracle(x, up_weights, format, column);
                const double silu =
                    gate_sum / (1.0 + std::exp(-gate_sum));
                const std::uint16_t expected =
                    output_half(silu * up_sum);
                row_expected[(static_cast<std::size_t>(row) * 2u +
                              expert_slot) *
                                 kHidden +
                             column] = expected;
                if (check_weight_rounding) {
                    const double unrounded_gate_sum =
                        dot_oracle(x, gate_weights, format, column, false);
                    const double unrounded_up_sum =
                        dot_oracle(x, up_weights, format, column, false);
                    const double unrounded_silu =
                        unrounded_gate_sum /
                        (1.0 + std::exp(-unrounded_gate_sum));
                    const std::uint16_t unrounded_expected =
                        output_half(unrounded_silu * unrounded_up_sum);
                    weight_rounding_changes_output |=
                        half_ulp_distance(expected, unrounded_expected) > 4u;
                }
            }
        }
    }
    if (check_weight_rounding) {
        check(weight_rounding_changes_output,
              label + " F16 weight rounding changes the scalar SwiGLU output "
                      "beyond the four-ULP tolerance");
    }
    for (std::size_t entry = 0u; entry < entries; ++entry) {
        const std::size_t group_index =
            entry < first_group_entries ? 0u : 1u;
        const std::size_t original_route = grouped_to_route[entry];
        const std::size_t row = original_route / 10u;
        const std::size_t expert_slot = group_index;
        for (std::size_t column = 0u; column < kHidden; ++column) {
            const std::uint16_t expected =
                (missing_second_expert && expert_slot == 1u) ||
                        (missing_first_expert && expert_slot == 0u)
                    ? 0u
                    : row_expected[(row * 2u + expert_slot) * kHidden +
                                   column];
            const std::uint16_t distance = half_ulp_distance(
                actual[entry * kHidden + column], expected);
            maximum_distance = std::max(maximum_distance, distance);
            if (distance > 4u) oracle_matches = false;
        }
        const auto gathered_values = typed<std::uint16_t>(gathered);
        const std::size_t row_base = row * kWidth;
        const std::size_t gathered_base = entry * kWidth;
        if (!std::equal(input_view.begin() + row_base,
                        input_view.begin() + row_base + kWidth,
                        gathered_values.begin() + gathered_base)) {
            gather_matches = false;
        }
    }
    check(gather_matches, label + " gathers the mapped source row");
    check(oracle_matches, label + " multi-expert " + format_name +
                              "/SwiGLU oracle",
          "maximum F16 distance=" + std::to_string(maximum_distance) +
              " ULP (limit 4)");
    if (format == WeightFormat::iq2_xxs && rows == 1u && caller_owned &&
        !missing_second_expert && !missing_first_expert) {
        // Frozen from the literal 32-word block, ggml dq_iq2_xxs semantics,
        // and an FP64 dot over the first 256 synthetic F16 activation values.
        const std::uint16_t literal_expected = 0x2c29u;
        const std::uint16_t distance =
            half_ulp_distance(actual[0], literal_expected);
        check(distance <= 4u,
              label + " matches the fixed packed-block Prefill golden",
              "F16 ULP distance=" + std::to_string(distance));
    }
    if (format == WeightFormat::iq2_xs && rows == 1u && caller_owned &&
        !missing_second_expert && !missing_first_expert) {
        // Filled with the literal IQ2_XS block and frozen after independent
        // dq_iq2_xs dequantization plus an FP64 dot over the synthetic input.
        const std::uint16_t literal_expected = 0x2738u;
        const std::uint16_t distance =
            half_ulp_distance(actual[0], literal_expected);
        check(distance <= 4u,
              label + " matches the fixed packed-block Prefill golden",
              "F16 ULP distance=" + std::to_string(distance));
    }
    if (format == WeightFormat::iq2_s && rows == 1u && caller_owned &&
        !missing_second_expert && !missing_first_expert) {
        // Frozen from the literal IQ2_S block and signed half input after
        // canonical dq_iq2_s dequantization and an FP64 dot.
        const std::uint16_t literal_expected = 0x1d33u;
        const std::uint16_t distance =
            half_ulp_distance(actual[0], literal_expected);
        check(distance <= 4u,
              label + " matches the fixed packed-block Prefill golden",
              "F16 ULP distance=" + std::to_string(distance));
    }
    const auto status_values = typed<std::uint32_t>(status);
    const bool status_matches =
        status_values[0] ==
        ((missing_second_expert || missing_first_expert) ? 1u : 0u);
    check(status_matches, label + " reports missing active expert address");

    const bool immutable =
        std::equal(input_before.begin(), input_before.end(),
                   mixed.payload().begin()) &&
        std::equal(gate0_before.begin(), gate0_before.end(),
                   gate0.payload().begin()) &&
        std::equal(up0_before.begin(), up0_before.end(),
                   up0.payload().begin()) &&
        std::equal(gate3_before.begin(), gate3_before.end(),
                   gate3.payload().begin()) &&
        std::equal(up3_before.begin(), up3_before.end(),
                   up3.payload().begin()) &&
        std::equal(ids_before.begin(), ids_before.end(),
                   ids.payload().begin());
    check(immutable, label + " preserves model weights and inputs");
    check(tail_is_poison(hidden, hidden_bytes),
          label + " preserves the output tail");
    check(tail_is_poison(status, sizeof(std::uint32_t)),
          label + " preserves the status tail");
    bool guards = true;
    for (twin::GuardedBuffer* buffer :
         {&ids, &resident, &mixed, &histogram, &plans, &tiles, &source,
          &destination, &gathered, &addresses, &address_snapshot, &gate0,
          &up0, &gate3, &up3, &hidden, &status}) {
        guards &= guards_intact(*buffer);
    }
    check(guards, label + " preserves allocation guards");

    std::vector<std::uint8_t> valid_plan_bytes(
        static_cast<const std::uint8_t*>(plan_slot.contents),
        static_cast<const std::uint8_t*>(plan_slot.contents) + plan_slot.length);
    const std::vector<std::uint8_t> valid_tile_bytes(
        tiles.payload().begin(), tiles.payload().end());
    bool malformed_expert_order_ok = true;
    bool malformed_tile_ok = true;
    std::fill(hidden.payload().begin(), hidden.payload().end(), kPoison);
    std::fill(status.payload().begin(), status.payload().end(), kPoison);
    auto* writable_plan = reinterpret_cast<PfPlan*>(plan_slot.contents);
    writable_plan->bad = 1u;
    auto invalid_plan_command =
        context.make_command_buffer(label + " invalid plan status", error);
    bool invalid_plan_completed = invalid_plan_command != nullptr;
    if (invalid_plan_completed) {
        invalid_plan_completed =
            encode_expert(format, context, *invalid_plan_command, rows, kExperts, kLayer, expert,
                error) &&
            invalid_plan_command->commit(error) &&
            invalid_plan_command->wait_until_completed(error);
    }
    const auto status_after_invalid_plan = typed<std::uint32_t>(status);
    const bool invalid_plan_outputs_zeroed =
        std::all_of(hidden.payload().begin(),
                    hidden.payload().begin() + hidden_bytes,
                    [](std::uint8_t byte) { return byte == 0u; });
    const bool invalid_plan_status_ok =
        invalid_plan_completed && status_after_invalid_plan[0] == 2u &&
        invalid_plan_outputs_zeroed &&
        tail_is_poison(hidden, hidden_bytes) &&
        tail_is_poison(status, sizeof(std::uint32_t));
    check(invalid_plan_status_ok,
          label + " reports an invalid plan and zeroes all hidden output",
          error);
    writable_plan->bad = 0u;
    writable_plan->n_tiles = 0u;
    std::fill(hidden.payload().begin(), hidden.payload().end(), kPoison);
    std::fill(status.payload().begin(), status.payload().end(), kPoison);
    auto missing_tiles_command =
        context.make_command_buffer(label + " missing plan tiles", error);
    bool missing_tiles_completed = missing_tiles_command != nullptr;
    if (missing_tiles_completed) {
        missing_tiles_completed =
            encode_expert(format, context, *missing_tiles_command, rows, kExperts, kLayer,
                expert, error) &&
            missing_tiles_command->commit(error) &&
            missing_tiles_command->wait_until_completed(error);
    }
    const auto status_after_missing_tiles = typed<std::uint32_t>(status);
    const bool missing_tiles_outputs_zeroed =
        std::all_of(hidden.payload().begin(),
                    hidden.payload().begin() + hidden_bytes,
                    [](std::uint8_t byte) { return byte == 0u; });
    const bool missing_tiles_status_ok =
        missing_tiles_completed && status_after_missing_tiles[0] == 2u &&
        missing_tiles_outputs_zeroed &&
        tail_is_poison(hidden, hidden_bytes) &&
        tail_is_poison(status, sizeof(std::uint32_t));
    check(missing_tiles_status_ok,
          label + " rejects a plan with missing tiles and zeroes output",
          error);
    std::memcpy(writable_plan, valid_plan_bytes.data(),
                valid_plan_bytes.size());
    auto malformed_tile_values = typed<Tile>(tiles);
    malformed_tile_values[0].first_entry = 1u;
    std::fill(hidden.payload().begin(), hidden.payload().end(), kPoison);
    std::fill(status.payload().begin(), status.payload().end(), kPoison);
    auto malformed_tile_command =
        context.make_command_buffer(label + " malformed plan tile", error);
    bool malformed_tile_completed = malformed_tile_command != nullptr;
    if (malformed_tile_completed) {
        malformed_tile_completed =
            encode_expert(format, context, *malformed_tile_command, rows, kExperts, kLayer,
                expert, error) &&
            malformed_tile_command->commit(error) &&
            malformed_tile_command->wait_until_completed(error);
    }
    const auto malformed_tile_status = typed<std::uint32_t>(status);
    const bool malformed_tile_outputs_zeroed =
        std::all_of(hidden.payload().begin(),
                    hidden.payload().begin() + hidden_bytes,
                    [](std::uint8_t byte) { return byte == 0u; });
    malformed_tile_ok =
        malformed_tile_completed && malformed_tile_status[0] == 2u &&
        malformed_tile_outputs_zeroed &&
        tail_is_poison(hidden, hidden_bytes) &&
        tail_is_poison(status, sizeof(std::uint32_t));
    check(malformed_tile_ok,
          label + " rejects a malformed tile with a valid header and zeroes output",
          error);
    std::copy(valid_tile_bytes.begin(), valid_tile_bytes.end(),
              tiles.payload().begin());
    if (writable_plan->n_groups > 1u) {
        const std::int32_t first_expert = writable_plan->grp_e[0];
        const std::int32_t second_expert = writable_plan->grp_e[1];
        writable_plan->grp_e[1] = first_expert;
        std::fill(hidden.payload().begin(), hidden.payload().end(), kPoison);
        std::fill(status.payload().begin(), status.payload().end(), kPoison);
        auto duplicate_expert_command =
            context.make_command_buffer(label + " duplicate plan experts",
                                        error);
        bool duplicate_expert_completed =
            duplicate_expert_command != nullptr;
        if (duplicate_expert_completed) {
            duplicate_expert_completed =
                encode_expert(format, context, *duplicate_expert_command, rows, kExperts,
                    kLayer, expert, error) &&
                duplicate_expert_command->commit(error) &&
                duplicate_expert_command->wait_until_completed(error);
        }
        const auto duplicate_status = typed<std::uint32_t>(status);
        const bool duplicate_outputs_zeroed =
            std::all_of(hidden.payload().begin(),
                        hidden.payload().begin() + hidden_bytes,
                        [](std::uint8_t byte) { return byte == 0u; });
        const bool duplicate_plan_ok =
            duplicate_expert_completed && duplicate_status[0] == 2u &&
            duplicate_outputs_zeroed &&
            tail_is_poison(hidden, hidden_bytes) &&
            tail_is_poison(status, sizeof(std::uint32_t));
        check(duplicate_plan_ok,
              label + " rejects duplicate expert groups and zeroes output",
              error);
        malformed_expert_order_ok &= duplicate_plan_ok;

        std::memcpy(writable_plan, valid_plan_bytes.data(),
                    valid_plan_bytes.size());
        writable_plan->grp_e[0] = second_expert;
        writable_plan->grp_e[1] = first_expert;
        std::fill(hidden.payload().begin(), hidden.payload().end(), kPoison);
        std::fill(status.payload().begin(), status.payload().end(), kPoison);
        auto unordered_expert_command =
            context.make_command_buffer(label + " unordered plan experts",
                                        error);
        bool unordered_expert_completed =
            unordered_expert_command != nullptr;
        if (unordered_expert_completed) {
            unordered_expert_completed =
                encode_expert(format, context, *unordered_expert_command, rows, kExperts,
                    kLayer, expert, error) &&
                unordered_expert_command->commit(error) &&
                unordered_expert_command->wait_until_completed(error);
        }
        const auto unordered_status = typed<std::uint32_t>(status);
        const bool unordered_outputs_zeroed =
            std::all_of(hidden.payload().begin(),
                        hidden.payload().begin() + hidden_bytes,
                        [](std::uint8_t byte) { return byte == 0u; });
        const bool unordered_plan_ok =
            unordered_expert_completed && unordered_status[0] == 2u &&
            unordered_outputs_zeroed &&
            tail_is_poison(hidden, hidden_bytes) &&
            tail_is_poison(status, sizeof(std::uint32_t));
        check(unordered_plan_ok,
              label + " rejects out-of-order expert groups and zeroes output",
              error);
        malformed_expert_order_ok &= unordered_plan_ok;
    }
    auto* writable_addresses = typed<PrefillMoeExpertAddress>(addresses).data();
    const std::size_t other_layer_offset =
        static_cast<std::size_t>(kLayer - 1u) * kExperts;
    const std::size_t selected_row_bytes =
        kExperts * sizeof(PrefillMoeExpertAddress);
    const std::vector<PrefillMoeExpertAddress> other_layer_before(
        writable_addresses + other_layer_offset,
        writable_addresses + other_layer_offset + kExperts);
    writable_addresses[other_layer_offset].role[0] =
        gate3.view().gpu_address;
    writable_addresses[other_layer_offset].role[1] =
        up3.view().gpu_address;
    writable_addresses[other_layer_offset + kSecondExpert].role[0] =
        gate0.view().gpu_address;
    writable_addresses[other_layer_offset + kSecondExpert].role[1] =
        up0.view().gpu_address;
    const std::vector<std::uint8_t> hidden_before_scratch_reuse(
        hidden.payload().begin(), hidden.payload().end());
    const std::vector<std::uint8_t> status_before_scratch_reuse(
        status.payload().begin(), status.payload().end());
    auto scratch_reuse_command =
        context.make_command_buffer(label + " staging scratch reuse", error);
    bool scratch_reuse_rejected = false;
    if (scratch_reuse_command != nullptr) {
        const bool encoded = Encoder::encode(
            context, *scratch_reuse_command,
            label + " staging scratch reuse",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_expert(format, context, encoder, rows, kExperts, kLayer, expert,
                           encode_error) &&
                       encode_expert(format, context, encoder, rows, kExperts, kLayer - 1u,
                           expert, encode_error);
            },
            error);
        scratch_reuse_rejected =
            !encoded &&
            error.find("reuses overlapping staging scratch") !=
                std::string::npos &&
            scratch_reuse_command->unsafe_native_handle() == nullptr &&
            std::equal(hidden_before_scratch_reuse.begin(),
                       hidden_before_scratch_reuse.end(),
                       hidden.payload().begin()) &&
            std::equal(status_before_scratch_reuse.begin(),
                       status_before_scratch_reuse.end(),
                       status.payload().begin());
    }
    std::copy(other_layer_before.begin(), other_layer_before.end(),
              writable_addresses + other_layer_offset);
    check(scratch_reuse_rejected,
          label + " rejects snapshot scratch reuse across two pending layers",
          error);
    const std::size_t selected_row_offset =
        static_cast<std::size_t>(kLayer) * kExperts;
    const auto* selected_row_begin =
        reinterpret_cast<const std::uint8_t*>(
            writable_addresses + selected_row_offset);
    const std::vector<std::uint8_t> selected_row_before(
        selected_row_begin, selected_row_begin + selected_row_bytes);
    auto alias_grouping = grouping;
    alias_grouping.histogram = addresses.view().slice(
        selected_row_offset * sizeof(PrefillMoeExpertAddress),
        histogram_bytes);
    std::fill(hidden.payload().begin(), hidden.payload().end(), kPoison);
    std::fill(status.payload().begin(), status.payload().end(), kPoison);
    auto address_alias_command =
        context.make_command_buffer(label + " GPU address-row alias", error);
    bool address_alias_encoded = false;
    if (address_alias_command != nullptr) {
        address_alias_encoded = Encoder::encode(
            context, *address_alias_command, label + " GPU address-row alias",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_moe_grouping(
                           context, encoder, rows, kExperts, false, kLayer,
                           alias_grouping, encode_error) &&
                       encode_expert(format, context, encoder, rows, kExperts, kLayer, expert,
                           encode_error);
            },
            error);
    }
    check(address_alias_encoded,
          label + " snapshots addresses after GPU alias preflight", error);
    bool address_alias_synchronized = false;
    bool address_alias_completed = false;
    if (address_alias_encoded) {
        const bool committed = address_alias_command->commit(error);
        if (committed) {
            address_alias_completed =
                address_alias_command->wait_until_completed(error);
            address_alias_synchronized = true;
        }
    }
    const auto status_after_address_alias = typed<std::uint32_t>(status);
    const std::size_t aliased_address_bytes =
        std::min(histogram_bytes, selected_row_bytes);
    const bool gpu_overwrote_address_row =
        !std::equal(selected_row_before.begin(),
                    selected_row_before.begin() + aliased_address_bytes,
                    selected_row_begin);
    bool address_alias_oracle_matches = address_alias_completed;
    const auto address_alias_output = typed<std::uint16_t>(hidden);
    if (address_alias_oracle_matches) {
        for (std::size_t entry = 0u; entry < entries; ++entry) {
            const std::size_t group_index =
                entry < first_group_entries ? 0u : 1u;
            const std::size_t original_route = grouped_to_route[entry];
            const std::size_t row = original_route / 10u;
            for (std::size_t column = 0u; column < kHidden; ++column) {
                const bool missing_group =
                    (missing_first_expert && group_index == 0u) ||
                    (missing_second_expert && group_index == 1u);
                const std::uint16_t expected =
                    missing_group
                        ? 0u
                        : row_expected[(row * 2u + group_index) * kHidden +
                                       column];
                if (half_ulp_distance(
                        address_alias_output[entry * kHidden + column],
                        expected) > 4u) {
                    address_alias_oracle_matches = false;
                }
            }
        }
    }
    const std::uint32_t expected_address_alias_status =
        (missing_first_expert || missing_second_expert) ? 1u : 0u;
    const bool address_snapshot_isolated =
        address_alias_completed && gpu_overwrote_address_row &&
        status_after_address_alias[0] == expected_address_alias_status &&
        std::equal(selected_row_before.begin(), selected_row_before.end(),
                   address_snapshot.payload().begin()) &&
        address_alias_oracle_matches;
    if (address_alias_synchronized) {
        std::memcpy(writable_addresses + selected_row_offset,
                    selected_row_before.data(), selected_row_bytes);
    } else if (address_alias_command != nullptr) {
        address_alias_command->discard();
    }
    check(address_snapshot_isolated,
          label + " uses the preflight address snapshot after a prior GPU "
                  "dispatch overwrites its source row",
          error);
    return tile_ranges_valid && gather_matches && oracle_matches &&
           status_matches && immutable && tail_is_poison(hidden, hidden_bytes) &&
           tail_is_poison(status, sizeof(std::uint32_t)) && guards &&
           rollback_ok && late_rollback_ok && invalid_plan_status_ok &&
           missing_tiles_status_ok && malformed_tile_ok &&
           malformed_expert_order_ok && scratch_reuse_rejected &&
           address_snapshot_isolated;
}

bool test_decode_probe(Context& context) {
    constexpr std::uint32_t values = kWidth;
    const std::size_t weight_bytes =
        kQ2BlocksPerRow * kQ2BlockBytes;
    const std::size_t output_bytes =
        static_cast<std::size_t>(values) * sizeof(std::uint16_t);
    twin::GuardedBuffer weights, decoded;
    std::string error;
    const bool allocated =
        weights.allocate(context, "Q2_0 decoder probe weights",
                         weight_bytes, error) &&
        decoded.allocate(context, "Q2_0 decoder probe output",
                         output_bytes + 64u, error);
    check(allocated, "Q2_0 decoder probe allocate", error);
    if (!allocated) return false;

    auto blocks = typed<block_q2_0>(weights);
    for (std::size_t block_index = 0u; block_index < blocks.size();
         ++block_index) {
        block_q2_0& block = blocks[block_index];
        block.d = 0x2800u;
        std::fill(std::begin(block.qs), std::end(block.qs), 0u);
        for (std::size_t k = 0u; k < 64u; ++k) {
            store_code(block, k, static_cast<std::uint8_t>(
                                     (block_index + k) % 4u));
        }
    }
    std::fill(decoded.payload().begin(), decoded.payload().end(), kPoison);
    decoded.capture_before_dispatch();

    twin::Dispatch work;
    work.label = "Q2_0 in-place decoder bitwise probe";
    work.buffers = {{0u, weights.view()}, {1u, decoded.view()}};
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {
        static_cast<std::uint32_t>(round_up_div(values, 256u)), 1u, 1u};
    work.threads_per_group = {256u, 1u, 1u};
    work.parameters = &values;
    work.parameter_bytes = sizeof(values);
    work.parameter_index = 2u;
    const bool dispatched = twin::dispatch_manifest(
        context, "strata_pf_q2_0_decode_probe", work, error);
    check(dispatched, "Q2_0 decoder probe GPU completion", error);
    if (!dispatched) return false;

    const auto actual = typed<std::uint16_t>(decoded);
    std::array<bool, 4u> observed_codes{};
    bool matches = true;
    for (std::size_t index = 0u; index < values; ++index) {
        const block_q2_0& block = blocks[index / 64u];
        const std::uint8_t packed = block.qs[(index % 64u) / 4u];
        const std::uint8_t code = static_cast<std::uint8_t>(
            (packed >> (2u * (index % 4u))) & 0x03u);
        observed_codes[code] = true;
        const std::uint16_t expected =
            half(f32(block.d) * static_cast<float>(static_cast<int>(code) - 1));
        if (actual[index] != expected) matches = false;
    }
    check(matches,
          "Q2_0 debug decoder matches host F16 bits for every weight",
          "codes 0/1/2/3");
    check(std::all_of(observed_codes.begin(), observed_codes.end(),
                      [](bool seen) { return seen; }),
          "Q2_0 decoder probe covers all four packed codes");
    const bool tail = tail_is_poison(decoded, output_bytes);
    check(tail, "Q2_0 decoder probe preserves output tail");
    const bool guards = guards_intact(weights) && guards_intact(decoded);
    check(guards, "Q2_0 decoder probe preserves allocation guards");
    return matches && std::all_of(observed_codes.begin(),
                                  observed_codes.end(),
                                  [](bool seen) { return seen; }) &&
           tail && guards;
}

bool test_validation(Context& context, Context& foreign_context,
                     WeightFormat format) {
    const std::string format_name(weight_format_name(format));
    const std::size_t weight_bytes = weight_format_expert_bytes(format);
    twin::GuardedBuffer addresses, gathered, plan, tiles, hidden, status, foreign;
    twin::GuardedBuffer weight_storage, address_snapshot;
    std::string error;
    const std::size_t entries = 10u;
    const std::size_t tile_capacity = 257u;
    const bool allocated =
        addresses.allocate(context, "validation addresses",
                           (kLayer + 1u) * kExperts * 24u, error) &&
        address_snapshot.allocate(
            context, "validation address snapshot",
            kExperts * sizeof(PrefillMoeExpertAddress), error) &&
        gathered.allocate(context, "validation gathered",
                          entries * kWidth * sizeof(std::uint16_t), error) &&
        plan.allocate(context, "validation plan", kPrefillMoePlanBytes, error) &&
        tiles.allocate(context, "validation tiles",
                       tile_capacity * sizeof(Tile), error) &&
        hidden.allocate(context, "validation hidden",
                        entries * kHidden * sizeof(std::uint16_t), error) &&
        status.allocate(context, "validation status", 64u, error) &&
        weight_storage.allocate(
            context, "validation weight storage",
            weight_bytes, error) &&
        foreign.allocate(foreign_context, "validation foreign",
                         entries * kWidth * sizeof(std::uint16_t), error);
    check(allocated, "Prefill " + format_name + " validation allocate",
          error);
    if (!allocated) return false;
    std::fill(addresses.payload().begin(), addresses.payload().end(), 0u);
    std::fill(status.payload().begin(), status.payload().end(), kPoison);
    const auto valid_plan = context.buffers().read_only(plan.view());
    const auto stable_weight_view =
        context.buffers().read_only(weight_storage.view());
    check(!stable_weight_view.is_tiered_slot_arena(),
          "ordinary Context weights start outside tiered-slot provenance");
    const std::array<PrefillMoeWeightStorageView, 1u> valid_weight_storage{{
        {stable_weight_view},
    }};
    const PrefillMoeExpertTileBuffers valid{
        context.buffers().read_only(addresses.view()),
        address_snapshot.view(),
        valid_weight_storage,
        nullptr,
        context.buffers().read_only(gathered.view()),
        valid_plan,
        context.buffers().read_only(tiles.view()),
        hidden.view(),
        status.view(),
    };
    auto rejects = [&](std::uint32_t rows, std::uint32_t experts,
                       std::uint32_t layer,
                       const PrefillMoeExpertTileBuffers& buffers,
                       std::string_view expected,
                       const std::string& label) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool result =
            encode_expert(format, context, *command, rows, experts, layer,
                          buffers, error);
        const bool rejected =
            !result && error.find(expected) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    passed &= rejects(0u, kExperts, kLayer, valid, "row count",
                      "rejects zero rows");
    passed &= rejects(1u, 128u, kLayer, valid, "256 or 512",
                      "rejects unsupported expert count");
    passed &= rejects(1u, kExperts, kPrefillMoePlanLayers, valid,
                      "outside the PfPlan", "rejects out-of-range layer");
    auto short_addresses = valid;
    short_addresses.addresses =
        context.buffers().read_only(addresses.view().slice(0u, 100u));
    passed &= rejects(1u, kExperts, kLayer, short_addresses, "undersized",
                      "rejects short address table");
    auto short_address_snapshot = valid;
    short_address_snapshot.address_snapshot = address_snapshot.view().slice(
        0u, kExperts * sizeof(PrefillMoeExpertAddress) - 1u);
    passed &= rejects(1u, kExperts, kLayer, short_address_snapshot,
                      "undersized or misaligned",
                      "rejects short address snapshot");
    auto aliased_address_snapshot = valid;
    aliased_address_snapshot.address_snapshot = addresses.view().slice(
        0u, kExperts * sizeof(PrefillMoeExpertAddress));
    passed &= rejects(1u, kExperts, kLayer, aliased_address_snapshot,
                      "address snapshot overlaps",
                      "rejects address snapshot alias");
    auto short_gather = valid;
    short_gather.gathered_rows = context.buffers().read_only(
        gathered.view().slice(0u, entries * kWidth * 2u - 2u));
    passed &= rejects(1u, kExperts, kLayer, short_gather, "undersized",
                      "rejects short gathered input");
    auto short_status = valid;
    short_status.status = status.view().slice(0u, 2u);
    passed &= rejects(1u, kExperts, kLayer, short_status, "undersized",
                      "rejects short status output");
    auto unbacked_address = valid;
    auto* selected_address =
        typed<PrefillMoeExpertAddress>(addresses)
            [kLayer * kExperts]
                .role;
    selected_address[0] =
        weight_storage.view().gpu_address + sizeof(std::uint16_t);
    selected_address[1] =
        weight_storage.view().gpu_address + sizeof(std::uint16_t);
    passed &= rejects(1u, kExperts, kLayer, unbacked_address,
                      "outside provided storage",
                      "rejects unbounded expert weight address");
    selected_address[0] = weight_storage.view().gpu_address;
    selected_address[1] = weight_storage.view().gpu_address;
    selected_address[0] = 0u;
    selected_address[1] = 0u;
    auto output_input_alias = valid;
    output_input_alias.hidden = gathered.view();
    passed &= rejects(1u, kExperts, kLayer, output_input_alias,
                      "output overlaps active gathered input",
                      "rejects output/input alias");
    auto status_input_alias = valid;
    status_input_alias.status = gathered.view().slice(0u, sizeof(std::uint32_t));
    passed &= rejects(1u, kExperts, kLayer, status_input_alias,
                      "output overlaps active gathered input",
                      "rejects status/input alias");
    auto foreign_input = valid;
    foreign_input.gathered_rows =
        foreign_context.buffers().read_only(foreign.view());
    passed &= rejects(1u, kExperts, kLayer, foreign_input,
                      "not owned by this Context",
                      "rejects foreign Context input");
    const std::array<PrefillMoeWeightStorageView, 1u> foreign_weight_storage{{
        {foreign_context.buffers().read_only(foreign.view())},
    }};
    auto foreign_weights = valid;
    foreign_weights.weight_storage = foreign_weight_storage;
    passed &= rejects(1u, kExperts, kLayer, foreign_weights,
                      "not owned by this Context",
                      "rejects foreign Context weight storage");
    twin::GuardedBuffer tiered_weight_storage;
    const bool tiered_allocated = tiered_weight_storage.allocate(
        context, "validation tiered weight storage",
        weight_storage.view().length, error);
    check(tiered_allocated, "allocate validation tiered weight storage",
          error);
    if (tiered_allocated) {
        const BufferRef tiered_view = tiered_weight_storage.view();
        const bool marked = context.buffers().mark_tiered_slot_arena(
            tiered_view, error);
        check(marked, "mark validation slot arena provenance", error);
        check(marked &&
                  context.buffers()
                      .read_only(tiered_view)
                      .is_tiered_slot_arena(),
              "read-only slices observe owner-stamped tiered provenance");
        selected_address[0] = tiered_view.gpu_address;
        selected_address[1] = tiered_view.gpu_address;
        const std::array<PrefillMoeWeightStorageView, 1u> tiered_storage{{
            {context.buffers().read_only(tiered_view)},
        }};
        auto missing_lease = valid;
        missing_lease.weight_storage = tiered_storage;
        passed &= marked &&
                  rejects(1u, kExperts, kLayer, missing_lease,
                          "ExpertLeaseSet",
                          "rejects tiered storage without a prepared lease");
    } else {
        passed = false;
    }
    selected_address[0] = weight_storage.view().gpu_address;
    selected_address[1] = weight_storage.view().gpu_address;
    if (std::getenv("STRATA_METAL_DISPATCH") != nullptr &&
        std::string_view(std::getenv("STRATA_METAL_DISPATCH")) == "serial") {
        check(true, "open-group refusal is covered in concurrent mode");
        return passed;
    }

    auto command = context.make_command_buffer(
        "Prefill " + format_name + " expert open group", error);
    if (!command) {
        check(false, "Prefill " + format_name +
                         " expert open-group command buffer",
              error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill " + format_name + " expert open group",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_expert(format, context, encoder, 1u, kExperts,
                                 kLayer, valid, encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, "rejects an open concurrent dispatch group", error);
    return passed && rejected;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_moe_expert_test strata.metallib\n");
        return 2;
    }
    std::string error;
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;
    ContextOptions foreign_options = options;
    foreign_options.pipelines = false;
    auto foreign = Context::open(foreign_options, error);
    check(foreign != nullptr, "open foreign Metal Context", error);
    if (!foreign) return 1;

    bool passed = true;
    for (const WeightFormat format :
         {WeightFormat::q2_0, WeightFormat::iq4_xs,
          WeightFormat::iq2_xxs, WeightFormat::iq2_xs,
          WeightFormat::iq2_s}) {
        passed &= test_validation(*context, *foreign, format);
    }
    passed &= test_decode_probe(*context);
    passed &= test_iq4_xs_literal_decode_anchor();
    passed &= test_iq2_xxs_literal_decode_anchor();
    passed &= test_iq2_xs_literal_decode_anchor();
    passed &= test_iq2_s_literal_decode_anchor();
    for (const WeightFormat format :
         {WeightFormat::q2_0, WeightFormat::iq4_xs,
          WeightFormat::iq2_xxs, WeightFormat::iq2_xs,
          WeightFormat::iq2_s}) {
        passed &= run_case(*context, format, 1u, true);
        passed &= run_case(*context, format, 7u, false);
        passed &= run_case(*context, format, 1u, false, true);
        passed &= run_case(*context, format, 7u, false, false, true);
    }
    std::printf("%s Prefill MoE expert tile GPU twin (%d failures)\n",
                passed ? "PASS" : "FAIL", g_failures);
    return passed && g_failures == 0 ? 0 : 1;
}
