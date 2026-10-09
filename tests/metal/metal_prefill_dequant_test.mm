// GPU oracle and buffer-contract tests for native Prefill dequantization.
#include "metallib_sha256.inc"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_dequant_dispatch.hpp"

#include "../../src/metal/kernels/common/types.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::size_t kGuardBytes = 4096u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint32_t kColumns = 257u;
constexpr std::uint32_t kK = 512u;

static_assert(sizeof(block_q4_0) == 18u);
static_assert(sizeof(block_q5_0) == 22u);
static_assert(sizeof(block_q8_0) == 34u);
static_assert(sizeof(block_q2_0) == 18u);
static_assert(sizeof(block_iq4_nl) == 18u);
static_assert(sizeof(block_q4_K) == 144u);
static_assert(sizeof(block_q5_K) == 176u);

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

class TestBuffer {
public:
    bool allocate(Context& context, const std::string& label,
                  std::size_t payload_bytes, std::string& error) {
        if (payload_bytes > std::numeric_limits<std::size_t>::max() -
                                2u * kGuardBytes) {
            error = label + " test-buffer size overflow";
            return false;
        }
        storage_ = context.buffers().allocate(
            label, payload_bytes + 2u * kGuardBytes, error);
        if (!storage_) return false;
        std::memset(storage_.contents, kPoison, storage_.length);
        view_ = storage_.slice(kGuardBytes, payload_bytes);
        if (!view_) {
            error = label + " test-buffer payload view is invalid";
            return false;
        }
        payload_bytes_ = payload_bytes;
        return true;
    }

    BufferRef view() const noexcept { return view_; }
    std::span<std::uint8_t> payload() noexcept {
        return {static_cast<std::uint8_t*>(view_.contents), payload_bytes_};
    }
    std::span<const std::uint8_t> allocation() const noexcept {
        return {static_cast<const std::uint8_t*>(storage_.contents),
                storage_.length};
    }
    bool guards_intact() const {
        const auto bytes = allocation();
        const std::size_t suffix = kGuardBytes + payload_bytes_;
        return std::all_of(bytes.begin(), bytes.begin() + kGuardBytes,
                           [](std::uint8_t value) {
                               return value == kPoison;
                           }) &&
               std::all_of(bytes.begin() + suffix, bytes.end(),
                           [](std::uint8_t value) {
                               return value == kPoison;
                           });
    }

private:
    BufferRef storage_{};
    BufferRef view_{};
    std::size_t payload_bytes_ = 0u;
};

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     std::string& error) {
    return command && command->commit(error) &&
           command->wait_until_completed(error);
}

template <typename Block>
std::size_t required_weight_bytes(std::uint32_t n_out, std::uint32_t k,
                                  std::uint32_t values_per_block) {
    return static_cast<std::size_t>(n_out) * (k / values_per_block) *
           sizeof(Block);
}

std::size_t required_output_bytes(std::uint32_t n_out, std::uint32_t k) {
    return static_cast<std::size_t>(n_out) * k * sizeof(std::uint16_t);
}

using EncodeCommandBuffer = bool (*)(
    Context&, CommandBuffer&, const PrefillDequantParams&,
    const PrefillDequantBuffers&, std::string&);
using EncodeEncoder = bool (*)(Context&, Encoder&,
                               const PrefillDequantParams&,
                               const PrefillDequantBuffers&, std::string&);

template <typename Block>
struct FormatTraits;

template <>
struct FormatTraits<block_q2_0> {
    static constexpr const char* name = "Q2_0";
    static constexpr std::uint32_t values_per_block = 64u;
    static constexpr std::uint32_t code_count = 4u;
    static constexpr EncodeCommandBuffer encode = &encode_prefill_dequant_q2_0;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q2_0;

    // Literal packed bytes and logical codes deliberately use separate
    // representations, so the fixture does not repeat the shader decoder.
    static constexpr std::array<std::uint8_t, 16> packed_pattern_a{
        0xe4u, 0xb4u, 0xd8u, 0x78u, 0x9cu, 0x6cu, 0xe1u, 0xb1u,
        0xc9u, 0x39u, 0x8du, 0x2du, 0xd2u, 0x72u, 0xc6u, 0x36u};
    static constexpr std::array<std::array<std::uint8_t, 4>, 16>
        logical_codes_a{{
            {{0u, 1u, 2u, 3u}}, {{0u, 1u, 3u, 2u}},
            {{0u, 2u, 1u, 3u}}, {{0u, 2u, 3u, 1u}},
            {{0u, 3u, 1u, 2u}}, {{0u, 3u, 2u, 1u}},
            {{1u, 0u, 2u, 3u}}, {{1u, 0u, 3u, 2u}},
            {{1u, 2u, 0u, 3u}}, {{1u, 2u, 3u, 0u}},
            {{1u, 3u, 0u, 2u}}, {{1u, 3u, 2u, 0u}},
            {{2u, 0u, 1u, 3u}}, {{2u, 0u, 3u, 1u}},
            {{2u, 1u, 0u, 3u}}, {{2u, 1u, 3u, 0u}},
        }};

    static void store_fixture(block_q2_0& block, std::uint32_t column,
                              std::uint32_t block_index) {
        const bool complement = ((column + block_index) & 1u) != 0u;
        for (std::size_t byte = 0u; byte < packed_pattern_a.size(); ++byte) {
            const std::uint8_t packed = packed_pattern_a[byte];
            block.qs[byte] = static_cast<strata_u8>(
                complement ? packed ^ 0xffu : packed);
        }
    }

    static std::uint32_t oracle_code(std::uint32_t column,
                                     std::uint32_t block_index,
                                     std::uint32_t value) {
        const std::uint32_t code =
            logical_codes_a[value >> 2u][value & 3u];
        const bool complement = ((column + block_index) & 1u) != 0u;
        return complement ? 3u - code : code;
    }

    static float dequantized_value(std::uint32_t code) {
        return static_cast<float>(static_cast<int>(code) - 1);
    }
};

template <>
struct FormatTraits<block_iq4_nl> {
    static constexpr const char* name = "IQ4_NL";
    static constexpr std::uint32_t values_per_block = 32u;
    static constexpr std::uint32_t code_count = 16u;
    static constexpr EncodeCommandBuffer encode =
        &encode_prefill_dequant_iq4_nl;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_iq4_nl;

    static constexpr std::array<std::uint8_t, 16> packed_pattern_a{
        0x10u, 0x32u, 0x54u, 0x76u, 0x98u, 0xbau, 0xdcu, 0xfeu,
        0xefu, 0xcdu, 0xabu, 0x89u, 0x67u, 0x45u, 0x23u, 0x01u};
    // Expected nibble indices are listed in K order, independently of the
    // packed-byte decoder and the GPU's codebook lookup.
    static constexpr std::array<std::uint8_t, 32> logical_codes_a{
        0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u, 15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u,
        1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u, 14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u};
    static constexpr std::array<std::int8_t, 16> iq4_nl_values{
        -127, -104, -83, -65, -49, -35, -22, -10,
        1,    13,   25,  38,  53,  69,  89,  113};

    static void store_fixture(block_iq4_nl& block, std::uint32_t column,
                              std::uint32_t block_index) {
        const bool complement = ((column + block_index) & 1u) != 0u;
        for (std::size_t byte = 0u; byte < packed_pattern_a.size(); ++byte) {
            const std::uint8_t packed = packed_pattern_a[byte];
            block.qs[byte] = static_cast<strata_u8>(
                complement ? packed ^ 0xffu : packed);
        }
    }

    static std::uint32_t oracle_code(std::uint32_t column,
                                     std::uint32_t block_index,
                                     std::uint32_t value) {
        const std::uint32_t code = logical_codes_a[value];
        const bool complement = ((column + block_index) & 1u) != 0u;
        return complement ? 15u - code : code;
    }

    static float dequantized_value(std::uint32_t code) {
        return static_cast<float>(iq4_nl_values[code]);
    }
};

template <>
struct FormatTraits<block_q4_K> {
    static constexpr const char* name = "Q4_K";
    static constexpr std::uint32_t values_per_block = 256u;
    static constexpr std::uint32_t code_count = 16u;
    static constexpr EncodeCommandBuffer encode =
        &encode_prefill_dequant_q4_k;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q4_k;

    // The packed bytes are literal. Expected codes below are listed in output
    // order and do not reuse the decoder's byte/nibble address calculation.
    static constexpr std::array<std::array<std::uint8_t, 16>, 8>
        packed_patterns{{
            {{0x10u, 0x32u, 0x54u, 0x76u, 0x98u, 0xbau, 0xdcu, 0xfeu,
              0xefu, 0xcdu, 0xabu, 0x89u, 0x67u, 0x45u, 0x23u, 0x01u}},
            {{0xf0u, 0xe1u, 0xd2u, 0xc3u, 0xb4u, 0xa5u, 0x96u, 0x87u,
              0x78u, 0x69u, 0x5au, 0x4bu, 0x3cu, 0x2du, 0x1eu, 0x0fu}},
            {{0x0fu, 0x1eu, 0x2du, 0x3cu, 0x4bu, 0x5au, 0x69u, 0x78u,
              0x87u, 0x96u, 0xa5u, 0xb4u, 0xc3u, 0xd2u, 0xe1u, 0xf0u}},
            {{0x78u, 0xf0u, 0x3cu, 0xb4u, 0x5au, 0xd2u, 0x1eu, 0x96u,
              0x69u, 0xe1u, 0x2du, 0xa5u, 0x4bu, 0xc3u, 0x0fu, 0x87u}},
            {{0x01u, 0x23u, 0x45u, 0x67u, 0x89u, 0xabu, 0xcdu, 0xefu,
              0xfeu, 0xdcu, 0xbau, 0x98u, 0x76u, 0x54u, 0x32u, 0x10u}},
            {{0xe1u, 0xd2u, 0xc3u, 0xb4u, 0xa5u, 0x96u, 0x87u, 0x78u,
              0x69u, 0x5au, 0x4bu, 0x3cu, 0x2du, 0x1eu, 0x0fu, 0xf0u}},
            {{0xf0u, 0x78u, 0xe1u, 0x69u, 0xd2u, 0x5au, 0xc3u, 0x4bu,
              0xb4u, 0x3cu, 0xa5u, 0x2du, 0x96u, 0x1eu, 0x87u, 0x0fu}},
            {{0x87u, 0x0fu, 0xc3u, 0x4bu, 0xa5u, 0x2du, 0xe1u, 0x69u,
              0x96u, 0x1eu, 0xd2u, 0x5au, 0xb4u, 0x3cu, 0xf0u, 0x78u}},
        }};
    static constexpr std::array<std::array<std::uint8_t, 16>, 8>
        logical_low_codes{{
            {{0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u,
              15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u}},
            {{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
              8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}},
            {{15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u,
              7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u}},
            {{8u, 0u, 12u, 4u, 10u, 2u, 14u, 6u,
              9u, 1u, 13u, 5u, 11u, 3u, 15u, 7u}},
            {{1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u,
              14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u}},
            {{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u,
              9u, 10u, 11u, 12u, 13u, 14u, 15u, 0u}},
            {{0u, 8u, 1u, 9u, 2u, 10u, 3u, 11u,
              4u, 12u, 5u, 13u, 6u, 14u, 7u, 15u}},
            {{7u, 15u, 3u, 11u, 5u, 13u, 1u, 9u,
              6u, 14u, 2u, 10u, 4u, 12u, 0u, 8u}},
        }};
    static constexpr std::array<std::array<std::uint8_t, 16>, 8>
        logical_high_codes{{
            {{1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u,
              14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u}},
            {{15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u,
              7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u}},
            {{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
              8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}},
            {{7u, 15u, 3u, 11u, 5u, 13u, 1u, 9u,
              6u, 14u, 2u, 10u, 4u, 12u, 0u, 8u}},
            {{0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u,
              15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u}},
            {{14u, 13u, 12u, 11u, 10u, 9u, 8u, 7u,
              6u, 5u, 4u, 3u, 2u, 1u, 0u, 15u}},
            {{15u, 7u, 14u, 6u, 13u, 5u, 12u, 4u,
              11u, 3u, 10u, 2u, 9u, 1u, 8u, 0u}},
            {{8u, 0u, 12u, 4u, 10u, 2u, 14u, 6u,
              9u, 1u, 13u, 5u, 11u, 3u, 15u, 7u}},
        }};
    static constexpr std::array<std::uint8_t, 12> packed_scales{
        0x85u, 0x4au, 0xcbu, 0x3eu, 0x75u, 0xdau,
        0x61u, 0xb6u, 0x9cu, 0xe2u, 0x5du, 0xa7u};
    static constexpr std::array<std::uint8_t, 8> logical_scales{
        5u, 10u, 11u, 62u, 44u, 18u, 61u, 7u};
    static constexpr std::array<std::uint8_t, 8> logical_mins{
        53u, 26u, 33u, 54u, 25u, 62u, 21u, 42u};

    static void store_fixture(block_q4_K& block, std::uint32_t column,
                              std::uint32_t block_index,
                              std::uint16_t d_bits,
                              std::uint16_t dmin_bits) {
        block.dm = static_cast<std::uint32_t>(d_bits) |
                   (static_cast<std::uint32_t>(dmin_bits) << 16u);
        std::copy(packed_scales.begin(), packed_scales.end(), block.scales);
        const bool complement = ((column + block_index) & 1u) != 0u;
        for (std::uint32_t chunk = 0u; chunk < 4u; ++chunk) {
            for (std::uint32_t byte = 0u; byte < 32u; ++byte) {
                const std::uint32_t pattern = chunk * 2u + byte / 16u;
                const std::uint8_t packed =
                    packed_patterns[pattern][byte % 16u];
                block.qs[chunk * 32u + byte] = static_cast<strata_u8>(
                    complement ? packed ^ 0xffu : packed);
            }
        }
    }

    static std::uint32_t oracle_code(std::uint32_t column,
                                    std::uint32_t block_index,
                                    std::uint32_t value) {
        const std::uint32_t qs_chunk = value / 64u;
        const std::uint32_t within_half = value % 64u;
        const std::uint32_t pattern =
            qs_chunk * 2u + (within_half % 32u) / 16u;
        const std::uint32_t code =
            within_half < 32u
                ? logical_low_codes[pattern][within_half % 16u]
                : logical_high_codes[pattern][(within_half - 32u) % 16u];
        const bool complement = ((column + block_index) & 1u) != 0u;
        return complement ? 15u - code : code;
    }

    static float expanded_value(std::uint32_t code, std::uint32_t value,
                                float d, float dmin) {
        const std::uint32_t group = value / 32u;
        const float d1 = d * static_cast<float>(logical_scales[group]);
        const float m1 = dmin * static_cast<float>(logical_mins[group]);
        return std::fma(d1, static_cast<float>(code), -m1);
    }
};

template <>
struct FormatTraits<block_q5_K> {
    static constexpr const char* name = "Q5_K";
    static constexpr std::uint32_t values_per_block = 256u;
    static constexpr std::uint32_t code_count = 32u;
    static constexpr EncodeCommandBuffer encode =
        &encode_prefill_dequant_q5_k;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q5_k;

    // Literal packed bytes encode the independent logical-code formula below.
    static constexpr std::array<std::uint8_t, 12> packed_scales{
        0x81u, 0x88u, 0xd0u, 0x1fu, 0xc0u, 0xc1u,
        0x50u, 0x1fu, 0xf0u, 0x0fu, 0x1fu, 0x2fu};
    static constexpr std::array<std::uint8_t, 32> packed_high_bits{
        0xe0u, 0xf8u, 0x3fu, 0x07u, 0xc1u, 0xf0u, 0x7cu, 0x1fu,
        0x03u, 0xe0u, 0xf8u, 0x3eu, 0x0fu, 0x81u, 0xf0u, 0x7cu,
        0x07u, 0xc1u, 0xf0u, 0x7cu, 0x1fu, 0x03u, 0xe0u, 0xf8u,
        0x3eu, 0x0fu, 0x81u, 0xf0u, 0x7cu, 0x1fu, 0x07u, 0xc0u};
    static constexpr std::array<std::uint8_t, 128> packed_low_bits{
        0x52u, 0xc9u, 0x30u, 0xa7u, 0x1eu, 0x85u, 0xfcu, 0x63u,
        0xdau, 0x41u, 0xb8u, 0x2fu, 0x96u, 0x0du, 0x74u, 0xebu,
        0xa7u, 0x1eu, 0x85u, 0xfcu, 0x63u, 0xdau, 0x41u, 0xb8u,
        0x2fu, 0x96u, 0x0du, 0x74u, 0xebu, 0x52u, 0xc9u, 0x30u,
        0xb8u, 0x2fu, 0x96u, 0x0du, 0x74u, 0xebu, 0x52u, 0xc9u,
        0x30u, 0xa7u, 0x1eu, 0x85u, 0xfcu, 0x63u, 0xdau, 0x41u,
        0x0du, 0x74u, 0xebu, 0x52u, 0xc9u, 0x30u, 0xa7u, 0x1eu,
        0x85u, 0xfcu, 0x63u, 0xdau, 0x41u, 0xb8u, 0x2fu, 0x96u,
        0x1eu, 0x85u, 0xfcu, 0x63u, 0xdau, 0x41u, 0xb8u, 0x2fu,
        0x96u, 0x0du, 0x74u, 0xebu, 0x52u, 0xc9u, 0x30u, 0xa7u,
        0x63u, 0xdau, 0x41u, 0xb8u, 0x2fu, 0x96u, 0x0du, 0x74u,
        0xebu, 0x52u, 0xc9u, 0x30u, 0xa7u, 0x1eu, 0x85u, 0xfcu,
        0x74u, 0xebu, 0x52u, 0xc9u, 0x30u, 0xa7u, 0x1eu, 0x85u,
        0xfcu, 0x63u, 0xdau, 0x41u, 0xb8u, 0x2fu, 0x96u, 0x0du,
        0xc9u, 0x30u, 0xa7u, 0x1eu, 0x85u, 0xfcu, 0x63u, 0xdau,
        0x41u, 0xb8u, 0x2fu, 0x96u, 0x0du, 0x74u, 0xebu, 0x52u};
    static constexpr std::array<std::uint8_t, 8> logical_scales{
        1u, 8u, 16u, 31u, 32u, 47u, 63u, 15u};
    static constexpr std::array<std::uint8_t, 8> logical_mins{
        0u, 1u, 16u, 31u, 63u, 48u, 17u, 2u};

    static void store_fixture(block_q5_K& block, std::uint32_t column,
                              std::uint32_t block_index,
                              std::uint16_t d_bits,
                              std::uint16_t dmin_bits) {
        block.dm = static_cast<std::uint32_t>(d_bits) |
                   (static_cast<std::uint32_t>(dmin_bits) << 16u);
        std::copy(packed_scales.begin(), packed_scales.end(), block.scales);
        const bool complement = ((column + block_index) & 1u) != 0u;
        for (std::uint32_t byte = 0u; byte < packed_high_bits.size(); ++byte) {
            block.qh[byte] = static_cast<strata_u8>(
                complement ? packed_high_bits[byte] ^ 0xffu
                           : packed_high_bits[byte]);
        }
        for (std::uint32_t byte = 0u; byte < packed_low_bits.size(); ++byte) {
            block.qs[byte] = static_cast<strata_u8>(
                complement ? packed_low_bits[byte] ^ 0xffu
                           : packed_low_bits[byte]);
        }
    }

    static std::uint32_t oracle_code(std::uint32_t column,
                                    std::uint32_t block_index,
                                    std::uint32_t value) {
        const std::uint32_t scale_group = value / 32u;
        const std::uint32_t within_group = value % 32u;
        const std::uint32_t code =
            (value * 7u + scale_group * 3u +
             (within_group / 16u) * 5u + 2u) %
            32u;
        const bool complement = ((column + block_index) & 1u) != 0u;
        return complement ? 31u - code : code;
    }

    static float expanded_value(std::uint32_t code, std::uint32_t value,
                                float d, float dmin) {
        const std::uint32_t group = value / 32u;
        const float d1 = d * static_cast<float>(logical_scales[group]);
        const float m1 = dmin * static_cast<float>(logical_mins[group]);
        return std::fma(d1, static_cast<float>(code), -m1);
    }
};

template <>
struct FormatTraits<block_q4_0> {
    static constexpr const char* name = "Q4_0";
    static constexpr std::uint32_t values_per_block = 32u;
    static constexpr std::uint32_t code_count = 16u;
    static constexpr EncodeCommandBuffer encode = &encode_prefill_dequant_q4_0;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q4_0;

    static void store(block_q4_0& block, std::uint32_t value,
                      std::uint32_t code) {
        const std::uint32_t shift = (value >> 4u) * 4u;
        block.qs[value & 15u] |=
            static_cast<strata_u8>((code & 0x0fu) << shift);
    }

    static float dequantized_value(std::uint32_t code) {
        return static_cast<float>(static_cast<int>(code) - 8);
    }
};

template <>
struct FormatTraits<block_q5_0> {
    static constexpr const char* name = "Q5_0";
    static constexpr std::uint32_t values_per_block = 32u;
    static constexpr std::uint32_t code_count = 32u;
    static constexpr EncodeCommandBuffer encode = &encode_prefill_dequant_q5_0;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q5_0;

    static void store(block_q5_0& block, std::uint32_t value,
                      std::uint32_t code) {
        const std::uint32_t shift = (value >> 4u) * 4u;
        block.qs[value & 15u] |=
            static_cast<strata_u8>((code & 0x0fu) << shift);
        block.qh[value >> 3u] |=
            static_cast<strata_u8>(((code >> 4u) & 1u) << (value & 7u));
    }

    static float dequantized_value(std::uint32_t code) {
        return static_cast<float>(static_cast<int>(code) - 16);
    }
};

template <>
struct FormatTraits<block_q8_0> {
    static constexpr const char* name = "Q8_0";
    static constexpr std::uint32_t values_per_block = 32u;
    static constexpr std::uint32_t code_count = 255u;
    static constexpr EncodeCommandBuffer encode = &encode_prefill_dequant_q8_0;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q8_0;

    static void store(block_q8_0& block, std::uint32_t value,
                      std::uint32_t code) {
        block.qs[value] =
            static_cast<strata_i8>(static_cast<int>(code) - 127);
    }

    static float dequantized_value(std::uint32_t code) {
        return static_cast<float>(static_cast<int>(code) - 127);
    }
};

bool expect_rejected(Context& context, EncodeCommandBuffer encode,
                     const PrefillDequantParams& params,
                     const PrefillDequantBuffers& buffers,
                     std::string_view expected_error,
                     const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode(context, *command, params, buffers, error);
    const bool rejected =
        !encoded && command->unsafe_native_handle() == nullptr &&
        error.find(expected_error) != std::string::npos;
    check(rejected, label, error);
    return rejected;
}

template <typename Block>
bool test_dequant_format(Context& context) {
    using Format = FormatTraits<Block>;
    const std::string prefix =
        std::string("Prefill ") + Format::name + " dequant";
    constexpr std::size_t blocks_per_row =
        kK / Format::values_per_block;
    const std::size_t weight_bytes =
        required_weight_bytes<Block>(kColumns, kK,
                                     Format::values_per_block);
    const std::size_t output_bytes =
        required_output_bytes(kColumns, kK);
    std::string error;
    TestBuffer weights, output;
    const bool allocated =
        weights.allocate(context, prefix + " weights", weight_bytes + 2u,
                         error) &&
        output.allocate(context, prefix + " K-major F16", output_bytes + 2u,
                        error);
    check(allocated, prefix + " allocate buffers", error);
    if (!allocated) return false;

    constexpr std::array<std::uint16_t, 5> scales{
        0x0001u, 0x0400u, 0x3555u, 0x3c00u, 0x7bffu};
    std::vector<Block> source(kColumns * blocks_per_row);
    std::vector<std::uint16_t> expected(kColumns * kK);
    for (std::uint32_t column = 0u; column < kColumns; ++column) {
        for (std::uint32_t block_index = 0u;
             block_index < blocks_per_row; ++block_index) {
            Block& block =
                source[static_cast<std::size_t>(column) * blocks_per_row +
                       block_index];
            block = {};
            const std::uint16_t scale =
                scales[(column + block_index) % scales.size()];
            const float expanded_scale =
                strata::kernels::f32_from_f16(scale);
            constexpr std::array<std::uint16_t, 5> min_scales{
                0x0001u, 0x0400u, 0x3000u, 0x3800u, 0x6bffu};
            const std::uint16_t min_scale =
                min_scales[(column * 3u + block_index) %
                           min_scales.size()];
            const float expanded_min_scale =
                strata::kernels::f32_from_f16(min_scale);
            if constexpr (std::is_same_v<Block, block_q4_K> ||
                          std::is_same_v<Block, block_q5_K>) {
                Format::store_fixture(block, column, block_index, scale,
                                      min_scale);
            } else {
                block.d = static_cast<strata_half>(scale);
            }
            for (std::uint32_t value = 0u;
                 value < Format::values_per_block; ++value) {
                const std::uint32_t code =
                    (column * 13u + block_index * 31u + value * 7u +
                     (value >> 4u) * 3u) %
                    Format::code_count;
                std::uint32_t oracle_code = code;
                if constexpr (std::is_same_v<Block, block_q2_0> ||
                              std::is_same_v<Block, block_iq4_nl>) {
                    if (value == 0u) {
                        Format::store_fixture(block, column, block_index);
                    }
                    oracle_code =
                        Format::oracle_code(column, block_index, value);
                } else if constexpr (std::is_same_v<Block, block_q4_K>) {
                    oracle_code =
                        Format::oracle_code(column, block_index, value);
                } else if constexpr (std::is_same_v<Block, block_q5_K>) {
                    oracle_code =
                        Format::oracle_code(column, block_index, value);
                } else {
                    Format::store(block, value, code);
                }
                const std::uint32_t k =
                    block_index * Format::values_per_block + value;
                float expanded_value = 0.0f;
                if constexpr (std::is_same_v<Block, block_q4_K> ||
                              std::is_same_v<Block, block_q5_K>) {
                    expanded_value = Format::expanded_value(
                        oracle_code, value, expanded_scale,
                        expanded_min_scale);
                } else {
                    expanded_value =
                        expanded_scale *
                        Format::dequantized_value(oracle_code);
                }
                expected[static_cast<std::size_t>(k) * kColumns + column] =
                    strata::kernels::f16_from_f32(expanded_value);
            }
        }
    }
    std::memcpy(weights.payload().data(), source.data(), weight_bytes);
    std::fill(output.payload().begin(), output.payload().end(), kPoison);
    const std::vector<std::uint8_t> weights_before(
        weights.allocation().begin(), weights.allocation().end());

    const PrefillDequantParams params{kColumns, kK};
    const PrefillDequantBuffers buffers{
        context.buffers().read_only(weights.view().slice(0u, weight_bytes)),
        output.view().slice(0u, output_bytes)};
    auto rollback_command =
        context.make_command_buffer(prefix + " caller-owned rollback", error);
    check(rollback_command != nullptr,
          prefix + " caller-owned rollback command buffer", error);
    if (!rollback_command) return false;
    const PrefillDequantParams invalid_params{0u, kK};
    bool prefix_dispatch_encoded = false;
    bool late_call_rejected = false;
    const bool transaction_succeeded = Encoder::encode(
        context, *rollback_command, prefix + " caller-owned rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            prefix_dispatch_encoded = Format::encode_encoder(
                context, encoder, params, buffers, encode_error);
            if (!prefix_dispatch_encoded) {
                return false;
            }
            late_call_rejected = !Format::encode_encoder(
                context, encoder, invalid_params, buffers, encode_error);
            return false;
        },
        error);
    const bool discard_verified = prefix_dispatch_encoded &&
        late_call_rejected && !transaction_succeeded &&
        rollback_command->unsafe_native_handle() == nullptr &&
        std::all_of(output.payload().begin(), output.payload().end(),
                    [](std::uint8_t value) { return value == kPoison; });
    check(discard_verified,
          prefix + " caller-owned late failure discards the earlier dispatch",
          error);
    if (!discard_verified) return false;

    auto command = context.make_command_buffer(prefix, error);
    check(command != nullptr, prefix + " command buffer", error);
    if (!command) return false;
    const bool encoded = Format::encode(context, *command, params, buffers,
                                        error);
    check(encoded, prefix + " encode", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(command, error);
    check(completed, prefix + " GPU dispatch", error);
    if (!completed) return false;

    const bool exact =
        std::memcmp(output.payload().data(), expected.data(), output_bytes) == 0;
    const bool input_unchanged =
        std::equal(weights_before.begin(), weights_before.end(),
                   weights.allocation().begin());
    const auto output_payload = output.payload();
    const bool trailing_bytes_untouched =
        output_payload[output_bytes] == kPoison &&
        output_payload[output_bytes + 1u] == kPoison;
    check(exact, std::string(Format::name) +
                     " expansion matches exact F16 rounding and K-major layout");
    check(input_unchanged, std::string(Format::name) +
                               " source weights remain unchanged");
    check(trailing_bytes_untouched,
          std::string(Format::name) +
              " dequant leaves unused output bytes untouched");
    check(weights.guards_intact() && output.guards_intact(),
          std::string(Format::name) + " dequant input and output guards");
    return exact && input_unchanged && trailing_bytes_untouched &&
           weights.guards_intact() && output.guards_intact();
}

template <typename Block>
bool test_validation_format(Context& context, Context& foreign_context) {
    using Format = FormatTraits<Block>;
    const std::string prefix =
        std::string("Prefill ") + Format::name + " dequant validation";
    const PrefillDequantParams valid{kColumns, kK};
    const std::size_t weight_bytes =
        required_weight_bytes<Block>(kColumns, kK,
                                     Format::values_per_block);
    const std::size_t output_bytes =
        required_output_bytes(kColumns, kK);
    std::string error;
    TestBuffer weights, output, alias, foreign_weights, foreign_output;
    const bool allocated =
        weights.allocate(context, prefix + " weights", weight_bytes + 2u,
                         error) &&
        output.allocate(context, prefix + " output", output_bytes + 2u,
                        error) &&
        alias.allocate(context, prefix + " aliased storage",
                       output_bytes + 2u, error) &&
        foreign_weights.allocate(foreign_context, prefix + " foreign weights",
                                 weight_bytes, error) &&
        foreign_output.allocate(foreign_context, prefix + " foreign output",
                                output_bytes, error);
    check(allocated, prefix + " allocate buffers", error);
    if (!allocated) return false;

    const PrefillDequantBuffers valid_buffers{
        context.buffers().read_only(weights.view().slice(0u, weight_bytes)),
        output.view().slice(0u, output_bytes)};
    bool passed = true;
    auto invalid = valid;
    invalid.n_out = 0u;
    passed &= expect_rejected(context, Format::encode, invalid, valid_buffers,
                              "positive signed extents", prefix + " zero N");
    invalid = valid;
    invalid.k = Format::values_per_block - 1u;
    passed &= expect_rejected(context, Format::encode, invalid, valid_buffers,
                              "positive signed extents", prefix + " short K");
    invalid = valid;
    invalid.k = Format::values_per_block + 1u;
    passed &= expect_rejected(context, Format::encode, invalid, valid_buffers,
                              "positive signed extents", prefix + " K remainder");
    invalid = valid;
    invalid.n_out =
        static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max()) +
        1u;
    passed &= expect_rejected(context, Format::encode, invalid, valid_buffers,
                              "positive signed extents", prefix + " signed N");
    invalid = valid;
    invalid.k =
        (static_cast<std::uint32_t>(
             std::numeric_limits<std::int32_t>::max()) /
         Format::values_per_block +
         1u) *
        Format::values_per_block;
    passed &= expect_rejected(context, Format::encode, invalid, valid_buffers,
                              "positive signed extents", prefix + " signed K");

    const PrefillDequantBuffers short_weights{
        context.buffers().read_only(
            weights.view().slice(0u, weight_bytes - 1u)),
        valid_buffers.k_major_f16};
    passed &= expect_rejected(context, Format::encode, valid, short_weights,
                              "weight buffer is undersized",
                              prefix + " short weights");
    const PrefillDequantBuffers short_output{
        valid_buffers.weights, output.view().slice(0u, output_bytes - 1u)};
    passed &= expect_rejected(context, Format::encode, valid, short_output,
                              "output is undersized", prefix + " short output");

    const PrefillDequantBuffers misaligned_weights{
        context.buffers().read_only(weights.view().slice(1u, weight_bytes)),
        valid_buffers.k_major_f16};
    passed &= expect_rejected(context, Format::encode, valid,
                              misaligned_weights, "misaligned",
                              prefix + " misaligned weights");
    const PrefillDequantBuffers misaligned_output{
        valid_buffers.weights, output.view().slice(1u, output_bytes)};
    passed &= expect_rejected(context, Format::encode, valid,
                              misaligned_output, "misaligned",
                              prefix + " misaligned output");

    const PrefillDequantBuffers overlapping{
        context.buffers().read_only(alias.view().slice(0u, weight_bytes)),
        alias.view().slice(0u, output_bytes)};
    passed &= expect_rejected(context, Format::encode, valid, overlapping,
                              "overlaps its weights", prefix + " input/output alias");
    const PrefillDequantBuffers partially_overlapping{
        context.buffers().read_only(alias.view().slice(0u, weight_bytes)),
        alias.view().slice(2u, output_bytes)};
    passed &= expect_rejected(
        context, Format::encode, valid, partially_overlapping,
        "overlaps its weights",
        prefix + " partial input/output overlap");

    const PrefillDequantBuffers foreign{
        foreign_context.buffers().read_only(foreign_weights.view()),
        valid_buffers.k_major_f16};
    passed &= expect_rejected(context, Format::encode, valid, foreign,
                              "not owned by this Context",
                              prefix + " foreign input");
    const PrefillDequantBuffers foreign_output_buffers{
        valid_buffers.weights, foreign_output.view()};
    passed &= expect_rejected(context, Format::encode, valid,
                              foreign_output_buffers,
                              "not owned by this Context",
                              prefix + " foreign output");
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_dequant_test strata.metallib\n");
        return 2;
    }

    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;
    ContextOptions foreign_options = options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal Context", error);
    if (!foreign_context) return 1;

    bool passed = true;
    passed &= test_dequant_format<block_q2_0>(*context);
    passed &= test_validation_format<block_q2_0>(*context,
                                                  *foreign_context);
    passed &= test_dequant_format<block_iq4_nl>(*context);
    passed &= test_validation_format<block_iq4_nl>(*context,
                                                    *foreign_context);
    passed &= test_dequant_format<block_q4_K>(*context);
    passed &= test_validation_format<block_q4_K>(*context,
                                                  *foreign_context);
    passed &= test_dequant_format<block_q5_K>(*context);
    passed &= test_validation_format<block_q5_K>(*context,
                                                  *foreign_context);
    passed &= test_dequant_format<block_q4_0>(*context);
    passed &= test_validation_format<block_q4_0>(*context,
                                                 *foreign_context);
    passed &= test_dequant_format<block_q5_0>(*context);
    passed &= test_validation_format<block_q5_0>(*context,
                                                 *foreign_context);
    passed &= test_dequant_format<block_q8_0>(*context);
    passed &= test_validation_format<block_q8_0>(*context,
                                                 *foreign_context);
    return passed && g_failures == 0 ? 0 : 1;
}
