// GPU oracle and buffer-contract tests for native Prefill dequantization.
#include "metallib_sha256.inc"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_dequant_dispatch.hpp"
#include "strata/metal/prefill_gemm_dispatch.hpp"
#include "strata/metal/prefill_native_projection.hpp"
#include "strata/metal/prefill_qsa_gate_dispatch.hpp"
#include "strata/metal/prefill_qsa_output_dispatch.hpp"

#include "../../src/metal/kernels/common/types.h"
#include "../../src/metal/kernels/common/codebooks.h"

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
#include <span>
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
static_assert(sizeof(block_q6_K) == 210u);
static_assert(sizeof(block_q3_K) == 110u);
static_assert(sizeof(block_iq4_xs) == 136u);
static_assert(sizeof(block_iq3_s) == 110u);

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

class ScopedEnvironmentOverride {
public:
    ScopedEnvironmentOverride(const char* name, const char* value)
        : name_(name) {
        const char* const previous = std::getenv(name);
        was_set_ = previous != nullptr;
        if (was_set_) previous_value_ = previous;
        ready_ = setenv(name_.c_str(), value, 1) == 0;
    }

    ~ScopedEnvironmentOverride() {
        if (!ready_) return;
        if (was_set_) {
            setenv(name_.c_str(), previous_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    bool ready() const noexcept { return ready_; }

private:
    std::string name_;
    std::string previous_value_;
    bool was_set_ = false;
    bool ready_ = false;
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
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q2_0;
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
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::IQ4_NL;
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
struct FormatTraits<block_iq4_xs> {
    static constexpr const char* name = "IQ4_XS";
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::IQ4_XS;
    static constexpr std::uint32_t values_per_block = 256u;
    static constexpr std::uint32_t code_count = 16u;
    static constexpr EncodeCommandBuffer encode =
        &encode_prefill_dequant_iq4_xs;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_iq4_xs;

    // Eight literal 16-byte segments make both nibble halves and each
    // 32-value group observably different.
    static constexpr std::array<std::uint8_t, 128> packed_qs{
        0xf0u, 0xe1u, 0xd2u, 0xc3u, 0xb4u, 0xa5u, 0x96u, 0x87u,
        0x78u, 0x69u, 0x5au, 0x4bu, 0x3cu, 0x2du, 0x1eu, 0x0fu,
        0x23u, 0x14u, 0x05u, 0xf6u, 0xe7u, 0xd8u, 0xc9u, 0xbau,
        0xabu, 0x9cu, 0x8du, 0x7eu, 0x6fu, 0x50u, 0x41u, 0x32u,
        0x56u, 0x47u, 0x38u, 0x29u, 0x1au, 0x0bu, 0xfcu, 0xedu,
        0xdeu, 0xcfu, 0xb0u, 0xa1u, 0x92u, 0x83u, 0x74u, 0x65u,
        0x89u, 0x7au, 0x6bu, 0x5cu, 0x4du, 0x3eu, 0x2fu, 0x10u,
        0x01u, 0xf2u, 0xe3u, 0xd4u, 0xc5u, 0xb6u, 0xa7u, 0x98u,
        0xbcu, 0xadu, 0x9eu, 0x8fu, 0x70u, 0x61u, 0x52u, 0x43u,
        0x34u, 0x25u, 0x16u, 0x07u, 0xf8u, 0xe9u, 0xdau, 0xcbu,
        0xefu, 0xd0u, 0xc1u, 0xb2u, 0xa3u, 0x94u, 0x85u, 0x76u,
        0x67u, 0x58u, 0x49u, 0x3au, 0x2bu, 0x1cu, 0x0du, 0xfeu,
        0x12u, 0x03u, 0xf4u, 0xe5u, 0xd6u, 0xc7u, 0xb8u, 0xa9u,
        0x9au, 0x8bu, 0x7cu, 0x6du, 0x5eu, 0x4fu, 0x30u, 0x21u,
        0x45u, 0x36u, 0x27u, 0x18u, 0x09u, 0xfau, 0xebu, 0xdcu,
        0xcdu, 0xbeu, 0xafu, 0x90u, 0x81u, 0x72u, 0x63u, 0x54u};
    static constexpr std::array<std::uint8_t, 8> logical_scales{
        0u, 1u, 2u, 15u, 16u, 17u, 31u, 63u};
    static constexpr std::array<std::int8_t, 16> iq4_nl_values{
        -127, -104, -83, -65, -49, -35, -22, -10,
        1,    13,   25,  38,  53,  69,  89,  113};

    static std::uint32_t scale_rotation(std::uint32_t column,
                                        std::uint32_t block_index) {
        return (column * 2u + block_index) % logical_scales.size();
    }

    static std::uint32_t scale_offset(std::uint32_t column,
                                      std::uint32_t block_index) {
        return (column * 2u + block_index) / logical_scales.size();
    }

    static std::uint32_t scale_epoch(std::uint32_t column,
                                     std::uint32_t block_index) {
        return (column * 2u + block_index) / 512u;
    }

    static std::uint32_t logical_scale(std::uint32_t column,
                                       std::uint32_t block_index,
                                       std::uint32_t group) {
        const std::uint32_t index =
            (group + scale_rotation(column, block_index)) %
            logical_scales.size();
        return (logical_scales[index] +
                scale_offset(column, block_index) +
                scale_epoch(column, block_index) * (7u + 11u * group)) &
               0x3fu;
    }

    static void store_fixture(block_iq4_xs& block, std::uint32_t column,
                              std::uint32_t block_index,
                              std::uint16_t d_bits) {
        block.d = d_bits;
        const bool complement = ((column + block_index) & 1u) != 0u;
        for (std::uint32_t byte = 0u; byte < packed_qs.size(); ++byte) {
            block.qs[byte] = static_cast<strata_u8>(
                complement ? packed_qs[byte] ^ 0xffu : packed_qs[byte]);
        }

        for (std::uint32_t group = 0u; group < 4u; ++group) {
            block.scales_l[group] = 0u;
        }
        std::uint16_t scales_high = 0u;
        for (std::uint32_t group = 0u; group < logical_scales.size();
             ++group) {
            const std::uint32_t scale =
                logical_scale(column, block_index, group);
            const std::uint32_t shift = 4u * (group % 2u);
            const std::uint32_t old =
                static_cast<std::uint32_t>(block.scales_l[group / 2u]);
            block.scales_l[group / 2u] = static_cast<strata_u8>(
                old | ((scale & 0x0fu) << shift));
            scales_high = static_cast<std::uint16_t>(
                scales_high |
                (((scale >> 4u) & 0x03u) << (2u * group)));
        }
        block.scales_h = static_cast<strata_u16>(scales_high);
    }

    static std::uint32_t oracle_code(std::uint32_t column,
                                    std::uint32_t block_index,
                                    std::uint32_t value) {
        const std::uint32_t group = value / 32u;
        const std::uint32_t within_group = value % 32u;
        const std::uint32_t base =
            within_group < 16u ? within_group : 31u - within_group;
        const std::uint32_t code = (base + 3u * group) & 0x0fu;
        const bool complement = ((column + block_index) & 1u) != 0u;
        return complement ? 15u - code : code;
    }

    static float expanded_value(std::uint32_t code, std::uint32_t value,
                                float d, std::uint32_t column,
                                std::uint32_t block_index) {
        const float scale =
            static_cast<float>(
                static_cast<std::int32_t>(
                    logical_scale(column, block_index, value / 32u)) -
                32);
        return (d * scale) * static_cast<float>(iq4_nl_values[code]);
    }
};

template <>
struct FormatTraits<block_q4_K> {
    static constexpr const char* name = "Q4_K";
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q4_K;
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
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q5_K;
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
struct FormatTraits<block_q6_K> {
    static constexpr const char* name = "Q6_K";
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q6_K;
    static constexpr std::uint32_t values_per_block = 256u;
    static constexpr std::uint32_t code_count = 64u;
    static constexpr EncodeCommandBuffer encode =
        &encode_prefill_dequant_q6_k;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q6_k;

    // Literal ql/qh planes encode all 64 codes. Every 16-byte half has
    // distinct bytes so a wrong sub-block base or lane half is observable.
    static constexpr std::array<std::uint8_t, 128> packed_ql{
        0x73u, 0x40u, 0x1du, 0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu,
        0xfbu, 0xc8u, 0x95u, 0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u,
        0xc8u, 0x95u, 0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u, 0x73u,
        0x40u, 0x1du, 0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu, 0xfbu,
        0x1du, 0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu, 0xfbu, 0xc8u,
        0x95u, 0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u, 0x73u, 0x40u,
        0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u, 0x73u, 0x40u, 0x1du,
        0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu, 0xfbu, 0xc8u, 0x95u,
        0xfbu, 0xc8u, 0x95u, 0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u,
        0x73u, 0x40u, 0x1du, 0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu,
        0x40u, 0x1du, 0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu, 0xfbu,
        0xc8u, 0x95u, 0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u, 0x73u,
        0x95u, 0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u, 0x73u, 0x40u,
        0x1du, 0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu, 0xfbu, 0xc8u,
        0xeau, 0xb7u, 0x84u, 0x51u, 0x2eu, 0xfbu, 0xc8u, 0x95u,
        0x62u, 0x3fu, 0x0cu, 0xd9u, 0xa6u, 0x73u, 0x40u, 0x1du};
    static constexpr std::array<std::uint8_t, 64> packed_qh{
        0x18u, 0x2du, 0x71u, 0xb6u, 0xcbu, 0x18u, 0x2du, 0x71u,
        0xb6u, 0xcbu, 0x18u, 0x6du, 0x71u, 0x86u, 0xcbu, 0x1cu,
        0x61u, 0xb2u, 0xc7u, 0xdbu, 0x2cu, 0x61u, 0xb6u, 0xc7u,
        0xd8u, 0x2cu, 0x61u, 0xb6u, 0xc7u, 0xd8u, 0x2cu, 0x61u,
        0xb6u, 0xcbu, 0x18u, 0x6du, 0x71u, 0x86u, 0xcbu, 0x1cu,
        0x6du, 0x72u, 0x86u, 0xcbu, 0x1cu, 0x6du, 0x72u, 0x86u,
        0xd8u, 0x2cu, 0x61u, 0xb6u, 0xc7u, 0xd8u, 0x2cu, 0x61u,
        0xb6u, 0xc7u, 0x18u, 0x2cu, 0x71u, 0xb6u, 0xcbu, 0x18u};
    static constexpr std::array<std::int8_t, 16> logical_scales{
        1, -1, 2, -2, 3, -3, 5, -5,
        7, -7, 15, -15, 31, -31, 63, -63};
    static constexpr std::uint32_t scale_count = 16u;

    static std::uint32_t scale_rotation(std::uint32_t column,
                                        std::uint32_t block_index) {
        return (column * 3u + block_index * 5u) % scale_count;
    }

    static void store_fixture(block_q6_K& block, std::uint32_t column,
                              std::uint32_t block_index,
                              std::uint16_t d_bits) {
        block.d = d_bits;
        const bool complement = ((column + block_index) & 1u) != 0u;
        for (std::uint32_t byte = 0u; byte < packed_ql.size(); ++byte) {
            block.ql[byte] = static_cast<strata_u8>(
                complement ? packed_ql[byte] ^ 0xffu : packed_ql[byte]);
        }
        for (std::uint32_t byte = 0u; byte < packed_qh.size(); ++byte) {
            block.qh[byte] = static_cast<strata_u8>(
                complement ? packed_qh[byte] ^ 0xffu : packed_qh[byte]);
        }
        const std::uint32_t rotation = scale_rotation(column, block_index);
        for (std::uint32_t group = 0u; group < logical_scales.size();
             ++group) {
            block.scales[group] = static_cast<strata_i8>(
                logical_scales[(group + rotation) % logical_scales.size()]);
        }
    }

    static std::uint32_t oracle_code(std::uint32_t column,
                                    std::uint32_t block_index,
                                    std::uint32_t value) {
        const std::uint32_t code =
            (value * 13u + (value / 16u) * 5u + 3u) % 64u;
        const bool complement = ((column + block_index) & 1u) != 0u;
        return complement ? 63u - code : code;
    }

    static float expanded_value(std::uint32_t code, std::uint32_t value,
                                float d, std::uint32_t column,
                                std::uint32_t block_index) {
        const std::uint32_t scale_group = value / 16u;
        const std::uint32_t scale_index =
            (scale_group + scale_rotation(column, block_index)) %
            scale_count;
        const float scale = static_cast<float>(logical_scales[scale_index]);
        return (d * scale) * static_cast<float>(static_cast<int>(code) - 32);
    }
};

template <>
struct FormatTraits<block_q3_K> {
    static constexpr const char* name = "Q3_K";
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q3_K;
    static constexpr std::uint32_t values_per_block = 256u;
    static constexpr std::uint32_t code_count = 8u;
    static constexpr EncodeCommandBuffer encode =
        &encode_prefill_dequant_q3_k;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_q3_k;

    // Literal hmask/qs bytes encode all eight signed codes and distinguish
    // every lane and 32-byte qs half; the separate logical-code oracle does
    // not unpack these.
    static constexpr std::array<std::uint8_t, 32> packed_hmask{
        0x00u, 0x21u, 0x42u, 0x63u, 0x84u, 0xa5u, 0xc6u, 0xe7u,
        0xa8u, 0x89u, 0xeau, 0xcbu, 0x2cu, 0x0du, 0x6eu, 0x4fu,
        0x50u, 0x71u, 0x12u, 0x33u, 0xd4u, 0xf5u, 0x96u, 0xb7u,
        0xf8u, 0xd9u, 0xbau, 0x9bu, 0x7cu, 0x5du, 0x3eu, 0x1fu};
    static constexpr std::array<std::uint8_t, 64> packed_qs{
        0x6cu, 0x1bu, 0xc6u, 0xb1u, 0x55u, 0x00u, 0xffu, 0xaau,
        0x93u, 0x4eu, 0x39u, 0xe4u, 0x88u, 0x77u, 0x22u, 0xddu,
        0x1bu, 0xc6u, 0xb1u, 0x6cu, 0x00u, 0xffu, 0xaau, 0x55u,
        0x4eu, 0x39u, 0xe4u, 0x93u, 0x77u, 0x22u, 0xddu, 0x88u,
        0x39u, 0x4eu, 0x93u, 0xe4u, 0x00u, 0x55u, 0xaau, 0xffu,
        0xc6u, 0x1bu, 0x6cu, 0xb1u, 0xddu, 0x22u, 0x77u, 0x88u,
        0x4eu, 0x93u, 0xe4u, 0x39u, 0x55u, 0xaau, 0xffu, 0x00u,
        0x1bu, 0x6cu, 0xb1u, 0xc6u, 0x22u, 0x77u, 0x88u, 0xddu};
    static constexpr std::array<std::uint8_t, 12> packed_scales{
        0x10u, 0xf1u, 0x02u, 0x1fu, 0xd0u, 0xe1u,
        0xffu, 0xe0u, 0xe4u, 0xe4u, 0xf4u, 0x78u};
    static constexpr std::array<std::uint8_t, 16> logical_scales{
        0u,  1u,  2u,  15u, 16u, 17u, 31u, 32u,
        33u, 47u, 48u, 49u, 61u, 62u, 63u, 30u};

    static std::uint32_t scale_rotation(std::uint32_t column,
                                        std::uint32_t block_index) {
        return (column * 2u + block_index) % logical_scales.size();
    }

    static std::uint32_t scale_offset(std::uint32_t column,
                                      std::uint32_t block_index) {
        return (column * 2u + block_index) / logical_scales.size();
    }

    static void store_fixture(block_q3_K& block, std::uint32_t column,
                              std::uint32_t block_index,
                              std::uint16_t d_bits) {
        block.d = d_bits;
        const bool complement = ((column + block_index) & 1u) != 0u;
        for (std::uint32_t byte = 0u; byte < packed_hmask.size(); ++byte) {
            block.hmask[byte] = static_cast<strata_u8>(
                complement ? packed_hmask[byte] ^ 0xffu
                           : packed_hmask[byte]);
        }
        for (std::uint32_t byte = 0u; byte < packed_qs.size(); ++byte) {
            block.qs[byte] = static_cast<strata_u8>(
                complement ? packed_qs[byte] ^ 0xffu : packed_qs[byte]);
        }
        const std::uint32_t row_index = column * 2u + block_index;
        if (row_index == 0u) {
            std::copy(packed_scales.begin(), packed_scales.end(),
                      block.scales);
            return;
        }

        std::array<std::uint8_t, 16> scales{};
        std::array<std::uint8_t, 12> packed{};
        const std::uint32_t rotation = scale_rotation(column, block_index);
        const std::uint32_t offset = scale_offset(column, block_index);
        for (std::uint32_t group = 0u; group < scales.size(); ++group) {
            scales[group] = static_cast<std::uint8_t>(
                (logical_scales[(group + rotation) % logical_scales.size()] +
                 offset) &
                0x3fu);
        }
        for (std::uint32_t group = 0u; group < 4u; ++group) {
            packed[group] = static_cast<std::uint8_t>(
                (scales[group] & 0x0fu) |
                ((scales[group + 8u] & 0x0fu) << 4u));
            packed[group + 4u] = static_cast<std::uint8_t>(
                (scales[group + 4u] & 0x0fu) |
                ((scales[group + 12u] & 0x0fu) << 4u));
            packed[group + 8u] = static_cast<std::uint8_t>(
                ((scales[group] >> 4u) & 0x03u) |
                (((scales[group + 4u] >> 4u) & 0x03u) << 2u) |
                (((scales[group + 8u] >> 4u) & 0x03u) << 4u) |
                (((scales[group + 12u] >> 4u) & 0x03u) << 6u));
        }
        std::copy(packed.begin(), packed.end(), block.scales);
    }

    static bool positive(std::uint32_t group, std::uint32_t lane) {
        if (group < 5u) {
            return ((lane >> group) & 1u) != 0u;
        }
        if (group == 5u) {
            return ((lane ^ (lane >> 3u)) & 1u) != 0u;
        }
        if (group == 6u) {
            return (((lane >> 1u) ^ (lane >> 4u)) & 1u) != 0u;
        }
        return (((lane >> 2u) ^ (lane >> 3u)) & 1u) != 0u;
    }

    static std::uint32_t oracle_code(std::uint32_t column,
                                    std::uint32_t block_index,
                                    std::uint32_t value) {
        const std::uint32_t group = value / 32u;
        const std::uint32_t lane = value % 32u;
        const std::uint32_t q2 =
            (value * 3u + group * 7u + lane / 16u +
             (lane / 4u) * (group + 1u) + lane / 8u) %
            4u;
        // The two qs halves encode separate 128-value regions. Toggle the
        // low q bit in the latter so a missing 32-byte half offset fails.
        const std::uint32_t half_adjusted_q2 = q2 ^ (group / 4u);
        std::int32_t code =
            static_cast<std::int32_t>(half_adjusted_q2) -
            (positive(group, lane) ? 0 : 4);
        if (((column + block_index) & 1u) != 0u) {
            code = -1 - code;
        }
        return static_cast<std::uint32_t>(code + 4);
    }

    static float expanded_value(std::uint32_t code, std::uint32_t value,
                                float d, std::uint32_t column,
                                std::uint32_t block_index) {
        const std::uint32_t scale_group = value / 16u;
        const std::uint32_t scale_index =
            (scale_group + scale_rotation(column, block_index)) %
            logical_scales.size();
        const std::uint32_t scale =
            (logical_scales[scale_index] +
             scale_offset(column, block_index)) &
            0x3fu;
        const float signed_code =
            static_cast<float>(static_cast<std::int32_t>(code) - 4);
        return (d * (static_cast<float>(scale) - 32.0f)) * signed_code;
    }
};

template <>
struct FormatTraits<block_q4_0> {
    static constexpr const char* name = "Q4_0";
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q4_0;
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
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q5_0;
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
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::Q8_0;
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

template <>
struct FormatTraits<block_iq3_s> {
    static constexpr const char* name = "IQ3_S";
    static constexpr PrefillNativeWeightFormat native_format =
        PrefillNativeWeightFormat::IQ3_S;
    static constexpr std::uint32_t values_per_block = 256u;
    static constexpr std::uint32_t code_count = 512u;
    static constexpr EncodeCommandBuffer encode =
        &encode_prefill_dequant_iq3_s;
    static constexpr EncodeEncoder encode_encoder =
        &encode_prefill_dequant_iq3_s;

    // Row zero retains the literal packed representation. Other rows shift
    // each grid index and rotate/offset scale nibbles to make block addressing
    // observable across all 257 output columns and both K blocks.
    static constexpr std::array<std::uint8_t, 64> packed_qs{
        0x0bu, 0x30u, 0x55u, 0x7au, 0x9fu, 0xc4u, 0xe9u, 0x0eu,
        0x33u, 0x58u, 0x7du, 0xa2u, 0xc7u, 0xecu, 0x11u, 0x36u,
        0x5bu, 0x80u, 0xa5u, 0xcau, 0xefu, 0x14u, 0x39u, 0x5eu,
        0x83u, 0xa8u, 0xcdu, 0xf2u, 0x17u, 0x3cu, 0x61u, 0x86u,
        0xabu, 0xd0u, 0xf5u, 0x1au, 0x3fu, 0x64u, 0x89u, 0xaeu,
        0xd3u, 0xf8u, 0x1du, 0x42u, 0x67u, 0x8cu, 0xb1u, 0xd6u,
        0xfbu, 0x20u, 0x45u, 0x6au, 0x8fu, 0xb4u, 0xd9u, 0xfeu,
        0x23u, 0x48u, 0x6du, 0x92u, 0xb7u, 0xdcu, 0x01u, 0x26u};
    static constexpr std::array<std::uint8_t, 8> packed_qh{
        0x80u, 0x3fu, 0xe0u, 0x0fu, 0xf8u, 0x03u, 0xfeu, 0xc0u};
    static constexpr std::array<std::uint8_t, 32> sign_masks{
        0x5du, 0xa6u, 0xefu, 0x38u, 0x81u, 0xcau, 0x13u, 0x5cu,
        0xa5u, 0xeeu, 0x37u, 0x80u, 0xc9u, 0x12u, 0x5bu, 0xa4u,
        0xedu, 0x36u, 0x7fu, 0xc8u, 0x11u, 0x5au, 0xa3u, 0xecu,
        0x35u, 0x7eu, 0xc7u, 0x10u, 0x59u, 0xa2u, 0xebu, 0x34u};
    static constexpr std::array<std::uint8_t, 8> base_scales{
        0u, 1u, 2u, 4u, 7u, 9u, 12u, 15u};
    static constexpr std::array<std::uint8_t, 4> packed_scales{
        0x10u, 0x42u, 0x97u, 0xfcu};

    static std::uint32_t row_id(std::uint32_t column,
                                std::uint32_t block_index) {
        return column * 2u + block_index;
    }

    static std::uint32_t logical_scale(std::uint32_t column,
                                       std::uint32_t block_index,
                                       std::uint32_t group) {
        const std::uint32_t row = row_id(column, block_index);
        const std::uint32_t rotation = row % base_scales.size();
        const std::uint32_t offset = (row / base_scales.size()) % 16u;
        const std::uint32_t epoch = row / 128u;
        return (base_scales[(group + rotation) % base_scales.size()] +
                offset + epoch * (3u + 5u * group)) &
               0x0fu;
    }

    static std::uint32_t logical_grid_index(std::uint32_t row,
                                            std::uint32_t record) {
        return ((record * 37u + 11u) % 512u + row * 13u) % 512u;
    }

    static void store_fixture(block_iq3_s& block, std::uint32_t column,
                              std::uint32_t block_index,
                              std::uint16_t d_bits) {
        block.d = d_bits;
        const std::uint32_t row = row_id(column, block_index);
        const bool complement = (row & 1u) != 0u;
        if (row == 0u) {
            std::copy(packed_qs.begin(), packed_qs.end(), block.qs);
            std::copy(packed_qh.begin(), packed_qh.end(), block.qh);
        } else {
            for (std::uint32_t record = 0u; record < packed_qs.size();
                 ++record) {
                const std::uint32_t index =
                    logical_grid_index(row, record);
                block.qs[record] =
                    static_cast<strata_u8>(index & 0xffu);
                block.qh[record / 8u] |= static_cast<strata_u8>(
                    ((index >> 8u) & 1u) << (record % 8u));
            }
        }
        for (std::uint32_t group = 0u; group < sign_masks.size(); ++group) {
            block.signs[group] = static_cast<strata_u8>(
                complement ? sign_masks[group] ^ 0xffu : sign_masks[group]);
        }
        if (row == 0u) {
            std::copy(packed_scales.begin(), packed_scales.end(),
                      block.scales);
        } else {
            for (std::uint32_t group = 0u; group < 8u; ++group) {
                block.scales[group / 2u] |= static_cast<strata_u8>(
                    logical_scale(column, block_index, group)
                    << (4u * (group % 2u)));
            }
        }
    }

    static std::array<std::uint32_t, 256> oracle_codes(
        const block_iq3_s& block) {
        std::array<std::uint32_t, 256> codes{};
        // Decode the bytes actually stored in the fixture. Walk two packed
        // four-value grid words per eight-value subgroup, then scatter those
        // quartets into K order. This differs from the GPU's value-indexed
        // record/lane calculation and does not reuse the fixture generator.
        for (std::uint32_t ib = 0u; ib < 8u; ++ib) {
            for (std::uint32_t il = 0u; il < 4u; ++il) {
                const std::uint32_t qs_base = 8u * ib + 2u * il;
                const std::uint32_t sign_group = 4u * ib + il;
                for (std::uint32_t pair = 0u; pair < 2u; ++pair) {
                    const std::uint32_t qh_bit =
                        (uint(block.qh[ib]) >> (2u * il + pair)) & 1u;
                    const std::uint32_t grid_index =
                        uint(block.qs[qs_base + pair]) | (qh_bit << 8u);
                    const std::uint32_t grid_word = iq3s_grid[grid_index];
                    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
                        const std::uint32_t subgroup_value =
                            4u * pair + byte;
                        const std::uint32_t k =
                            32u * ib + 8u * il + subgroup_value;
                        const std::uint32_t magnitude =
                            (grid_word >> (8u * byte)) & 0xffu;
                        const bool negative =
                            (uint(block.signs[sign_group]) &
                             (1u << subgroup_value)) != 0u;
                        codes[k] =
                            magnitude | (negative ? 0x100u : 0u);
                    }
                }
            }
        }
        return codes;
    }

    static float expanded_value(std::uint32_t code, std::uint32_t value,
                                float d, std::uint32_t column,
                                std::uint32_t block_index) {
        const std::uint32_t scale_group = value / 32u;
        const float scale =
            d * static_cast<float>(
                    1u + 2u * logical_scale(column, block_index,
                                             scale_group));
        const float scaled = scale * static_cast<float>(code & 0xffu);
        return (code & 0x100u) != 0u ? -scaled : scaled;
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
bool test_native_projection_format(
    Context& context, TestBuffer& weights, TestBuffer& direct_dequant,
    std::size_t weight_bytes,
    const std::vector<std::uint16_t>& expected_dequant) {
    using Format = FormatTraits<Block>;
    constexpr std::uint32_t rows = 3u;
    constexpr std::uint32_t col0 = 3u;
    constexpr std::uint32_t ldy = kColumns + col0 + 5u;
    constexpr std::size_t scratch_bytes =
        static_cast<std::size_t>(kColumns) * kK * sizeof(std::uint16_t);
    constexpr std::size_t x_bytes =
        static_cast<std::size_t>(rows) * kK * sizeof(std::uint16_t);
    constexpr std::size_t y_elements =
        static_cast<std::size_t>(rows - 1u) * ldy + col0 + kColumns;
    constexpr std::size_t y_bytes = y_elements * sizeof(float);
    const std::string prefix =
        std::string("Prefill ") + Format::name + " native projection";
    std::string error;
    TestBuffer x, direct_y, projection_scratch, projection_y, failed_y;
    const bool allocated =
        x.allocate(context, prefix + " F16 activations", x_bytes, error) &&
        direct_y.allocate(context, prefix + " direct F32 output",
                          y_bytes + 8u, error) &&
        projection_scratch.allocate(context, prefix + " F16 scratch",
                                    scratch_bytes, error) &&
        projection_y.allocate(context, prefix + " composed F32 output",
                              y_bytes + 8u, error) &&
        failed_y.allocate(context, prefix + " rollback F32 output",
                          y_bytes + 8u, error);
    check(allocated, prefix + " allocate buffers", error);
    if (!allocated) return false;

    std::vector<std::uint16_t> x_values(x_bytes / sizeof(std::uint16_t));
    for (std::size_t index = 0u; index < x_bytes / sizeof(std::uint16_t);
         ++index) {
        const int centered = static_cast<int>(index % 17u) - 8;
        x_values[index] = strata::kernels::f16_from_f32(
            static_cast<float>(centered) * 0.015625f);
    }
    std::memcpy(x.payload().data(), x_values.data(), x_bytes);
    const std::vector<std::uint8_t> x_before(
        x.payload().begin(), x.payload().end());
    const std::vector<std::uint8_t> weights_before(
        weights.payload().begin(), weights.payload().begin() + weight_bytes);

    const PrefillGemmMPPParams params{
        rows, kColumns, kK, ldy, col0, false};
    const auto x_read =
        context.buffers().read_only(x.view().slice(0u, x_bytes));
    const auto direct_weights = context.buffers().read_only(
        direct_dequant.view().slice(0u, scratch_bytes));
    const PrefillGemmMPPBuffers direct_buffers{
        x_read, direct_weights, direct_y.view()};
    auto direct_command =
        context.make_command_buffer(prefix + " component chain", error);
    check(direct_command != nullptr,
          prefix + " reference command buffer", error);
    if (!direct_command) return false;
    PrefillGemmBackend direct_backend = PrefillGemmBackend::SIMD;
    const bool direct_encoded = encode_prefill_gemm(
        context, *direct_command, params, direct_buffers,
        PrefillGemmFormat::F16, error, &direct_backend);
    check(direct_encoded, prefix + " direct GEMM encode", error);
    if (!direct_encoded) return false;
    const bool direct_completed = submit_and_wait(direct_command, error);
    check(direct_completed, prefix + " direct GEMM completion", error);
    if (!direct_completed) return false;

    std::memset(projection_scratch.payload().data(), kPoison, scratch_bytes);
    std::memset(projection_y.payload().data(), kPoison,
                projection_y.payload().size());
    const PrefillNativeProjectionBuffers projection_buffers{
        x_read,
        context.buffers().read_only(
            weights.view().slice(0u, weight_bytes)),
        projection_scratch.view(),
        projection_y.view()};
    PrefillGemmBackend projection_backend = PrefillGemmBackend::SIMD;
    bool projection_encoded = false;
    if constexpr (std::is_same_v<Block, block_iq3_s>) {
        auto projection_command =
            context.make_command_buffer(prefix + " caller-owned", error);
        check(projection_command != nullptr,
              prefix + " caller-owned command buffer", error);
        if (!projection_command) return false;
        projection_encoded = Encoder::encode(
            context, *projection_command, prefix + " caller-owned",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_native_projection(
                    context, encoder, params, Format::native_format,
                    projection_buffers, encode_error, &projection_backend);
            },
            error);
        check(projection_encoded, prefix + " caller-owned encode", error);
        if (!projection_encoded) return false;
        const bool completed = submit_and_wait(projection_command, error);
        check(completed, prefix + " caller-owned completion", error);
        if (!completed) return false;
    } else {
        auto projection_command =
            context.make_command_buffer(prefix + " standalone", error);
        check(projection_command != nullptr,
              prefix + " standalone command buffer", error);
        if (!projection_command) return false;
        projection_encoded = encode_prefill_native_projection(
            context, *projection_command, params, Format::native_format,
            projection_buffers, error, &projection_backend);
        check(projection_encoded, prefix + " standalone encode", error);
        if (!projection_encoded) return false;
        const bool completed = submit_and_wait(projection_command, error);
        check(completed, prefix + " standalone completion", error);
        if (!completed) return false;
    }

    const bool backend_matches = projection_backend == direct_backend;
    const bool scratch_matches =
        std::memcmp(projection_scratch.payload().data(),
                    expected_dequant.data(), scratch_bytes) == 0;
    const bool output_matches =
        std::memcmp(projection_y.payload().data(), direct_y.payload().data(),
                    projection_y.payload().size()) == 0;
    const bool x_unchanged =
        std::equal(x_before.begin(), x_before.end(), x.payload().begin());
    const bool weights_unchanged =
        std::equal(weights_before.begin(), weights_before.end(),
                   weights.payload().begin());
    const auto output_is_poison = [](std::span<const std::uint8_t> bytes,
                                     std::size_t begin, std::size_t end) {
        return std::all_of(bytes.begin() + begin, bytes.begin() + end,
                           [](std::uint8_t byte) { return byte == kPoison; });
    };
    bool output_padding_untouched = true;
    const auto reference_bytes =
        std::span<const std::uint8_t>(direct_y.payload().data(),
                                      direct_y.payload().size());
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t row_begin =
            static_cast<std::size_t>(row) * ldy * sizeof(float);
        const std::size_t active_begin = row_begin + col0 * sizeof(float);
        const std::size_t active_end =
            active_begin + static_cast<std::size_t>(kColumns) * sizeof(float);
        output_padding_untouched &=
            output_is_poison(reference_bytes, row_begin, active_begin);
        if (row + 1u < rows) {
            const std::size_t row_end =
                static_cast<std::size_t>(row + 1u) * ldy * sizeof(float);
            output_padding_untouched &=
                output_is_poison(reference_bytes, active_end, row_end);
        } else {
            output_padding_untouched &=
                output_is_poison(reference_bytes, active_end,
                                 reference_bytes.size());
        }
    }
    check(backend_matches,
          prefix + " uses the same selected GEMM backend as its components");
    check(scratch_matches,
          prefix + " dequantizes exact K-major F16 weights before GEMM");
    check(output_matches,
          prefix + " matches the direct dequant-to-GEMM component chain");
    check(output_padding_untouched,
          prefix + " leaves output column, row and tail padding untouched");
    check(x_unchanged && weights_unchanged,
          prefix + " preserves activations and native weights");

    std::memset(projection_scratch.payload().data(), kPoison, scratch_bytes);
    std::memset(failed_y.payload().data(), kPoison, failed_y.payload().size());
    PrefillGemmMPPParams invalid_params = params;
    invalid_params.ldy = col0 + kColumns - 1u;
    const PrefillNativeProjectionBuffers rollback_buffers{
        x_read,
        context.buffers().read_only(
            weights.view().slice(0u, weight_bytes)),
        projection_scratch.view(),
        failed_y.view()};
    auto rollback_command =
        context.make_command_buffer(prefix + " late failure rollback", error);
    check(rollback_command != nullptr,
          prefix + " late failure command buffer", error);
    if (!rollback_command) return false;
    PrefillGemmBackend unchanged_backend = PrefillGemmBackend::MPP;
    const bool invalid_encoded = encode_prefill_native_projection(
        context, *rollback_command, invalid_params, Format::native_format,
        rollback_buffers, error, &unchanged_backend);
    const bool rollback_discarded =
        !invalid_encoded &&
        error.find("output extent or stride is invalid") != std::string::npos &&
        rollback_command->unsafe_native_handle() == nullptr &&
        unchanged_backend == PrefillGemmBackend::MPP &&
        std::all_of(projection_scratch.payload().begin(),
                    projection_scratch.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(failed_y.payload().begin(), failed_y.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; });
    check(rollback_discarded,
          prefix + " late GEMM rejection discards dequant and preserves buffers",
          error);

    bool caller_prefix_proven = true;
    if constexpr (std::is_same_v<Block, block_iq3_s>) {
        std::memset(projection_scratch.payload().data(), kPoison,
                    scratch_bytes);
        std::memset(failed_y.payload().data(), kPoison,
                    failed_y.payload().size());
        auto probe_command =
            context.make_command_buffer(prefix + " late-failure probe", error);
        check(probe_command != nullptr,
              prefix + " late-failure probe command buffer", error);
        if (!probe_command) return false;
        PrefillGemmBackend probe_backend = PrefillGemmBackend::MPP;
        bool coordinator_rejected_after_prefix = false;
        const bool probe_encoded = Encoder::encode(
            context, *probe_command, prefix + " late-failure probe",
            [&](Encoder& encoder, std::string& encode_error) {
                coordinator_rejected_after_prefix =
                    !encode_prefill_native_projection(
                        context, encoder, invalid_params, Format::native_format,
                        rollback_buffers, encode_error, &probe_backend) &&
                    encode_error.find("output extent or stride is invalid") !=
                        std::string::npos;
                // This probe deliberately submits the successfully encoded
                // dequant prefix so the test can prove the stride error is
                // reached after it. The rollback case below propagates false.
                encode_error.clear();
                return coordinator_rejected_after_prefix;
            },
            error);
        if (probe_encoded) {
            caller_prefix_proven =
                submit_and_wait(probe_command, error) &&
                probe_backend == PrefillGemmBackend::MPP &&
                std::memcmp(projection_scratch.payload().data(),
                            expected_dequant.data(), scratch_bytes) == 0 &&
                std::all_of(failed_y.payload().begin(), failed_y.payload().end(),
                            [](std::uint8_t byte) { return byte == kPoison; });
        } else {
            caller_prefix_proven = false;
        }
        check(caller_prefix_proven,
              prefix + " late GEMM rejection follows a completed dequant prefix",
              error);
    }

    bool caller_owned_rollback = true;
    if constexpr (std::is_same_v<Block, block_iq3_s>) {
        std::memset(projection_scratch.payload().data(), kPoison,
                    scratch_bytes);
        std::memset(failed_y.payload().data(), kPoison,
                    failed_y.payload().size());
        auto caller_command =
            context.make_command_buffer(prefix + " caller-owned rollback",
                                        error);
        check(caller_command != nullptr,
              prefix + " caller-owned rollback command buffer", error);
        if (!caller_command) return false;
        PrefillGemmBackend caller_backend = PrefillGemmBackend::MPP;
        bool projection_rejected = false;
        const bool caller_segment_succeeded = Encoder::encode(
            context, *caller_command, prefix + " caller-owned rollback",
            [&](Encoder& encoder, std::string& encode_error) {
                projection_rejected = !encode_prefill_native_projection(
                    context, encoder, invalid_params, Format::native_format,
                    rollback_buffers, encode_error, &caller_backend);
                return !projection_rejected;
            },
            error);
        caller_owned_rollback =
            !caller_segment_succeeded && projection_rejected &&
            error.find("output extent or stride is invalid") !=
                std::string::npos &&
            caller_command->unsafe_native_handle() == nullptr &&
            caller_backend == PrefillGemmBackend::MPP &&
            std::all_of(projection_scratch.payload().begin(),
                        projection_scratch.payload().end(),
                        [](std::uint8_t byte) { return byte == kPoison; }) &&
            std::all_of(failed_y.payload().begin(), failed_y.payload().end(),
                        [](std::uint8_t byte) { return byte == kPoison; });
        check(caller_owned_rollback,
              prefix + " caller-owned late failure discards the segment",
              error);
    }

    const bool guards_intact =
        x.guards_intact() && direct_y.guards_intact() &&
        projection_scratch.guards_intact() && projection_y.guards_intact() &&
        failed_y.guards_intact();
    check(guards_intact, prefix + " buffer guards");
    return backend_matches && scratch_matches && output_matches &&
           output_padding_untouched && x_unchanged && weights_unchanged &&
           caller_prefix_proven &&
           rollback_discarded && caller_owned_rollback && guards_intact;
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
            } else if constexpr (std::is_same_v<Block, block_q6_K>) {
                Format::store_fixture(block, column, block_index, scale);
            } else if constexpr (std::is_same_v<Block, block_q3_K>) {
                Format::store_fixture(block, column, block_index, scale);
            } else if constexpr (std::is_same_v<Block, block_iq4_xs>) {
                Format::store_fixture(block, column, block_index, scale);
            } else if constexpr (std::is_same_v<Block, block_iq3_s>) {
                Format::store_fixture(block, column, block_index, scale);
            } else {
                block.d = static_cast<strata_half>(scale);
            }
            std::array<std::uint32_t, Format::values_per_block>
                iq3_s_oracle_codes{};
            if constexpr (std::is_same_v<Block, block_iq3_s>) {
                iq3_s_oracle_codes = Format::oracle_codes(block);
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
                } else if constexpr (std::is_same_v<Block, block_q6_K>) {
                    oracle_code =
                        Format::oracle_code(column, block_index, value);
                } else if constexpr (std::is_same_v<Block, block_q3_K>) {
                    oracle_code =
                        Format::oracle_code(column, block_index, value);
                } else if constexpr (std::is_same_v<Block, block_iq4_xs>) {
                    oracle_code =
                        Format::oracle_code(column, block_index, value);
                } else if constexpr (std::is_same_v<Block, block_iq3_s>) {
                    oracle_code = iq3_s_oracle_codes[value];
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
                } else if constexpr (std::is_same_v<Block, block_q6_K>) {
                    expanded_value = Format::expanded_value(
                        oracle_code, value, expanded_scale,
                        column, block_index);
                } else if constexpr (std::is_same_v<Block, block_q3_K>) {
                    expanded_value = Format::expanded_value(
                        oracle_code, value, expanded_scale,
                        column, block_index);
                } else if constexpr (std::is_same_v<Block, block_iq4_xs>) {
                    expanded_value = Format::expanded_value(
                        oracle_code, value, expanded_scale,
                        column, block_index);
                } else if constexpr (std::is_same_v<Block, block_iq3_s>) {
                    expanded_value = Format::expanded_value(
                        oracle_code, value, expanded_scale,
                        column, block_index);
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
    const bool projection =
        test_native_projection_format<Block>(context, weights, output,
                                             weight_bytes, expected);
    return exact && input_unchanged && trailing_bytes_untouched &&
           weights.guards_intact() && output.guards_intact() && projection;
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

bool expect_native_projection_rejected(
    Context& context, const PrefillGemmMPPParams& params,
    PrefillNativeWeightFormat format,
    const PrefillNativeProjectionBuffers& buffers,
    std::string_view expected_error, const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    PrefillGemmBackend selected = PrefillGemmBackend::MPP;
    const bool encoded = encode_prefill_native_projection(
        context, *command, params, format, buffers, error, &selected);
    const bool rejected =
        !encoded && command->unsafe_native_handle() == nullptr &&
        selected == PrefillGemmBackend::MPP &&
        error.find(expected_error) != std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_native_projection_validation(Context& context,
                                       Context& foreign_context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t n = 64u;
    constexpr std::uint32_t k = 64u;
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(n) * (k / 64u) * sizeof(block_q2_0);
    constexpr std::size_t x_bytes =
        static_cast<std::size_t>(rows) * k * sizeof(std::uint16_t);
    constexpr std::size_t scratch_bytes =
        static_cast<std::size_t>(n) * k * sizeof(std::uint16_t);
    constexpr std::size_t y_bytes =
        static_cast<std::size_t>(rows) * n * sizeof(float);
    const PrefillGemmMPPParams params{rows, n, k, n, 0u, false};
    std::string error;
    TestBuffer x, weights, scratch, y, shared, foreign_weights;
    const std::size_t shared_bytes =
        std::max({weight_bytes, x_bytes, scratch_bytes, y_bytes});
    const bool allocated =
        x.allocate(context, "native projection validation X", x_bytes, error) &&
        weights.allocate(context, "native projection validation W",
                         weight_bytes, error) &&
        scratch.allocate(context, "native projection validation scratch",
                         scratch_bytes, error) &&
        y.allocate(context, "native projection validation Y", y_bytes, error) &&
        shared.allocate(context, "native projection validation aliases",
                        shared_bytes, error) &&
        foreign_weights.allocate(foreign_context,
                                 "native projection foreign W",
                                 weight_bytes, error);
    check(allocated, "native projection validation allocate buffers", error);
    if (!allocated) return false;

    const auto x_read = context.buffers().read_only(x.view());
    const auto weights_read = context.buffers().read_only(weights.view());
    const PrefillNativeProjectionBuffers valid{
        x_read, weights_read, scratch.view(), y.view()};
    bool passed = true;
    passed &= expect_native_projection_rejected(
        context, params, static_cast<PrefillNativeWeightFormat>(0xffu), valid,
        "format is invalid", "native projection rejects an invalid format");

    const PrefillNativeProjectionBuffers scratch_aliases_x{
        context.buffers().read_only(
            shared.view().slice(0u, x_bytes)),
        weights_read, shared.view().slice(0u, scratch_bytes), y.view()};
    passed &= expect_native_projection_rejected(
        context, params, PrefillNativeWeightFormat::Q2_0, scratch_aliases_x,
        "scratch overlaps an input",
        "native projection rejects scratch/X overlap before dispatch");

    const PrefillNativeProjectionBuffers y_aliases_weights{
        x_read,
        context.buffers().read_only(shared.view().slice(0u, weight_bytes)),
        scratch.view(), shared.view().slice(0u, y_bytes)};
    passed &= expect_native_projection_rejected(
        context, params, PrefillNativeWeightFormat::Q2_0, y_aliases_weights,
        "output overlaps another buffer",
        "native projection rejects Y/native-weight overlap before dispatch");

    const PrefillNativeProjectionBuffers y_aliases_x{
        context.buffers().read_only(shared.view().slice(0u, x_bytes)),
        weights_read, scratch.view(), shared.view().slice(0u, y_bytes)};
    passed &= expect_native_projection_rejected(
        context, params, PrefillNativeWeightFormat::Q2_0, y_aliases_x,
        "output overlaps another buffer",
        "native projection rejects Y/X overlap before dispatch");

    const PrefillNativeProjectionBuffers y_aliases_scratch{
        x_read, weights_read, shared.view().slice(0u, scratch_bytes),
        shared.view().slice(0u, y_bytes)};
    passed &= expect_native_projection_rejected(
        context, params, PrefillNativeWeightFormat::Q2_0, y_aliases_scratch,
        "output overlaps another buffer",
        "native projection rejects Y/scratch overlap before dispatch");

    const PrefillNativeProjectionBuffers foreign_input{
        x_read, foreign_context.buffers().read_only(foreign_weights.view()),
        scratch.view(), y.view()};
    passed &= expect_native_projection_rejected(
        context, params, PrefillNativeWeightFormat::Q2_0, foreign_input,
        "not owned by this Context",
        "native projection rejects foreign weights");

    auto invalid_geometry = params;
    invalid_geometry.k = 96u;
    passed &= expect_native_projection_rejected(
        context, invalid_geometry, PrefillNativeWeightFormat::Q2_0, valid,
        "divisible by the native block size",
        "native projection rejects native-block K remainder");
    return passed;
}

bool test_qsa_output(Context& context, Context& foreign_context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t k = 24u * 256u;
    constexpr std::uint32_t n = 2560u;
    constexpr std::size_t attention_bytes =
        static_cast<std::size_t>(rows) * k * sizeof(float);
    constexpr std::size_t q_full_bytes =
        static_cast<std::size_t>(rows) * 24u * 512u * sizeof(float);
    constexpr std::size_t gate_bytes =
        static_cast<std::size_t>(rows) * k * sizeof(std::uint16_t);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(n) * (k / 32u) * sizeof(block_q8_0);
    constexpr std::size_t dequantized_bytes =
        static_cast<std::size_t>(n) * k * sizeof(std::uint16_t);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(rows) * n * sizeof(float);
    const std::string prefix = "Prefill QSA output";
    std::string error;
    TestBuffer attention, q_full, gate, weights, dequantized, output;
    TestBuffer foreign_weights;
    const bool allocated =
        attention.allocate(context, prefix + " attention", attention_bytes,
                           error) &&
        q_full.allocate(context, prefix + " Qf", q_full_bytes, error) &&
        gate.allocate(context, prefix + " gated F16", gate_bytes, error) &&
        weights.allocate(context, prefix + " Q8_0 weights", weight_bytes,
                         error) &&
        dequantized.allocate(context, prefix + " dequantized F16",
                             dequantized_bytes, error) &&
        output.allocate(context, prefix + " F32 output", output_bytes,
                        error) &&
        foreign_weights.allocate(foreign_context,
                                 prefix + " foreign weights", weight_bytes,
                                 error);
    check(allocated, prefix + " allocate buffers", error);
    if (!allocated) return false;

    std::vector<float> attention_values(attention_bytes / sizeof(float));
    std::vector<float> q_values(q_full_bytes / sizeof(float));
    for (std::size_t index = 0u; index < attention_values.size(); ++index) {
        const int centered = static_cast<int>(index % 19u) - 9;
        attention_values[index] = static_cast<float>(centered) * 0.03125f;
    }
    for (std::size_t index = 0u; index < q_values.size(); ++index) {
        const std::size_t within_head = index % 512u;
        const int centered =
            static_cast<int>((index * 7u + within_head * 3u) % 29u) - 14;
        q_values[index] = static_cast<float>(centered) * 0.0625f;
    }
    std::memcpy(attention.payload().data(), attention_values.data(),
                attention_bytes);
    std::memcpy(q_full.payload().data(), q_values.data(), q_full_bytes);

    constexpr std::size_t blocks_per_output = k / 32u;
    std::vector<block_q8_0> weight_values(
        static_cast<std::size_t>(n) * blocks_per_output);
    for (std::uint32_t column = 0u; column < n; ++column) {
        for (std::uint32_t block_index = 0u;
             block_index < blocks_per_output; ++block_index) {
            block_q8_0& block =
                weight_values[static_cast<std::size_t>(column) *
                                  blocks_per_output +
                              block_index];
            block = {};
            block.d = static_cast<strata_half>(0x3000u);
            for (std::uint32_t value = 0u; value < 32u; ++value) {
                const int code = static_cast<int>(
                                     (column * 5u + block_index * 3u +
                                      value * 7u) %
                                     11u) -
                                 5;
                block.qs[value] = static_cast<strata_i8>(code);
            }
        }
    }
    std::memcpy(weights.payload().data(), weight_values.data(), weight_bytes);
    const std::vector<std::uint8_t> attention_before(
        attention.payload().begin(), attention.payload().end());
    const std::vector<std::uint8_t> q_full_before(
        q_full.payload().begin(), q_full.payload().end());
    const std::vector<std::uint8_t> weights_before(
        weights.payload().begin(), weights.payload().end());

    const PrefillQsaOutputBuffers buffers{
        {context.buffers().read_only(attention.view()),
         context.buffers().read_only(q_full.view()), gate.view()},
        context.buffers().read_only(weights.view()), dequantized.view(),
        output.view()};
    const auto expect_alias_rejected =
        [&](const PrefillQsaOutputBuffers& invalid,
            const std::string& label) {
            auto command = context.make_command_buffer(label, error);
            if (!command) {
                check(false, label + " command buffer", error);
                return false;
            }
            error.clear();
            const bool rejected = !encode_prefill_qsa_output(
                context, *command, rows, PrefillNativeWeightFormat::Q8_0,
                invalid, error);
            const bool clean = rejected &&
                command->unsafe_native_handle() == nullptr &&
                error.find("overlap another active stage buffer") !=
                    std::string::npos;
            check(clean, label, error);
            return clean;
        };
    PrefillQsaOutputBuffers gate_aliases_weights = buffers;
    gate_aliases_weights.gate.attn_h =
        weights.view().slice(0u, gate_bytes);
    const bool gate_weight_alias_rejected =
        expect_alias_rejected(gate_aliases_weights,
                              prefix + " rejects gate/weight overlap");
    PrefillQsaOutputBuffers output_aliases_q = buffers;
    output_aliases_q.y_f32 = q_full.view().slice(0u, output_bytes);
    const bool output_q_alias_rejected =
        expect_alias_rejected(output_aliases_q,
                              prefix + " rejects output/Qf overlap");

    PrefillGemmBackend standalone_backend = PrefillGemmBackend::SIMD;
    auto standalone =
        context.make_command_buffer(prefix + " standalone", error);
    check(standalone != nullptr, prefix + " standalone command buffer", error);
    if (!standalone) return false;
    const bool standalone_encoded = encode_prefill_qsa_output(
        context, *standalone, rows, PrefillNativeWeightFormat::Q8_0, buffers,
        error, &standalone_backend);
    check(standalone_encoded, prefix + " standalone encode", error);
    if (!standalone_encoded) return false;
    const bool standalone_completed = submit_and_wait(standalone, error);
    check(standalone_completed, prefix + " standalone completion", error);
    if (!standalone_completed) return false;
    const std::vector<std::uint8_t> expected_output(
        output.payload().begin(), output.payload().end());
    const std::vector<std::uint8_t> expected_gate(
        gate.payload().begin(), gate.payload().end());

    std::fill(gate.payload().begin(), gate.payload().end(), kPoison);
    std::fill(dequantized.payload().begin(), dequantized.payload().end(),
              kPoison);
    std::fill(output.payload().begin(), output.payload().end(), kPoison);
    PrefillGemmBackend caller_backend = PrefillGemmBackend::SIMD;
    auto caller_owned =
        context.make_command_buffer(prefix + " caller-owned", error);
    check(caller_owned != nullptr,
          prefix + " caller-owned command buffer", error);
    if (!caller_owned) return false;
    const bool caller_encoded = Encoder::encode(
        context, *caller_owned, prefix + " caller-owned",
        [&](Encoder& encoder, std::string& body_error) {
            return encode_prefill_qsa_output(
                context, encoder, rows, PrefillNativeWeightFormat::Q8_0,
                buffers, body_error, &caller_backend);
        },
        error);
    check(caller_encoded, prefix + " caller-owned encode", error);
    if (!caller_encoded) return false;
    const bool caller_completed = submit_and_wait(caller_owned, error);
    check(caller_completed, prefix + " caller-owned completion", error);
    if (!caller_completed) return false;
    const bool caller_matches =
        output.payload().size() == expected_output.size() &&
        std::equal(expected_output.begin(), expected_output.end(),
                   output.payload().begin()) &&
        std::equal(expected_gate.begin(), expected_gate.end(),
                   gate.payload().begin()) &&
        caller_backend == standalone_backend;
    check(caller_matches,
          prefix + " caller-owned gate and projection match standalone");

    std::fill(gate.payload().begin(), gate.payload().end(), kPoison);
    std::fill(dequantized.payload().begin(), dequantized.payload().end(),
              kPoison);
    std::fill(output.payload().begin(), output.payload().end(), kPoison);
    auto components = context.make_command_buffer(prefix + " components",
                                                   error);
    check(components != nullptr, prefix + " component command buffer", error);
    if (!components) return false;
    PrefillGemmBackend component_backend = PrefillGemmBackend::SIMD;
    const PrefillGemmMPPParams projection_params{
        rows, n, k, n, 0u, false};
    const PrefillNativeProjectionBuffers projection_buffers{
        context.buffers().read_only(gate.view()),
        context.buffers().read_only(weights.view()), dequantized.view(),
        output.view()};
    const bool components_encoded = Encoder::encode(
        context, *components, prefix + " component reference",
        [&](Encoder& encoder, std::string& body_error) {
            if (!encode_prefill_qsa_gate_attn(context, encoder, rows,
                                              buffers.gate, body_error)) {
                return false;
            }
            return encode_prefill_native_projection(
                context, encoder, projection_params,
                PrefillNativeWeightFormat::Q8_0, projection_buffers,
                body_error, &component_backend);
        },
        error);
    check(components_encoded, prefix + " component reference encode", error);
    if (!components_encoded) return false;
    const bool components_completed = submit_and_wait(components, error);
    check(components_completed, prefix + " component completion", error);
    if (!components_completed) return false;
    const bool component_matches =
        std::equal(expected_output.begin(), expected_output.end(),
                   output.payload().begin()) &&
        std::equal(expected_gate.begin(), expected_gate.end(),
                   gate.payload().begin()) &&
        component_backend == standalone_backend;
    check(component_matches,
          prefix + " composed APIs match gate then native projection");

    std::vector<float> output_values(output_bytes / sizeof(float));
    std::memcpy(output_values.data(), output.payload().data(), output_bytes);
    bool output_finite = true;
    bool output_nonzero = false;
    for (const float value : output_values) {
        output_finite &= std::isfinite(value);
        output_nonzero |= value != 0.0f;
    }
    const bool input_immutable =
        std::equal(attention_before.begin(), attention_before.end(),
                   attention.payload().begin()) &&
        std::equal(q_full_before.begin(), q_full_before.end(),
                   q_full.payload().begin()) &&
        std::equal(weights_before.begin(), weights_before.end(),
                   weights.payload().begin());
    check(output_nonzero, prefix + " writes a nonzero projection result");
    check(output_finite, prefix + " projection output is finite");
    check(input_immutable, prefix + " preserves attention, Qf and weights");

    PrefillQsaOutputBuffers foreign_input = buffers;
    foreign_input.native_weights =
        foreign_context.buffers().read_only(foreign_weights.view());
    std::fill(gate.payload().begin(), gate.payload().end(), kPoison);
    std::fill(dequantized.payload().begin(), dequantized.payload().end(),
              kPoison);
    std::fill(output.payload().begin(), output.payload().end(), kPoison);
    auto foreign_standalone =
        context.make_command_buffer(prefix + " foreign-weight preflight",
                                    error);
    check(foreign_standalone != nullptr,
          prefix + " foreign-weight command buffer", error);
    if (!foreign_standalone) return false;
    error.clear();
    const bool foreign_standalone_rejected = !encode_prefill_qsa_output(
        context, *foreign_standalone, rows,
        PrefillNativeWeightFormat::Q8_0, foreign_input, error);
    const bool foreign_standalone_clean =
        foreign_standalone_rejected &&
        foreign_standalone->unsafe_native_handle() == nullptr &&
        error.find("weights are not owned by this Context") !=
            std::string::npos &&
        std::all_of(gate.payload().begin(), gate.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(dequantized.payload().begin(), dequantized.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(output.payload().begin(), output.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; });
    check(foreign_standalone_clean,
          prefix + " rejects foreign weights before dispatch", error);

    auto foreign_caller =
        context.make_command_buffer(prefix + " caller foreign-weight preflight",
                                    error);
    check(foreign_caller != nullptr,
          prefix + " caller foreign-weight command buffer", error);
    if (!foreign_caller) return false;
    error.clear();
    const bool foreign_caller_rejected = !Encoder::encode(
        context, *foreign_caller, prefix + " caller foreign-weight preflight",
        [&](Encoder& encoder, std::string& body_error) {
            return encode_prefill_qsa_output(
                context, encoder, rows, PrefillNativeWeightFormat::Q8_0,
                foreign_input, body_error);
        },
        error);
    const bool foreign_caller_clean =
        foreign_caller_rejected &&
        foreign_caller->unsafe_native_handle() == nullptr &&
        error.find("weights are not owned by this Context") !=
            std::string::npos &&
        std::all_of(gate.payload().begin(), gate.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(dequantized.payload().begin(), dequantized.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(output.payload().begin(), output.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; });
    check(foreign_caller_clean,
          prefix + " caller-owned rejects foreign weights before dispatch",
          error);

    ScopedEnvironmentOverride invalid_selector("STRATA_METAL_GEMM",
                                                "invalid");
    check(invalid_selector.ready(),
          prefix + " install late-invalid GEMM selector");
    if (!invalid_selector.ready()) return false;
    auto standalone_rollback =
        context.make_command_buffer(prefix + " standalone selector rollback",
                                    error);
    check(standalone_rollback != nullptr,
          prefix + " standalone selector command buffer", error);
    if (!standalone_rollback) return false;
    error.clear();
    const bool standalone_rejected = !encode_prefill_qsa_output(
        context, *standalone_rollback, rows,
        PrefillNativeWeightFormat::Q8_0, buffers, error);
    const bool standalone_rollback_ok =
        standalone_rejected &&
        standalone_rollback->unsafe_native_handle() == nullptr &&
        error == "STRATA_METAL_GEMM must be 'mpp' or 'simd'" &&
        std::all_of(gate.payload().begin(), gate.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(dequantized.payload().begin(), dequantized.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(output.payload().begin(), output.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; });
    check(standalone_rollback_ok,
          prefix + " standalone late selector failure discards both stages",
          error);

    auto caller_rollback =
        context.make_command_buffer(prefix + " caller selector rollback",
                                    error);
    check(caller_rollback != nullptr,
          prefix + " caller selector command buffer", error);
    if (!caller_rollback) return false;
    error.clear();
    const bool caller_rejected = !Encoder::encode(
        context, *caller_rollback, prefix + " caller selector rollback",
        [&](Encoder& encoder, std::string& body_error) {
            return encode_prefill_qsa_output(
                context, encoder, rows, PrefillNativeWeightFormat::Q8_0,
                buffers, body_error);
        },
        error);
    const bool caller_rollback_ok =
        caller_rejected && caller_rollback->unsafe_native_handle() == nullptr &&
        error == "STRATA_METAL_GEMM must be 'mpp' or 'simd'" &&
        std::all_of(gate.payload().begin(), gate.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(dequantized.payload().begin(), dequantized.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; }) &&
        std::all_of(output.payload().begin(), output.payload().end(),
                    [](std::uint8_t byte) { return byte == kPoison; });
    check(caller_rollback_ok,
          prefix + " caller-owned late selector failure discards both stages",
          error);

    const bool guards_intact =
        attention.guards_intact() && q_full.guards_intact() &&
        gate.guards_intact() && weights.guards_intact() &&
        dequantized.guards_intact() && output.guards_intact() &&
        foreign_weights.guards_intact();
    check(guards_intact, prefix + " buffer guards");
    return gate_weight_alias_rejected && output_q_alias_rejected &&
           caller_matches && component_matches && output_nonzero &&
           output_finite && input_immutable && foreign_standalone_clean &&
           foreign_caller_clean && standalone_rollback_ok &&
           caller_rollback_ok && guards_intact;
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
    passed &= test_dequant_format<block_iq4_xs>(*context);
    passed &= test_validation_format<block_iq4_xs>(*context,
                                                    *foreign_context);
    passed &= test_dequant_format<block_q4_K>(*context);
    passed &= test_validation_format<block_q4_K>(*context,
                                                  *foreign_context);
    passed &= test_dequant_format<block_q5_K>(*context);
    passed &= test_validation_format<block_q5_K>(*context,
                                                  *foreign_context);
    passed &= test_dequant_format<block_q6_K>(*context);
    passed &= test_validation_format<block_q6_K>(*context,
                                                  *foreign_context);
    passed &= test_dequant_format<block_q3_K>(*context);
    passed &= test_validation_format<block_q3_K>(*context,
                                                  *foreign_context);
    passed &= test_dequant_format<block_iq3_s>(*context);
    passed &= test_validation_format<block_iq3_s>(*context,
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
    passed &= test_native_projection_validation(*context, *foreign_context);
    passed &= test_qsa_output(*context, *foreign_context);
    return passed && g_failures == 0 ? 0 : 1;
}
