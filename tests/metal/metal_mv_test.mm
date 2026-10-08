// Q8_1 activation quantizer tests (docs/mac/04-metal-kernels.md §5.8).
#include "oracles/q8_1.hpp"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q4_0_mv.hpp"
#include "oracles/q5_0_mv.hpp"
#include "oracles/q2_0_mv.hpp"
#include "oracles/iq4_nl_mv.hpp"
#include "oracles/iq4_xs_mv.hpp"
#include "oracles/iq4_xs_moe_fixture.hpp"
#include "oracles/iq2_xxs_mv.hpp"
#include "oracles/iq2_xxs_moe_fixture.hpp"
#include "oracles/iq2_xs_moe_fixture.hpp"
#include "oracles/iq2_xs_mv.hpp"
#include "oracles/iq2_s_mv.hpp"
#include "oracles/iq2_s_moe_fixture.hpp"
#include "oracles/iq3_xxs_mv.hpp"
#include "oracles/iq3_xxs_moe_fixture.hpp"
#include "oracles/iq1_m_mv.hpp"
#include "oracles/iq3_s_mv.hpp"
#include "oracles/iq3_s_moe_fixture.hpp"
#include "oracles/q5_k_mv.hpp"
#include "oracles/q3_k_mv.hpp"
#include "oracles/q4_k_mv.hpp"
#include "oracles/q6_k_mv.hpp"
#include "oracles/bf16_mv.hpp"
#include "synth.hpp"
#include "twin.hpp"

#include "metallib_sha256.inc"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/window_output_projection.hpp"
#include "../../src/metal/kernels/common/types.h"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "../../src/metal/kernels/common/codebooks.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unistd.h>

using namespace strata::metal;

namespace {

struct Q81Params {
    std::uint32_t n;
    std::uint32_t rows;
};
struct BF16MVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct Q8MVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct Q4MVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct Q5MVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct Q2MVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ4NLParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ4XSParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ3SParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ3XXSParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ1MParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ2XXSParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ2XSParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct IQ2SParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct Q6MVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct Q3MVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
struct Q4KMVParams {
    std::uint32_t n_in;
    std::uint32_t n_out;
};
static_assert(sizeof(Q81Params) == 8);
static_assert(sizeof(BF16MVParams) == 8);
static_assert(sizeof(Q8MVParams) == 8);
static_assert(sizeof(Q4MVParams) == 8);
static_assert(sizeof(Q5MVParams) == 8);
static_assert(sizeof(Q2MVParams) == 8);
static_assert(sizeof(IQ4NLParams) == 8);
static_assert(sizeof(IQ4XSParams) == 8);
static_assert(sizeof(IQ3SParams) == 8);
static_assert(sizeof(IQ3XXSParams) == 8);
static_assert(sizeof(IQ1MParams) == 8);
static_assert(sizeof(IQ2XXSParams) == 8);
static_assert(sizeof(IQ2XSParams) == 8);
static_assert(sizeof(IQ2SParams) == 8);
static_assert(sizeof(Q6MVParams) == 8);
static_assert(sizeof(Q3MVParams) == 8);
static_assert(sizeof(Q4KMVParams) == 8);
static_assert(sizeof(Q81) == 36);
static_assert(sizeof(block_q8_0) == 34);
static_assert(sizeof(block_q4_0) == 18);
static_assert(sizeof(block_q5_0) == 22);
static_assert(sizeof(block_q2_0) == 18);
static_assert(sizeof(block_iq4_nl) == 18);
static_assert(sizeof(block_iq4_xs) == 136);
static_assert(sizeof(block_iq3_s) == 110);
static_assert(sizeof(block_iq2_xxs) == 66);
static_assert(sizeof(block_iq2_xs) == 74);
static_assert(sizeof(block_iq2_s) == 82);
static_assert(sizeof(block_iq3_xxs) == 98);
static_assert(sizeof(block_iq1_m) == 56);
static_assert(sizeof(block_q3_K) == 110);
static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_q6_K) == 210);
static_assert(offsetof(block_iq4_nl, d) == 0);
static_assert(offsetof(block_iq4_nl, qs) == 2);
static_assert(offsetof(block_iq4_xs, d) == 0);
static_assert(offsetof(block_iq4_xs, scales_h) == 2);
static_assert(offsetof(block_iq4_xs, scales_l) == 4);
static_assert(offsetof(block_iq4_xs, qs) == 8);
static_assert(offsetof(block_iq3_s, d) == 0);
static_assert(offsetof(block_iq3_s, qs) == 2);
static_assert(offsetof(block_iq3_s, qh) == 66);
static_assert(offsetof(block_iq3_s, signs) == 74);
static_assert(offsetof(block_iq3_s, scales) == 106);
static_assert(offsetof(block_iq2_xxs, d) == 0);
static_assert(offsetof(block_iq2_xxs, qs) == 2);
static_assert(offsetof(block_iq2_xs, d) == 0);
static_assert(offsetof(block_iq2_xs, qs) == 2);
static_assert(offsetof(block_iq2_xs, scales) == 66);
static_assert(offsetof(block_iq2_s, d) == 0);
static_assert(offsetof(block_iq2_s, qs) == 2);
static_assert(offsetof(block_iq2_s, qh) == 66);
static_assert(offsetof(block_iq2_s, scales) == 74);
static_assert(offsetof(block_iq3_xxs, d) == 0);
static_assert(offsetof(block_iq3_xxs, qs) == 2);
static_assert(offsetof(block_iq1_m, qs) == 0);
static_assert(offsetof(block_iq1_m, qh) == 32);
static_assert(offsetof(block_iq1_m, scales) == 48);
static_assert(offsetof(block_q3_K, hmask) == 0);
static_assert(offsetof(block_q3_K, qs) == 32);
static_assert(offsetof(block_q3_K, scales) == 96);
static_assert(offsetof(block_q3_K, d) == 108);
static_assert(offsetof(block_q4_K, dm) == 0);
static_assert(offsetof(block_q4_K, scales) == 4);
static_assert(offsetof(block_q4_K, qs) == 16);
static_assert(offsetof(block_q6_K, ql) == 0);
static_assert(offsetof(block_q6_K, qh) == 128);
static_assert(offsetof(block_q6_K, scales) == 192);
static_assert(offsetof(block_q6_K, d) == 208);

int g_failures = 0;
constexpr std::array<int, 16> k_iq4_nl_golden_values = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};
constexpr std::array<int, 16> k_iq4_xs_golden_values = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};

void check(bool ok, const std::string& label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !ok;
}

bool run_window_output_weight_mapping_tests() {
    const std::array<std::uint64_t, 2> shape{
        kWindowOutputInputWidth, kWindowOutputVocabularySize};
    const std::array<std::pair<std::uint32_t, MatVecWeightFormat>, 3> cases{{
        {13u, MatVecWeightFormat::q5_k},
        {14u, MatVecWeightFormat::q6_k},
        {23u, MatVecWeightFormat::iq4_xs},
    }};
    bool passed = true;
    for (const auto& [ggml_type, expected_format] : cases) {
        MatVecWeightFormat format{};
        std::string error = "stale";
        const bool accepted = window_output_weight_format(
            ggml_type, shape, format, error);
        check(accepted && format == expected_format && error.empty(),
              "Window output.weight maps supported GGML type " +
                  std::to_string(ggml_type),
              error);
        passed &= accepted && format == expected_format && error.empty();
    }

    MatVecWeightFormat format{};
    std::string error;
    const bool rejects_type =
        !window_output_weight_format(30u, shape, format, error) &&
        error.find("Q5_K, Q6_K or IQ4_XS") != std::string::npos;
    check(rejects_type, "Window output.weight rejects unsupported GGML type",
          error);
    passed &= rejects_type;

    const std::array<std::uint64_t, 2> bad_shape{
        kWindowOutputInputWidth - 1u, kWindowOutputVocabularySize};
    const bool rejects_shape =
        !window_output_weight_format(13u, bad_shape, format, error) &&
        error.find("[2560, 248320]") != std::string::npos;
    check(rejects_shape, "Window output.weight rejects unexpected dimensions",
          error);
    passed &= rejects_shape;

    const std::array<std::uint64_t, 1> bad_rank{
        kWindowOutputInputWidth};
    const bool rejects_rank =
        !window_output_weight_format(23u, bad_rank, format, error) &&
        error.find("[2560, 248320]") != std::string::npos;
    check(rejects_rank, "Window output.weight rejects unexpected rank", error);
    passed &= rejects_rank;
    return passed;
}

void set_q5_0_code(block_q5_0& weight, std::uint32_t index, std::uint32_t code) {
    const std::uint32_t packed_index = index & 15u;
    if (index < 16u) {
        weight.qs[packed_index] = static_cast<std::uint8_t>(
            (weight.qs[packed_index] & 0xf0u) | (code & 0x0fu));
    } else {
        weight.qs[packed_index] = static_cast<std::uint8_t>(
            (weight.qs[packed_index] & 0x0fu) | ((code & 0x0fu) << 4u));
    }
    if ((code & 0x10u) != 0u) {
        weight.qh[index >> 3u] |= static_cast<std::uint8_t>(1u << (index & 7u));
    }
}

void set_q2_0_code(block_q2_0& weight, std::uint32_t index, std::uint32_t code) {
    const std::uint32_t byte_index = index >> 2u;
    const std::uint32_t shift = 2u * (index & 3u);
    const std::uint8_t mask = static_cast<std::uint8_t>(3u << shift);
    weight.qs[byte_index] = static_cast<std::uint8_t>(
        (weight.qs[byte_index] & static_cast<std::uint8_t>(~mask)) |
        static_cast<std::uint8_t>((code & 3u) << shift));
}

void set_iq4_nl_code(block_iq4_nl& weight, std::uint32_t index, std::uint32_t code) {
    const std::uint32_t byte_index = index & 15u;
    if (index < 16u) {
        weight.qs[byte_index] = static_cast<std::uint8_t>(
            (weight.qs[byte_index] & 0xf0u) | (code & 0x0fu));
    } else {
        weight.qs[byte_index] = static_cast<std::uint8_t>(
            (weight.qs[byte_index] & 0x0fu) | ((code & 0x0fu) << 4u));
    }
}

void set_iq4_xs_code(block_iq4_xs& weight, std::uint32_t index, std::uint32_t code) {
    const std::uint32_t subblock = index / 32u;
    const std::uint32_t local = index % 32u;
    const std::uint32_t byte_index =
        16u * subblock + (local < 16u ? local : local - 16u);
    if (local < 16u) {
        weight.qs[byte_index] = static_cast<std::uint8_t>(
            (weight.qs[byte_index] & 0xf0u) | (code & 0x0fu));
    } else {
        weight.qs[byte_index] = static_cast<std::uint8_t>(
            (weight.qs[byte_index] & 0x0fu) | ((code & 0x0fu) << 4u));
    }
}

void set_iq4_xs_scale(block_iq4_xs& weight, std::uint32_t subblock,
                      std::uint32_t code) {
    const std::uint32_t low_shift = 4u * (subblock % 2u);
    const std::uint8_t low_mask = static_cast<std::uint8_t>(0x0fu << low_shift);
    weight.scales_l[subblock / 2u] = static_cast<std::uint8_t>(
        (weight.scales_l[subblock / 2u] & static_cast<std::uint8_t>(~low_mask)) |
        static_cast<std::uint8_t>((code & 0x0fu) << low_shift));
    const std::uint32_t high_shift = 2u * subblock;
    const std::uint16_t high_mask = static_cast<std::uint16_t>(0x03u << high_shift);
    weight.scales_h = static_cast<std::uint16_t>(
        (weight.scales_h & static_cast<std::uint16_t>(~high_mask)) |
        static_cast<std::uint16_t>(((code >> 4u) & 0x03u) << high_shift));
}

void set_q6_k_code(block_q6_K& weight, std::uint32_t index, std::uint32_t code) {
    const std::uint32_t half = index / 128u;
    const std::uint32_t local = index % 128u;
    const std::uint32_t group = local / 32u;
    const std::uint32_t lane = local % 32u;
    const std::uint32_t ql_index =
        64u * half + 32u * (group % 2u) + lane;
    const std::uint32_t ql_shift = group >= 2u ? 4u : 0u;
    const std::uint8_t ql_mask = static_cast<std::uint8_t>(0x0fu << ql_shift);
    weight.ql[ql_index] = static_cast<std::uint8_t>(
        (weight.ql[ql_index] & static_cast<std::uint8_t>(~ql_mask)) |
        static_cast<std::uint8_t>((code & 0x0fu) << ql_shift));
    const std::uint32_t qh_index = 32u * half + lane;
    const std::uint32_t qh_shift = 2u * group;
    const std::uint8_t qh_mask = static_cast<std::uint8_t>(0x03u << qh_shift);
    weight.qh[qh_index] = static_cast<std::uint8_t>(
        (weight.qh[qh_index] & static_cast<std::uint8_t>(~qh_mask)) |
        static_cast<std::uint8_t>(((code >> 4u) & 0x03u) << qh_shift));
}

void set_q6_k_scale(block_q6_K& weight, std::uint32_t index, std::int32_t scale) {
    weight.scales[index] = static_cast<std::int8_t>(scale);
}

void set_q3_k_code(block_q3_K& weight, std::uint32_t index, std::int32_t code) {
    const std::uint32_t half = index / 128u;
    const std::uint32_t local = index % 128u;
    const std::uint32_t group = local / 32u;
    const std::uint32_t lane = local % 32u;
    const std::uint32_t qs_index = 32u * half + lane;
    const std::uint32_t qs_shift = 2u * group;
    const std::uint8_t qs_mask = static_cast<std::uint8_t>(0x03u << qs_shift);
    weight.qs[qs_index] = static_cast<std::uint8_t>(
        (weight.qs[qs_index] & static_cast<std::uint8_t>(~qs_mask)) |
        static_cast<std::uint8_t>((static_cast<std::uint32_t>(code) & 0x03u)
                                  << qs_shift));
    const std::uint32_t hmask_bit = 4u * half + group;
    const std::uint8_t hmask_mask =
        static_cast<std::uint8_t>(1u << hmask_bit);
    if (code >= 0) {
        weight.hmask[lane] |= hmask_mask;
    } else {
        weight.hmask[lane] &= static_cast<std::uint8_t>(~hmask_mask);
    }
}

void set_q3_k_scale(block_q3_K& weight, std::uint32_t index,
                    std::int32_t scale) {
    const std::uint32_t code =
        static_cast<std::uint32_t>(scale + 32);
    const std::uint32_t low_byte = index % 8u;
    const std::uint32_t low_shift = 4u * (index / 8u);
    const std::uint8_t low_mask =
        static_cast<std::uint8_t>(0x0fu << low_shift);
    weight.scales[low_byte] = static_cast<std::uint8_t>(
        (weight.scales[low_byte] & static_cast<std::uint8_t>(~low_mask)) |
        static_cast<std::uint8_t>((code & 0x0fu) << low_shift));
    const std::uint32_t high_byte = 8u + index % 4u;
    const std::uint32_t high_shift = 2u * (index / 4u);
    const std::uint8_t high_mask =
        static_cast<std::uint8_t>(0x03u << high_shift);
    weight.scales[high_byte] = static_cast<std::uint8_t>(
        (weight.scales[high_byte] & static_cast<std::uint8_t>(~high_mask)) |
        static_cast<std::uint8_t>(((code >> 4u) & 0x03u) << high_shift));
}

void set_q4_k_code(block_q4_K& weight, std::uint32_t index, std::uint32_t code) {
    const std::uint32_t group = index / 32u;
    const std::uint32_t local = index % 32u;
    const std::uint32_t byte_index = 32u * (group / 2u) + local;
    const std::uint32_t shift = 4u * (group % 2u);
    const std::uint8_t mask = static_cast<std::uint8_t>(0x0fu << shift);
    weight.qs[byte_index] = static_cast<std::uint8_t>(
        (weight.qs[byte_index] & static_cast<std::uint8_t>(~mask)) |
        static_cast<std::uint8_t>((code & 0x0fu) << shift));
}

void set_q4_k_scale_min(block_q4_K& weight, std::uint32_t group,
                        std::uint32_t scale, std::uint32_t min) {
    if (group < 4u) {
        weight.scales[group] = static_cast<std::uint8_t>(
            (weight.scales[group] & 0xc0u) | (scale & 0x3fu));
        weight.scales[group + 4u] = static_cast<std::uint8_t>(
            (weight.scales[group + 4u] & 0xc0u) | (min & 0x3fu));
        return;
    }
    const std::uint32_t packed_index = group + 4u;
    weight.scales[packed_index] = static_cast<std::uint8_t>(
        (weight.scales[packed_index] & 0xf0u) | (scale & 0x0fu));
    weight.scales[packed_index] = static_cast<std::uint8_t>(
        (weight.scales[packed_index] & 0x0fu) | ((min & 0x0fu) << 4u));
    weight.scales[group - 4u] = static_cast<std::uint8_t>(
        (weight.scales[group - 4u] & 0x3fu) | (((scale >> 4u) & 0x03u) << 6u));
    weight.scales[group] = static_cast<std::uint8_t>(
        (weight.scales[group] & 0x3fu) | (((min >> 4u) & 0x03u) << 6u));
}

void set_q5_k_code(block_q5_K& weight, std::uint32_t index,
                   std::uint32_t code) {
    const std::uint32_t group = index / 32u;
    const std::uint32_t local = index % 32u;
    const std::uint32_t byte_index = 32u * (group / 2u) + local;
    const std::uint32_t shift = 4u * (group % 2u);
    const std::uint8_t mask = static_cast<std::uint8_t>(0x0fu << shift);
    weight.qs[byte_index] = static_cast<std::uint8_t>(
        (weight.qs[byte_index] & static_cast<std::uint8_t>(~mask)) |
        static_cast<std::uint8_t>((code & 0x0fu) << shift));
    const std::uint8_t high_mask =
        static_cast<std::uint8_t>(1u << group);
    weight.qh[local] = static_cast<std::uint8_t>(
        (weight.qh[local] & static_cast<std::uint8_t>(~high_mask)) |
        static_cast<std::uint8_t>(((code >> 4u) & 1u) << group));
}

void set_q5_k_scale_min(block_q5_K& weight, std::uint32_t group,
                        std::uint32_t scale, std::uint32_t min) {
    if (group < 4u) {
        weight.scales[group] = static_cast<std::uint8_t>(
            (weight.scales[group] & 0xc0u) | (scale & 0x3fu));
        weight.scales[group + 4u] = static_cast<std::uint8_t>(
            (weight.scales[group + 4u] & 0xc0u) | (min & 0x3fu));
        return;
    }
    const std::uint32_t packed_index = group + 4u;
    weight.scales[packed_index] = static_cast<std::uint8_t>(
        (weight.scales[packed_index] & 0xf0u) | (scale & 0x0fu));
    weight.scales[packed_index] = static_cast<std::uint8_t>(
        (weight.scales[packed_index] & 0x0fu) | ((min & 0x0fu) << 4u));
    weight.scales[group - 4u] = static_cast<std::uint8_t>(
        (weight.scales[group - 4u] & 0x3fu) | (((scale >> 4u) & 0x03u) << 6u));
    weight.scales[group] = static_cast<std::uint8_t>(
        (weight.scales[group] & 0x3fu) | (((min >> 4u) & 0x03u) << 6u));
}

void install_q8_1_edge_blocks(std::vector<float>& input, std::uint32_t n, std::uint32_t rows) {
    constexpr std::array<float, 12> ties = {
        127.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 63.5f, -63.5f, 126.5f, -126.5f, -127.0f};
    constexpr std::array<std::uint32_t, 8> subnormal_bits = {
        0u, 1u, 0x80000001u, 2u, 0x80000002u, 1u, 0x80000001u, 0u};
    constexpr std::array<std::uint32_t, 8> subnormal_scale_bits = {
        0u, 1u, 0x80000001u, 32u, 0x80000020u, 63u, 0x8000003fu, 64u};
    constexpr std::array<std::uint32_t, 8> rounded_subnormal_scale_bits = {
        190u, 0x800000beu, 189u, 0x800000bdu, 127u, 0x8000007fu, 1u, 0x80000001u};

    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row) * n;
        for (std::size_t block = 0; block < n / 32u; ++block) {
            const std::size_t block_base = row_base + block * 32u;
            auto block_input = input.begin() + block_base;
            switch (block) {
                case 0:
                    std::fill_n(block_input, 32, 0.0f);
                    break;
                case 1:
                    for (std::size_t lane = 0; lane < 32; ++lane) {
                        block_input[lane] = ties[lane % ties.size()];
                    }
                    break;
                case 2:
                    for (std::size_t lane = 0; lane < 32; ++lane) {
                        block_input[lane] = (lane & 1u) == 0 ? 1.0f : -1.0f;
                    }
                    break;
                case 3:
                    for (std::size_t lane = 0; lane < 32; ++lane) {
                        block_input[lane] =
                            std::bit_cast<float>(subnormal_bits[lane % subnormal_bits.size()]);
                    }
                    break;
                case 4:
                    for (std::size_t lane = 0; lane < 32; ++lane) {
                        block_input[lane] = std::bit_cast<float>(
                            subnormal_scale_bits[lane % subnormal_scale_bits.size()]);
                    }
                    break;
                case 5:
                    for (std::size_t lane = 0; lane < 32; ++lane) {
                        block_input[lane] = std::bit_cast<float>(
                            rounded_subnormal_scale_bits[lane % rounded_subnormal_scale_bits.size()]);
                    }
                    break;
                case 6:
                    std::fill_n(block_input, 32, 0.0f);
                    block_input[0] = std::bit_cast<float>(0x03fe0000u);  // amax / 127 == min normal
                    block_input[1] = std::bit_cast<float>(0x007fffffu);  // largest subnormal xi
                    break;
                case 7:
                    std::fill_n(block_input, 32, 0.0f);
                    block_input[0] = std::bit_cast<float>(0x04000000u);
                    block_input[1] = std::bit_cast<float>(0x007fffffu);
                    break;
                case 8:
                    std::fill_n(block_input, 32, 0.0f);
                    block_input[0] = std::bit_cast<float>(0x04800000u);
                    block_input[1] = std::bit_cast<float>(0x007fffffu);
                    break;
                default:
                    break;
            }
        }
    }
}

bool test_shape(Context& context, std::uint32_t n, std::uint32_t rows,
                twin::InputSet set, std::mt19937_64& rng,
                const twin::Options& options) {
    const std::string shape = "Q8_1 n=" + std::to_string(n) + " rows=" + std::to_string(rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    std::vector<float> input_values = twin::synthesize_floats(set, static_cast<std::size_t>(n) * rows, rng, err);
    if (!err.empty()) {
        check(false, shape + " input synthesis", err);
        return false;
    }
    if (set == twin::InputSet::edge) install_q8_1_edge_blocks(input_values, n, rows);

    std::vector<Q81> expected;
    check(twin::quantize_q8_1_reference(input_values, n, rows, expected, err),
          shape + " CPU Q8_1 oracle", err);
    if (expected.empty()) return false;

    const std::size_t input_bytes = input_values.size() * sizeof(float);
    const std::size_t output_bytes = expected.size() * sizeof(Q81);
    twin::GuardedBuffer input;
    twin::GuardedBuffer output;
    check(input.allocate(context, shape + " input", input_bytes, err), shape + " input buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err), shape + " output buffer", err);
    if (!input.view() || !output.view()) return false;
    std::memcpy(input.payload().data(), input_values.data(), input_bytes);
    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    check(output.poison_bytes_against(expected_bytes, err), shape + " output poison", err);
    input.capture_before_dispatch();
    output.capture_before_dispatch();

    const Q81Params params{n, rows};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n + 255u) / 256u, rows, 1};
    work.threads_per_group = {0, 0, 0};
    work.buffers = {{0, input.view()}, {1, output.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 2;
    const bool dispatched = twin::dispatch_manifest(context, "strata_q_q8_1", work, err);
    check(dispatched, shape + " GPU quantizer dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_bytes);
    if (output_failure) {
        twin::report_failure(shape.c_str(), options, rows, *output_failure);
        const auto* actual_blocks = static_cast<const Q81*>(output.view().contents);
        const std::size_t block = output_failure->index / sizeof(Q81);
        const std::size_t byte_in_block = output_failure->index % sizeof(Q81);
        const std::size_t row = block / (n / 32u);
        const std::size_t block_in_row = block % (n / 32u);
        const std::size_t input_base = row * n + block_in_row * 32u;
        std::printf("  block=%zu row=%zu block_in_row=%zu byte_in_block=%zu "
                    "ds actual=0x%08x expected=0x%08x\n",
                    block, row, block_in_row, byte_in_block,
                    actual_blocks[block].ds, expected[block].ds);
        float amax = 0.0f;
        for (std::size_t lane = 0; lane < 32; ++lane) {
            amax = std::max(amax, std::fabs(input_values[input_base + lane]));
        }
        const std::uint16_t actual_scale_bits = static_cast<std::uint16_t>(actual_blocks[block].ds);
        const std::uint16_t expected_scale_bits = static_cast<std::uint16_t>(expected[block].ds);
        std::printf("  amax=%a d_ref=%a d_gpu=%a d_bits=%04x/%04x\n", amax, amax / 127.0f,
                    strata::kernels::f32_from_f16(actual_scale_bits),
                    actual_scale_bits, expected_scale_bits);
        if (byte_in_block >= sizeof(actual_blocks[block].ds)) {
            const std::size_t lane = byte_in_block - sizeof(actual_blocks[block].ds);
            std::printf("  code mismatch lane=%zu actual=%d expected=%d\n", lane,
                        int(actual_blocks[block].qs[lane]), int(expected[block].qs[lane]));
        }
        std::printf("  input[0..3]=%a,%a,%a,%a\n", input_values[input_base],
                    input_values[input_base + 1], input_values[input_base + 2],
                    input_values[input_base + 3]);
    }
    check(!output_failure, shape + " exact Q8_1 bytes");
    const std::array<twin::ByteRange, 1> written = {{{0, output_bytes}}};
    check(!input.first_unwritten_change({}), shape + " input and guards remain unchanged");
    check(!output.first_unwritten_change(written), shape + " output guards remain unchanged");
    return true;
}

bool test_q8_0_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q8_0> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q8_0 n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 32u;
    const std::size_t weight_block_count = static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q8_0> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<int> random_code(-127, 127);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        constexpr std::array<std::int8_t, 12> edge_codes = {
            -127, 127, -1, 0, 1, 2, -2, 63, -63, 126, -126, 17};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            weights[block].d =
                strata::kernels::f16_from_f32(scales[block % scales.size()]);
            for (std::size_t k = 0; k < 32; ++k) {
                int code = 0;
                if (set == twin::InputSet::random) {
                    code = random_code(rng);
                } else if (set == twin::InputSet::edge) {
                    code = edge_codes[(k + block) % edge_codes.size()];
                } else {
                    code = static_cast<int>((k + block) % 7u) - 3;
                }
                weights[block].qs[k] = static_cast<std::int8_t>(code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q8_0_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q8_0 mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q8_0 known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q8_0);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q8MVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 32u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q8_0_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q8_0 mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q8_0 output bytes");
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q8_0 sink bytes");
    }
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_q4_0_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q4_0> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q4_0 n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 32u;
    const std::size_t weight_block_count = static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q4_0> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<int> random_nibble(0, 15);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        constexpr std::array<std::uint8_t, 16> edge_codes = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            weights[block].d =
                strata::kernels::f16_from_f32(scales[block % scales.size()]);
            for (std::size_t k = 0; k < 16; ++k) {
                std::uint8_t low = 0;
                std::uint8_t high = 0;
                if (set == twin::InputSet::random) {
                    low = static_cast<std::uint8_t>(random_nibble(rng));
                    high = static_cast<std::uint8_t>(random_nibble(rng));
                } else if (set == twin::InputSet::edge) {
                    low = edge_codes[(k + block) % edge_codes.size()];
                    high = edge_codes[(k + block + 7u) % edge_codes.size()];
                } else {
                    low = static_cast<std::uint8_t>((k + block) % 16u);
                    high = static_cast<std::uint8_t>((15u - k + block) % 16u);
                }
                weights[block].qs[k] = static_cast<std::uint8_t>(low | (high << 4u));
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q4_0_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q4_0 mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q4_0 known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q4_0);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q4MVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 64u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q4_0_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q4_0 mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q4_0 output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q4_0 sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_q5_0_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q5_0> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q5_0 n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 32u;
    const std::size_t weight_block_count = static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q5_0> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<int> random_code(0, 31);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            auto& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(scales[block % scales.size()]);
            std::fill_n(weight.qh, 4, std::uint8_t{0});
            std::fill_n(weight.qs, 16, std::uint8_t{0});
            for (std::uint32_t index = 0; index < 32; ++index) {
                std::uint32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = static_cast<std::uint32_t>(random_code(rng));
                } else if (set == twin::InputSet::edge) {
                    code = (index + static_cast<std::uint32_t>(block * 7u)) % 32u;
                } else {
                    code = (index * 7u + static_cast<std::uint32_t>(block) + 3u) % 32u;
                }
                set_q5_0_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q5_0_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q5_0 mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q5_0 known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q5_0);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q5MVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 64u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q5_0_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q5_0 mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q5_0 output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q5_0 sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_q2_0_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q2_0> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q2_0 n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 64u;
    const std::size_t activation_blocks_per_row = n_in / 32u;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q2_0> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<int> random_code(0, 3);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            auto& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(scales[block % scales.size()]);
            std::fill_n(weight.qs, 16, std::uint8_t{0});
            for (std::uint32_t index = 0; index < 64; ++index) {
                std::uint32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = static_cast<std::uint32_t>(random_code(rng));
                } else if (set == twin::InputSet::edge) {
                    code = (index + static_cast<std::uint32_t>(block)) % 4u;
                } else {
                    code = (index * 3u + static_cast<std::uint32_t>(block) + 1u) % 4u;
                }
                set_q2_0_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q2_0_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q2_0 mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q2_0 known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q2_0);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q2MVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 64u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q2_0_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q2_0 mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q2_0 output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q2_0 sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_iq4_nl_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                       std::uint32_t nc, std::uint32_t live_rows,
                       twin::InputSet set, std::mt19937_64& rng,
                       const twin::Options& options,
                       std::span<const float> fixed_inputs = {},
                       std::span<const block_iq4_nl> fixed_weights = {},
                       std::span<const float> known_answer = {},
                       std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ4_NL n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 32u;
    const std::size_t activation_blocks_per_row = blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq4_nl> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_code(0, 15);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            auto& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(scales[block % scales.size()]);
            std::fill_n(weight.qs, 16, std::uint8_t{0});
            for (std::uint32_t index = 0; index < 32; ++index) {
                std::uint32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = random_code(rng);
                } else if (set == twin::InputSet::edge) {
                    code = (index + static_cast<std::uint32_t>(block)) % 16u;
                } else {
                    code = (index * 5u + static_cast<std::uint32_t>(block) + 1u) % 16u;
                }
                set_iq4_nl_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq4_nl_matvec_reference(weights, activations, n_in, n_out,
                                        nc, live_rows, expected, err),
          shape + " CPU IQ4_NL mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic IQ4_NL known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq4_nl);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ4NLParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 64u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_iq4_nl_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ4_NL mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact IQ4_NL output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact IQ4_NL sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_iq4_xs_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                       std::uint32_t nc, std::uint32_t live_rows,
                       twin::InputSet set, std::mt19937_64& rng,
                       const twin::Options& options,
                       std::span<const float> fixed_inputs = {},
                       std::span<const block_iq4_xs> fixed_weights = {},
                       std::span<const float> known_answer = {},
                       std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ4_XS n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq4_xs> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_code(0, 15);
        std::uniform_int_distribution<std::uint32_t> random_scale(0, 63);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            auto& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(scales[block % scales.size()]);
            weight.scales_h = 0;
            std::fill_n(weight.scales_l, 4, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
            for (std::uint32_t subblock = 0; subblock < 8; ++subblock) {
                std::uint32_t scale = 0;
                if (set == twin::InputSet::random) {
                    scale = random_scale(rng);
                } else if (set == twin::InputSet::edge) {
                    scale = (subblock * 9u + static_cast<std::uint32_t>(block)) % 64u;
                } else {
                    scale = (subblock * 11u + static_cast<std::uint32_t>(block) * 3u + 1u) % 64u;
                }
                set_iq4_xs_scale(weight, subblock, scale);
            }
            for (std::uint32_t index = 0; index < 256; ++index) {
                std::uint32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = random_code(rng);
                } else if (set == twin::InputSet::edge) {
                    code = (index * 7u + static_cast<std::uint32_t>(block) * 3u) % 16u;
                } else {
                    code = (index * 5u + static_cast<std::uint32_t>(block) + 1u) % 16u;
                }
                set_iq4_xs_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq4_xs_matvec_reference(weights, activations, n_in, n_out,
                                        nc, live_rows, expected, err),
          shape + " CPU IQ4_XS mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic IQ4_XS known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq4_xs);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ4XSParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 16u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_iq4_xs_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ4_XS mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact IQ4_XS output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact IQ4_XS sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_iq3_s_shape(Context& context, std::uint32_t n_in,
                      std::uint32_t n_out, std::uint32_t nc,
                      std::uint32_t live_rows, twin::InputSet set,
                      std::mt19937_64& rng, const twin::Options& options,
                      std::span<const float> fixed_inputs = {},
                      std::span<const block_iq3_s> fixed_weights = {},
                      std::span<const float> known_answer = {},
                      std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ3_S n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq3_s> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_byte(0u, 255u);
        constexpr std::array<float, 5> scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq3_s& weight = weights[block];
            float weight_scale = 1.0f;
            if (set == twin::InputSet::random) {
                weight_scale = scales[random_byte(rng) % scales.size()];
            } else if (set == twin::InputSet::edge) {
                weight_scale = scales[(block * 3u + 1u) % scales.size()];
            } else {
                weight_scale = scales[(block * 2u + 3u) % scales.size()];
            }
            weight.d = strata::kernels::f16_from_f32(weight_scale);
            for (std::uint32_t i = 0; i < 64u; ++i) {
                std::uint32_t qs = 0;
                std::uint32_t qh = 0;
                if (set == twin::InputSet::random) {
                    qs = random_byte(rng);
                    qh = random_byte(rng);
                } else if (set == twin::InputSet::edge) {
                    qs = (i * 73u + static_cast<std::uint32_t>(block) * 29u +
                          0x5bu) & 0xffu;
                    qh = (i * 41u + static_cast<std::uint32_t>(block) * 67u +
                          0xa7u) & 0xffu;
                } else {
                    qs = (i * 37u + static_cast<std::uint32_t>(block) * 11u +
                          3u) & 0xffu;
                    qh = (i * 59u + static_cast<std::uint32_t>(block) * 31u +
                          0x5au) & 0xffu;
                }
                weight.qs[i] = static_cast<std::uint8_t>(qs);
                weight.qh[i / 8u] |= static_cast<std::uint8_t>(
                    ((qh >> (i % 8u)) & 1u) << (i % 8u));
            }
            for (std::uint32_t i = 0; i < 32u; ++i) {
                std::uint32_t signs = 0;
                if (set == twin::InputSet::random) {
                    signs = random_byte(rng);
                } else if (set == twin::InputSet::edge) {
                    signs = (i * 29u + static_cast<std::uint32_t>(block) * 17u +
                             0x96u) & 0xffu;
                } else {
                    signs = (i * 47u + static_cast<std::uint32_t>(block) * 13u +
                             0x69u) & 0xffu;
                }
                weight.signs[i] = static_cast<std::uint8_t>(signs);
            }
            for (std::uint32_t i = 0; i < 4u; ++i) {
                std::uint32_t packed_scales = 0;
                if (set == twin::InputSet::random) {
                    packed_scales = random_byte(rng);
                } else if (set == twin::InputSet::edge) {
                    packed_scales = (i * 0x35u +
                                     static_cast<std::uint32_t>(block) * 0x17u +
                                     0x8bu) & 0xffu;
                } else {
                    packed_scales = (i * 0x47u +
                                     static_cast<std::uint32_t>(block) * 0x19u +
                                     0x5du) & 0xffu;
                }
                weight.scales[i] = static_cast<std::uint8_t>(packed_scales);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq3_s_matvec_reference(weights, activations, n_in, n_out,
                                       nc, live_rows, expected, err),
          shape + " CPU IQ3_S mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) {
            twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        }
        check(!known_failure, shape + " independent IQ3_S known answer");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count =
        static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq3_s);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows",
                             sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ3SParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n_out + 3u) / 4u, 1u, 1u};
    work.threads_per_group = {32u, 4u, 1u};
    work.buffers = {{0u, input.view()}, {1u, weight.view()},
                    {2u, output.view()}, {3u, sink.view()},
                    {4u, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_iq3_s_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ3_S mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure =
        twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    }
    check(!output_failure, shape + " exact IQ3_S output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure =
            twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(shape.c_str(), options, nc - live_rows,
                                 *sink_failure);
        }
        check(!sink_failure, shape + " exact IQ3_S sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}),
          shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}),
          shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}),
          shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written),
          shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written),
          shape + " sink guards remain unchanged");
    return true;
}

bool test_iq2_xxs_shape(Context& context, std::uint32_t n_in,
                        std::uint32_t n_out, std::uint32_t nc,
                        std::uint32_t live_rows, twin::InputSet set,
                        std::mt19937_64& rng, const twin::Options& options,
                        std::span<const float> fixed_inputs = {},
                        std::span<const block_iq2_xxs> fixed_weights = {},
                        std::span<const float> known_answer = {},
                        std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ2_XXS n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (inputs.size() != input_count) {
        check(false, shape + " fixed input size");
        return false;
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq2_xxs> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_word(0u, 65535u);
        constexpr std::array<float, 5> scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq2_xxs& weight = weights[block];
            float weight_scale = 1.0f;
            if (set == twin::InputSet::random) {
                weight_scale = scales[random_word(rng) % scales.size()];
            } else if (set == twin::InputSet::edge) {
                weight_scale = scales[(block * 3u + 1u) % scales.size()];
            } else {
                weight_scale = scales[(block * 2u + 3u) % scales.size()];
            }
            weight.d = strata::kernels::f16_from_f32(weight_scale);
            for (std::uint32_t i = 0; i < 32u; ++i) {
                std::uint32_t word = 0u;
                if (set == twin::InputSet::random) {
                    word = random_word(rng);
                } else if (set == twin::InputSet::edge) {
                    word = (i * 0x2d5bu +
                            static_cast<std::uint32_t>(block) * 0x13a7u +
                            0x8e31u) & 0xffffu;
                } else {
                    word = (i * 0x17cbu +
                            static_cast<std::uint32_t>(block) * 0x4f21u +
                            0x35a9u) & 0xffffu;
                }
                weight.qs[i] = static_cast<std::uint16_t>(word);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }
    if (weights.size() != weight_block_count) {
        check(false, shape + " fixed weight size");
        return false;
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq2_xxs_matvec_reference(weights, activations, n_in, n_out,
                                         nc, live_rows, expected, err),
          shape + " CPU IQ2_XXS mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        if (known_answer.size() != expected.size()) {
            check(false, shape + " independent known-answer size");
            return false;
        }
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) {
            twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                std::fprintf(stderr,
                             "IQ2_XXS known[%zu]: oracle=%a (0x%08x) "
                             "literal=%a (0x%08x)\n",
                             i, expected[i],
                             std::bit_cast<std::uint32_t>(expected[i]),
                             known_answer[i],
                             std::bit_cast<std::uint32_t>(known_answer[i]));
            }
        }
        check(!known_failure, shape + " independent IQ2_XXS known answer");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count =
        static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq2_xxs);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0u) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows",
                             sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ2XXSParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n_out + 3u) / 4u, 1u, 1u};
    work.threads_per_group = {32u, 4u, 1u};
    work.buffers = {{0u, input.view()}, {1u, weight.view()},
                    {2u, output.view()}, {3u, sink.view()},
                    {4u, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_iq2_xxs_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ2_XXS mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure =
        twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
        for (std::size_t i = 0; i < output_count; ++i) {
            float actual_value = 0.0f;
            std::memcpy(&actual_value,
                        output.payload().data() + i * sizeof(float),
                        sizeof(actual_value));
            std::fprintf(stderr,
                         "IQ2_XXS output[%zu]: actual=%a (0x%08x) "
                         "expected=%a (0x%08x)\n",
                         i, actual_value,
                         std::bit_cast<std::uint32_t>(actual_value),
                         expected[i],
                         std::bit_cast<std::uint32_t>(expected[i]));
        }
    }
    check(!output_failure, shape + " exact IQ2_XXS output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0u) {
        const auto sink_failure =
            twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(shape.c_str(), options, nc - live_rows,
                                 *sink_failure);
        }
        check(!sink_failure, shape + " exact IQ2_XXS sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}),
          shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}),
          shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}),
          shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written),
          shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written),
          shape + " sink guards remain unchanged");
    return true;
}

bool test_iq2_xs_shape(Context& context, std::uint32_t n_in,
                       std::uint32_t n_out, std::uint32_t nc,
                       std::uint32_t live_rows, twin::InputSet set,
                       std::mt19937_64& rng, const twin::Options& options,
                       std::span<const float> fixed_inputs = {},
                       std::span<const block_iq2_xs> fixed_weights = {},
                       std::span<const float> known_answer = {},
                       std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ2_XS n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (inputs.size() != input_count) {
        check(false, shape + " fixed input size");
        return false;
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq2_xs> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_word(0u, 65535u);
        constexpr std::array<float, 5> scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq2_xs& weight = weights[block];
            float weight_scale = 1.0f;
            if (set == twin::InputSet::random) {
                weight_scale = scales[random_word(rng) % scales.size()];
            } else if (set == twin::InputSet::edge) {
                weight_scale = scales[(block * 3u + 1u) % scales.size()];
            } else {
                weight_scale = scales[(block * 2u + 3u) % scales.size()];
            }
            weight.d = strata::kernels::f16_from_f32(weight_scale);
            for (std::uint32_t i = 0; i < 32u; ++i) {
                std::uint32_t word = 0u;
                if (set == twin::InputSet::random) {
                    word = random_word(rng);
                } else if (set == twin::InputSet::edge) {
                    word = (i * 0x2d5bu +
                            static_cast<std::uint32_t>(block) * 0x13a7u +
                            0x8e31u) & 0xffffu;
                } else {
                    word = (i * 0x17cbu +
                            static_cast<std::uint32_t>(block) * 0x4f21u +
                            0x35a9u) & 0xffffu;
                }
                weight.qs[i] = static_cast<std::uint16_t>(word);
            }
            for (std::uint32_t part = 0; part < 8u; ++part) {
                std::uint32_t scale_pair = 0u;
                if (set == twin::InputSet::random) {
                    scale_pair = random_word(rng) & 0xffu;
                } else if (set == twin::InputSet::edge) {
                    scale_pair = (block * 53u + part * 29u + 0x37u) & 0xffu;
                } else {
                    scale_pair = (block * 31u + part * 47u + 0x59u) & 0xffu;
                }
                weight.scales[part] = static_cast<std::uint8_t>(scale_pair);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }
    if (weights.size() != weight_block_count) {
        check(false, shape + " fixed weight size");
        return false;
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq2_xs_matvec_reference(weights, activations, n_in, n_out,
                                        nc, live_rows, expected, err),
          shape + " CPU IQ2_XS mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        if (known_answer.size() != expected.size()) {
            check(false, shape + " independent known-answer size");
            return false;
        }
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) {
            twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                std::fprintf(stderr,
                             "IQ2_XS known[%zu]: oracle=%a (0x%08x) "
                             "literal=%a (0x%08x)\n",
                             i, expected[i],
                             std::bit_cast<std::uint32_t>(expected[i]),
                             known_answer[i],
                             std::bit_cast<std::uint32_t>(known_answer[i]));
            }
        }
        check(!known_failure, shape + " independent IQ2_XS known answer");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count =
        static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq2_xs);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0u) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows",
                             sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ2XSParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n_out + 3u) / 4u, 1u, 1u};
    work.threads_per_group = {32u, 4u, 1u};
    work.buffers = {{0u, input.view()}, {1u, weight.view()},
                    {2u, output.view()}, {3u, sink.view()},
                    {4u, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_iq2_xs_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ2_XS mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure =
        twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
        for (std::size_t i = 0; i < output_count; ++i) {
            float actual_value = 0.0f;
            std::memcpy(&actual_value,
                        output.payload().data() + i * sizeof(float),
                        sizeof(actual_value));
            std::fprintf(stderr,
                         "IQ2_XS output[%zu]: actual=%a (0x%08x) "
                         "expected=%a (0x%08x)\n",
                         i, actual_value,
                         std::bit_cast<std::uint32_t>(actual_value),
                         expected[i],
                         std::bit_cast<std::uint32_t>(expected[i]));
        }
    }
    check(!output_failure, shape + " exact IQ2_XS output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0u) {
        const auto sink_failure =
            twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(shape.c_str(), options, nc - live_rows,
                                 *sink_failure);
        }
        check(!sink_failure, shape + " exact IQ2_XS sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}),
          shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}),
          shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}),
          shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written),
          shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written),
          shape + " sink guards remain unchanged");
    return true;
}

bool test_iq2_s_shape(Context& context, std::uint32_t n_in,
                      std::uint32_t n_out, std::uint32_t nc,
                      std::uint32_t live_rows, twin::InputSet set,
                      std::mt19937_64& rng, const twin::Options& options,
                      std::span<const float> fixed_inputs = {},
                      std::span<const block_iq2_s> fixed_weights = {},
                      std::span<const float> known_answer = {},
                      std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ2_S n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (inputs.size() != input_count) {
        check(false, shape + " fixed input size");
        return false;
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq2_s> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_byte(0u, 255u);
        constexpr std::array<float, 5> scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq2_s& weight = weights[block];
            float weight_scale = 1.0f;
            if (set == twin::InputSet::random) {
                weight_scale = scales[random_byte(rng) % scales.size()];
            } else if (set == twin::InputSet::edge) {
                weight_scale = scales[(block * 3u + 1u) % scales.size()];
            } else {
                weight_scale = scales[(block * 2u + 3u) % scales.size()];
            }
            weight.d = strata::kernels::f16_from_f32(weight_scale);
            for (std::uint32_t part = 0; part < 8u; ++part) {
                std::uint32_t scale_pair = 0u;
                if (set == twin::InputSet::random) {
                    scale_pair = random_byte(rng);
                    weight.qh[part] =
                        static_cast<std::uint8_t>(random_byte(rng));
                } else if (set == twin::InputSet::edge) {
                    scale_pair =
                        (block * 53u + part * 29u + 0x37u) & 0xffu;
                    weight.qh[part] = static_cast<std::uint8_t>(
                        (block * 0x6du + part * 0x35u + 0x96u) & 0xffu);
                } else {
                    scale_pair =
                        (block * 31u + part * 47u + 0x59u) & 0xffu;
                    weight.qh[part] = static_cast<std::uint8_t>(
                        (block * 0x35u + part * 0x6du + 0x4bu) & 0xffu);
                }
                weight.scales[part] = static_cast<std::uint8_t>(scale_pair);
                for (std::uint32_t code_slot = 0; code_slot < 4u;
                     ++code_slot) {
                    std::uint32_t low = 0u;
                    std::uint32_t signs = 0u;
                    if (set == twin::InputSet::random) {
                        low = random_byte(rng);
                        signs = random_byte(rng);
                    } else if (set == twin::InputSet::edge) {
                        low = (block * 0x5bu + part * 0x2du +
                               code_slot * 0x73u + 0x19u) & 0xffu;
                        signs = (block * 0x37u + part * 0x91u +
                                 code_slot * 0x4bu + 0xe3u) & 0xffu;
                    } else {
                        low = (block * 0x2du + part * 0x73u +
                               code_slot * 0x5bu + 0xa7u) & 0xffu;
                        signs = (block * 0x91u + part * 0x4bu +
                                 code_slot * 0x37u + 0x65u) & 0xffu;
                    }
                    weight.qs[4u * part + code_slot] =
                        static_cast<std::uint8_t>(low);
                    weight.qs[32u + 4u * part + code_slot] =
                        static_cast<std::uint8_t>(signs);
                }
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }
    if (weights.size() != weight_block_count) {
        check(false, shape + " fixed weight size");
        return false;
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq2_s_matvec_reference(weights, activations, n_in, n_out,
                                       nc, live_rows, expected, err),
          shape + " CPU IQ2_S mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        if (known_answer.size() != expected.size()) {
            check(false, shape + " independent known-answer size");
            return false;
        }
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure =
            twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) {
            twin::report_failure(shape.c_str(), options, live_rows,
                                 *known_failure);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                std::fprintf(stderr,
                             "IQ2_S known[%zu]: oracle=%a (0x%08x) "
                             "literal=%a (0x%08x)\n",
                             i, expected[i],
                             std::bit_cast<std::uint32_t>(expected[i]),
                             known_answer[i],
                             std::bit_cast<std::uint32_t>(known_answer[i]));
            }
        }
        check(!known_failure, shape + " independent IQ2_S known answer");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count =
        static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq2_s);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0u) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows",
                             sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ2SParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n_out + 3u) / 4u, 1u, 1u};
    work.threads_per_group = {32u, 4u, 1u};
    work.buffers = {{0u, input.view()}, {1u, weight.view()},
                    {2u, output.view()}, {3u, sink.view()},
                    {4u, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_iq2_s_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ2_S mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure =
        twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
        for (std::size_t i = 0; i < output_count; ++i) {
            float actual_value = 0.0f;
            std::memcpy(&actual_value,
                        output.payload().data() + i * sizeof(float),
                        sizeof(actual_value));
            std::fprintf(stderr,
                         "IQ2_S output[%zu]: actual=%a (0x%08x) "
                         "expected=%a (0x%08x)\n",
                         i, actual_value,
                         std::bit_cast<std::uint32_t>(actual_value),
                         expected[i],
                         std::bit_cast<std::uint32_t>(expected[i]));
        }
    }
    check(!output_failure, shape + " exact IQ2_S output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0u) {
        const auto sink_failure =
            twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(shape.c_str(), options, nc - live_rows,
                                 *sink_failure);
        }
        check(!sink_failure, shape + " exact IQ2_S sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}),
          shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}),
          shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}),
          shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written),
          shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written),
          shape + " sink guards remain unchanged");
    return true;
}

void pack_iq1_m_scales(block_iq1_m& weight, std::uint16_t half_scale_bits,
                       const std::array<std::uint8_t, 16>& local_scales) {
    std::array<std::uint16_t, 4> packed{};
    for (std::uint32_t part = 0; part < 8u; ++part) {
        const std::uint32_t word = part / 2u;
        const std::uint32_t shift = 6u * (part & 1u);
        packed[word] |= static_cast<std::uint16_t>(
            (std::uint16_t(local_scales[2u * part + 0u]) << shift) |
            (std::uint16_t(local_scales[2u * part + 1u]) << (shift + 3u)));
    }
    for (std::uint32_t word = 0; word < packed.size(); ++word) {
        packed[word] |= static_cast<std::uint16_t>(
            ((half_scale_bits >> (4u * word)) & 0x0fu) << 12u);
        weight.scales[2u * word + 0u] =
            static_cast<std::uint8_t>(packed[word] & 0xffu);
        weight.scales[2u * word + 1u] =
            static_cast<std::uint8_t>(packed[word] >> 8u);
    }
}

bool test_iq1_m_shape(Context& context, std::uint32_t n_in,
                      std::uint32_t n_out, std::uint32_t nc,
                      std::uint32_t live_rows, twin::InputSet set,
                      std::mt19937_64& rng, const twin::Options& options,
                      std::span<const float> fixed_inputs = {},
                      std::span<const block_iq1_m> fixed_weights = {},
                      std::span<const float> known_answer = {},
                      std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ1_M n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (inputs.size() != input_count) {
        check(false, shape + " fixed input size");
        return false;
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq1_m> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_byte(0u, 255u);
        std::uniform_int_distribution<std::uint32_t> random_grid(0u, 2047u);
        constexpr std::array<float, 5> scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq1_m& weight = weights[block];
            std::array<std::uint8_t, 16> local_scales{};
            float weight_scale = 1.0f;
            if (set == twin::InputSet::random) {
                weight_scale = scales[random_byte(rng) % scales.size()];
            } else if (set == twin::InputSet::edge) {
                weight_scale = scales[(block * 3u + 1u) % scales.size()];
            } else {
                weight_scale = scales[(block * 2u + 3u) % scales.size()];
            }
            const std::uint16_t half_scale_bits =
                strata::kernels::f16_from_f32(weight_scale);
            for (std::uint32_t part = 0; part < 8u; ++part) {
                for (std::uint32_t codeword = 0; codeword < 4u;
                     ++codeword) {
                    std::uint32_t grid_index = 0u;
                    std::uint32_t qhl = 0u;
                    if (set == twin::InputSet::random) {
                        grid_index = random_grid(rng);
                        qhl = grid_index >> 8u;
                        if ((random_byte(rng) & 1u) != 0u) qhl |= 0x08u;
                    } else if (set == twin::InputSet::edge) {
                        grid_index = (static_cast<std::uint32_t>(block) * 0x1d3u +
                                      part * 0x11bu + codeword * 0x5du +
                                      0x7f3u) & 0x7ffu;
                        qhl = (grid_index >> 8u) |
                              (((block + part + codeword) & 1u) << 3u);
                    } else {
                        grid_index = (static_cast<std::uint32_t>(block) * 0x1a5u +
                                      part * 0x16du + codeword * 0x73u +
                                      0x35bu) & 0x7ffu;
                        qhl = (grid_index >> 8u) |
                              (((block * 3u + part + codeword * 5u) & 1u) << 3u);
                    }
                    weight.qs[4u * part + codeword] =
                        static_cast<std::uint8_t>(grid_index & 0xffu);
                    const std::uint32_t qh_byte_index =
                        2u * part + codeword / 2u;
                    const std::uint32_t qh_shift =
                        4u * (codeword & 1u);
                    weight.qh[qh_byte_index] |=
                        static_cast<std::uint8_t>(qhl << qh_shift);
                }
                for (std::uint32_t half = 0; half < 2u; ++half) {
                    const std::uint32_t scale_index = 2u * part + half;
                    if (set == twin::InputSet::random) {
                        local_scales[scale_index] = random_byte(rng) & 0x07u;
                    } else if (set == twin::InputSet::edge) {
                        local_scales[scale_index] =
                            (static_cast<std::uint32_t>(block) * 5u +
                             part * 3u + half * 7u + 7u) & 0x07u;
                    } else {
                        local_scales[scale_index] =
                            (static_cast<std::uint32_t>(block) * 3u +
                             part * 5u + half * 3u + 1u) & 0x07u;
                    }
                }
            }
            pack_iq1_m_scales(weight, half_scale_bits, local_scales);
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }
    if (weights.size() != weight_block_count) {
        check(false, shape + " fixed weight size");
        return false;
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq1_m_matvec_reference(weights, activations, n_in, n_out,
                                       nc, live_rows, expected, err),
          shape + " CPU IQ1_M mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        if (known_answer.size() != expected.size()) {
            check(false, shape + " independent known-answer size");
            return false;
        }
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure =
            twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) {
            twin::report_failure(shape.c_str(), options, live_rows,
                                 *known_failure);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                std::fprintf(stderr,
                             "IQ1_M known[%zu]: oracle=%a (0x%08x) "
                             "literal=%a (0x%08x)\n",
                             i, expected[i],
                             std::bit_cast<std::uint32_t>(expected[i]),
                             known_answer[i],
                             std::bit_cast<std::uint32_t>(known_answer[i]));
            }
        }
        check(!known_failure, shape + " independent IQ1_M known answer");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count =
        static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq1_m);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0u) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows",
                             sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ1MParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n_out + 3u) / 4u, 1u, 1u};
    work.threads_per_group = {32u, 4u, 1u};
    work.buffers = {{0u, input.view()}, {1u, weight.view()},
                    {2u, output.view()}, {3u, sink.view()},
                    {4u, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_iq1_m_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ1_M mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure =
        twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
        for (std::size_t i = 0; i < output_count; ++i) {
            float actual_value = 0.0f;
            std::memcpy(&actual_value,
                        output.payload().data() + i * sizeof(float),
                        sizeof(actual_value));
            std::fprintf(stderr,
                         "IQ1_M output[%zu]: actual=%a (0x%08x) "
                         "expected=%a (0x%08x)\n",
                         i, actual_value,
                         std::bit_cast<std::uint32_t>(actual_value),
                         expected[i],
                         std::bit_cast<std::uint32_t>(expected[i]));
        }
    }
    check(!output_failure, shape + " exact IQ1_M output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0u) {
        const auto sink_failure =
            twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(shape.c_str(), options, nc - live_rows,
                                 *sink_failure);
        }
        check(!sink_failure, shape + " exact IQ1_M sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}),
          shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}),
          shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}),
          shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written),
          shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written),
          shape + " sink guards remain unchanged");
    return true;
}

bool run_iq1_m_generated_decoder_sweep(Context& context,
                                      std::mt19937_64& rng,
                                      const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 4096u;
    constexpr std::uint32_t nc = 8u;
    constexpr std::array<std::uint16_t, 8> half_scales = {
        0x3c00u, 0x3800u, 0x4000u, 0x3555u,
        0x3e00u, 0x3400u, 0x3a00u, 0x3f00u};
    bool completed = true;

    for (std::uint32_t codeword = 0; codeword < 4u; ++codeword) {
        std::vector<block_iq1_m> weights(n_out);
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        std::vector<float> known_answer(static_cast<std::size_t>(n_out) * nc);
        std::array<std::uint32_t, 64> observed_grid_indices{};
        std::uint16_t observed_qh_nibbles = 0u;
        std::uint8_t observed_local_scales = 0u;
        bool raw_fields_match = true;

        for (std::uint32_t column = 0; column < nc; ++column) {
            inputs[static_cast<std::size_t>(column) * n_in +
                   8u * codeword + column] = 127.0f;
        }

        for (std::uint32_t row = 0; row < n_out; ++row) {
            const std::uint32_t grid_index = row / 2u;
            const std::uint32_t delta_bit = row & 1u;
            const std::uint32_t qhl =
                ((grid_index >> 8u) & 0x07u) | (delta_bit << 3u);
            const std::uint32_t qh_byte = codeword / 2u;
            const std::uint32_t qh_shift = 4u * (codeword & 1u);
            const std::uint32_t local_scale_index =
                codeword < 2u ? 0u : 1u;
            const std::uint8_t local_scale =
                static_cast<std::uint8_t>(row & 0x07u);
            const std::uint16_t half_scale_bits =
                half_scales[(row + codeword) % half_scales.size()];

            block_iq1_m& weight = weights[row];
            weight.qs[codeword] =
                static_cast<std::uint8_t>(grid_index & 0xffu);
            weight.qh[qh_byte] =
                static_cast<std::uint8_t>(qhl << qh_shift);
            std::array<std::uint8_t, 16> local_scales{};
            local_scales[local_scale_index] = local_scale;
            pack_iq1_m_scales(weight, half_scale_bits, local_scales);

            const std::uint32_t actual_qhl =
                (weight.qh[qh_byte] >> qh_shift) & 0x0fu;
            const std::uint32_t decoded_index =
                static_cast<std::uint32_t>(weight.qs[codeword]) |
                ((actual_qhl & 0x07u) << 8u);
            const std::uint32_t packed_word =
                static_cast<std::uint32_t>(weight.scales[0]) |
                (static_cast<std::uint32_t>(
                     weight.scales[1])
                 << 8u);
            const std::uint32_t decoded_local_scale =
                (packed_word >> ((local_scale_index & 1u) * 3u)) &
                0x07u;
            raw_fields_match &= decoded_index == grid_index &&
                                actual_qhl == qhl &&
                                decoded_local_scale == local_scale;
            observed_grid_indices[grid_index / 32u] |=
                1u << (grid_index & 31u);
            observed_qh_nibbles |=
                static_cast<std::uint16_t>(1u << actual_qhl);
            observed_local_scales |=
                static_cast<std::uint8_t>(1u << decoded_local_scale);

            const std::uint32_t grid = iq1s_grid_gpu[grid_index];
            const float delta =
                delta_bit != 0u ? -1.125f : -0.875f;
            const float weight_scale =
                strata::kernels::f32_from_f16(half_scale_bits);
            const float odd_local_scale =
                static_cast<float>(2u * local_scale + 1u);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const std::uint32_t digit =
                    column < 4u
                        ? (grid >> (8u * column)) & 0x0fu
                        : (grid >> (8u * (column - 4u) + 4u)) & 0x0fu;
                const float scaled_dot =
                    (static_cast<float>(digit * 127u) +
                     std::fma(delta, 127.0f, 0.0f)) *
                    odd_local_scale;
                known_answer[static_cast<std::size_t>(column) * n_out + row] =
                    std::fma(weight_scale, scaled_dot, 0.0f);
            }
        }

        const std::string shape =
            "IQ1_M generated decoder sweep codeword=" +
            std::to_string(codeword);
        check(raw_fields_match, shape + " raw qs/qh/scale fields");
        bool all_grid_indices_seen = true;
        for (const std::uint32_t observed : observed_grid_indices) {
            all_grid_indices_seen &= observed == 0xffffffffu;
        }
        check(all_grid_indices_seen,
              shape + " covers all 2048 grid indices");
        check(observed_qh_nibbles == 0xffffu,
              shape + " covers all 16 qh nibble states");
        check(observed_local_scales == 0xffu,
              shape + " covers all eight local scale values");
        completed &= test_iq1_m_shape(
            context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq1_m_packed_golden(Context& context, std::mt19937_64& rng,
                             const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 1u;
    constexpr std::uint32_t nc = 8u;
    constexpr std::array<std::uint8_t, 8> raw_scales = {
        0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0xc0u, 0x40u, 0x35u};
    constexpr std::array<float, 32> known_positions = {
        -1222.375f, -1222.375f, -1222.375f, -1222.375f,
        -1222.375f, -1222.375f, -1222.375f, -1222.375f,
        -174.625f, 1222.375f, -174.625f, -1571.625f,
        -1571.625f, -174.625f, 1222.375f, -1571.625f,
        714.375f, 79.375f, 714.375f, -555.625f,
        -555.625f, 79.375f, 79.375f, 714.375f,
        -79.375f, -714.375f, 555.625f, -79.375f,
        -79.375f, 555.625f, 555.625f, 555.625f};

    block_iq1_m weight{};
    weight.qs[28] = 0x00u;
    weight.qs[29] = 0xffu;
    weight.qs[30] = 0xa5u;
    weight.qs[31] = 0xe7u;
    weight.qh[14] = 0x90u;
    weight.qh[15] = 0xf6u;
    std::copy(raw_scales.begin(), raw_scales.end(), weight.scales);
    check(std::equal(raw_scales.begin(), raw_scales.end(), weight.scales),
          "IQ1_M literal packed golden retains raw tail scale bytes");
    constexpr std::array<std::uint32_t, 4> expected_indices = {
        0x000u, 0x1ffu, 0x6a5u, 0x7e7u};
    constexpr std::array<std::uint32_t, 4> expected_deltas = {0u, 1u, 0u, 1u};
    bool tail_fields_match = true;
    for (std::uint32_t codeword = 0; codeword < 4u; ++codeword) {
        const std::uint32_t qhl =
            (weight.qh[14u + codeword / 2u] >>
             (4u * (codeword & 1u))) &
            0x0fu;
        const std::uint32_t grid_index =
            static_cast<std::uint32_t>(weight.qs[28u + codeword]) |
            ((qhl & 7u) << 8u);
        tail_fields_match &= grid_index == expected_indices[codeword] &&
                             (qhl >> 3u) == expected_deltas[codeword];
    }
    check(tail_fields_match,
          "IQ1_M literal tail bytes decode to all four fixed grid indices");

    const std::array<block_iq1_m, 1> weights = {weight};
    bool completed = true;
    for (std::uint32_t batch = 0; batch < 32u; batch += nc) {
        std::array<float, n_in * nc> inputs{};
        std::array<float, nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t position = 224u + batch + column;
            inputs[static_cast<std::size_t>(column) * n_in + position] =
                127.0f;
            known_answer[column] = known_positions[batch + column];
        }
        completed &= test_iq1_m_shape(
            context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq1_m_padded_fixture(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 3u;
    constexpr std::uint32_t nc = 4u;
    constexpr std::uint32_t live_rows = 2u;
    constexpr std::array<std::uint16_t, n_out> weight_scales = {
        0x3800u, 0x3c00u, 0x4000u};
    constexpr std::array<float, n_out * nc> known_answer = {
        71.4375f, 142.875f, 285.75f,
        71.4375f, 142.875f, 285.75f,
        71.4375f, 142.875f, 285.75f,
        71.4375f, 142.875f, 285.75f};

    std::array<block_iq1_m, n_out> weights{};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        weights[row].qs[0] = 1u;
        weights[row].qs[3] = 1u;
        std::array<std::uint8_t, 16> local_scales{};
        pack_iq1_m_scales(weights[row], weight_scales[row], local_scales);
    }
    std::array<float, n_in * nc> inputs{};
    inputs[0] = 127.0f;
    inputs[n_in + 24u] = 127.0f;
    inputs[2u * n_in] = -127.0f;
    inputs[3u * n_in] = 63.0f;
    return test_iq1_m_shape(
        context, n_in, n_out, nc, live_rows, twin::InputSet::fixture, rng,
        options, inputs, weights, known_answer);
}

bool run_iq1_m_fma_golden(Context& context, std::mt19937_64& rng,
                          const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9u, 13u};
    constexpr std::array<std::uint32_t, 13> grid_indices = {
        9u, 88u, 167u, 246u, 325u, 404u, 483u,
        562u, 641u, 720u, 799u, 878u, 957u};
    constexpr std::array<std::uint32_t, 13> local_scales = {
        1u, 4u, 7u, 2u, 5u, 0u, 3u, 6u, 1u, 4u, 7u, 2u, 5u};
    constexpr std::array<std::uint16_t, 13> half_scale_bits = {
        0x3d00u, 0x34f6u, 0x3f00u, 0x3a00u, 0x3452u,
        0x3e00u, 0x3800u, 0x3029u, 0x3d00u, 0x34f6u,
        0x3f00u, 0x3a00u, 0x3452u};
    constexpr std::array<std::uint32_t, 2> expected_bits = {
        0xc3272572u, 0xc32f4206u};

    bool completed = true;
    for (std::size_t test = 0; test < block_counts.size(); ++test) {
        const std::uint32_t blocks = block_counts[test];
        const std::uint32_t n_in = blocks * 256u;
        std::vector<block_iq1_m> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            const std::uint32_t index = grid_indices[block];
            const std::uint32_t delta_bit = (block * 7u + 36u * 3u + 1u) & 1u;
            const std::uint32_t local_scale = local_scales[block];
            const std::uint32_t amplitude_step =
                (block * 13u + 36u * 11u) % 19u;
            const float amplitude =
                3.7f + static_cast<float>(amplitude_step) * 0.03125f;
            block_iq1_m& weight = weights[block];
            weight.qs[0] = static_cast<std::uint8_t>(index & 0xffu);
            weight.qh[0] = static_cast<std::uint8_t>(
                ((index >> 8u) & 7u) | (delta_bit << 3u));
            std::array<std::uint8_t, 16> packed_local_scales{};
            packed_local_scales[0] =
                static_cast<std::uint8_t>(local_scale);
            pack_iq1_m_scales(weight, half_scale_bits[block],
                              packed_local_scales);
            inputs[static_cast<std::size_t>(block) * 256u] = amplitude;
        }
        const float known = std::bit_cast<float>(expected_bits[test]);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq1_m_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq1_m_long_k_golden(Context& context, std::mt19937_64& rng,
                             const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63u, 64u, 65u};
    bool completed = true;
    for (const std::uint32_t blocks : block_counts) {
        const std::uint32_t n_in = blocks * 256u;
        std::vector<block_iq1_m> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            weights[block].qs[0] = 1u;
            std::array<std::uint8_t, 16> local_scales{};
            pack_iq1_m_scales(weights[block], 0x3c00u, local_scales);
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const std::array<float, 1> known_answer = {
            142.875f * static_cast<float>(blocks)};
        completed &= test_iq1_m_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq1_m_multi_eq_single(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::array<std::uint32_t, 4> widths = {
        256u, 1024u, 2304u, 16384u};
    constexpr std::array<std::uint32_t, 3> column_counts = {2u, 4u, 8u};
    constexpr std::uint32_t n_out = 5u;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::size_t blocks_per_row = n_in / 256u;
        std::vector<block_iq1_m> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq1_m& weight = weights[block];
            const std::uint16_t half_scale_bits =
                strata::kernels::f16_from_f32(
                    0.25f + 0.125f * static_cast<float>(block % 7u));
            std::array<std::uint8_t, 16> local_scales{};
            for (std::uint32_t part = 0; part < 8u; ++part) {
                for (std::uint32_t codeword = 0; codeword < 4u;
                     ++codeword) {
                    const std::uint32_t index =
                        (static_cast<std::uint32_t>(block) * 0x1d3u +
                         part * 0x11bu + codeword * 0x05du + 0x7f3u) &
                        0x7ffu;
                    const std::uint32_t qhl =
                        ((index >> 8u) & 7u) |
                        (((block + part * 3u + codeword) & 1u) << 3u);
                    weight.qs[4u * part + codeword] =
                        static_cast<std::uint8_t>(index & 0xffu);
                    weight.qh[2u * part + codeword / 2u] |=
                        static_cast<std::uint8_t>(
                            qhl << (4u * (codeword & 1u)));
                }
                for (std::uint32_t half = 0; half < 2u; ++half) {
                    local_scales[2u * part + half] =
                        static_cast<std::uint8_t>(
                            (static_cast<std::uint32_t>(block) * 3u +
                             part * 5u + half * 3u + 1u) &
                            7u);
                }
            }
            pack_iq1_m_scales(weight, half_scale_bits, local_scales);
        }

        std::vector<float> inputs(static_cast<std::size_t>(n_in) * 8u);
        std::string err;
        inputs = twin::synthesize_floats(
            twin::InputSet::fixture, inputs.size(), rng, err);
        if (!err.empty()) {
            check(false, "IQ1_M mv_multi_eq_single input synthesis", err);
            return false;
        }
        for (const std::uint32_t nc : column_counts) {
            std::vector<float> multi;
            completed &= test_iq1_m_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                rng, options,
                std::span<const float>(
                    inputs.data(), static_cast<std::size_t>(n_in) * nc),
                weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ1_M multi-column result available");
                completed = false;
                continue;
            }
            for (std::uint32_t column = 0; column < nc; ++column) {
                std::vector<float> single;
                completed &= test_iq1_m_shape(
                    context, n_in, n_out, 1u, 1u,
                    twin::InputSet::fixture, rng, options,
                    std::span<const float>(
                        inputs.data() +
                            static_cast<std::size_t>(column) * n_in,
                        n_in),
                    weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ1_M single-column result available");
                    completed = false;
                    continue;
                }
                const auto multi_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() +
                        static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch =
                    twin::compare_bytes(multi_bytes, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ1_M mv_multi_eq_single",
                                         options, column, *mismatch);
                }
                check(!mismatch,
                      "IQ1_M NC column equals single-column dispatch");
            }
        }
    }

    std::array<block_iq1_m, 1> base_weight{};
    pack_iq1_m_scales(base_weight[0], 0x3c00u,
                      std::array<std::uint8_t, 16>{});
    std::array<block_iq1_m, 1> changed_weight = base_weight;
    changed_weight[0].qs[0] = 1u;
    std::array<float, 256> control_input{};
    control_input[0] = 127.0f;
    std::vector<float> base;
    std::vector<float> changed;
    completed &= test_iq1_m_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, base_weight, {}, &base);
    completed &= test_iq1_m_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, changed_weight, {}, &changed);
    if (base.size() == 1u && changed.size() == 1u) {
        check(std::bit_cast<std::uint32_t>(base[0]) !=
                  std::bit_cast<std::uint32_t>(changed[0]),
              "IQ1_M code-index negative control changes output");
    } else {
        check(false, "IQ1_M negative-control results available");
        completed = false;
    }
    return completed;
}

bool run_iq1_m_set(Context& context, twin::InputSet set,
                   const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x1a1a5u);
    constexpr std::array<std::uint32_t, 14> widths = {
        256u, 512u, 768u, 1024u, 1280u, 1792u, 2048u, 2304u, 2560u,
        6144u, 10240u, 16128u, 16384u, 16640u};
    constexpr std::array<std::uint32_t, 4> column_counts = {1u, 2u, 4u, 8u};
    constexpr std::uint32_t n_out = 7u;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq1_m_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_iq1_m_generated_decoder_sweep(
            context, rng, options);
        completed &= run_iq1_m_packed_golden(context, rng, options);
        completed &= run_iq1_m_padded_fixture(context, rng, options);
        completed &= run_iq1_m_fma_golden(context, rng, options);
        completed &= run_iq1_m_long_k_golden(context, rng, options);
        completed &= run_iq1_m_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool test_iq3_xxs_shape(Context& context, std::uint32_t n_in,
                        std::uint32_t n_out, std::uint32_t nc,
                        std::uint32_t live_rows, twin::InputSet set,
                        std::mt19937_64& rng, const twin::Options& options,
                        std::span<const float> fixed_inputs = {},
                        std::span<const block_iq3_xxs> fixed_weights = {},
                        std::span<const float> known_answer = {},
                        std::vector<float>* actual_output = nullptr) {
    const std::string shape = "IQ3_XXS n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (inputs.size() != input_count) {
        check(false, shape + " fixed input size");
        return false;
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_iq3_xxs> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_byte(0u, 255u);
        constexpr std::array<float, 5> scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq3_xxs& weight = weights[block];
            float weight_scale = 1.0f;
            if (set == twin::InputSet::random) {
                weight_scale = scales[random_byte(rng) % scales.size()];
            } else if (set == twin::InputSet::edge) {
                weight_scale = scales[(block * 3u + 1u) % scales.size()];
            } else {
                weight_scale = scales[(block * 2u + 3u) % scales.size()];
            }
            weight.d = strata::kernels::f16_from_f32(weight_scale);
            for (std::uint32_t part = 0; part < 8u; ++part) {
                std::uint32_t aux32 = 0u;
                for (std::uint32_t code_slot = 0; code_slot < 4u;
                     ++code_slot) {
                    std::uint32_t sign_index = 0u;
                    if (set == twin::InputSet::random) {
                        sign_index = random_byte(rng) & 0x7fu;
                    } else if (set == twin::InputSet::edge) {
                        sign_index =
                            (block * 0x25u + part * 0x39u +
                             code_slot * 0x5bu + 0x7du) & 0x7fu;
                    } else {
                        sign_index =
                            (block * 0x39u + part * 0x5bu +
                             code_slot * 0x25u + 0x17u) & 0x7fu;
                    }
                    aux32 |= sign_index << (7u * code_slot);
                }
                std::uint32_t scale = 0u;
                if (set == twin::InputSet::random) {
                    scale = random_byte(rng) & 0x0fu;
                } else if (set == twin::InputSet::edge) {
                    scale = (block * 11u + part * 7u + 13u) & 0x0fu;
                } else {
                    scale = (block * 5u + part * 9u + 3u) & 0x0fu;
                }
                aux32 |= scale << 28u;
                for (std::uint32_t grid_slot = 0; grid_slot < 8u;
                     ++grid_slot) {
                    std::uint32_t index = 0u;
                    if (set == twin::InputSet::random) {
                        index = random_byte(rng);
                    } else if (set == twin::InputSet::edge) {
                        index = (block * 0x67u + part * 0x2du +
                                 grid_slot * 0x53u + 0x91u) & 0xffu;
                    } else {
                        index = (block * 0x2du + part * 0x53u +
                                 grid_slot * 0x67u + 0x35u) & 0xffu;
                    }
                    weight.qs[8u * part + grid_slot] =
                        static_cast<std::uint8_t>(index);
                }
                for (std::uint32_t byte = 0; byte < 4u; ++byte) {
                    weight.qs[64u + 4u * part + byte] =
                        static_cast<std::uint8_t>(aux32 >> (8u * byte));
                }
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }
    if (weights.size() != weight_block_count) {
        check(false, shape + " fixed weight size");
        return false;
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::iq3_xxs_matvec_reference(weights, activations, n_in, n_out,
                                         nc, live_rows, expected, err),
          shape + " CPU IQ3_XXS mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        if (known_answer.size() != expected.size()) {
            check(false, shape + " independent known-answer size");
            return false;
        }
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure =
            twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) {
            twin::report_failure(shape.c_str(), options, live_rows,
                                 *known_failure);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                std::fprintf(stderr,
                             "IQ3_XXS known[%zu]: oracle=%a (0x%08x) "
                             "literal=%a (0x%08x)\n",
                             i, expected[i],
                             std::bit_cast<std::uint32_t>(expected[i]),
                             known_answer[i],
                             std::bit_cast<std::uint32_t>(known_answer[i]));
            }
        }
        check(!known_failure,
              shape + " independent IQ3_XXS known answer");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count =
        static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_iq3_xxs);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0u) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows",
                             sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const IQ3XXSParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {(n_out + 3u) / 4u, 1u, 1u};
    work.threads_per_group = {32u, 4u, 1u};
    work.buffers = {{0u, input.view()}, {1u, weight.view()},
                    {2u, output.view()}, {3u, sink.view()},
                    {4u, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5u;
    const std::string kernel = "strata_mv_iq3_xxs_n" + std::to_string(nc);
    const bool dispatched =
        twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU IQ3_XXS mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure =
        twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) {
        twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
        for (std::size_t i = 0; i < output_count; ++i) {
            float actual_value = 0.0f;
            std::memcpy(&actual_value,
                        output.payload().data() + i * sizeof(float),
                        sizeof(actual_value));
            std::fprintf(stderr,
                         "IQ3_XXS output[%zu]: actual=%a (0x%08x) "
                         "expected=%a (0x%08x)\n",
                         i, actual_value,
                         std::bit_cast<std::uint32_t>(actual_value),
                         expected[i],
                         std::bit_cast<std::uint32_t>(expected[i]));
        }
    }
    check(!output_failure, shape + " exact IQ3_XXS output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0u) {
        const auto sink_failure =
            twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) {
            twin::report_failure(shape.c_str(), options, nc - live_rows,
                                 *sink_failure);
        }
        check(!sink_failure, shape + " exact IQ3_XXS sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {
        {{0u, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}),
          shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}),
          shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}),
          shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written),
          shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written),
          shape + " sink guards remain unchanged");
    return true;
}

bool test_q4_k_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q4_K> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q4_K n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row =
        static_cast<std::size_t>(8u) * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q4_K> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_code(0, 15);
        std::uniform_int_distribution<std::uint32_t> random_scale(0, 63);
        constexpr std::array<float, 5> block_scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        constexpr std::array<std::uint32_t, 16> edge_scales = {
            0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 47, 48, 61, 62, 63, 8};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_q4_K& weight = weights[block];
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
            const float d = block_scales[block % block_scales.size()];
            const float dmin = block_scales[(block + 2u) % block_scales.size()];
            weight.dm =
                static_cast<std::uint32_t>(strata::kernels::f16_from_f32(d)) |
                (static_cast<std::uint32_t>(
                     strata::kernels::f16_from_f32(dmin)) << 16u);
            for (std::uint32_t group = 0; group < 8u; ++group) {
                std::uint32_t scale = 0;
                std::uint32_t min = 0;
                if (set == twin::InputSet::random) {
                    scale = random_scale(rng);
                    min = random_scale(rng);
                } else if (set == twin::InputSet::edge) {
                    scale = edge_scales[(group + block) % edge_scales.size()];
                    min = edge_scales[(group * 3u + block + 5u) %
                                      edge_scales.size()];
                } else {
                    scale = (group * 17u + static_cast<std::uint32_t>(block) * 7u +
                             3u) % 64u;
                    min = (group * 29u + static_cast<std::uint32_t>(block) * 11u +
                           1u) % 64u;
                }
                set_q4_k_scale_min(weight, group, scale, min);
            }
            for (std::uint32_t index = 0; index < 256u; ++index) {
                std::uint32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = random_code(rng);
                } else if (set == twin::InputSet::edge) {
                    code = (index * 5u + static_cast<std::uint32_t>(block) * 3u +
                            1u) % 16u;
                } else {
                    code = (index * 7u + static_cast<std::uint32_t>(block) * 11u +
                            index / 32u * 3u + 2u) % 16u;
                }
                set_q4_k_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q4_k_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q4_K mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q4_K known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q4_K);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q4KMVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 8u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q4_k_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q4_K mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q4_K output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q4_K sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_q5_k_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q5_K> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q5_K n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::uint32_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row =
        static_cast<std::size_t>(8u) * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q5_K> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_code(0, 31);
        std::uniform_int_distribution<std::uint32_t> random_scale(0, 63);
        constexpr std::array<float, 5> block_scales = {
            0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        constexpr std::array<std::uint32_t, 16> edge_scales = {
            0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 47, 48, 61, 62, 63, 8};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_q5_K& weight = weights[block];
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            std::fill_n(weight.qh, 32, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
            const float d = block_scales[block % block_scales.size()];
            const float dmin = block_scales[(block + 2u) % block_scales.size()];
            weight.dm =
                static_cast<std::uint32_t>(strata::kernels::f16_from_f32(d)) |
                (static_cast<std::uint32_t>(
                     strata::kernels::f16_from_f32(dmin)) << 16u);
            for (std::uint32_t group = 0; group < 8u; ++group) {
                std::uint32_t scale = 0;
                std::uint32_t min = 0;
                if (set == twin::InputSet::random) {
                    scale = random_scale(rng);
                    min = random_scale(rng);
                } else if (set == twin::InputSet::edge) {
                    scale = edge_scales[(group + block) % edge_scales.size()];
                    min = edge_scales[(group * 3u + block + 5u) %
                                      edge_scales.size()];
                } else {
                    scale = (group * 17u + static_cast<std::uint32_t>(block) * 7u +
                             3u) % 64u;
                    min = (group * 29u + static_cast<std::uint32_t>(block) * 11u +
                           1u) % 64u;
                }
                set_q5_k_scale_min(weight, group, scale, min);
            }
            for (std::uint32_t index = 0; index < 256u; ++index) {
                std::uint32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = random_code(rng);
                } else if (set == twin::InputSet::edge) {
                    code = (index * 5u + static_cast<std::uint32_t>(block) * 3u +
                            1u) % 32u;
                } else {
                    code = (index * 7u + static_cast<std::uint32_t>(block) * 11u +
                            index / 32u * 3u + 2u) % 32u;
                }
                set_q5_k_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q5_k_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q5_K mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q5_K known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q5_K);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q4KMVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 8u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q5_k_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q5_K mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q5_K output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q5_K sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_q3_k_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q3_K> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q3_K n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q3_K> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::int32_t> random_code(-4, 3);
        std::uniform_int_distribution<std::int32_t> random_scale(-32, 31);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        constexpr std::array<std::int32_t, 16> edge_scales = {
            -32, -31, -24, -16, -8, -1, 0, 1,
            2, 7, 15, 16, 24, 30, 31, -30};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_q3_K& weight = weights[block];
            std::fill_n(weight.hmask, 32, std::uint8_t{0});
            std::fill_n(weight.qs, 64, std::uint8_t{0});
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            weight.d = strata::kernels::f16_from_f32(scales[block % scales.size()]);
            for (std::uint32_t index = 0; index < 16; ++index) {
                std::int32_t scale = 0;
                if (set == twin::InputSet::random) {
                    scale = random_scale(rng);
                } else if (set == twin::InputSet::edge) {
                    scale = edge_scales[(index + block) % edge_scales.size()];
                } else {
                    scale = static_cast<std::int32_t>(
                        (index * 7u + static_cast<std::uint32_t>(block) * 3u) % 33u) - 16;
                }
                set_q3_k_scale(weight, index, scale);
            }
            for (std::uint32_t index = 0; index < 256; ++index) {
                std::int32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = random_code(rng);
                } else if (set == twin::InputSet::edge) {
                    code = static_cast<std::int32_t>(
                        (index * 5u + static_cast<std::uint32_t>(block) * 3u + 1u) % 8u) - 4;
                } else {
                    code = static_cast<std::int32_t>(
                        (index * 3u + static_cast<std::uint32_t>(block) * 5u + 2u) % 8u) - 4;
                }
                set_q3_k_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q3_k_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q3_K mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q3_K known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q3_K);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q3MVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 8u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q3_k_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q3_K mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q3_K output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q3_K sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool test_q6_k_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t nc, std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const block_q6_K> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "Q6_K n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    if (set == twin::InputSet::edge && fixed_inputs.empty()) {
        install_q8_1_edge_blocks(inputs, n_in, nc);
    }

    const std::size_t blocks_per_row = n_in / 256u;
    const std::size_t activation_blocks_per_row = 8u * blocks_per_row;
    const std::size_t weight_block_count =
        static_cast<std::size_t>(n_out) * blocks_per_row;
    std::vector<block_q6_K> weights;
    if (fixed_weights.empty()) {
        weights.resize(weight_block_count);
        std::uniform_int_distribution<std::uint32_t> random_code(0, 63);
        std::uniform_int_distribution<std::int32_t> random_scale(-128, 127);
        constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
        constexpr std::array<std::int32_t, 16> edge_scales = {
            -128, -64, -32, -16, -1, 0, 1, 2,
            7, 15, 31, 63, 64, 96, 126, 127};
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_q6_K& weight = weights[block];
            std::fill_n(weight.ql, 128, std::uint8_t{0});
            std::fill_n(weight.qh, 64, std::uint8_t{0});
            std::fill_n(weight.scales, 16, std::int8_t{0});
            weight.d = strata::kernels::f16_from_f32(scales[block % scales.size()]);
            for (std::uint32_t index = 0; index < 16; ++index) {
                std::int32_t scale = 0;
                if (set == twin::InputSet::random) {
                    scale = random_scale(rng);
                } else if (set == twin::InputSet::edge) {
                    scale = edge_scales[(index + block) % edge_scales.size()];
                } else {
                    scale = static_cast<std::int32_t>(
                        (index * 7u + static_cast<std::uint32_t>(block) * 3u) % 33u) - 16;
                }
                set_q6_k_scale(weight, index, scale);
            }
            for (std::uint32_t index = 0; index < 256; ++index) {
                std::uint32_t code = 0;
                if (set == twin::InputSet::random) {
                    code = random_code(rng);
                } else if (set == twin::InputSet::edge) {
                    code = (index * 13u + static_cast<std::uint32_t>(block) * 17u) % 64u;
                } else {
                    code = (index * 37u + static_cast<std::uint32_t>(block) * 11u + 3u) % 64u;
                }
                set_q6_k_code(weight, index, code);
            }
        }
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(inputs, n_in, nc, activations, err),
          shape + " CPU Q8_1 activation oracle", err);
    if (activations.size() != activation_blocks_per_row * nc) return false;

    std::vector<float> expected;
    check(twin::q6_k_matvec_reference(weights, activations, n_in, n_out,
                                      nc, live_rows, expected, err),
          shape + " CPU Q6_K mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic Q6_K known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(block_q6_K);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " activations", input_bytes, err),
          shape + " activation buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), activations.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const Q6MVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    const std::uint32_t rows_per_group = blocks_per_row < 4u ? 4u : 1u;
    work.grid = {(n_out + rows_per_group - 1u) / rows_per_group, 1, 1};
    work.threads_per_group = {32, 4, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_q6_k_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU Q6_K mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact Q6_K output bytes");
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact Q6_K sink bytes");
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " activation and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool run_q8_0_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 5> widths = {640, 992, 1024, 2560, 6144};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<std::int8_t, 8> weight_pattern = {
        -3, -2, -1, 1, 2, 3, 4, -4};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 3> scales = {0.25f, 0.5f, 1.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 32u;
        std::vector<block_q8_0> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                for (std::uint32_t k = 0; k < 32; ++k) {
                    weight.qs[k] =
                        weight_pattern[(row + block + k) % weight_pattern.size()];
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);
        weights[0].qs[0] = 1;

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_q8_0_shape(context, n_in, n_out, nc, nc,
                                         twin::InputSet::fixture, rng, options,
                                         inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q8_0 multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin = inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q8_0_shape(context, n_in, n_out, 1, 1,
                                             twin::InputSet::fixture, rng, options,
                                             single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q8_0 single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(multi.data() +
                                                          static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q8_0 mv_multi_eq_single", options, column, *mismatch);
                }
                check(!mismatch, "Q8_0 NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 640 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].qs[0] ^= 1;
                std::vector<float> flipped;
                completed &= test_q8_0_shape(context, n_in, n_out, nc, nc,
                                             twin::InputSet::fixture, rng, options,
                                             inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "Q8_0 mv_multi_eq_single negative control detects code-bit flip");
                } else {
                    check(false, "Q8_0 negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_q4_0_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 5> widths = {640, 2016, 2048, 2560, 6144};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 32u;
        std::vector<block_q4_0> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                for (std::uint32_t k = 0; k < 16; ++k) {
                    const std::uint8_t low =
                        static_cast<std::uint8_t>((row + block + k) % 16u);
                    const std::uint8_t high =
                        static_cast<std::uint8_t>((15u - row + block + k) % 16u);
                    weight.qs[k] = static_cast<std::uint8_t>(low | (high << 4u));
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);
        weights[0].qs[0] = static_cast<std::uint8_t>((weights[0].qs[0] & 0xf0u) | 8u);

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_q4_0_shape(context, n_in, n_out, nc, nc,
                                         twin::InputSet::fixture, rng, options,
                                         inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q4_0 multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q4_0_shape(context, n_in, n_out, 1, 1,
                                             twin::InputSet::fixture, rng, options,
                                             single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q4_0 single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q4_0 mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "Q4_0 NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 640 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].qs[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_q4_0_shape(context, n_in, n_out, nc, nc,
                                             twin::InputSet::fixture, rng, options,
                                             inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "Q4_0 mv_multi_eq_single negative control detects code-bit flip");
                } else {
                    check(false, "Q4_0 negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_q5_0_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 5> widths = {640, 2016, 2048, 2560, 6144};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 32u;
        std::vector<block_q5_0> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                std::fill_n(weight.qh, 4, std::uint8_t{0});
                std::fill_n(weight.qs, 16, std::uint8_t{0});
                for (std::uint32_t index = 0; index < 32; ++index) {
                    const std::uint32_t code =
                        (row * 5u + block * 3u + index * 7u) % 32u;
                    set_q5_0_code(weight, index, code);
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_q5_0_shape(context, n_in, n_out, nc, nc,
                                         twin::InputSet::fixture, rng, options,
                                         inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q5_0 multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q5_0_shape(context, n_in, n_out, 1, 1,
                                             twin::InputSet::fixture, rng, options,
                                             single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q5_0 single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q5_0 mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "Q5_0 NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 640 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].qh[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_q5_0_shape(context, n_in, n_out, nc, nc,
                                             twin::InputSet::fixture, rng, options,
                                             inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "Q5_0 mv_multi_eq_single negative control detects high-bit flip");
                } else {
                    check(false, "Q5_0 negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_q5_0_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                 const twin::Options& options) {
    constexpr std::uint32_t n_in = 640;
    constexpr std::uint32_t blocks_per_row = n_in / 32u;
    constexpr std::array<std::uint8_t, 4> qh_bytes = {
        0xa5, 0x3c, 0x5a, 0xc3};
    constexpr std::array<std::uint8_t, 16> qs_bytes = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    constexpr std::array<float, 32> known_answers = {
        0.0f, -1905.0f, 254.0f, -1651.0f, -1524.0f, 635.0f, -1270.0f, 889.0f,
        -1016.0f, -889.0f, 1270.0f, 1397.0f, 1524.0f, 1651.0f, -254.0f, -127.0f,
        -2032.0f, 127.0f, -1778.0f, 381.0f, 508.0f, -1397.0f, 762.0f, -1143.0f,
        1016.0f, 1143.0f, -762.0f, -635.0f, -508.0f, -381.0f, 1778.0f, 1905.0f};
    std::vector<block_q5_0> weights(blocks_per_row);
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
        auto& weight = weights[block];
        weight.d = strata::kernels::f16_from_f32(block == 0u ? 1.0f : 0.0f);
        std::fill_n(weight.qh, 4, std::uint8_t{0});
        std::fill_n(weight.qs, 16, std::uint8_t{0});
    }
    std::copy(qh_bytes.begin(), qh_bytes.end(), weights[0].qh);
    std::copy(qs_bytes.begin(), qs_bytes.end(), weights[0].qs);

    std::vector<float> inputs(n_in, 0.0f);
    bool completed = true;
    for (std::uint32_t index = 0; index < 32; ++index) {
        std::fill(inputs.begin(), inputs.end(), 0.0f);
        inputs[index] = 127.0f;
        const std::span<const float> answer(&known_answers[index], 1);
        completed &= test_q5_0_shape(context, n_in, 1, 1, 1,
                                     twin::InputSet::fixture, rng, options,
                                     inputs, weights, answer);
    }
    return completed;
}

bool run_q5_0_fma_golden(Context& context, std::mt19937_64& rng,
                         const twin::Options& options) {
    constexpr std::uint32_t n_in = 2080;
    constexpr std::uint32_t blocks_per_row = n_in / 32u;
    std::vector<float> inputs(n_in, 0.0f);
    std::fill_n(inputs.begin(), 32, 0.1f);
    std::fill_n(inputs.begin() + 64u * 32u, 32, 0.1f);
    std::vector<block_q5_0> weights(blocks_per_row);
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
        auto& weight = weights[block];
        weight.d = strata::kernels::f16_from_f32(0.0f);
        std::fill_n(weight.qh, 4, std::uint8_t{0});
        std::fill_n(weight.qs, 16, std::uint8_t{0});
    }
    weights[0].d = strata::kernels::f16_from_f32(0.125f);
    weights[64].d = strata::kernels::f16_from_f32(0.1f);
    std::fill_n(weights[64].qs, 16, std::uint8_t{0x11});
    constexpr std::array<float, 1> known_answer = {
        -11.19607448577880859375f};
    return test_q5_0_shape(context, n_in, 1, 1, 1,
                            twin::InputSet::fixture, rng, options,
                            inputs, weights, known_answer);
}

bool run_q2_0_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                 const twin::Options& options) {
    constexpr std::array<std::array<std::uint8_t, 16>, 3> raw_qs = {{
        {{0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4,
          0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4}},
        {{0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff,
          0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff}},
        {{0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55,
          0xaa, 0xaa, 0xaa, 0xaa, 0xff, 0xff, 0xff, 0xff}},
    }};
    constexpr auto expected_codes = [] {
        std::array<std::array<std::uint8_t, 64>, 3> codes{};
        for (std::uint32_t index = 0; index < 64; ++index) {
            codes[0][index] = static_cast<std::uint8_t>(index & 3u);
            codes[1][index] = static_cast<std::uint8_t>((index >> 2u) & 3u);
            codes[2][index] = static_cast<std::uint8_t>((index >> 4u) & 3u);
        }
        return codes;
    }();

    std::vector<block_q2_0> weights(1);
    weights[0].d = strata::kernels::f16_from_f32(1.0f);
    std::vector<float> inputs(64, 0.0f);
    bool completed = true;
    for (std::size_t pattern = 0; pattern < raw_qs.size(); ++pattern) {
        std::copy(raw_qs[pattern].begin(), raw_qs[pattern].end(),
                  weights[0].qs);
        for (std::uint32_t index = 0; index < 64; ++index) {
            std::fill(inputs.begin(), inputs.end(), 0.0f);
            inputs[index] = 127.0f;
            const float answer =
                (static_cast<int>(expected_codes[pattern][index]) - 1) * 127.0f;
            completed &= test_q2_0_shape(
                context, 64, 1, 1, 1, twin::InputSet::fixture, rng, options,
                inputs, weights, std::span<const float>(&answer, 1));
        }
    }
    return completed;
}

bool run_q2_0_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 8> widths = {
        64, 640, 2560, 4032, 4096, 4160, 6144, 10240};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 64u;
        std::vector<block_q2_0> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                std::fill_n(weight.qs, 16, std::uint8_t{0});
                for (std::uint32_t index = 0; index < 64; ++index) {
                    set_q2_0_code(
                        weight, index,
                        (row + block * 2u + index * 3u + 1u) % 4u);
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_q2_0_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q2_0 multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q2_0_shape(
                    context, n_in, n_out, 1, 1, twin::InputSet::fixture, rng,
                    options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q2_0 single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q2_0 mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "Q2_0 NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 640 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].qs[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_q2_0_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "Q2_0 mv_multi_eq_single negative control detects code-bit flip");
                } else {
                    check(false, "Q2_0 negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_iq4_nl_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                   const twin::Options& options) {
    bool completed = true;
    for (std::uint32_t pattern = 0; pattern < 2; ++pattern) {
        block_iq4_nl weight{};
        weight.d = strata::kernels::f16_from_f32(1.0f);
        for (std::uint32_t byte = 0; byte < 16; ++byte) {
            weight.qs[byte] = pattern == 0
                ? static_cast<std::uint8_t>((byte << 4u) | byte)
                : std::uint8_t{0x10u};
        }
        const std::array<block_iq4_nl, 1> weights = {weight};
        for (std::uint32_t index = 0; index < 32; ++index) {
            std::array<float, 32> inputs{};
            inputs[index] = 127.0f;
            const std::uint32_t code = pattern == 0
                ? index & 15u
                : (index < 16u ? 0u : 1u);
            const float answer =
                static_cast<float>(k_iq4_nl_golden_values[code] * 127);
            completed &= test_iq4_nl_shape(
                context, 32, 1, 1, 1, twin::InputSet::fixture, rng, options,
                inputs, weights, std::span<const float>(&answer, 1));
        }
    }
    return completed;
}

bool run_iq4_nl_fma_golden(Context& context, std::mt19937_64& rng,
                           const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> widths = {2080, 4128};
    constexpr std::array<float, 1> known_answer = {
        -40.222751617431640625f};
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 32u;
        const std::uint32_t final_block = blocks_per_row - 1u;
        std::vector<float> inputs(n_in, 0.0f);
        std::fill_n(inputs.begin(), 32, 0.1f);
        std::fill_n(inputs.begin() + static_cast<std::size_t>(final_block) * 32u,
                    32, 0.1f);
        std::vector<block_iq4_nl> weights(blocks_per_row);
        for (auto& weight : weights) {
            weight.d = strata::kernels::f16_from_f32(0.0f);
            std::fill_n(weight.qs, 16, std::uint8_t{0});
        }
        weights[0].d = strata::kernels::f16_from_f32(0.125f);
        weights[final_block].d = strata::kernels::f16_from_f32(0.1f);
        for (std::uint32_t index = 0; index < 32; ++index) {
            set_iq4_nl_code(weights[0], index, 8u);
            set_iq4_nl_code(weights[final_block], index, 0u);
        }
        completed &= test_iq4_nl_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq4_nl_multi_eq_single(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::array<std::uint32_t, 7> widths = {
        32, 640, 2016, 2048, 2080, 2560, 6144};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 32u;
        std::vector<block_iq4_nl> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                std::fill_n(weight.qs, 16, std::uint8_t{0});
                for (std::uint32_t index = 0; index < 32; ++index) {
                    set_iq4_nl_code(
                        weight, index,
                        (row + block * 3u + index * 5u + 1u) % 16u);
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_iq4_nl_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ4_NL multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_iq4_nl_shape(
                    context, n_in, n_out, 1, 1, twin::InputSet::fixture, rng,
                    options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ4_NL single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ4_NL mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "IQ4_NL NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 640 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].qs[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_iq4_nl_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "IQ4_NL multi-column negative control detects code-bit flip");
                } else {
                    check(false, "IQ4_NL negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_iq4_xs_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                   const twin::Options& options) {
    constexpr std::array<std::array<std::uint32_t, 8>, 2> scale_codes = {{
        {{1, 2, 4, 8, 16, 32, 48, 63}},
        {{0, 17, 34, 51, 12, 29, 46, 63}},
    }};
    constexpr std::array<std::array<std::uint8_t, 4>, 2> raw_scales_l = {{
        {{0x21, 0x84, 0x00, 0xf0}},
        {{0x10, 0x32, 0xdc, 0xfe}},
    }};
    constexpr std::array<std::uint16_t, 2> raw_scales_h = {0xf900, 0xe4e4};
    bool completed = true;
    for (std::uint32_t pattern = 0; pattern < 2; ++pattern) {
        block_iq4_xs weight{};
        weight.d = strata::kernels::f16_from_f32(1.0f);
        std::copy(raw_scales_l[pattern].begin(), raw_scales_l[pattern].end(),
                  weight.scales_l);
        weight.scales_h = raw_scales_h[pattern];
        for (std::uint32_t subblock = 0; subblock < 8; ++subblock) {
            for (std::uint32_t byte = 0; byte < 16; ++byte) {
                const std::uint32_t low = pattern == 0 ? byte : 15u - byte;
                const std::uint32_t high = byte;
                weight.qs[16u * subblock + byte] =
                    static_cast<std::uint8_t>((high << 4u) | low);
            }
        }
        const std::array<block_iq4_xs, 1> weights = {weight};
        for (std::uint32_t batch = 0; batch < 32; ++batch) {
            std::array<float, 256 * 8> inputs{};
            std::array<float, 8> known_answer{};
            for (std::uint32_t column = 0; column < 8; ++column) {
                const std::uint32_t index = batch * 8u + column;
                inputs[static_cast<std::size_t>(column) * 256u + index] = 127.0f;
                const std::uint32_t local = index % 32u;
                const std::uint32_t nibble = local < 16u
                    ? (pattern == 0 ? local : 15u - local)
                    : (pattern == 0 ? local - 16u : local - 16u);
                const std::uint32_t subblock = index / 32u;
                known_answer[column] = static_cast<float>(
                    k_iq4_xs_golden_values[nibble] * 127 *
                    (static_cast<int>(scale_codes[pattern][subblock]) - 32));
            }
            completed &= test_iq4_xs_shape(
                context, 256, 1, 8, 8, twin::InputSet::fixture, rng, options,
                inputs, weights, known_answer);
        }
    }
    return completed;
}

bool run_iq4_xs_fixture(Context& context, std::mt19937_64& rng,
                        const twin::Options& options) {
    constexpr std::uint32_t n_in = 256;
    constexpr std::uint32_t n_out = 3;
    constexpr std::uint32_t nc = 4;
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 1.5f};
    constexpr std::array<std::uint32_t, 8> scale_codes = {
        0, 7, 15, 16, 23, 32, 47, 63};
    std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
    std::vector<block_iq4_xs> weights(n_out);
    for (std::uint32_t row = 0; row < n_out; ++row) {
        block_iq4_xs& weight = weights[row];
        weight.d = strata::kernels::f16_from_f32(weight_scales[row]);
        weight.scales_h = 0;
        std::fill_n(weight.scales_l, 4, std::uint8_t{0});
        std::fill_n(weight.qs, 128, std::uint8_t{0});
        for (std::uint32_t subblock = 0; subblock < 8; ++subblock) {
            set_iq4_xs_scale(
                weight, subblock,
                (scale_codes[subblock] + row * 3u) % 64u);
            set_iq4_xs_code(weight, 32u * subblock, (row + subblock * 3u) % 16u);
        }
    }
    for (std::uint32_t column = 0; column < nc; ++column) {
        for (std::uint32_t subblock = 0; subblock < 8; ++subblock) {
            inputs[static_cast<std::size_t>(column) * n_in + subblock * 32u] =
                127.0f * static_cast<float>(column + 1u);
        }
    }

    std::array<float, n_out * nc> known_answer{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        const std::uint32_t input_column = column < 3u ? column : 0u;
        for (std::uint32_t row = 0; row < n_out; ++row) {
            float sum = 0.0f;
            for (std::uint32_t subblock = 0; subblock < 8; ++subblock) {
                const std::uint32_t code = (row + subblock * 3u) % 16u;
                const std::int32_t scale =
                    static_cast<std::int32_t>((scale_codes[subblock] + row * 3u) % 64u) - 32;
                sum += weight_scales[row] *
                       static_cast<float>(input_column + 1u) *
                       static_cast<float>(k_iq4_xs_golden_values[code] * 127 * scale);
            }
            known_answer[column * n_out + row] = sum;
        }
    }
    return test_iq4_xs_shape(
        context, n_in, n_out, nc, 3, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq4_xs_moe_onehot_fixture(Context& context, std::mt19937_64& rng,
                                   const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::array<float, 1> known_answer = {
        twin::kIQ4XSMoeOneHotAnswer};
    std::array<float, n_in> inputs{};
    inputs[twin::kIQ4XSMoeOneHotPosition] = 127.0f;
    const std::array<block_iq4_xs, 1> weights = {
        twin::iq4_xs_moe_fixture()};
    return test_iq4_xs_shape(
        context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq4_xs_fma_golden(Context& context, std::mt19937_64& rng,
                           const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> widths = {16640, 33024};
    constexpr std::array<float, 1> known_answer = {
        0.64050829410552978515625f};
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        const std::uint32_t final_block = blocks_per_row - 1u;
        std::vector<float> inputs(n_in, 0.0f);
        std::fill_n(inputs.begin(), 256, 0.1f);
        std::fill_n(
            inputs.begin() + static_cast<std::size_t>(final_block) * 256u,
            256, 0.1f);
        std::vector<block_iq4_xs> weights(blocks_per_row);
        for (block_iq4_xs& weight : weights) {
            weight.d = strata::kernels::f16_from_f32(0.0f);
            weight.scales_h = 0;
            std::fill_n(weight.scales_l, 4, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
        }
        for (const std::uint32_t block : {0u, final_block}) {
            block_iq4_xs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(block == 0u ? 0.125f : 0.1f);
            for (std::uint32_t subblock = 0; subblock < 8; ++subblock) {
                set_iq4_xs_scale(weight, subblock, block == 0u ? 33u : 31u);
            }
            for (std::uint32_t index = 0; index < 256; ++index) {
                set_iq4_xs_code(weight, index, 8u);
            }
        }
        completed &= test_iq4_xs_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq4_xs_multi_eq_single(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::array<std::uint32_t, 9> widths = {
        256, 2560, 3840, 4096, 6144, 10240, 16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<block_iq4_xs> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                block_iq4_xs& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                weight.scales_h = 0;
                std::fill_n(weight.scales_l, 4, std::uint8_t{0});
                std::fill_n(weight.qs, 128, std::uint8_t{0});
                for (std::uint32_t subblock = 0; subblock < 8; ++subblock) {
                    set_iq4_xs_scale(
                        weight, subblock,
                        (row + block * 3u + subblock * 11u + 1u) % 64u);
                }
                for (std::uint32_t index = 0; index < 256; ++index) {
                    set_iq4_xs_code(
                        weight, index,
                        (row + block * 3u + index * 5u + 1u) % 16u);
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_iq4_xs_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ4_XS multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_iq4_xs_shape(
                    context, n_in, n_out, 1, 1, twin::InputSet::fixture, rng,
                    options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ4_XS single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ4_XS mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "IQ4_XS NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 2560 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].qs[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_iq4_xs_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "IQ4_XS multi-column negative control detects code-bit flip");
                } else {
                    check(false, "IQ4_XS negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_q6_k_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                 const twin::Options& options) {
    constexpr std::array<std::int8_t, 16> raw_scales = {
        -128, -64, -32, -16, -1, 0, 1, 2,
        7, 15, 31, 63, 64, 96, 126, 127};
    bool completed = true;
    for (std::uint32_t pattern = 0; pattern < 4; ++pattern) {
        block_q6_K weight{};
        weight.d = strata::kernels::f16_from_f32(1.0f);
        std::copy(raw_scales.begin(), raw_scales.end(), weight.scales);
        for (std::uint32_t index = 0; index < 256; ++index) {
            set_q6_k_code(weight, index, (index + pattern * 16u) % 64u);
        }
        const std::array<block_q6_K, 1> weights = {weight};
        for (std::uint32_t batch = 0; batch < 32; ++batch) {
            std::array<float, 256 * 8> inputs{};
            std::array<float, 8> known_answer{};
            for (std::uint32_t column = 0; column < 8; ++column) {
                const std::uint32_t index = batch * 8u + column;
                inputs[static_cast<std::size_t>(column) * 256u + index] = 127.0f;
                const std::uint32_t code = (index + pattern * 16u) % 64u;
                const std::int32_t q = static_cast<std::int32_t>(code) - 32;
                const std::int32_t scale = raw_scales[index / 16u];
                known_answer[column] = static_cast<float>(q * scale * 127);
            }
            completed &= test_q6_k_shape(
                context, 256, 1, 8, 8, twin::InputSet::fixture, rng, options,
                inputs, weights, known_answer);
        }
    }
    return completed;
}

bool run_q6_k_raw_layout_golden(Context& context, std::mt19937_64& rng,
                                const twin::Options& options) {
    constexpr std::array<std::int8_t, 16> raw_scales = {
        -128, -64, -32, -16, -1, 0, 1, 2,
        7, 15, 31, 63, 64, 96, 126, 127};
    constexpr std::array<std::uint8_t, 128> raw_ql = {
        0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4, 0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6,
        0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4, 0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6,
        0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4, 0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6,
        0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4, 0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6,
        0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6, 0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4,
        0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6, 0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4,
        0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6, 0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4,
        0x81, 0x5e, 0x2b, 0xf8, 0xc5, 0x92, 0x6f, 0x3c, 0x09, 0xd6, 0xa3, 0x70, 0x4d, 0x1a, 0xe7, 0xb4};
    constexpr std::array<std::uint8_t, 64> raw_qh = {
        0x88, 0xdd, 0x2d, 0x72, 0x77, 0x88, 0xdd, 0x2d, 0x72, 0x77, 0x88, 0xdd, 0x2d, 0x72, 0x87, 0x88,
        0xdd, 0x22, 0x72, 0x87, 0x88, 0xdd, 0x22, 0x72, 0x87, 0x88, 0xdd, 0x22, 0x72, 0x87, 0xd8, 0xdd,
        0xdd, 0x2d, 0x72, 0x77, 0x88, 0xdd, 0x2d, 0x72, 0x87, 0x88, 0xdd, 0x22, 0x72, 0x87, 0x88, 0xdd,
        0x22, 0x72, 0x87, 0x88, 0xdd, 0x22, 0x72, 0x87, 0xd8, 0xdd, 0x22, 0x77, 0x87, 0xd8, 0xdd, 0x22};
    block_q6_K weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    std::copy(raw_scales.begin(), raw_scales.end(), weight.scales);
    std::copy(raw_ql.begin(), raw_ql.end(), weight.ql);
    std::copy(raw_qh.begin(), raw_qh.end(), weight.qh);

    const std::array<block_q6_K, 1> weights = {weight};
    bool completed = true;
    for (std::uint32_t batch = 0; batch < 256; batch += 8) {
        std::array<float, 256 * 8> inputs{};
        std::array<float, 8> known_answer{};
        for (std::size_t column = 0; column < 8; ++column) {
            const std::uint32_t position = batch + static_cast<std::uint32_t>(column);
            inputs[column * 256u + position] = 127.0f;
            const std::uint32_t code =
                (position * 13u + (position / 64u) * 7u + 3u) % 64u;
            const std::int32_t q = static_cast<std::int32_t>(code) - 32;
            const std::int32_t scale = raw_scales[position / 16u];
            known_answer[column] = static_cast<float>(q * scale * 127);
        }
        completed &= test_q6_k_shape(
            context, 256, 1, 8, 8, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q6_k_fixture(Context& context, std::mt19937_64& rng,
                      const twin::Options& options) {
    constexpr std::uint32_t n_in = 256;
    constexpr std::uint32_t n_out = 3;
    constexpr std::uint32_t nc = 4;
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 1.5f};
    std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
    std::vector<block_q6_K> weights(n_out);
    for (std::uint32_t row = 0; row < n_out; ++row) {
        block_q6_K& weight = weights[row];
        std::fill_n(weight.ql, 128, std::uint8_t{0});
        std::fill_n(weight.qh, 64, std::uint8_t{0});
        std::fill_n(weight.scales, 16, std::int8_t{0});
        weight.d = strata::kernels::f16_from_f32(weight_scales[row]);
        for (std::uint32_t group = 0; group < 16; ++group) {
            const std::int32_t scale =
                static_cast<std::int32_t>((row * 3u + group * 5u) % 17u) - 8;
            set_q6_k_scale(weight, group, scale);
            const std::uint32_t code = (row * 13u + group * 7u + 3u) % 64u;
            set_q6_k_code(weight, group * 16u, code);
        }
    }
    for (std::uint32_t column = 0; column < nc; ++column) {
        for (std::uint32_t group = 0; group < 16; ++group) {
            const std::uint32_t index = group * 16u;
            const float q8_code = group % 2u == 0u ? 127.0f : 63.0f;
            inputs[static_cast<std::size_t>(column) * n_in + index] =
                q8_code * static_cast<float>(column + 1u);
        }
    }

    std::array<float, n_out * nc> known_answer{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        const std::uint32_t input_column = column < 3u ? column : 0u;
        for (std::uint32_t row = 0; row < n_out; ++row) {
            float sum = 0.0f;
            for (std::uint32_t group = 0; group < 16; ++group) {
                const std::uint32_t code = (row * 13u + group * 7u + 3u) % 64u;
                const std::int32_t q = static_cast<std::int32_t>(code) - 32;
                const std::int32_t scale =
                    static_cast<std::int32_t>((row * 3u + group * 5u) % 17u) - 8;
                const std::int32_t activation_code =
                    group % 2u == 0u ? 127 : 63;
                sum += weight_scales[row] *
                       static_cast<float>(input_column + 1u) *
                       static_cast<float>(q * scale * activation_code);
            }
            known_answer[column * n_out + row] = sum;
        }
    }
    return test_q6_k_shape(
        context, n_in, n_out, nc, 3, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_q6_k_fma_golden(Context& context, std::mt19937_64& rng,
                         const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> widths = {1280, 2304};
    constexpr std::array<float, 1> known_answer = {
        -266.151519775390625f};
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        const std::uint32_t final_block = blocks_per_row - 1u;
        std::vector<float> inputs(n_in, 0.0f);
        std::fill_n(inputs.begin(), 256, 0.1f);
        std::fill_n(
            inputs.begin() + static_cast<std::size_t>(final_block) * 256u,
            256, 0.1f);
        std::vector<block_q6_K> weights(blocks_per_row);
        for (block_q6_K& weight : weights) {
            std::fill_n(weight.ql, 128, std::uint8_t{0});
            std::fill_n(weight.qh, 64, std::uint8_t{0});
            std::fill_n(weight.scales, 16, std::int8_t{0});
            weight.d = strata::kernels::f16_from_f32(0.0f);
        }
        for (const std::uint32_t block : {0u, final_block}) {
            block_q6_K& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(block == 0u ? 0.125f : 0.1f);
            std::fill_n(weight.scales, 16,
                        static_cast<std::int8_t>(block == 0u ? 1 : 2));
        }
        completed &= test_q6_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q6_k_long_k_golden(Context& context, std::mt19937_64& rng,
                            const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> widths = {16128, 16384, 16640};
    constexpr std::array<float, 3> known_answers = {
        -32004.0f, -30988.0f, -30480.0f};
    bool completed = true;
    for (std::size_t shape = 0; shape < widths.size(); ++shape) {
        const std::uint32_t n_in = widths[shape];
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<float> inputs(n_in, 0.0f);
        std::vector<block_q6_K> weights(blocks_per_row);
        for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
            block_q6_K& weight = weights[block];
            std::fill_n(weight.ql, 128, std::uint8_t{0});
            std::fill_n(weight.qh, 64, std::uint8_t{0});
            std::fill_n(weight.scales, 16, std::int8_t{0});
            weight.d = strata::kernels::f16_from_f32(0.125f);
            weight.scales[0] = static_cast<std::int8_t>(
                static_cast<std::int32_t>(block % 7u) - 2);
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        completed &= test_q6_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights,
            std::span<const float>(&known_answers[shape], 1));
    }
    return completed;
}

bool run_q6_k_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 11> widths = {
        256, 512, 768, 1024, 1280, 2560, 6144, 10240, 16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<block_q6_K> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                block_q6_K& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                std::fill_n(weight.ql, 128, std::uint8_t{0});
                std::fill_n(weight.qh, 64, std::uint8_t{0});
                std::fill_n(weight.scales, 16, std::int8_t{0});
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                for (std::uint32_t index = 0; index < 16; ++index) {
                    const std::int32_t scale =
                        static_cast<std::int32_t>(
                            (row * 3u + block * 7u + index * 5u + 1u) % 31u) - 15;
                    set_q6_k_scale(weight, index, scale);
                }
                for (std::uint32_t index = 0; index < 256; ++index) {
                    set_q6_k_code(
                        weight, index,
                        (row + block * 5u + index * 7u + 3u) % 64u);
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_q6_k_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q6_K multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q6_k_shape(
                    context, n_in, n_out, 1, 1, twin::InputSet::fixture, rng,
                    options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q6_K single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q6_K mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "Q6_K NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 2560 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].ql[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_q6_k_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "Q6_K multi-column negative control detects code-bit flip");
                } else {
                    check(false, "Q6_K negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool test_bf16_shape(Context& context, std::uint32_t n_in, std::uint32_t n_out,
                     std::uint32_t block_size, std::uint32_t nc,
                     std::uint32_t live_rows,
                     twin::InputSet set, std::mt19937_64& rng,
                     const twin::Options& options,
                     std::span<const float> fixed_inputs = {},
                     std::span<const std::uint16_t> fixed_weights = {},
                     std::span<const float> known_answer = {},
                     std::vector<float>* actual_output = nullptr) {
    const std::string shape = "BF16 n_in=" + std::to_string(n_in) +
                              " n_out=" + std::to_string(n_out) +
                              " bs=" + std::to_string(block_size) +
                              " nc=" + std::to_string(nc) +
                              " live_rows=" + std::to_string(live_rows) +
                              " set=" + twin::input_set_name(set);
    std::string err;
    const std::size_t input_count = static_cast<std::size_t>(n_in) * nc;
    const std::size_t weight_count = static_cast<std::size_t>(n_in) * n_out;
    std::vector<float> inputs;
    if (fixed_inputs.empty()) {
        inputs = twin::synthesize_floats(set, input_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " input synthesis", err);
            return false;
        }
    } else {
        inputs.assign(fixed_inputs.begin(), fixed_inputs.end());
    }
    std::vector<std::uint16_t> weights;
    if (fixed_weights.empty()) {
        const std::vector<float> weight_values =
            twin::synthesize_floats(set, weight_count, rng, err);
        if (!err.empty()) {
            check(false, shape + " weight synthesis", err);
            return false;
        }
        weights.resize(weight_count);
        std::transform(weight_values.begin(), weight_values.end(), weights.begin(),
                       strata::kernels::bf16_from_f32);
    } else {
        weights.assign(fixed_weights.begin(), fixed_weights.end());
    }

    std::vector<float> expected;
    check(twin::bf16_matvec_reference(weights, inputs, n_in, n_out, nc,
                                      live_rows, block_size, expected, err),
          shape + " CPU BF16 mat-vec oracle", err);
    if (expected.size() != static_cast<std::size_t>(n_out) * nc) return false;
    if (!known_answer.empty()) {
        const auto expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data()),
            expected.size() * sizeof(float));
        const auto known_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(known_answer.data()),
            known_answer.size() * sizeof(float));
        const auto known_failure = twin::compare_bytes(expected_bytes, known_bytes);
        if (known_failure) twin::report_failure(shape.c_str(), options, live_rows, *known_failure);
        check(!known_failure, shape + " analytic BF16 known answer");
    }

    const std::size_t output_count = static_cast<std::size_t>(n_out) * live_rows;
    const std::size_t sink_count = static_cast<std::size_t>(n_out) * (nc - live_rows);
    const std::size_t input_bytes = inputs.size() * sizeof(float);
    const std::size_t weight_bytes = weights.size() * sizeof(std::uint16_t);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_bytes = std::max<std::size_t>(sizeof(float), sink_count * sizeof(float));
    const auto expected_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    std::vector<std::uint8_t> expected_sink_storage(sink_bytes, 0u);
    if (sink_count != 0) {
        std::memcpy(expected_sink_storage.data(),
                    expected.data() + output_count, sink_count * sizeof(float));
    }
    const auto expected_sink_bytes = std::span<const std::uint8_t>(
        expected_sink_storage.data(), expected_sink_storage.size());

    twin::GuardedBuffer input;
    twin::GuardedBuffer weight;
    twin::GuardedBuffer output;
    twin::GuardedBuffer sink;
    twin::GuardedBuffer row_count;
    check(input.allocate(context, shape + " input", input_bytes, err),
          shape + " input buffer", err);
    check(weight.allocate(context, shape + " weights", weight_bytes, err),
          shape + " weight buffer", err);
    check(output.allocate(context, shape + " output", output_bytes, err),
          shape + " output buffer", err);
    check(sink.allocate(context, shape + " sink", sink_bytes, err),
          shape + " sink buffer", err);
    check(row_count.allocate(context, shape + " rows", sizeof(std::uint32_t), err),
          shape + " row-count buffer", err);
    if (!input.view() || !weight.view() || !output.view() ||
        !sink.view() || !row_count.view()) {
        return false;
    }
    std::memcpy(input.payload().data(), inputs.data(), input_bytes);
    std::memcpy(weight.payload().data(), weights.data(), weight_bytes);
    std::memcpy(row_count.payload().data(), &live_rows, sizeof(live_rows));
    check(output.poison_bytes_against(expected_output_bytes, err),
          shape + " output poison", err);
    check(sink.poison_bytes_against(expected_sink_bytes, err),
          shape + " sink poison", err);
    input.capture_before_dispatch();
    weight.capture_before_dispatch();
    output.capture_before_dispatch();
    sink.capture_before_dispatch();
    row_count.capture_before_dispatch();

    const BF16MVParams params{n_in, n_out};
    twin::Dispatch work;
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {n_out, 1, 1};
    work.threads_per_group = {block_size, 1, 1};
    work.buffers = {{0, input.view()}, {1, weight.view()}, {2, output.view()},
                    {3, sink.view()}, {4, row_count.view()}};
    work.parameters = &params;
    work.parameter_bytes = sizeof(params);
    work.parameter_index = 5;
    const std::string kernel = "strata_mv_bf16_b" + std::to_string(block_size) +
                               "_n" + std::to_string(nc);
    const bool dispatched = twin::dispatch_manifest(context, kernel.c_str(), work, err);
    check(dispatched, shape + " GPU BF16 mat-vec dispatch", err);
    if (!dispatched) return true;

    const auto output_failure = twin::compare_bytes(output.payload(), expected_output_bytes);
    if (output_failure) twin::report_failure(shape.c_str(), options, live_rows, *output_failure);
    check(!output_failure, shape + " exact BF16 output bytes");
    if (sink_count != 0) {
        const auto sink_failure = twin::compare_bytes(sink.payload(), expected_sink_bytes);
        if (sink_failure) twin::report_failure(shape.c_str(), options, nc - live_rows, *sink_failure);
        check(!sink_failure, shape + " exact BF16 sink bytes");
    }
    if (actual_output != nullptr) {
        actual_output->resize(output_count);
        std::memcpy(actual_output->data(), output.payload().data(), output_bytes);
    }
    const std::array<twin::ByteRange, 1> output_written = {{{0, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written = {{{0, sink_count * sizeof(float)}}};
    check(!input.first_unwritten_change({}), shape + " input and guards remain unchanged");
    check(!weight.first_unwritten_change({}), shape + " weights and guards remain unchanged");
    check(!row_count.first_unwritten_change({}), shape + " row count and guards remain unchanged");
    check(!output.first_unwritten_change(output_written), shape + " output guards remain unchanged");
    check(!sink.first_unwritten_change(sink_written), shape + " sink guards remain unchanged");
    return true;
}

bool run_bf16_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 3> shapes = {
        std::pair<std::uint32_t, std::uint32_t>{320, 160},
        {2560, 256}, {10240, 256}};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 5> weight_pattern = {
        -0.25f, -0.125f, 0.125f, 0.25f, 0.375f};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const auto [n_in, block_size] : shapes) {
        std::vector<std::uint16_t> weights(static_cast<std::size_t>(n_in) * n_out, 0u);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t k = 0; k < n_in; ++k) {
                const float value = static_cast<float>(row + 1u) *
                                    weight_pattern[k % weight_pattern.size()];
                weights[static_cast<std::size_t>(row) * n_in + k] =
                    strata::kernels::bf16_from_f32(value);
            }
        }
        // 128 -> 129 after one low-bit flip, leaving a robust negative control
        // while the rest of each row still exercises every K iteration.
        weights[0] = strata::kernels::bf16_from_f32(128.0f);
        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        input_pattern[(k + column) % input_pattern.size()] *
                        static_cast<float>(column + 1u);
                }
            }

            std::vector<float> multi;
            completed &= test_bf16_shape(context, n_in, n_out, block_size, nc, nc,
                                         twin::InputSet::fixture, rng, options,
                                         inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "BF16 multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin = inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_bf16_shape(context, n_in, n_out, block_size, 1, 1,
                                             twin::InputSet::fixture, rng, options,
                                             single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "BF16 single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(multi.data() +
                                                          static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("BF16 mv_multi_eq_single", options, column, *mismatch);
                }
                check(!mismatch, "BF16 NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 320 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_bf16_shape(context, n_in, n_out, block_size, nc, nc,
                                             twin::InputSet::fixture, rng, options,
                                             inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "BF16 mv_multi_eq_single negative control detects one-bit weight flip");
                } else {
                    check(false, "BF16 negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_set(Context& context, twin::InputSet set, const twin::Options& options) {
    std::mt19937_64 rng(options.seed);
    constexpr std::array<std::uint32_t, 3> widths = {640, 2560, 6144};
    constexpr std::array<std::uint32_t, 3> row_counts = {1, 3, 8};
    bool completed = true;
    for (const std::uint32_t n : widths) {
        for (const std::uint32_t rows : row_counts) {
            completed &= test_shape(context, n, rows, set, rng, options);
        }
    }
    return completed;
}

bool run_q8_0_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x80b10cu);
    constexpr std::array<std::uint32_t, 5> widths = {640, 992, 1024, 2560, 6144};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q8_0_shape(context, n_in, n_out, nc,
                                             live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::uint32_t n_in = 640;
        constexpr std::uint32_t n_out_fixture = 3;
        constexpr std::uint32_t nc = 4;
        constexpr std::uint32_t blocks_per_row = n_in / 32;
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
        for (std::uint32_t column = 0; column < nc; ++column) {
            std::fill_n(inputs.begin() + static_cast<std::size_t>(column) * n_in,
                        n_in, 127.0f * static_cast<float>(column + 1u));
        }
        std::vector<block_q8_0> weights(
            static_cast<std::size_t>(n_out_fixture) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight = weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(1.0f);
                std::fill_n(weight.qs, 32, std::int8_t{0});
                if (block == 0) {
                    std::fill_n(weight.qs, 32, static_cast<std::int8_t>(row + 1u));
                }
            }
        }
        std::array<float, n_out_fixture * nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t input_column = column < 3u ? column : 0u;
            for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
                known_answer[column * n_out_fixture + row] =
                    4064.0f * static_cast<float>((input_column + 1u) * (row + 1u));
            }
        }
        completed &= test_q8_0_shape(context, n_in, n_out_fixture, nc, 3,
                                     set, rng, options, inputs, weights, known_answer);
        completed &= run_q8_0_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_q4_0_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x40b10cu);
    constexpr std::array<std::uint32_t, 5> widths = {640, 2016, 2048, 2560, 6144};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q4_0_shape(context, n_in, n_out, nc,
                                             live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::uint32_t n_in = 640;
        constexpr std::uint32_t n_out_fixture = 3;
        constexpr std::uint32_t nc = 4;
        constexpr std::uint32_t blocks_per_row = n_in / 32;
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        for (std::uint32_t column = 0; column < nc; ++column) {
            for (std::uint32_t k = 0; k < 32; ++k) {
                const std::uint32_t phase = (k & 15u) + column;
                const bool active = k < 16u ? phase % 4u != 0u : phase % 4u == 0u;
                if (active) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u);
                }
            }
        }
        std::vector<block_q4_0> weights(
            static_cast<std::size_t>(n_out_fixture) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight = weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(block == 0 ? 1.0f : 0.0f);
                std::fill_n(weight.qs, 16, std::uint8_t{0x88});
                if (block == 0) {
                    const std::uint8_t low = static_cast<std::uint8_t>(row + 9u);
                    const std::uint8_t high = static_cast<std::uint8_t>(row + 5u);
                    std::fill_n(weight.qs, 16,
                                static_cast<std::uint8_t>(low | (high << 4u)));
                }
            }
        }
        std::array<float, n_out_fixture * nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t input_column = column < 3u ? column : 0u;
            for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
                known_answer[column * n_out_fixture + row] =
                    2032.0f * static_cast<float>(input_column + 1u) *
                    static_cast<float>(row);
            }
        }
        completed &= test_q4_0_shape(context, n_in, n_out_fixture, nc, 3,
                                     set, rng, options, inputs, weights, known_answer);
        completed &= run_q4_0_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_q5_0_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x50b10cu);
    constexpr std::array<std::uint32_t, 5> widths = {640, 2016, 2048, 2560, 6144};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q5_0_shape(context, n_in, n_out, nc,
                                             live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::uint32_t n_in = 640;
        constexpr std::uint32_t n_out_fixture = 3;
        constexpr std::uint32_t nc = 4;
        constexpr std::uint32_t blocks_per_row = n_in / 32;
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        for (std::uint32_t column = 0; column < nc; ++column) {
            const float value = 127.0f * static_cast<float>(column + 1u);
            inputs[static_cast<std::size_t>(column) * n_in] = value;
            inputs[static_cast<std::size_t>(column) * n_in + 17u] = value;
            inputs[static_cast<std::size_t>(column) * n_in + 31u] = value;
        }
        std::vector<block_q5_0> weights(
            static_cast<std::size_t>(n_out_fixture) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                auto& weight = weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    block == 0 ? static_cast<float>(row + 1u) : 0.0f);
                std::fill_n(weight.qh, 4, std::uint8_t{0});
                std::fill_n(weight.qs, 16, std::uint8_t{0});
                if (block == 0) {
                    set_q5_0_code(weight, 0, 31);
                    set_q5_0_code(weight, 17, 17);
                    set_q5_0_code(weight, 31, 14);
                }
            }
        }
        std::array<float, n_out_fixture * nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t input_column = column < 3u ? column : 0u;
            for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
                known_answer[column * n_out_fixture + row] =
                    1778.0f * static_cast<float>(input_column + 1u) *
                    static_cast<float>(row + 1u);
            }
        }
        completed &= test_q5_0_shape(context, n_in, n_out_fixture, nc, 3,
                                     set, rng, options, inputs, weights, known_answer);
        completed &= run_q5_0_raw_decoder_golden(context, rng, options);
        completed &= run_q5_0_fma_golden(context, rng, options);
        completed &= run_q5_0_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_q2_0_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x20b10cu);
    constexpr std::array<std::uint32_t, 8> widths = {
        64, 640, 2560, 4032, 4096, 4160, 6144, 10240};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q2_0_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::array<std::uint32_t, 2> multi_block_widths = {4160, 8256};
        for (const std::uint32_t multi_block_n_in : multi_block_widths) {
            std::vector<float> multi_block_input(multi_block_n_in, 127.0f);
            std::vector<block_q2_0> multi_block_weight(multi_block_n_in / 64u);
            for (std::size_t block = 0; block < multi_block_weight.size(); ++block) {
                auto& weight = multi_block_weight[block];
                weight.d = strata::kernels::f16_from_f32(
                    block + 1u == multi_block_weight.size() ? 2.0f : 1.0f);
                std::fill_n(weight.qs, 16, std::uint8_t{0xaau});
            }
            std::fill_n(multi_block_input.end() - 64, 64, 31.75f);
            const float multi_block_answer =
                multi_block_n_in == 4160 ? 524256.0f : 1044448.0f;
            completed &= test_q2_0_shape(
                context, multi_block_n_in, 1, 1, 1, set, rng, options,
                multi_block_input, multi_block_weight,
                std::span<const float>(&multi_block_answer, 1));
        }

        constexpr std::uint32_t n_in = 64;
        constexpr std::uint32_t n_out_fixture = 3;
        constexpr std::uint32_t nc = 4;
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        for (std::uint32_t column = 0; column < nc; ++column) {
            const float multiplier = static_cast<float>(column + 1u);
            inputs[static_cast<std::size_t>(column) * n_in] =
                127.0f * multiplier;
            inputs[static_cast<std::size_t>(column) * n_in + 32u] =
                63.5f * multiplier;
        }
        std::vector<block_q2_0> weights(n_out_fixture);
        for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
            auto& weight = weights[row];
            weight.d = strata::kernels::f16_from_f32(
                0.5f * static_cast<float>(row + 1u));
            std::fill_n(weight.qs, 16, std::uint8_t{0x55});
            set_q2_0_code(weight, 0, 0);
            set_q2_0_code(weight, 32, 2);
        }
        std::array<float, n_out_fixture * nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t input_column = column < 3u ? column : 0u;
            for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
                known_answer[column * n_out_fixture + row] =
                    -31.75f * static_cast<float>(input_column + 1u) *
                    static_cast<float>(row + 1u);
            }
        }
        completed &= test_q2_0_shape(
            context, n_in, n_out_fixture, nc, 3, set, rng, options,
            inputs, weights, known_answer);
        completed &= run_q2_0_raw_decoder_golden(context, rng, options);
        completed &= run_q2_0_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_iq4_nl_set(Context& context, twin::InputSet set,
                    const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x1a4a1u);
    constexpr std::array<std::uint32_t, 7> widths = {
        32, 640, 2016, 2048, 2080, 2560, 6144};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq4_nl_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::uint32_t n_in = 32;
        constexpr std::uint32_t n_out_fixture = 3;
        constexpr std::uint32_t nc = 4;
        constexpr std::array<float, n_out_fixture> weight_scales = {
            0.5f, 1.0f, 1.5f};
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        for (std::uint32_t column = 0; column < nc; ++column) {
            inputs[static_cast<std::size_t>(column) * n_in] =
                127.0f * static_cast<float>(column + 1u);
        }
        std::vector<block_iq4_nl> weights(n_out_fixture);
        for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
            auto& weight = weights[row];
            weight.d = strata::kernels::f16_from_f32(weight_scales[row]);
            std::fill_n(weight.qs, 16, std::uint8_t{0});
            set_iq4_nl_code(weight, 0, row);
        }
        std::array<float, n_out_fixture * nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t input_column = column < 3u ? column : 0u;
            for (std::uint32_t row = 0; row < n_out_fixture; ++row) {
                known_answer[column * n_out_fixture + row] =
                    weight_scales[row] * static_cast<float>(input_column + 1u) *
                    static_cast<float>(k_iq4_nl_golden_values[row] * 127);
            }
        }
        completed &= test_iq4_nl_shape(
            context, n_in, n_out_fixture, nc, 3, set, rng, options,
            inputs, weights, known_answer);
        completed &= run_iq4_nl_raw_decoder_golden(context, rng, options);
        completed &= run_iq4_nl_fma_golden(context, rng, options);
        completed &= run_iq4_nl_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_iq4_xs_set(Context& context, twin::InputSet set,
                    const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x1a4a5u);
    constexpr std::array<std::uint32_t, 9> widths = {
        256, 2560, 3840, 4096, 6144, 10240, 16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq4_xs_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_iq4_xs_fixture(context, rng, options);
        completed &= run_iq4_xs_moe_onehot_fixture(context, rng, options);
        completed &= run_iq4_xs_raw_decoder_golden(context, rng, options);
        completed &= run_iq4_xs_fma_golden(context, rng, options);
        completed &= run_iq4_xs_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_iq2_xxs_generated_decoder_sweep(Context& context,
                                         std::mt19937_64& rng,
                                         const twin::Options& options) {
    constexpr std::uint32_t n_in = 2048u;
    constexpr std::uint32_t n_out = 1u;
    constexpr std::uint32_t nc = 8u;
    std::array<block_iq2_xxs, 8> weights{};
    for (std::uint32_t block = 0; block < weights.size(); ++block) {
        block_iq2_xxs& weight = weights[block];
        weight.d = strata::kernels::f16_from_f32(
            0.5f + 0.25f * static_cast<float>(block));
        for (std::uint32_t part = 0; part < 8u; ++part) {
            std::uint32_t indices = 0u;
            std::uint32_t aux32 = ((block * 8u + part) & 0x0fu) << 28u;
            for (std::uint32_t code_slot = 0; code_slot < 4u; ++code_slot) {
                const std::uint32_t code =
                    block * 32u + part * 4u + code_slot;
                indices |= code << (8u * code_slot);
                const std::uint32_t sign_bits =
                    (code * 29u + code_slot * 36u + 0x35u) & 0x7fu;
                aux32 |= sign_bits << (7u * code_slot);
            }
            const std::uint32_t word = 4u * part;
            weight.qs[word + 0u] =
                static_cast<std::uint16_t>(indices & 0xffffu);
            weight.qs[word + 1u] =
                static_cast<std::uint16_t>(indices >> 16u);
            weight.qs[word + 2u] =
                static_cast<std::uint16_t>(aux32 & 0xffffu);
            weight.qs[word + 3u] =
                static_cast<std::uint16_t>(aux32 >> 16u);
        }
    }

    std::array<std::uint32_t, 8> observed_codes{};
    std::array<std::uint8_t, 8> observed_sign_states{};
    std::uint16_t observed_scales = 0u;
    bool raw_words_match = true;
    for (std::uint32_t block = 0; block < weights.size(); ++block) {
        const block_iq2_xxs& weight = weights[block];
        for (std::uint32_t part = 0; part < 8u; ++part) {
            const std::uint32_t word = 4u * part;
            const std::uint32_t indices =
                static_cast<std::uint32_t>(weight.qs[word]) |
                (static_cast<std::uint32_t>(weight.qs[word + 1u]) << 16u);
            const std::uint32_t aux32 =
                static_cast<std::uint32_t>(weight.qs[word + 2u]) |
                (static_cast<std::uint32_t>(weight.qs[word + 3u]) << 16u);
            const std::uint32_t expected_scale =
                (block * 8u + part) & 0x0fu;
            raw_words_match &= (aux32 >> 28u) == expected_scale;
            observed_scales |= static_cast<std::uint16_t>(1u << (aux32 >> 28u));
            for (std::uint32_t code_slot = 0; code_slot < 4u; ++code_slot) {
                const std::uint32_t code =
                    (indices >> (8u * code_slot)) & 0xffu;
                raw_words_match &= code ==
                    block * 32u + part * 4u + code_slot;
                observed_codes[code / 32u] |= 1u << (code % 32u);
                const std::uint32_t raw_signs =
                    (aux32 >> (7u * code_slot)) & 0xffu;
                const std::uint32_t signs = raw_signs ^
                    ((std::popcount(raw_signs) & 1u) << 7u);
                for (std::uint32_t component = 0; component < 8u; ++component) {
                    observed_sign_states[component] |= static_cast<std::uint8_t>(
                        ((signs >> component) & 1u) != 0u ? 0x02u : 0x01u);
                }
            }
        }
    }
    check(raw_words_match,
          "IQ2_XXS generated u16 words expose every code and scale nibble");
    bool all_codes_seen = true;
    for (const std::uint32_t observed : observed_codes) {
        all_codes_seen &= observed == 0xffffffffu;
    }
    check(all_codes_seen, "IQ2_XXS raw words cover all 256 codebook indices");
    bool both_signs_seen = true;
    for (const std::uint8_t observed : observed_sign_states) {
        both_signs_seen &= observed == 0x03u;
    }
    check(both_signs_seen,
          "IQ2_XXS raw words cover both signs for all eight components");
    check(observed_scales == 0xffffu,
          "IQ2_XXS raw words cover all 16 scale nibbles");

    bool completed = true;
    for (std::uint32_t batch = 0; batch < n_in; batch += nc) {
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        std::array<float, nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t value_index = batch + column;
            const std::uint32_t block = value_index / 256u;
            const std::uint32_t position = value_index % 256u;
            const std::uint32_t part = position / 32u;
            const std::uint32_t within_part = position % 32u;
            const std::uint32_t code_slot = within_part / 8u;
            const std::uint32_t component = within_part % 8u;
            const block_iq2_xxs& weight = weights[block];
            const std::uint32_t word = 4u * part;
            const std::uint32_t indices =
                static_cast<std::uint32_t>(weight.qs[word]) |
                (static_cast<std::uint32_t>(weight.qs[word + 1u]) << 16u);
            const std::uint32_t aux32 =
                static_cast<std::uint32_t>(weight.qs[word + 2u]) |
                (static_cast<std::uint32_t>(weight.qs[word + 3u]) << 16u);
            const std::uint32_t code =
                (indices >> (8u * code_slot)) & 0xffu;
            const std::uint32_t raw_signs =
                (aux32 >> (7u * code_slot)) & 0xffu;
            const std::uint32_t signs = raw_signs ^
                ((std::popcount(raw_signs) & 1u) << 7u);
            const int magnitude = static_cast<int>(
                (iq2xxs_grid[code] >> (8u * component)) & 0xffu);
            const int signed_magnitude =
                ((signs >> component) & 1u) != 0u ? -magnitude : magnitude;
            const int odd_scale = static_cast<int>((aux32 >> 27u) | 1u);
            const int scaled_dot = (signed_magnitude * 127 * odd_scale) / 8;
            const float d = strata::kernels::f32_from_f16(weight.d);
            known_answer[column] =
                std::fma(d, static_cast<float>(scaled_dot), 0.0f);
            inputs[static_cast<std::size_t>(column) * n_in + value_index] =
                127.0f;
        }
        completed &= test_iq2_xxs_shape(
            context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq2_xxs_fixed_packed_golden(Context& context,
                                     std::mt19937_64& rng,
                                     const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 1u;
    constexpr std::uint32_t nc = 8u;

    block_iq2_xxs weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    std::copy(twin::kIQ2XXSMoePackedWords.begin(),
              twin::kIQ2XXSMoePackedWords.end(), weight.qs);
    const std::array<block_iq2_xxs, 1> weights = {weight};
    std::array<float, n_in * nc> inputs{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        inputs[static_cast<std::size_t>(column) * n_in +
               twin::kIQ2XXSMoeOneHotPositions[column]] = 127.0f;
    }
    return test_iq2_xxs_shape(
        context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng, options,
        inputs, weights, twin::kIQ2XXSMoeOneHotAnswers);
}

bool run_iq2_xxs_signed_division_fixture(Context& context,
                                         std::mt19937_64& rng,
                                         const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 1u;
    constexpr std::uint32_t nc = 2u;
    block_iq2_xxs weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    weight.qs[0] = 1u;  // code 1 starts with magnitude 43; code 0 starts with 8.
    const std::array<block_iq2_xxs, 1> weights = {weight};
    std::array<float, n_in * nc> inputs{};
    inputs[0] = 1.0f;
    inputs[1] = 127.0f;
    inputs[2] = -127.0f;
    inputs[n_in] = -1.0f;
    inputs[n_in + 1u] = 127.0f;
    inputs[n_in + 2u] = -127.0f;
    constexpr std::array<float, nc> known_answer = {5.0f, -5.0f};
    return test_iq2_xxs_shape(
        context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq2_xxs_padded_fixture(Context& context, std::mt19937_64& rng,
                                const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 3u;
    constexpr std::uint32_t nc = 4u;
    constexpr std::uint32_t live_rows = 2u;
    std::array<block_iq2_xxs, n_out> weights{};
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 2.0f};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        block_iq2_xxs& weight = weights[row];
        weight.d = strata::kernels::f16_from_f32(weight_scales[row]);
        for (std::uint32_t part = 0; part < 8u; ++part) {
            weight.qs[4u * part] = 0x0101u;
            weight.qs[4u * part + 1u] = 0x0101u;
        }
    }
    std::array<float, n_in * nc> inputs{};
    inputs[0] = 127.0f;
    inputs[n_in + 1u] = 127.0f;
    inputs[2u * n_in] = -127.0f;
    inputs[3u * n_in + 1u] = -127.0f;
    constexpr std::array<float, n_out * nc> known_answer = {
        341.0f, 682.0f, 1364.0f,
        63.5f, 127.0f, 254.0f,
        341.0f, 682.0f, 1364.0f,
        341.0f, 682.0f, 1364.0f};
    return test_iq2_xxs_shape(
        context, n_in, n_out, nc, live_rows, twin::InputSet::fixture, rng,
        options, inputs, weights, known_answer);
}

bool run_iq2_xxs_fma_golden(Context& context, std::mt19937_64& rng,
                            const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9u, 13u};
    constexpr std::array<std::uint32_t, 2> expected_bits = {
        0x46c73542u, 0x4707fa52u};
    bool completed = true;
    for (std::size_t test = 0; test < block_counts.size(); ++test) {
        const std::uint32_t blocks = block_counts[test];
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq2_xxs> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq2_xxs& weight = weights[block];
            const std::array<float, 3> scale_values = {0.13f, 0.27f, 0.31f};
            weight.d = strata::kernels::f16_from_f32(
                scale_values[block % scale_values.size()]);
            const std::uint32_t nibble = (block * 5u + 3u) & 0x0fu;
            const std::uint32_t aux32 = nibble << 28u;
            weight.qs[0] = 1u;
            weight.qs[2] = static_cast<std::uint16_t>(aux32 & 0xffffu);
            weight.qs[3] = static_cast<std::uint16_t>(aux32 >> 16u);
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const float known = std::bit_cast<float>(expected_bits[test]);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq2_xxs_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq2_xxs_long_k_golden(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63u, 64u, 65u};
    bool completed = true;
    for (const std::uint32_t blocks : block_counts) {
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq2_xxs> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq2_xxs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(1.0f);
            weight.qs[0] = 1u;
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const float known = 682.0f * static_cast<float>(blocks);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq2_xxs_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq2_xxs_multi_eq_single(Context& context, std::mt19937_64& rng,
                                 const twin::Options& options) {
    constexpr std::array<std::uint32_t, 4> widths = {
        256u, 1024u, 2304u, 16384u};
    constexpr std::array<std::uint32_t, 3> column_counts = {2u, 4u, 8u};
    constexpr std::uint32_t n_out = 5u;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        std::vector<block_iq2_xxs> weights(
            static_cast<std::size_t>(n_out) * (n_in / 256u));
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq2_xxs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(
                0.25f + 0.125f * static_cast<float>(block % 7u));
            for (std::uint32_t i = 0; i < 32u; ++i) {
                weight.qs[i] = static_cast<std::uint16_t>(
                    (i * 0x139du + block * 0x2b17u + 0x5a31u) & 0xffffu);
            }
        }
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * 8u);
        std::string err;
        inputs = twin::synthesize_floats(
            twin::InputSet::fixture, inputs.size(), rng, err);
        if (!err.empty()) {
            check(false, "IQ2_XXS mv_multi_eq_single input synthesis", err);
            return false;
        }
        for (const std::uint32_t nc : column_counts) {
            std::vector<float> multi;
            completed &= test_iq2_xxs_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                rng, options,
                std::span<const float>(inputs.data(),
                                       static_cast<std::size_t>(n_in) * nc),
                weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ2_XXS multi-column result available");
                completed = false;
                continue;
            }
            for (std::uint32_t column = 0; column < nc; ++column) {
                std::vector<float> single;
                completed &= test_iq2_xxs_shape(
                    context, n_in, n_out, 1u, 1u,
                    twin::InputSet::fixture, rng, options,
                    std::span<const float>(
                        inputs.data() + static_cast<std::size_t>(column) * n_in,
                        n_in),
                    weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ2_XXS single-column result available");
                    completed = false;
                    continue;
                }
                const auto multi_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_bytes, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ2_XXS mv_multi_eq_single",
                                         options, column, *mismatch);
                }
                check(!mismatch,
                      "IQ2_XXS NC column equals single-column dispatch");
            }
        }
    }

    std::array<block_iq2_xxs, 1> base_weight{};
    base_weight[0].d = strata::kernels::f16_from_f32(1.0f);
    std::array<block_iq2_xxs, 1> changed_weight = base_weight;
    changed_weight[0].qs[0] = 1u;
    std::array<float, 256> control_input{};
    control_input[0] = 127.0f;
    std::vector<float> base;
    std::vector<float> changed;
    completed &= test_iq2_xxs_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, base_weight, {}, &base);
    completed &= test_iq2_xxs_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, changed_weight, {}, &changed);
    if (base.size() == 1u && changed.size() == 1u) {
        check(std::bit_cast<std::uint32_t>(base[0]) !=
                  std::bit_cast<std::uint32_t>(changed[0]),
              "IQ2_XXS mv_multi_eq_single negative control changes output");
    } else {
        check(false, "IQ2_XXS negative-control results available");
        completed = false;
    }
    return completed;
}

bool run_iq2_xxs_set(Context& context, twin::InputSet set,
                     const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x2a2a5u);
    constexpr std::array<std::uint32_t, 14> widths = {
        256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq2_xxs_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_iq2_xxs_generated_decoder_sweep(context, rng, options);
        completed &= run_iq2_xxs_fixed_packed_golden(context, rng, options);
        completed &= run_iq2_xxs_signed_division_fixture(context, rng, options);
        completed &= run_iq2_xxs_padded_fixture(context, rng, options);
        completed &= run_iq2_xxs_fma_golden(context, rng, options);
        completed &= run_iq2_xxs_long_k_golden(context, rng, options);
        completed &= run_iq2_xxs_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_iq2_xs_generated_decoder_sweep(Context& context,
                                        std::mt19937_64& rng,
                                        const twin::Options& options) {
    constexpr std::uint32_t n_in = 4096u;
    constexpr std::uint32_t nc = 8u;
    constexpr std::uint32_t block_count = n_in / 256u;
    std::array<block_iq2_xs, block_count> weights{};
    for (std::uint32_t block = 0; block < block_count; ++block) {
        block_iq2_xs& weight = weights[block];
        weight.d = strata::kernels::f16_from_f32(
            0.5f + 0.125f * static_cast<float>(block % 4u));
        for (std::uint32_t part = 0; part < 8u; ++part) {
            const std::uint32_t scale_index = block * 8u + part;
            const std::uint32_t ls0 = scale_index & 0x0fu;
            const std::uint32_t ls1 = (scale_index * 7u + 3u) & 0x0fu;
            weight.scales[part] =
                static_cast<std::uint8_t>(ls0 | (ls1 << 4u));
            for (std::uint32_t code_slot = 0; code_slot < 4u; ++code_slot) {
                const std::uint32_t code =
                    block * 32u + part * 4u + code_slot;
                const std::uint32_t sign_index =
                    (code * 37u + 0x35u) & 0x7fu;
                weight.qs[4u * part + code_slot] =
                    static_cast<std::uint16_t>(code | (sign_index << 9u));
            }
        }
    }

    std::array<std::uint32_t, 16> observed_code_indices{};
    std::array<std::uint8_t, 8> observed_sign_states{};
    std::uint16_t observed_ls0 = 0u;
    std::uint16_t observed_ls1 = 0u;
    bool generated_words_match = true;
    for (std::uint32_t block = 0; block < block_count; ++block) {
        const block_iq2_xs& weight = weights[block];
        for (std::uint32_t part = 0; part < 8u; ++part) {
            const std::uint32_t scale_index = block * 8u + part;
            const std::uint32_t ls0 = scale_index & 0x0fu;
            const std::uint32_t ls1 = (scale_index * 7u + 3u) & 0x0fu;
            generated_words_match &=
                weight.scales[part] == (ls0 | (ls1 << 4u));
            observed_ls0 |= static_cast<std::uint16_t>(1u << ls0);
            observed_ls1 |= static_cast<std::uint16_t>(1u << ls1);
            for (std::uint32_t code_slot = 0; code_slot < 4u; ++code_slot) {
                const std::uint16_t codeword =
                    weight.qs[4u * part + code_slot];
                const std::uint32_t code = codeword & 0x1ffu;
                const std::uint32_t sign_index = codeword >> 9u;
                const std::uint32_t expected_code =
                    block * 32u + part * 4u + code_slot;
                const std::uint32_t expected_sign =
                    (expected_code * 37u + 0x35u) & 0x7fu;
                generated_words_match &=
                    code == expected_code && sign_index == expected_sign;
                observed_code_indices[code / 32u] |=
                    1u << (code % 32u);
                const std::uint32_t signs = ksigns_iq2xs[sign_index];
                for (std::uint32_t component = 0; component < 8u;
                     ++component) {
                    observed_sign_states[component] |=
                        (signs & kmask_iq2xs[component]) != 0u ? 0x02u : 0x01u;
                }
            }
        }
    }
    check(generated_words_match,
          "IQ2_XS generated packed words expose all indices and scales");
    bool all_codes_seen = true;
    for (const std::uint32_t observed : observed_code_indices) {
        all_codes_seen &= observed == 0xffffffffu;
    }
    check(all_codes_seen, "IQ2_XS generated sweep covers all 512 grid indices");
    bool both_signs_seen = true;
    for (const std::uint8_t observed : observed_sign_states) {
        both_signs_seen &= observed == 0x03u;
    }
    check(both_signs_seen,
          "IQ2_XS generated sweep covers both signs at all eight components");
    check(observed_ls0 == 0xffffu && observed_ls1 == 0xffffu,
          "IQ2_XS generated sweep covers all low and high scale nibbles");

    bool completed = true;
    for (std::uint32_t batch = 0; batch < n_in; batch += nc) {
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        for (std::uint32_t column = 0; column < nc; ++column) {
            inputs[static_cast<std::size_t>(column) * n_in + batch + column] =
                127.0f;
        }
        completed &= test_iq2_xs_shape(
            context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
            inputs, weights);
    }
    return completed;
}

bool run_iq2_xs_fixed_packed_golden(Context& context,
                                    std::mt19937_64& rng,
                                    const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t nc = 8u;
    constexpr auto& raw_qs = twin::kIQ2XSMoePackedWords;
    constexpr auto& raw_scales = twin::kIQ2XSMoeScales;
    constexpr auto& positions = twin::kIQ2XSMoeOneHotPositions;
    constexpr auto& known_answer = twin::kIQ2XSMoeOneHotAnswers;

    block_iq2_xs weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    std::copy(raw_qs.begin(), raw_qs.end(), weight.qs);
    std::copy(raw_scales.begin(), raw_scales.end(), weight.scales);
    const std::array<block_iq2_xs, 1> weights = {weight};
    std::array<float, n_in * nc> inputs{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        inputs[static_cast<std::size_t>(column) * n_in + positions[column]] =
            127.0f;
    }
    return test_iq2_xs_shape(
        context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq2_xs_signed_division_fixture(Context& context,
                                        std::mt19937_64& rng,
                                        const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t nc = 2u;
    block_iq2_xs weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    weight.qs[0] = 1u;  // IQ2_XS grid index 1 starts with magnitude 43.
    const std::array<block_iq2_xs, 1> weights = {weight};
    std::array<float, n_in * nc> inputs{};
    inputs[0] = 127.0f;
    inputs[n_in] = -127.0f;
    constexpr std::array<float, nc> known_answer = {682.0f, -682.0f};
    return test_iq2_xs_shape(
        context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq2_xs_padded_fixture(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 3u;
    constexpr std::uint32_t nc = 4u;
    constexpr std::uint32_t live_rows = 2u;
    std::array<block_iq2_xs, n_out> weights{};
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 2.0f};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        weights[row].d = strata::kernels::f16_from_f32(weight_scales[row]);
        std::fill_n(weights[row].scales, 8u, std::uint8_t{0x53u});
    }
    std::array<float, n_in * nc> inputs{};
    for (std::uint32_t part = 0; part < 8u; ++part) {
        inputs[part * 32u] = 127.0f;
        inputs[n_in + part * 32u + 16u] = 127.0f;
        inputs[2u * n_in + part * 32u] = -127.0f;
        inputs[3u * n_in + part * 32u + 16u] = -127.0f;
    }
    constexpr std::array<float, n_out * nc> known_answer = {
        3556.0f, 7112.0f, 14224.0f,
        5588.0f, 11176.0f, 22352.0f,
        3556.0f, 7112.0f, 14224.0f,
        3556.0f, 7112.0f, 14224.0f,
    };
    return test_iq2_xs_shape(
        context, n_in, n_out, nc, live_rows, twin::InputSet::fixture, rng,
        options, inputs, weights, known_answer);
}

bool run_iq2_xs_fma_golden(Context& context, std::mt19937_64& rng,
                           const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9u, 13u};
    constexpr std::array<std::uint32_t, 2> expected_bits = {
        0x46c73542u, 0x4707fa52u};
    bool completed = true;
    for (std::size_t test = 0; test < block_counts.size(); ++test) {
        const std::uint32_t blocks = block_counts[test];
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq2_xs> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq2_xs& weight = weights[block];
            constexpr std::array<float, 3> scale_values = {
                0.13f, 0.27f, 0.31f};
            weight.d = strata::kernels::f16_from_f32(
                scale_values[block % scale_values.size()]);
            weight.qs[0] = 1u;
            weight.scales[0] =
                static_cast<std::uint8_t>((block * 5u + 3u) & 0x0fu);
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const float known = std::bit_cast<float>(expected_bits[test]);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq2_xs_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq2_xs_long_k_golden(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63u, 64u, 65u};
    bool completed = true;
    for (const std::uint32_t blocks : block_counts) {
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq2_xs> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq2_xs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(1.0f);
            weight.qs[0] = 1u;
            weight.scales[0] = 3u;
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const float known = 4778.0f * static_cast<float>(blocks);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq2_xs_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq2_xs_multi_eq_single(Context& context, std::mt19937_64& rng,
                                const twin::Options& options) {
    constexpr std::array<std::uint32_t, 4> widths = {
        256u, 1024u, 2304u, 16384u};
    constexpr std::array<std::uint32_t, 3> column_counts = {2u, 4u, 8u};
    constexpr std::uint32_t n_out = 5u;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        std::vector<block_iq2_xs> weights(
            static_cast<std::size_t>(n_out) * (n_in / 256u));
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq2_xs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(
                0.25f + 0.125f * static_cast<float>(block % 7u));
            for (std::uint32_t i = 0; i < 32u; ++i) {
                weight.qs[i] = static_cast<std::uint16_t>(
                    (i * 0x139du + block * 0x2b17u + 0x5a31u) & 0xffffu);
            }
            for (std::uint32_t part = 0; part < 8u; ++part) {
                weight.scales[part] = static_cast<std::uint8_t>(
                    (block * 31u + part * 47u + 0x59u) & 0xffu);
            }
        }

        std::vector<float> inputs(static_cast<std::size_t>(n_in) * 8u);
        std::string err;
        inputs = twin::synthesize_floats(
            twin::InputSet::fixture, inputs.size(), rng, err);
        if (!err.empty()) {
            check(false, "IQ2_XS mv_multi_eq_single input synthesis", err);
            return false;
        }
        for (const std::uint32_t nc : column_counts) {
            std::vector<float> multi;
            completed &= test_iq2_xs_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                rng, options,
                std::span<const float>(inputs.data(),
                                       static_cast<std::size_t>(n_in) * nc),
                weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ2_XS multi-column result available");
                completed = false;
                continue;
            }
            for (std::uint32_t column = 0; column < nc; ++column) {
                std::vector<float> single;
                completed &= test_iq2_xs_shape(
                    context, n_in, n_out, 1u, 1u,
                    twin::InputSet::fixture, rng, options,
                    std::span<const float>(
                        inputs.data() + static_cast<std::size_t>(column) * n_in,
                        n_in),
                    weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ2_XS single-column result available");
                    completed = false;
                    continue;
                }
                const auto multi_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch =
                    twin::compare_bytes(multi_bytes, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ2_XS mv_multi_eq_single",
                                         options, column, *mismatch);
                }
                check(!mismatch,
                      "IQ2_XS NC column equals single-column dispatch");
            }
        }
    }

    std::array<block_iq2_xs, 1> base_weight{};
    base_weight[0].d = strata::kernels::f16_from_f32(1.0f);
    base_weight[0].scales[0] = 3u;
    std::array<block_iq2_xs, 1> changed_weight = base_weight;
    changed_weight[0].qs[0] = 1u;
    std::array<float, 256> control_input{};
    control_input[0] = 127.0f;
    std::vector<float> base;
    std::vector<float> changed;
    completed &= test_iq2_xs_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, base_weight, {}, &base);
    completed &= test_iq2_xs_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, changed_weight, {}, &changed);
    if (base.size() == 1u && changed.size() == 1u) {
        check(std::bit_cast<std::uint32_t>(base[0]) !=
                  std::bit_cast<std::uint32_t>(changed[0]),
              "IQ2_XS mv_multi_eq_single negative control changes output");
    } else {
        check(false, "IQ2_XS negative-control results available");
        completed = false;
    }
    return completed;
}

bool run_iq2_xs_set(Context& context, twin::InputSet set,
                    const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x2a2a6u);
    constexpr std::array<std::uint32_t, 14> widths = {
        256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq2_xs_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_iq2_xs_generated_decoder_sweep(context, rng, options);
        completed &= run_iq2_xs_fixed_packed_golden(context, rng, options);
        completed &= run_iq2_xs_signed_division_fixture(context, rng, options);
        completed &= run_iq2_xs_padded_fixture(context, rng, options);
        completed &= run_iq2_xs_fma_golden(context, rng, options);
        completed &= run_iq2_xs_long_k_golden(context, rng, options);
        completed &= run_iq2_xs_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_iq2_s_generated_decoder_sweep(Context& context,
                                       std::mt19937_64& rng,
                                       const twin::Options& options) {
    constexpr std::uint32_t n_in = 8192u;
    constexpr std::uint32_t nc = 8u;
    constexpr std::uint32_t block_count = n_in / 256u;
    std::array<block_iq2_s, block_count> weights{};
    for (std::uint32_t block = 0; block < block_count; ++block) {
        block_iq2_s& weight = weights[block];
        weight.d = strata::kernels::f16_from_f32(
            0.5f + 0.125f * static_cast<float>(block % 4u));
        for (std::uint32_t part = 0; part < 8u; ++part) {
            const std::uint32_t scale_index = block * 8u + part;
            const std::uint32_t ls0 = scale_index & 0x0fu;
            const std::uint32_t ls1 = (scale_index * 7u + 3u) & 0x0fu;
            weight.scales[part] =
                static_cast<std::uint8_t>(ls0 | (ls1 << 4u));
            std::uint32_t high_codes = 0u;
            for (std::uint32_t code_slot = 0; code_slot < 4u; ++code_slot) {
                const std::uint32_t code =
                    block * 32u + part * 4u + code_slot;
                const std::uint32_t signs =
                    (code * 73u + 0xa5u) & 0xffu;
                weight.qs[4u * part + code_slot] =
                    static_cast<std::uint8_t>(code & 0xffu);
                weight.qs[32u + 4u * part + code_slot] =
                    static_cast<std::uint8_t>(signs);
                high_codes |= ((code >> 8u) & 0x03u) << (2u * code_slot);
            }
            weight.qh[part] = static_cast<std::uint8_t>(high_codes);
        }
    }

    std::array<std::uint32_t, 32> observed_code_indices{};
    std::array<std::uint8_t, 4> observed_high_codes{};
    std::array<std::uint8_t, 8> observed_sign_states{};
    std::uint16_t observed_ls0 = 0u;
    std::uint16_t observed_ls1 = 0u;
    bool generated_bytes_match = true;
    for (std::uint32_t block = 0; block < block_count; ++block) {
        const block_iq2_s& weight = weights[block];
        for (std::uint32_t part = 0; part < 8u; ++part) {
            const std::uint32_t scale_index = block * 8u + part;
            const std::uint32_t ls0 = scale_index & 0x0fu;
            const std::uint32_t ls1 = (scale_index * 7u + 3u) & 0x0fu;
            generated_bytes_match &=
                weight.scales[part] == (ls0 | (ls1 << 4u));
            observed_ls0 |= static_cast<std::uint16_t>(1u << ls0);
            observed_ls1 |= static_cast<std::uint16_t>(1u << ls1);
            for (std::uint32_t code_slot = 0; code_slot < 4u; ++code_slot) {
                const std::uint32_t code =
                    block * 32u + part * 4u + code_slot;
                const std::uint32_t low = weight.qs[4u * part + code_slot];
                const std::uint32_t high =
                    (weight.qh[part] >> (2u * code_slot)) & 0x03u;
                const std::uint32_t signs =
                    weight.qs[32u + 4u * part + code_slot];
                generated_bytes_match &=
                    low == (code & 0xffu) && high == (code >> 8u) &&
                    signs == ((code * 73u + 0xa5u) & 0xffu);
                observed_code_indices[code / 32u] |=
                    1u << (code % 32u);
                observed_high_codes[code_slot] |=
                    static_cast<std::uint8_t>(1u << high);
                for (std::uint32_t component = 0; component < 8u;
                     ++component) {
                    observed_sign_states[component] |=
                        (signs & (1u << component)) != 0u ? 0x02u : 0x01u;
                }
            }
        }
    }
    check(generated_bytes_match,
          "IQ2_S generated packed bytes expose all grid and sign fields");
    bool all_codes_seen = true;
    for (const std::uint32_t observed : observed_code_indices) {
        all_codes_seen &= observed == 0xffffffffu;
    }
    check(all_codes_seen, "IQ2_S generated sweep covers all 1024 grid indices");
    bool all_high_codes_seen = true;
    for (const std::uint8_t observed : observed_high_codes) {
        all_high_codes_seen &= observed == 0x0fu;
    }
    check(all_high_codes_seen,
          "IQ2_S generated sweep covers all qh values in each code slot");
    bool both_signs_seen = true;
    for (const std::uint8_t observed : observed_sign_states) {
        both_signs_seen &= observed == 0x03u;
    }
    check(both_signs_seen,
          "IQ2_S generated sweep covers both signs at all eight components");
    check(observed_ls0 == 0xffffu && observed_ls1 == 0xffffu,
          "IQ2_S generated sweep covers all low and high scale nibbles");

    bool completed = true;
    for (std::uint32_t batch = 0; batch < n_in; batch += nc) {
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        for (std::uint32_t column = 0; column < nc; ++column) {
            inputs[static_cast<std::size_t>(column) * n_in + batch + column] =
                127.0f;
        }
        completed &= test_iq2_s_shape(
            context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
            inputs, weights);
    }
    return completed;
}

bool run_iq2_s_fixed_packed_golden(Context& context,
                                   std::mt19937_64& rng,
                                   const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t nc = 8u;
    constexpr auto& raw_qs = twin::kIQ2SMoePackedBytes;
    constexpr auto& raw_qh = twin::kIQ2SMoeHighCodes;
    constexpr auto& raw_scales = twin::kIQ2SMoeScales;
    constexpr auto& positions = twin::kIQ2SMoeOneHotPositions;
    constexpr auto& known_answer = twin::kIQ2SMoeOneHotAnswers;

    block_iq2_s weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    std::copy(raw_qs.begin(), raw_qs.end(), weight.qs);
    std::copy(raw_qh.begin(), raw_qh.end(), weight.qh);
    std::copy(raw_scales.begin(), raw_scales.end(), weight.scales);
    const std::array<block_iq2_s, 1> weights = {weight};
    std::array<float, n_in * nc> inputs{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        inputs[static_cast<std::size_t>(column) * n_in + positions[column]] =
            127.0f;
    }
    return test_iq2_s_shape(
        context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq2_s_signed_division_fixture(Context& context,
                                      std::mt19937_64& rng,
                                      const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t nc = 2u;
    block_iq2_s weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    weight.qs[0] = 1u;
    weight.scales[0] = 1u;
    const std::array<block_iq2_s, 1> weights = {weight};
    std::array<float, n_in * nc> inputs{};
    inputs[0] = 127.0f;
    inputs[n_in] = -127.0f;
    constexpr std::array<float, nc> known_answer = {2047.0f, -2047.0f};
    return test_iq2_s_shape(
        context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq2_s_padded_fixture(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 3u;
    constexpr std::uint32_t nc = 4u;
    constexpr std::uint32_t live_rows = 2u;
    std::array<block_iq2_s, n_out> weights{};
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 2.0f};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        weights[row].d = strata::kernels::f16_from_f32(weight_scales[row]);
        std::fill_n(weights[row].scales, 8u, std::uint8_t{0x21u});
        for (std::uint32_t part = 0; part < 8u; ++part) {
            weights[row].qs[4u * part] = 1u;
            weights[row].qs[4u * part + 2u] = 1u;
        }
    }
    std::array<float, n_in * nc> inputs{};
    for (std::uint32_t part = 0; part < 8u; ++part) {
        inputs[part * 32u] = 127.0f;
        inputs[n_in + part * 32u + 16u] = 127.0f;
        inputs[2u * n_in + part * 32u] = -127.0f;
        inputs[3u * n_in + part * 32u + 16u] = -127.0f;
    }
    constexpr std::array<float, n_out * nc> known_answer = {
        8188.0f, 16376.0f, 32752.0f,
        13652.0f, 27304.0f, 54608.0f,
        8188.0f, 16376.0f, 32752.0f,
        8188.0f, 16376.0f, 32752.0f,
    };
    return test_iq2_s_shape(
        context, n_in, n_out, nc, live_rows, twin::InputSet::fixture, rng,
        options, inputs, weights, known_answer);
}

bool run_iq2_s_fma_golden(Context& context, std::mt19937_64& rng,
                          const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9u, 13u};
    constexpr std::array<std::uint32_t, 2> expected_bits = {
        0x46c73542u, 0x4707fa52u};
    bool completed = true;
    for (std::size_t test = 0; test < block_counts.size(); ++test) {
        const std::uint32_t blocks = block_counts[test];
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq2_s> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq2_s& weight = weights[block];
            constexpr std::array<float, 3> scale_values = {
                0.13f, 0.27f, 0.31f};
            weight.d = strata::kernels::f16_from_f32(
                scale_values[block % scale_values.size()]);
            weight.qs[0] = 1u;
            weight.scales[0] =
                static_cast<std::uint8_t>((block * 5u + 3u) & 0x0fu);
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const float known = std::bit_cast<float>(expected_bits[test]);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq2_s_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq2_s_long_k_golden(Context& context, std::mt19937_64& rng,
                             const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63u, 64u, 65u};
    bool completed = true;
    for (const std::uint32_t blocks : block_counts) {
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq2_s> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq2_s& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(1.0f);
            weight.qs[0] = 1u;
            weight.scales[0] = 3u;
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const float known = 4778.0f * static_cast<float>(blocks);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq2_s_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq2_s_multi_eq_single(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::array<std::uint32_t, 4> widths = {
        256u, 1024u, 2304u, 16384u};
    constexpr std::array<std::uint32_t, 3> column_counts = {2u, 4u, 8u};
    constexpr std::uint32_t n_out = 5u;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::size_t blocks_per_row = n_in / 256u;
        std::vector<block_iq2_s> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq2_s& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(
                0.25f + 0.125f * static_cast<float>(block % 7u));
            for (std::uint32_t part = 0; part < 8u; ++part) {
                weight.qh[part] = static_cast<std::uint8_t>(
                    (block * 0x35u + part * 0x6du + 0x4bu) & 0xffu);
                weight.scales[part] = static_cast<std::uint8_t>(
                    (block * 31u + part * 47u + 0x59u) & 0xffu);
                for (std::uint32_t code_slot = 0; code_slot < 4u;
                     ++code_slot) {
                    weight.qs[4u * part + code_slot] =
                        static_cast<std::uint8_t>(
                            (block * 0x2du + part * 0x73u +
                             code_slot * 0x5bu + 0xa7u) & 0xffu);
                    weight.qs[32u + 4u * part + code_slot] =
                        static_cast<std::uint8_t>(
                            (block * 0x91u + part * 0x4bu +
                             code_slot * 0x37u + 0x65u) & 0xffu);
                }
            }
        }

        std::vector<float> inputs(static_cast<std::size_t>(n_in) * 8u);
        std::string err;
        inputs = twin::synthesize_floats(
            twin::InputSet::fixture, inputs.size(), rng, err);
        if (!err.empty()) {
            check(false, "IQ2_S mv_multi_eq_single input synthesis", err);
            return false;
        }
        for (const std::uint32_t nc : column_counts) {
            std::vector<float> multi;
            completed &= test_iq2_s_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                rng, options,
                std::span<const float>(inputs.data(),
                                       static_cast<std::size_t>(n_in) * nc),
                weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ2_S multi-column result available");
                completed = false;
                continue;
            }
            for (std::uint32_t column = 0; column < nc; ++column) {
                std::vector<float> single;
                completed &= test_iq2_s_shape(
                    context, n_in, n_out, 1u, 1u,
                    twin::InputSet::fixture, rng, options,
                    std::span<const float>(
                        inputs.data() + static_cast<std::size_t>(column) * n_in,
                        n_in),
                    weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ2_S single-column result available");
                    completed = false;
                    continue;
                }
                const auto multi_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch =
                    twin::compare_bytes(multi_bytes, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ2_S mv_multi_eq_single",
                                         options, column, *mismatch);
                }
                check(!mismatch,
                      "IQ2_S NC column equals single-column dispatch");
            }
        }
    }

    std::array<block_iq2_s, 1> base_weight{};
    base_weight[0].d = strata::kernels::f16_from_f32(1.0f);
    base_weight[0].scales[0] = 3u;
    std::array<block_iq2_s, 1> changed_weight = base_weight;
    changed_weight[0].qs[0] = 1u;
    std::array<float, 256> control_input{};
    control_input[0] = 127.0f;
    std::vector<float> base;
    std::vector<float> changed;
    completed &= test_iq2_s_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, base_weight, {}, &base);
    completed &= test_iq2_s_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, changed_weight, {}, &changed);
    if (base.size() == 1u && changed.size() == 1u) {
        check(std::bit_cast<std::uint32_t>(base[0]) !=
                  std::bit_cast<std::uint32_t>(changed[0]),
              "IQ2_S mv_multi_eq_single negative control changes output");
    } else {
        check(false, "IQ2_S negative-control results available");
        completed = false;
    }
    return completed;
}

bool run_iq2_s_set(Context& context, twin::InputSet set,
                   const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x2a2a7u);
    constexpr std::array<std::uint32_t, 14> widths = {
        256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq2_s_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_iq2_s_generated_decoder_sweep(context, rng, options);
        completed &= run_iq2_s_fixed_packed_golden(context, rng, options);
        completed &= run_iq2_s_signed_division_fixture(context, rng, options);
        completed &= run_iq2_s_padded_fixture(context, rng, options);
        completed &= run_iq2_s_fma_golden(context, rng, options);
        completed &= run_iq2_s_long_k_golden(context, rng, options);
        completed &= run_iq2_s_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_iq3_xxs_generated_decoder_sweep(Context& context,
                                         std::mt19937_64& rng,
                                         const twin::Options& options) {
    constexpr std::uint32_t n_in = 1024u;
    constexpr std::uint32_t nc = 8u;
    constexpr std::uint32_t block_count = n_in / 256u;
    std::array<block_iq3_xxs, block_count> weights{};
    for (std::uint32_t block = 0; block < block_count; ++block) {
        block_iq3_xxs& weight = weights[block];
        weight.d = strata::kernels::f16_from_f32(
            0.5f + 0.125f * static_cast<float>(block % 4u));
        for (std::uint32_t part = 0; part < 8u; ++part) {
            const std::uint32_t scale_index = block * 8u + part;
            const std::uint32_t scale =
                (scale_index * 7u + 3u) & 0x0fu;
            std::uint32_t aux32 = scale << 28u;
            for (std::uint32_t code_slot = 0; code_slot < 4u;
                 ++code_slot) {
                const std::uint32_t sign_slot =
                    block * 32u + part * 4u + code_slot;
                const std::uint32_t sign_index =
                    (sign_slot * 37u + 0x35u) & 0x7fu;
                aux32 |= sign_index << (7u * code_slot);
            }
            for (std::uint32_t grid_slot = 0; grid_slot < 8u; ++grid_slot) {
                const std::uint32_t grid_index =
                    block * 64u + part * 8u + grid_slot;
                weight.qs[8u * part + grid_slot] =
                    static_cast<std::uint8_t>(grid_index);
            }
            for (std::uint32_t byte = 0; byte < 4u; ++byte) {
                weight.qs[64u + 4u * part + byte] =
                    static_cast<std::uint8_t>(aux32 >> (8u * byte));
            }
        }
    }

    std::array<std::uint32_t, 8> observed_grid_indices{};
    std::array<std::uint8_t, 128> observed_sign_indices{};
    std::array<std::uint8_t, 8> observed_sign_states{};
    std::uint16_t observed_scales = 0u;
    bool generated_bytes_match = true;
    for (std::uint32_t block = 0; block < block_count; ++block) {
        const block_iq3_xxs& weight = weights[block];
        for (std::uint32_t part = 0; part < 8u; ++part) {
            const std::uint32_t scale_index = block * 8u + part;
            const std::uint32_t expected_scale =
                (scale_index * 7u + 3u) & 0x0fu;
            const std::uint32_t aux_offset = 64u + 4u * part;
            const std::uint32_t aux32 =
                std::uint32_t(weight.qs[aux_offset + 0u]) |
                (std::uint32_t(weight.qs[aux_offset + 1u]) << 8u) |
                (std::uint32_t(weight.qs[aux_offset + 2u]) << 16u) |
                (std::uint32_t(weight.qs[aux_offset + 3u]) << 24u);
            const std::uint32_t scale = aux32 >> 28u;
            generated_bytes_match &= scale == expected_scale;
            observed_scales |= static_cast<std::uint16_t>(1u << scale);
            for (std::uint32_t code_slot = 0; code_slot < 4u;
                 ++code_slot) {
                const std::uint32_t sign_slot =
                    block * 32u + part * 4u + code_slot;
                const std::uint32_t expected_sign =
                    (sign_slot * 37u + 0x35u) & 0x7fu;
                const std::uint32_t sign_index =
                    (aux32 >> (7u * code_slot)) & 0x7fu;
                generated_bytes_match &= sign_index == expected_sign;
                observed_sign_indices[sign_index] = 1u;
                const std::uint32_t parity =
                    std::popcount(sign_index) & 1u;
                const std::uint32_t signs =
                    sign_index ^ (parity << 7u);
                for (std::uint32_t component = 0; component < 8u;
                     ++component) {
                    observed_sign_states[component] |=
                        (signs & (1u << component)) != 0u ? 0x02u : 0x01u;
                }
            }
            for (std::uint32_t grid_slot = 0; grid_slot < 8u; ++grid_slot) {
                const std::uint32_t expected_grid =
                    block * 64u + part * 8u + grid_slot;
                const std::uint32_t grid_index =
                    weight.qs[8u * part + grid_slot];
                generated_bytes_match &= grid_index == expected_grid;
                observed_grid_indices[grid_index / 32u] |=
                    1u << (grid_index % 32u);
            }
        }
    }
    check(generated_bytes_match,
          "IQ3_XXS generated bytes expose every index, sign and scale field");
    bool all_grid_indices_seen = true;
    for (const std::uint32_t observed : observed_grid_indices) {
        all_grid_indices_seen &= observed == 0xffffffffu;
    }
    check(all_grid_indices_seen,
          "IQ3_XXS generated sweep covers all 256 grid indices");
    bool all_sign_indices_seen = true;
    for (const std::uint8_t observed : observed_sign_indices) {
        all_sign_indices_seen &= observed == 1u;
    }
    check(all_sign_indices_seen,
          "IQ3_XXS generated sweep covers all 128 sign indices");
    bool both_component_signs_seen = true;
    for (const std::uint8_t observed : observed_sign_states) {
        both_component_signs_seen &= observed == 0x03u;
    }
    check(both_component_signs_seen,
          "IQ3_XXS generated sweep covers both signs at all eight components");
    check(observed_scales == 0xffffu,
          "IQ3_XXS generated sweep covers all 16 scale nibbles");

    bool completed = true;
    for (std::uint32_t batch = 0; batch < n_in; batch += nc) {
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        for (std::uint32_t column = 0; column < nc; ++column) {
            inputs[static_cast<std::size_t>(column) * n_in + batch + column] =
                127.0f;
        }
        completed &= test_iq3_xxs_shape(
            context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
            inputs, weights);
    }
    return completed;
}

bool run_iq3_xxs_fixed_packed_golden(Context& context,
                                    std::mt19937_64& rng,
                                    const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t nc = 8u;

    block_iq3_xxs weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    std::copy(twin::kIQ3XXSMoeIndices.begin(),
              twin::kIQ3XXSMoeIndices.end(), weight.qs + 56u);
    std::copy(twin::kIQ3XXSMoeAux.begin(), twin::kIQ3XXSMoeAux.end(),
              weight.qs + 92u);
    const std::array<block_iq3_xxs, 1> weights = {weight};
    bool completed = true;
    for (std::uint32_t batch = 0; batch < 4u; ++batch) {
        std::array<float, n_in * nc> inputs{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t position =
                twin::kIQ3XXSMoeOneHotPositions[batch * nc + column];
            inputs[static_cast<std::size_t>(column) * n_in + position] =
                127.0f;
        }
        completed &= test_iq3_xxs_shape(
            context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
            inputs, weights,
            std::span<const float>(twin::kIQ3XXSMoeOneHotAnswers.data() +
                                       batch * nc,
                                   nc));
    }
    return completed;
}

bool run_iq3_xxs_signed_division_fixture(Context& context,
                                         std::mt19937_64& rng,
                                         const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t nc = 2u;
    block_iq3_xxs weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    weight.qs[0] = 5u;  // component 0 of grid 5 is 62.
    weight.qs[67] = 0x10u;  // scale 1; sign index 0.
    const std::array<block_iq3_xxs, 1> weights = {weight};
    std::array<float, n_in * nc> inputs{};
    inputs[0] = 127.0f;
    inputs[n_in] = -127.0f;
    constexpr std::array<float, nc> known_answer = {5905.0f, -5905.0f};
    return test_iq3_xxs_shape(
        context, n_in, 1u, nc, nc, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_iq3_xxs_padded_fixture(Context& context, std::mt19937_64& rng,
                                const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 3u;
    constexpr std::uint32_t nc = 4u;
    constexpr std::uint32_t live_rows = 2u;
    std::array<block_iq3_xxs, n_out> weights{};
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 2.0f};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        weights[row].d = strata::kernels::f16_from_f32(weight_scales[row]);
        weights[row].qs[0] = 5u;
        weights[row].qs[67] = 0x10u;
    }
    std::array<float, n_in * nc> inputs{};
    inputs[0] = 127.0f;
    inputs[1] = 127.0f;
    inputs[n_in] = 127.0f;
    inputs[n_in + 1u] = 127.0f;
    inputs[2u * n_in] = -127.0f;
    inputs[3u * n_in + 1u] = -127.0f;
    constexpr std::array<float, n_out * nc> known_answer = {
        3524.0f, 7048.0f, 14096.0f,
        3524.0f, 7048.0f, 14096.0f,
        3524.0f, 7048.0f, 14096.0f,
        3524.0f, 7048.0f, 14096.0f,
    };
    return test_iq3_xxs_shape(
        context, n_in, n_out, nc, live_rows, twin::InputSet::fixture, rng,
        options, inputs, weights, known_answer);
}

bool run_iq3_xxs_fma_golden(Context& context, std::mt19937_64& rng,
                            const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9u, 13u};
    constexpr std::array<std::uint32_t, 2> expected_bits = {
        0x45fbc643u, 0x46440c05u};
    constexpr std::array<float, 9> scales9 = {
        0.130503539f, 2.09138784f, 2.95873839f, 2.47694827f, 0.317397426f,
        0.546569387f, 1.92889927f, 2.44278875f, 0.166486383f};
    constexpr std::array<std::uint8_t, 9> grids9 = {
        202u, 61u, 163u, 184u, 5u, 161u, 165u, 194u, 40u};
    constexpr std::array<std::uint8_t, 9> ls9 = {
        5u, 13u, 15u, 7u, 11u, 10u, 8u, 3u, 10u};
    constexpr std::array<float, 13> scales13 = {
        2.59982809f, 0.95108646f, 1.60765085f, 0.264913803f,
        1.69767525f, 2.6836436f, 1.79056702f, 0.135110662f,
        1.69256776f, 2.70743131f, 0.778015597f, 1.26746434f, 1.6546217f};
    constexpr std::array<std::uint8_t, 13> grids13 = {
        168u, 175u, 255u, 144u, 105u, 175u, 241u, 148u, 48u, 80u, 40u,
        211u, 247u};
    constexpr std::array<std::uint8_t, 13> ls13 = {
        3u, 14u, 7u, 15u, 12u, 13u, 4u, 0u, 10u, 2u, 3u, 11u, 8u};

    // 3.7f produces a non-unit Q8_1 half scale while each nonzero code remains 127.
    constexpr float activation_value = 3.7f;
    bool completed = true;
    for (std::size_t test = 0; test < block_counts.size(); ++test) {
        const std::uint32_t blocks = block_counts[test];
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq3_xxs> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq3_xxs& weight = weights[block];
            const float d = test == 0u ? scales9[block] : scales13[block];
            const std::uint8_t grid = test == 0u ? grids9[block] : grids13[block];
            const std::uint8_t scale = test == 0u ? ls9[block] : ls13[block];
            weight.d = strata::kernels::f16_from_f32(d);
            weight.qs[0] = grid;
            weight.qs[67] = static_cast<std::uint8_t>(scale << 4u);
            inputs[static_cast<std::size_t>(block) * 256u] = activation_value;
        }
        const float known = std::bit_cast<float>(expected_bits[test]);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq3_xxs_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq3_xxs_long_k_golden(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63u, 64u, 65u};
    bool completed = true;
    for (const std::uint32_t blocks : block_counts) {
        const std::uint32_t n_in = 256u * blocks;
        std::vector<block_iq3_xxs> weights(blocks);
        std::vector<float> inputs(n_in, 0.0f);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq3_xxs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(1.0f);
            weight.qs[0] = 5u;
            weight.qs[67] = 0x10u;
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        const float known = 5905.0f * static_cast<float>(blocks);
        const std::array<float, 1> known_answer = {known};
        completed &= test_iq3_xxs_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq3_xxs_multi_eq_single(Context& context, std::mt19937_64& rng,
                                const twin::Options& options) {
    constexpr std::array<std::uint32_t, 4> widths = {
        256u, 1024u, 2304u, 16384u};
    constexpr std::array<std::uint32_t, 3> column_counts = {2u, 4u, 8u};
    constexpr std::uint32_t n_out = 5u;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::size_t blocks_per_row = n_in / 256u;
        std::vector<block_iq3_xxs> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::size_t block = 0; block < weights.size(); ++block) {
            block_iq3_xxs& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(
                0.25f + 0.125f * static_cast<float>(block % 7u));
            for (std::uint32_t part = 0; part < 8u; ++part) {
                std::uint32_t aux32 = 0u;
                for (std::uint32_t code_slot = 0; code_slot < 4u;
                     ++code_slot) {
                    const std::uint32_t sign_index =
                        (block * 0x39u + part * 0x5bu +
                         code_slot * 0x25u + 0x17u) & 0x7fu;
                    aux32 |= sign_index << (7u * code_slot);
                }
                const std::uint32_t scale =
                    (block * 31u + part * 47u + 0x59u) & 0x0fu;
                aux32 |= scale << 28u;
                for (std::uint32_t grid_slot = 0; grid_slot < 8u;
                     ++grid_slot) {
                    weight.qs[8u * part + grid_slot] =
                        static_cast<std::uint8_t>(
                            (block * 0x2du + part * 0x53u +
                             grid_slot * 0x67u + 0x35u) & 0xffu);
                }
                for (std::uint32_t byte = 0; byte < 4u; ++byte) {
                    weight.qs[64u + 4u * part + byte] =
                        static_cast<std::uint8_t>(aux32 >> (8u * byte));
                }
            }
        }

        std::vector<float> inputs(static_cast<std::size_t>(n_in) * 8u);
        std::string err;
        inputs = twin::synthesize_floats(
            twin::InputSet::fixture, inputs.size(), rng, err);
        if (!err.empty()) {
            check(false, "IQ3_XXS mv_multi_eq_single input synthesis", err);
            return false;
        }
        for (const std::uint32_t nc : column_counts) {
            std::vector<float> multi;
            completed &= test_iq3_xxs_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                rng, options,
                std::span<const float>(inputs.data(),
                                       static_cast<std::size_t>(n_in) * nc),
                weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ3_XXS multi-column result available");
                completed = false;
                continue;
            }
            for (std::uint32_t column = 0; column < nc; ++column) {
                std::vector<float> single;
                completed &= test_iq3_xxs_shape(
                    context, n_in, n_out, 1u, 1u,
                    twin::InputSet::fixture, rng, options,
                    std::span<const float>(
                        inputs.data() + static_cast<std::size_t>(column) * n_in,
                        n_in),
                    weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ3_XXS single-column result available");
                    completed = false;
                    continue;
                }
                const auto multi_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch =
                    twin::compare_bytes(multi_bytes, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ3_XXS mv_multi_eq_single",
                                         options, column, *mismatch);
                }
                check(!mismatch,
                      "IQ3_XXS NC column equals single-column dispatch");
            }
        }
    }

    std::array<block_iq3_xxs, 1> base_weight{};
    base_weight[0].d = strata::kernels::f16_from_f32(1.0f);
    base_weight[0].qs[67] = 0x10u;
    std::array<block_iq3_xxs, 1> changed_weight = base_weight;
    base_weight[0].qs[0] = 4u;
    changed_weight[0].qs[0] = 5u;
    std::array<float, 256> control_input{};
    control_input[0] = 127.0f;
    std::vector<float> base;
    std::vector<float> changed;
    completed &= test_iq3_xxs_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, base_weight, {}, &base);
    completed &= test_iq3_xxs_shape(
        context, 256u, 1u, 1u, 1u, twin::InputSet::fixture, rng, options,
        control_input, changed_weight, {}, &changed);
    if (base.size() == 1u && changed.size() == 1u) {
        check(std::bit_cast<std::uint32_t>(base[0]) !=
                  std::bit_cast<std::uint32_t>(changed[0]),
              "IQ3_XXS mv_multi_eq_single negative control changes output");
    } else {
        check(false, "IQ3_XXS negative-control results available");
        completed = false;
    }
    return completed;
}

bool run_iq3_xxs_set(Context& context, twin::InputSet set,
                     const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x3a3a8u);
    constexpr std::array<std::uint32_t, 14> widths = {
        256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq3_xxs_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_iq3_xxs_generated_decoder_sweep(
            context, rng, options);
        completed &= run_iq3_xxs_fixed_packed_golden(
            context, rng, options);
        completed &= run_iq3_xxs_signed_division_fixture(
            context, rng, options);
        completed &= run_iq3_xxs_padded_fixture(context, rng, options);
        completed &= run_iq3_xxs_fma_golden(context, rng, options);
        completed &= run_iq3_xxs_long_k_golden(context, rng, options);
        completed &= run_iq3_xxs_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_iq3_s_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                  const twin::Options& options) {
    constexpr std::uint32_t n_in = 2048u;
    constexpr std::uint32_t n_out = 1u;
    constexpr std::uint32_t nc = 8u;
    constexpr std::array<std::uint8_t, 64> raw_qs_low = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
        0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
        0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
        0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f};
    constexpr std::array<std::uint8_t, 8> raw_qh_clear = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    constexpr std::array<std::uint8_t, 8> raw_qh_set = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    constexpr std::array<std::uint8_t, 32> raw_signs = {
        0x01, 0x23, 0x45, 0x67, 0x01, 0x23, 0x45, 0x67,
        0x01, 0x23, 0x45, 0x67, 0x01, 0x23, 0x45, 0x67,
        0x01, 0x23, 0x45, 0x67, 0x01, 0x23, 0x45, 0x67,
        0x01, 0x23, 0x45, 0x67, 0x01, 0x23, 0x45, 0x67};
    constexpr std::array<std::uint8_t, 4> raw_scales_low = {
        0x10, 0x32, 0x54, 0x76};
    constexpr std::array<std::uint8_t, 4> raw_scales_high = {
        0x98, 0xba, 0xdc, 0xfe};
    std::array<block_iq3_s, 8> weights{};
    for (std::uint32_t block = 0; block < weights.size(); ++block) {
        block_iq3_s& weight = weights[block];
        weight.d = strata::kernels::f16_from_f32(
            0.5f * static_cast<float>(block % 4u + 1u));
        const std::uint32_t low_base = (block % 4u) * 64u;
        for (std::uint32_t i = 0; i < raw_qs_low.size(); ++i) {
            weight.qs[i] = static_cast<std::uint8_t>(low_base + raw_qs_low[i]);
        }
        const auto& high_bits = block < 4u ? raw_qh_clear : raw_qh_set;
        std::copy(high_bits.begin(), high_bits.end(), weight.qh);
        for (std::uint32_t i = 0; i < raw_signs.size(); ++i) {
            weight.signs[i] = static_cast<std::uint8_t>(
                raw_signs[i] ^ (block < 4u ? 0u : 0xffu));
        }
        const auto& packed_scales =
            block % 2u == 0u ? raw_scales_low : raw_scales_high;
        std::copy(packed_scales.begin(), packed_scales.end(), weight.scales);
    }

    std::array<std::uint32_t, 16> observed_codebooks{};
    std::array<std::uint8_t, 16> observed_qh_states{};
    std::array<std::uint8_t, 8> observed_sign_states{};
    std::uint16_t observed_scales = 0;
    bool raw_bytes_match = true;
    for (std::uint32_t block = 0; block < weights.size(); ++block) {
        const block_iq3_s& weight = weights[block];
        for (std::uint32_t position = 0; position < 256u; ++position) {
            const std::uint32_t ib = position / 32u;
            const std::uint32_t il = (position % 32u) / 8u;
            const std::uint32_t pos = position % 8u;
            const std::uint32_t g = pos / 4u;
            const std::uint32_t component = pos % 4u;
            const std::uint32_t qs_index = 8u * ib + 2u * il + g;
            const std::uint32_t expected_low =
                (block % 4u) * 64u + raw_qs_low[qs_index];
            const std::uint32_t actual_low = weight.qs[qs_index];
            const std::uint32_t qh_bit =
                (static_cast<std::uint32_t>(weight.qh[ib]) >>
                 (2u * il + g)) & 1u;
            const std::uint32_t expected_high = block < 4u ? 0u : 1u;
            raw_bytes_match &= actual_low == expected_low &&
                               qh_bit == expected_high;
            const std::uint32_t code = actual_low | (qh_bit << 8u);
            observed_codebooks[code / 32u] |= 1u << (code % 32u);
            observed_qh_states[2u * ib + qh_bit] |=
                static_cast<std::uint8_t>(1u << (2u * il + g));

            const std::uint32_t sign_byte =
                static_cast<std::uint32_t>(weight.signs[4u * ib + il]);
            const bool negative = ((sign_byte >> pos) & 1u) != 0u;
            observed_sign_states[pos] |=
                static_cast<std::uint8_t>(negative ? 0x02u : 0x01u);
            const std::uint32_t scale_byte = weight.scales[position / 64u];
            const std::uint32_t shift = 4u * ((position / 32u) & 1u);
            const std::uint32_t scale =
                (scale_byte >> shift) & 0x0fu;
            observed_scales |= static_cast<std::uint16_t>(1u << scale);
            (void)component;
        }
    }
    check(raw_bytes_match,
          "IQ3_S literal qh/qs bytes decode with every expected bit");
    bool all_codebooks_seen = true;
    for (const std::uint32_t observed : observed_codebooks) {
        all_codebooks_seen &= observed == 0xffffffffu;
    }
    check(all_codebooks_seen, "IQ3_S raw bytes cover all 512 codebook indices");
    bool all_qh_bits_seen = true;
    for (const std::uint8_t observed : observed_qh_states) {
        all_qh_bits_seen &= observed == 0xffu;
    }
    check(all_qh_bits_seen, "IQ3_S raw bytes exercise all qh bit positions");
    bool both_signs_seen = true;
    for (const std::uint8_t observed : observed_sign_states) {
        both_signs_seen &= observed == 0x03u;
    }
    check(both_signs_seen, "IQ3_S raw signs exercise both signs at each position");
    check(observed_scales == 0xffffu,
          "IQ3_S raw scale bytes cover all 16 scale nibbles");

    bool completed = true;
    for (std::uint32_t batch = 0; batch < n_in; batch += nc) {
        std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
        std::array<float, nc> known_answer{};
        for (std::uint32_t column = 0; column < nc; ++column) {
            const std::uint32_t value_index = batch + column;
            const std::uint32_t block = value_index / 256u;
            const std::uint32_t position = value_index % 256u;
            const block_iq3_s& weight = weights[block];
            inputs[static_cast<std::size_t>(column) * n_in + value_index] =
                127.0f;
            const std::uint32_t ib = position / 32u;
            const std::uint32_t il = (position % 32u) / 8u;
            const std::uint32_t pos = position % 8u;
            const std::uint32_t g = pos / 4u;
            const std::uint32_t component = pos % 4u;
            const std::uint32_t qs_index = 8u * ib + 2u * il + g;
            const std::uint32_t qh_bit =
                (static_cast<std::uint32_t>(weight.qh[ib]) >>
                 (2u * il + g)) & 1u;
            const std::uint32_t code =
                static_cast<std::uint32_t>(weight.qs[qs_index]) |
                (qh_bit << 8u);
            const int magnitude = static_cast<int>(
                (iq3s_grid[code] >> (8u * component)) & 0xffu);
            const int sign = ((weight.signs[4u * ib + il] >> pos) & 1u) != 0u
                                 ? -1
                                 : 1;
            const std::uint32_t scale_byte = weight.scales[ib / 2u];
            const std::uint32_t scale =
                1u + 2u * ((scale_byte >> (4u * (ib % 2u))) & 0x0fu);
            const float d = strata::kernels::f32_from_f16(weight.d);
            known_answer[column] =
                std::fma(d, static_cast<float>(sign * magnitude *
                                                static_cast<int>(scale) * 127),
                         0.0f);
        }
        completed &= test_iq3_s_shape(
            context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq3_s_fma_golden(Context& context, std::mt19937_64& rng,
                          const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9u, 13u};
    // Independently derived with a scalar lane/warp calculator, not the IQ3_S oracle:
    // 0x48a33700 for 9 blocks and 0x48b6e750 for 13 blocks.
    constexpr std::array<std::uint32_t, 2> answer_bits = {
        0x48a33700u, 0x48b6e750u};
    bool completed = true;
    for (std::size_t shape = 0; shape < block_counts.size(); ++shape) {
        const std::uint32_t blocks = block_counts[shape];
        const std::uint32_t n_in = 256u * blocks;
        std::vector<float> inputs(n_in, 127.0f);
        std::vector<block_iq3_s> weights(blocks);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_iq3_s& weight = weights[block];
            const bool active =
                block == 0u || block == 4u || block + 1u == blocks;
            const float d = block == 0u ? 0.13f
                            : block == 4u ? 0.27f
                            : block + 1u == blocks ? 0.31f
                                                   : 0.0f;
            weight.d = strata::kernels::f16_from_f32(active ? d : 0.0f);
            std::fill_n(weight.qs, 64u, std::uint8_t{0});
            std::fill_n(weight.qh, 8u, std::uint8_t{0});
            std::fill_n(weight.signs, 32u, std::uint8_t{0});
            std::fill_n(weight.scales, 4u, std::uint8_t{0});
            for (std::uint32_t part = 0; part < 8u; ++part) {
                const std::uint32_t nibble =
                    (block * 7u + part * 3u + 5u) % 16u;
                weight.scales[part / 2u] |= static_cast<std::uint8_t>(
                    nibble << (4u * (part % 2u)));
            }
        }
        const float answer = std::bit_cast<float>(answer_bits[shape]);
        const std::array<float, 1> known_answer = {answer};
        completed &= test_iq3_s_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq3_s_long_k_golden(Context& context, std::mt19937_64& rng,
                             const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63u, 64u, 65u};
    constexpr std::array<float, 3> known_answers = {
        1024128.0f, 1040384.0f, 1056640.0f};
    bool completed = true;
    for (std::size_t shape = 0; shape < block_counts.size(); ++shape) {
        const std::uint32_t blocks = block_counts[shape];
        const std::uint32_t n_in = blocks * 256u;
        std::vector<float> inputs(n_in, 127.0f);
        std::vector<block_iq3_s> weights(blocks);
        for (block_iq3_s& weight : weights) {
            weight.d = strata::kernels::f16_from_f32(0.5f);
            std::fill_n(weight.qs, 64u, std::uint8_t{0});
            std::fill_n(weight.qh, 8u, std::uint8_t{0});
            std::fill_n(weight.signs, 32u, std::uint8_t{0});
            std::fill_n(weight.scales, 4u, std::uint8_t{0});
        }
        const std::array<float, 1> known_answer = {known_answers[shape]};
        completed &= test_iq3_s_shape(
            context, n_in, 1u, 1u, 1u, twin::InputSet::fixture, rng,
            options, inputs, weights, known_answer);
    }
    return completed;
}

bool run_iq3_s_fixture(Context& context, std::mt19937_64& rng,
                       const twin::Options& options) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 3u;
    constexpr std::uint32_t nc = 4u;
    std::array<block_iq3_s, n_out> weights{};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        block_iq3_s& weight = weights[row];
        weight.d = strata::kernels::f16_from_f32(
            0.5f * static_cast<float>(row + 1u));
        if (row == 0u) {
            std::copy(twin::kIQ3SMoeQs.begin(), twin::kIQ3SMoeQs.end(),
                      weight.qs);
            std::copy(twin::kIQ3SMoeQh.begin(), twin::kIQ3SMoeQh.end(),
                      weight.qh);
            std::copy(twin::kIQ3SMoeSigns.begin(), twin::kIQ3SMoeSigns.end(),
                      weight.signs);
            std::copy(twin::kIQ3SMoeScales.begin(),
                      twin::kIQ3SMoeScales.end(), weight.scales);
        } else {
            for (std::uint32_t i = 0; i < 64u; ++i) {
                weight.qs[i] = static_cast<std::uint8_t>(
                    (i * 29u + row * 37u + 7u) & 0xffu);
            }
            for (std::uint32_t i = 0; i < 8u; ++i) {
                weight.qh[i] = static_cast<std::uint8_t>(
                    i * 31u + row * 17u + 0x53u);
            }
            for (std::uint32_t i = 0; i < 32u; ++i) {
                weight.signs[i] = static_cast<std::uint8_t>(
                    i * 43u + row * 19u + 0x96u);
            }
            for (std::uint32_t i = 0; i < 4u; ++i) {
                weight.scales[i] = static_cast<std::uint8_t>(
                    i * 0x35u + row * 0x17u + 0x8bu);
            }
        }
    }
    std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
    for (std::uint32_t column = 0; column < nc; ++column) {
        inputs[static_cast<std::size_t>(column) * n_in] =
            127.0f * static_cast<float>(column + 1u);
    }
    std::array<float, n_out * nc> known_answer{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        const std::uint32_t input_column = column < n_out ? column : 0u;
        for (std::uint32_t row = 0; row < n_out; ++row) {
            const block_iq3_s& weight = weights[row];
            const std::uint32_t code =
                static_cast<std::uint32_t>(weight.qs[0]) |
                ((static_cast<std::uint32_t>(weight.qh[0]) & 1u) << 8u);
            const int magnitude = static_cast<int>(iq3s_grid[code] & 0xffu);
            const int sign = (weight.signs[0] & 1u) != 0u ? -1 : 1;
            const std::uint32_t scale =
                1u + 2u * (static_cast<std::uint32_t>(weight.scales[0]) & 0x0fu);
            const float d = strata::kernels::f32_from_f16(weight.d);
            known_answer[column * n_out + row] = std::fma(
                d * static_cast<float>(input_column + 1u),
                static_cast<float>(sign * magnitude * static_cast<int>(scale) *
                                   127),
                0.0f);
        }
    }
    return test_iq3_s_shape(context, n_in, n_out, nc, 3u,
                            twin::InputSet::fixture, rng, options, inputs,
                            weights, known_answer);
}

bool run_iq3_s_multi_eq_single(Context& context, std::mt19937_64& rng,
                               const twin::Options& options) {
    constexpr std::array<std::uint32_t, 14> widths = {
        256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> weight_scales = {
        0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3u;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<block_iq3_s> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                block_iq3_s& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                weight.d = strata::kernels::f16_from_f32(
                    weight_scales[(row + 2u * block) % weight_scales.size()]);
                for (std::uint32_t i = 0; i < 64u; ++i) {
                    weight.qs[i] = static_cast<std::uint8_t>(
                        (i * 37u + row * 13u + block * 19u + 5u) & 0xffu);
                }
                for (std::uint32_t i = 0; i < 8u; ++i) {
                    weight.qh[i] = static_cast<std::uint8_t>(
                        i * 53u + row * 29u + block * 17u + 0x81u);
                }
                for (std::uint32_t i = 0; i < 32u; ++i) {
                    weight.signs[i] = static_cast<std::uint8_t>(
                        i * 31u + row * 23u + block * 11u + 0x49u);
                }
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    weight.scales[i] = static_cast<std::uint8_t>(
                        i * 0x35u + row * 0x17u + block * 0x29u + 0x13u);
                }
            }
        }

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + 3u * column) %
                                      input_pattern.size()];
                }
            }
            std::vector<float> multi;
            completed &= test_iq3_s_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "IQ3_S multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                const std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_iq3_s_shape(
                    context, n_in, n_out, 1u, 1u, twin::InputSet::fixture,
                    rng, options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "IQ3_S single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch =
                    twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("IQ3_S mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "IQ3_S NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() +
                              static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 2560u && nc == 4u) {
                auto flipped_weights = weights;
                flipped_weights[0].qh[0] ^= 1u;
                std::vector<float> flipped_multi;
                completed &= test_iq3_s_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped_multi);
                if (flipped_multi.size() == singles.size()) {
                    std::vector<float> flipped_singles(singles.size());
                    bool singles_available = true;
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const auto begin =
                            inputs.begin() +
                            static_cast<std::size_t>(column) * n_in;
                        const std::vector<float> single_input(begin,
                                                             begin + n_in);
                        std::vector<float> single;
                        completed &= test_iq3_s_shape(
                            context, n_in, n_out, 1u, 1u,
                            twin::InputSet::fixture, rng, options,
                            single_input, flipped_weights, {}, &single);
                        if (single.size() != n_out) {
                            singles_available = false;
                            break;
                        }
                        std::copy(single.begin(), single.end(),
                                  flipped_singles.begin() +
                                      static_cast<std::size_t>(column) *
                                          n_out);
                    }
                    check(singles_available,
                          "IQ3_S flipped-weight single-column results available");
                    if (singles_available) {
                        const auto multi_bytes = std::span<const std::uint8_t>(
                            reinterpret_cast<const std::uint8_t*>(
                                flipped_multi.data()),
                            flipped_multi.size() * sizeof(float));
                        const auto single_bytes = std::span<const std::uint8_t>(
                            reinterpret_cast<const std::uint8_t*>(
                                flipped_singles.data()),
                            flipped_singles.size() * sizeof(float));
                        check(!twin::compare_bytes(multi_bytes, single_bytes),
                              "IQ3_S flipped weights preserve NC parity");
                        check(twin::compare_bytes(
                                  std::span<const std::uint8_t>(
                                      reinterpret_cast<const std::uint8_t*>(
                                          multi.data()),
                                      multi.size() * sizeof(float)),
                                  multi_bytes).has_value(),
                              "IQ3_S qh-bit negative control changes output");
                        auto divergent_singles = flipped_singles;
                        const auto first_bits =
                            std::bit_cast<std::uint32_t>(divergent_singles[0]);
                        divergent_singles[0] =
                            std::bit_cast<float>(first_bits ^ 1u);
                        const auto divergent_bytes =
                            std::span<const std::uint8_t>(
                                reinterpret_cast<const std::uint8_t*>(
                                    divergent_singles.data()),
                                divergent_singles.size() * sizeof(float));
                        check(twin::compare_bytes(multi_bytes, divergent_bytes)
                                  .has_value(),
                              "IQ3_S NC parity comparator detects one-bit divergence");
                    }
                } else {
                    check(false,
                          "IQ3_S flipped multi-column result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_iq3_s_set(Context& context, twin::InputSet set,
                   const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x1a3b5u);
    constexpr std::array<std::uint32_t, 14> widths = {
        256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_iq3_s_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_iq3_s_fixture(context, rng, options);
        completed &= run_iq3_s_raw_decoder_golden(context, rng, options);
        completed &= run_iq3_s_fma_golden(context, rng, options);
        completed &= run_iq3_s_long_k_golden(context, rng, options);
        completed &= run_iq3_s_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_q3_k_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                 const twin::Options& options) {
    constexpr std::array<std::int32_t, 16> signed_scales = {
        -32, -31, -24, -16, -8, -1, 3, 1,
        2, 7, 15, 16, 24, 30, 31, -30};
    constexpr std::array<std::uint8_t, 32> raw_hmask = {
        0x3c, 0x87, 0xf0, 0x1e, 0xc3, 0x78, 0x0f, 0xe1,
        0x1e, 0xc3, 0x78, 0x0f, 0xe1, 0x3c, 0x87, 0xf0,
        0x0f, 0xe1, 0x3c, 0x87, 0xf0, 0x1e, 0xc3, 0x78,
        0x87, 0xf0, 0x1e, 0xc3, 0x78, 0x0f, 0xe1, 0x3c};
    constexpr std::array<std::uint8_t, 64> raw_qs = {
        0x4e, 0x39, 0xe4, 0x93, 0x4e, 0x39, 0xe4, 0x93,
        0x93, 0x4e, 0x39, 0xe4, 0x93, 0x4e, 0x39, 0xe4,
        0xe4, 0x93, 0x4e, 0x39, 0xe4, 0x93, 0x4e, 0x39,
        0x39, 0xe4, 0x93, 0x4e, 0x39, 0xe4, 0x93, 0x4e,
        0x4e, 0x39, 0xe4, 0x93, 0x4e, 0x39, 0xe4, 0x93,
        0x93, 0x4e, 0x39, 0xe4, 0x93, 0x4e, 0x39, 0xe4,
        0xe4, 0x93, 0x4e, 0x39, 0xe4, 0x93, 0x4e, 0x39,
        0x39, 0xe4, 0x93, 0x4e, 0x39, 0xe4, 0x93, 0x4e};
    constexpr std::array<std::uint8_t, 12> raw_scales = {
        0x20, 0x71, 0xf8, 0x00, 0x88, 0xef,
        0xf3, 0x21, 0xe4, 0xe4, 0xe8, 0x39};
    block_q3_K weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);
    std::copy(raw_hmask.begin(), raw_hmask.end(), weight.hmask);
    std::copy(raw_qs.begin(), raw_qs.end(), weight.qs);
    std::copy(raw_scales.begin(), raw_scales.end(), weight.scales);
    const std::array<block_q3_K, 1> weights = {weight};

    bool completed = true;
    for (std::uint32_t batch = 0; batch < 256; batch += 8) {
        std::array<float, 256 * 8> inputs{};
        std::array<float, 8> known_answer{};
        for (std::uint32_t column = 0; column < 8; ++column) {
            const std::uint32_t position = batch + column;
            inputs[static_cast<std::size_t>(column) * 256u + position] = 127.0f;
            const std::uint32_t raw_code =
                (position * 3u + position / 8u + (position / 32u) * 5u + 2u) % 8u;
            const std::int32_t q = static_cast<std::int32_t>(raw_code) - 4;
            known_answer[column] = static_cast<float>(
                q * signed_scales[position / 16u] * 127);
        }
        completed &= test_q3_k_shape(
            context, 256, 1, 8, 8, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q3_k_fixture(Context& context, std::mt19937_64& rng,
                      const twin::Options& options) {
    constexpr std::uint32_t n_in = 256;
    constexpr std::uint32_t n_out = 3;
    constexpr std::uint32_t nc = 4;
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 1.5f};
    constexpr std::array<std::int32_t, n_out> code_values = {-4, -3, 2};
    constexpr std::array<std::int32_t, n_out> scale_values = {2, 0, 3};
    std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
    std::vector<block_q3_K> weights(n_out);
    for (std::uint32_t row = 0; row < n_out; ++row) {
        block_q3_K& weight = weights[row];
        std::fill_n(weight.hmask, 32, std::uint8_t{0});
        std::fill_n(weight.qs, 64, std::uint8_t{0});
        std::fill_n(weight.scales, 12, std::uint8_t{0});
        weight.d = strata::kernels::f16_from_f32(weight_scales[row]);
        set_q3_k_scale(weight, 0, scale_values[row]);
        set_q3_k_code(weight, 0, code_values[row]);
    }
    for (std::uint32_t column = 0; column < nc; ++column) {
        inputs[static_cast<std::size_t>(column) * n_in] =
            127.0f * static_cast<float>(column + 1u);
    }
    std::array<float, n_out * nc> known_answer{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        const std::uint32_t input_column = column < 3u ? column : 0u;
        for (std::uint32_t row = 0; row < n_out; ++row) {
            known_answer[column * n_out + row] =
                weight_scales[row] * static_cast<float>(scale_values[row] *
                                                        code_values[row] * 127) *
                static_cast<float>(input_column + 1u);
        }
    }
    return test_q3_k_shape(
        context, n_in, n_out, nc, 3, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_q3_k_fma_golden(Context& context, std::mt19937_64& rng,
                         const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> widths = {2304, 3328};
    constexpr std::array<float, 2> known_answers = {
        -33.268939971923828125f, -33.2689361572265625f};
    bool completed = true;
    for (std::size_t shape = 0; shape < widths.size(); ++shape) {
        const std::uint32_t n_in = widths[shape];
        const std::uint32_t blocks_per_row = n_in / 256u;
        const std::uint32_t final_block = blocks_per_row - 1u;
        std::vector<float> inputs(n_in, 0.0f);
        std::fill_n(inputs.begin(), 256, 0.1f);
        std::fill_n(
            inputs.begin() + static_cast<std::size_t>(final_block) * 256u,
            256, 0.1f);
        std::vector<block_q3_K> weights(blocks_per_row);
        for (block_q3_K& weight : weights) {
            std::fill_n(weight.hmask, 32, std::uint8_t{0});
            std::fill_n(weight.qs, 64, std::uint8_t{0});
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            weight.d = strata::kernels::f16_from_f32(0.0f);
        }
        for (const std::uint32_t block : {0u, final_block}) {
            block_q3_K& weight = weights[block];
            weight.d = strata::kernels::f16_from_f32(block == 0u ? 0.125f : 0.1f);
            const std::int32_t scale = block == 0u ? 1 : 2;
            for (std::uint32_t index = 0; index < 16; ++index) {
                set_q3_k_scale(weight, index, scale);
            }
        }
        completed &= test_q3_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights,
            std::span<const float>(&known_answers[shape], 1));
    }
    return completed;
}

bool run_q3_k_long_k_golden(Context& context, std::mt19937_64& rng,
                            const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> widths = {16128, 16384, 16640};
    constexpr std::array<float, 3> known_answers = {
        -4000.5f, -3873.5f, -3810.0f};
    bool completed = true;
    for (std::size_t shape = 0; shape < widths.size(); ++shape) {
        const std::uint32_t n_in = widths[shape];
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<float> inputs(n_in, 0.0f);
        std::vector<block_q3_K> weights(blocks_per_row);
        for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
            block_q3_K& weight = weights[block];
            std::fill_n(weight.hmask, 32, std::uint8_t{0});
            std::fill_n(weight.qs, 64, std::uint8_t{0});
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            weight.d = strata::kernels::f16_from_f32(0.125f);
            set_q3_k_scale(
                weight, 0,
                static_cast<std::int32_t>(block % 7u) - 2);
            inputs[static_cast<std::size_t>(block) * 256u] = 127.0f;
        }
        completed &= test_q3_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights,
            std::span<const float>(&known_answers[shape], 1));
    }
    return completed;
}

bool run_q3_k_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 12> widths = {
        256, 512, 768, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<block_q3_K> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                block_q3_K& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                std::fill_n(weight.hmask, 32, std::uint8_t{0});
                std::fill_n(weight.qs, 64, std::uint8_t{0});
                std::fill_n(weight.scales, 12, std::uint8_t{0});
                weight.d = strata::kernels::f16_from_f32(
                    scales[(row + block) % scales.size()]);
                for (std::uint32_t index = 0; index < 16; ++index) {
                    const std::int32_t scale =
                        static_cast<std::int32_t>(
                            (row * 3u + block * 7u + index * 5u + 1u) % 63u) - 31;
                    set_q3_k_scale(weight, index, scale);
                }
                for (std::uint32_t index = 0; index < 256; ++index) {
                    const std::int32_t code =
                        static_cast<std::int32_t>(
                            (row + block * 5u + index * 3u + 3u) % 8u) - 4;
                    set_q3_k_code(weight, index, code);
                }
            }
        }
        weights[0].d = strata::kernels::f16_from_f32(1.0f);

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_q3_k_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q3_K multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q3_k_shape(
                    context, n_in, n_out, 1, 1, twin::InputSet::fixture, rng,
                    options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q3_K single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q3_K mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "Q3_K NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 2560 && nc == 4) {
                auto flipped_weights = weights;
                flipped_weights[0].qs[0] ^= 1u;
                std::vector<float> flipped;
                completed &= test_q3_k_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped);
                if (flipped.size() == singles.size()) {
                    const auto flipped_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(flipped.data()),
                        flipped.size() * sizeof(float));
                    const auto singles_bytes = std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t*>(singles.data()),
                        singles.size() * sizeof(float));
                    check(twin::compare_bytes(flipped_bytes, singles_bytes).has_value(),
                          "Q3_K multi-column negative control detects code-bit flip");
                } else {
                    check(false, "Q3_K negative-control result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_q4_k_fixture(Context& context, std::mt19937_64& rng,
                      const twin::Options& options) {
    constexpr std::uint32_t n_in = 256;
    constexpr std::uint32_t n_out = 3;
    constexpr std::uint32_t nc = 4;
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 1.5f};
    constexpr std::array<float, n_out> minimum_scales = {0.25f, 0.5f, 0.75f};
    constexpr std::array<std::uint32_t, 8> scales = {
        1, 7, 15, 31, 32, 47, 63, 21};
    constexpr std::array<std::uint32_t, 8> mins = {
        63, 48, 33, 17, 1, 8, 16, 31};
    std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
    std::vector<block_q4_K> weights(n_out);
    std::array<std::array<std::uint32_t, 8>, n_out> row_codes{};
    std::array<std::array<std::uint32_t, 8>, n_out> row_scales{};
    std::array<std::array<std::uint32_t, 8>, n_out> row_mins{};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        block_q4_K& weight = weights[row];
        std::fill_n(weight.scales, 12, std::uint8_t{0});
        std::fill_n(weight.qs, 128, std::uint8_t{0});
        weight.dm =
            static_cast<std::uint32_t>(
                strata::kernels::f16_from_f32(weight_scales[row])) |
            (static_cast<std::uint32_t>(
                 strata::kernels::f16_from_f32(minimum_scales[row])) << 16u);
        for (std::uint32_t group = 0; group < 8u; ++group) {
            row_scales[row][group] = (scales[group] + 3u * row) % 64u;
            row_mins[row][group] = (mins[group] + 5u * row) % 64u;
            row_codes[row][group] = (row * 7u + group * 11u + 3u) % 16u;
            set_q4_k_scale_min(weight, group, row_scales[row][group],
                               row_mins[row][group]);
            for (std::uint32_t local = 0; local < 32u; ++local) {
                set_q4_k_code(
                    weight, group * 32u + local,
                    (row_codes[row][group] + 5u * local) % 16u);
            }
        }
    }
    for (std::uint32_t column = 0; column < nc; ++column) {
        for (std::uint32_t group = 0; group < 8u; ++group) {
            inputs[static_cast<std::size_t>(column) * n_in + group * 32u] =
                127.0f * static_cast<float>(column + 1u);
        }
    }

    std::array<float, n_out * nc> known_answer{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        const std::uint32_t input_column = column < n_out ? column : 0u;
        for (std::uint32_t row = 0; row < n_out; ++row) {
            float sum = 0.0f;
            for (std::uint32_t group = 0; group < 8u; ++group) {
                const float dot = static_cast<float>(
                    127 * static_cast<int>(row_codes[row][group]) *
                    static_cast<int>(row_scales[row][group]));
                const float min_dot = static_cast<float>(
                    127 * static_cast<int>(row_mins[row][group]));
                const float d8 = static_cast<float>(input_column + 1u);
                const float scaled_dot =
                    std::fma(d8, dot, 0.0f);
                const float scaled_min =
                    std::fma(d8, min_dot, 0.0f);
                sum += std::fma(
                    -minimum_scales[row], scaled_min,
                    weight_scales[row] * scaled_dot);
            }
            known_answer[column * n_out + row] = sum;
        }
    }
    return test_q4_k_shape(
        context, n_in, n_out, nc, 3, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_q4_k_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                 const twin::Options& options) {
    constexpr std::array<std::uint32_t, 8> raw_scales = {
        1, 8, 16, 31, 32, 47, 63, 15};
    constexpr std::array<std::uint32_t, 8> raw_mins = {
        0, 1, 16, 31, 63, 48, 17, 2};
    constexpr std::array<std::uint8_t, 12> packed_scales = {
        0x81, 0x88, 0xd0, 0x1f, 0xc0, 0xc1,
        0x50, 0x1f, 0xf0, 0x0f, 0x1f, 0x2f};
    constexpr std::array<std::uint8_t, 128> raw_qs = {
        0x51, 0x62, 0x73, 0x84, 0x95, 0xa6, 0xb7, 0xc8,
        0xea, 0xfb, 0x0c, 0x1d, 0x2e, 0x3f, 0x40, 0x51,
        0x73, 0x84, 0x95, 0xa6, 0xb7, 0xc8, 0xd9, 0xea,
        0x0c, 0x1d, 0x2e, 0x3f, 0x40, 0x51, 0x62, 0x73,
        0xd9, 0xea, 0xfb, 0x0c, 0x1d, 0x2e, 0x3f, 0x40,
        0x62, 0x73, 0x84, 0x95, 0xa6, 0xb7, 0xc8, 0xd9,
        0xfb, 0x0c, 0x1d, 0x2e, 0x3f, 0x40, 0x51, 0x62,
        0x84, 0x95, 0xa6, 0xb7, 0xc8, 0xd9, 0xea, 0xfb,
        0x51, 0x62, 0x73, 0x84, 0x95, 0xa6, 0xb7, 0xc8,
        0xea, 0xfb, 0x0c, 0x1d, 0x2e, 0x3f, 0x40, 0x51,
        0x73, 0x84, 0x95, 0xa6, 0xb7, 0xc8, 0xd9, 0xea,
        0x0c, 0x1d, 0x2e, 0x3f, 0x40, 0x51, 0x62, 0x73,
        0xd9, 0xea, 0xfb, 0x0c, 0x1d, 0x2e, 0x3f, 0x40,
        0x62, 0x73, 0x84, 0x95, 0xa6, 0xb7, 0xc8, 0xd9,
        0xfb, 0x0c, 0x1d, 0x2e, 0x3f, 0x40, 0x51, 0x62,
        0x84, 0x95, 0xa6, 0xb7, 0xc8, 0xd9, 0xea, 0xfb};
    block_q4_K weight{};
    weight.dm =
        static_cast<std::uint32_t>(strata::kernels::f16_from_f32(1.0f)) |
        (static_cast<std::uint32_t>(
             strata::kernels::f16_from_f32(0.5f)) << 16u);
    std::copy(packed_scales.begin(), packed_scales.end(), weight.scales);
    std::copy(raw_qs.begin(), raw_qs.end(), weight.qs);
    const std::array<block_q4_K, 1> weights = {weight};

    std::array<std::uint16_t, 8> observed_codes{};
    for (std::uint32_t position = 0; position < 256u; ++position) {
        const std::uint32_t code =
            (position + position / 8u + 1u) % 16u;
        observed_codes[position / 32u] |=
            static_cast<std::uint16_t>(1u << code);
        if (position + 8u < 256u) {
            check(code != ((position + 8u + (position + 8u) / 8u + 1u) % 16u),
                  "Q4_K raw code changes across adjacent lane groups");
        }
        if (position % 32u < 16u) {
            const std::uint32_t other = position + 16u;
            check(code != ((other + other / 8u + 1u) % 16u),
                  "Q4_K raw low/high nibble codes are distinct");
        }
    }
    for (const std::uint16_t mask : observed_codes) {
        check(mask == 0xffffu, "Q4_K raw group covers all 16 codes");
    }

    bool completed = true;
    for (std::uint32_t batch = 0; batch < 256u; batch += 8u) {
        std::array<float, 256 * 8> inputs{};
        std::array<float, 8> known_answer{};
        for (std::uint32_t column = 0; column < 8u; ++column) {
            const std::uint32_t position = batch + column;
            const std::uint32_t group = position / 32u;
            const std::uint32_t code =
                (position + position / 8u + 1u) % 16u;
            inputs[static_cast<std::size_t>(column) * 256u + position] = 127.0f;
            const float dot = static_cast<float>(
                127 * static_cast<int>(code) *
                static_cast<int>(raw_scales[group]));
            const float min_dot =
                static_cast<float>(127 * static_cast<int>(raw_mins[group]));
            known_answer[column] =
                std::fma(-0.5f, min_dot, dot);
        }
        completed &= test_q4_k_shape(
            context, 256, 1, 8, 8, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q4_k_long_k_golden(Context& context, std::mt19937_64& rng,
                            const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63, 64, 65};
    constexpr std::array<float, 3> known_answers = {
        9729216.0f, 9883648.0f, 10038080.0f};
    bool completed = true;
    for (std::size_t shape = 0; shape < block_counts.size(); ++shape) {
        const std::uint32_t n_in = 256u * block_counts[shape];
        std::vector<float> inputs(n_in, 127.0f);
        std::vector<block_q4_K> weights(block_counts[shape]);
        for (block_q4_K& weight : weights) {
            weight.dm =
                static_cast<std::uint32_t>(
                    strata::kernels::f16_from_f32(0.5f)) |
                (static_cast<std::uint32_t>(
                     strata::kernels::f16_from_f32(0.25f)) << 16u);
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
            for (std::uint32_t group = 0; group < 8u; ++group) {
                set_q4_k_scale_min(weight, group, 2u, 1u);
                for (std::uint32_t local = 0; local < 32u; ++local) {
                    set_q4_k_code(weight, group * 32u + local, 5u);
                }
            }
        }
        const std::array<float, 1> known_answer = {known_answers[shape]};
        completed &= test_q4_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q4_k_fma_golden(Context& context, std::mt19937_64& rng,
                         const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9, 13};
    // Fixed results independently derived by a scalar lane/warp calculator
    // using the quantized Q8_1 blocks and raw Q4_K bytes, not the Q4_K oracle.
    constexpr std::array<float, 2> known_answers = {
        0x1.7f220ep+10f, 0x1.0de62cp+11f};
    bool completed = true;
    for (std::size_t shape = 0; shape < block_counts.size(); ++shape) {
        const std::uint32_t blocks = block_counts[shape];
        const std::uint32_t n_in = 256u * blocks;
        std::vector<float> inputs(n_in, 0.0f);
        std::vector<block_q4_K> weights(blocks);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_q4_K& weight = weights[block];
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
            const bool active = block == 0u || block + 1u == blocks;
            const float d = block == 0u ? 0.13f : 0.27f;
            const float dmin = block == 0u ? 0.31f : 0.17f;
            weight.dm =
                static_cast<std::uint32_t>(strata::kernels::f16_from_f32(
                    active ? d : 0.0f)) |
                (static_cast<std::uint32_t>(
                     strata::kernels::f16_from_f32(active ? dmin : 0.0f))
                 << 16u);
            for (std::uint32_t group = 0; group < 8u; ++group) {
                set_q4_k_scale_min(
                    weight, group,
                    (group * 13u + block * 7u + 9u) % 64u,
                    (group * 23u + block * 11u + 5u) % 64u);
            }
            for (std::uint32_t index = 0; index < 256u; ++index) {
                set_q4_k_code(
                    weight, index,
                    (index * 7u + block * 3u + index / 32u * 5u + 2u) % 16u);
                if (active) {
                    const std::int32_t centered =
                        static_cast<std::int32_t>((index * 17u + block * 13u) %
                                                  97u) -
                        48;
                    inputs[static_cast<std::size_t>(block) * 256u + index] =
                        0.1f + static_cast<float>(centered) * 0.00037f;
                }
            }
        }
        const std::array<float, 1> known_answer = {known_answers[shape]};
        completed &= test_q4_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q4_k_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 12> widths = {
        256, 512, 768, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::array<float, 5> minimum_scales = {
        0.0625f, 0.125f, 0.25f, 0.5f, 1.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<block_q4_K> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                block_q4_K& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                std::fill_n(weight.scales, 12, std::uint8_t{0});
                std::fill_n(weight.qs, 128, std::uint8_t{0});
                weight.dm =
                    static_cast<std::uint32_t>(strata::kernels::f16_from_f32(
                        scales[(row + block) % scales.size()])) |
                    (static_cast<std::uint32_t>(
                         strata::kernels::f16_from_f32(
                             minimum_scales[(row + 2u * block) %
                                            minimum_scales.size()])) << 16u);
                for (std::uint32_t group = 0; group < 8u; ++group) {
                    set_q4_k_scale_min(
                        weight, group,
                        (row * 7u + block * 11u + group * 13u + 1u) % 64u,
                        (row * 17u + block * 5u + group * 19u + 3u) % 64u);
                }
                for (std::uint32_t index = 0; index < 256u; ++index) {
                    set_q4_k_code(
                        weight, index,
                        (row * 3u + block * 7u + index * 5u + 1u) % 16u);
                }
            }
        }

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }

            std::vector<float> multi;
            completed &= test_q4_k_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q4_K multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                const std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q4_k_shape(
                    context, n_in, n_out, 1, 1, twin::InputSet::fixture, rng,
                    options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q4_K single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q4_K mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "Q4_K NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 2560u && nc == 4u) {
                auto flipped_weights = weights;
                flipped_weights[0].qs[0] ^= 1u;
                std::vector<float> flipped_multi;
                completed &= test_q4_k_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped_multi);
                if (flipped_multi.size() == singles.size()) {
                    std::vector<float> flipped_singles(singles.size());
                    bool singles_available = true;
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const auto begin =
                            inputs.begin() + static_cast<std::size_t>(column) * n_in;
                        const std::vector<float> single_input(begin, begin + n_in);
                        std::vector<float> single;
                        completed &= test_q4_k_shape(
                            context, n_in, n_out, 1, 1,
                            twin::InputSet::fixture, rng, options,
                            single_input, flipped_weights, {}, &single);
                        if (single.size() != n_out) {
                            singles_available = false;
                            break;
                        }
                        std::copy(
                            single.begin(), single.end(),
                            flipped_singles.begin() +
                                static_cast<std::size_t>(column) * n_out);
                    }
                    check(singles_available,
                          "Q4_K flipped-weight single-column results available");
                    if (singles_available) {
                        const auto flipped_multi_bytes =
                            std::span<const std::uint8_t>(
                                reinterpret_cast<const std::uint8_t*>(
                                    flipped_multi.data()),
                                flipped_multi.size() * sizeof(float));
                        const auto flipped_singles_bytes =
                            std::span<const std::uint8_t>(
                                reinterpret_cast<const std::uint8_t*>(
                                    flipped_singles.data()),
                                flipped_singles.size() * sizeof(float));
                        const auto flipped_mismatch = twin::compare_bytes(
                            flipped_multi_bytes, flipped_singles_bytes);
                        check(!flipped_mismatch,
                              "Q4_K flipped weights preserve NC parity");

                        auto divergent_singles = flipped_singles;
                        const auto first_bits =
                            std::bit_cast<std::uint32_t>(divergent_singles[0]);
                        divergent_singles[0] =
                            std::bit_cast<float>(first_bits ^ 1u);
                        const auto divergent_bytes =
                            std::span<const std::uint8_t>(
                                reinterpret_cast<const std::uint8_t*>(
                                    divergent_singles.data()),
                                divergent_singles.size() * sizeof(float));
                        check(twin::compare_bytes(flipped_multi_bytes,
                                                  divergent_bytes)
                                  .has_value(),
                              "Q4_K NC parity comparator detects one-bit divergence");
                    }
                } else {
                    check(false, "Q4_K flipped multi-column result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_q5_k_fixture(Context& context, std::mt19937_64& rng,
                      const twin::Options& options) {
    constexpr std::uint32_t n_in = 256;
    constexpr std::uint32_t n_out = 3;
    constexpr std::uint32_t nc = 4;
    constexpr std::array<float, n_out> weight_scales = {0.5f, 1.0f, 1.5f};
    constexpr std::array<float, n_out> minimum_scales = {0.25f, 0.5f, 0.75f};
    constexpr std::array<std::uint32_t, 8> scales = {
        1, 7, 15, 31, 32, 47, 63, 21};
    constexpr std::array<std::uint32_t, 8> mins = {
        63, 48, 33, 17, 1, 8, 16, 31};
    std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc, 0.0f);
    std::vector<block_q5_K> weights(n_out);
    std::array<std::array<std::uint32_t, 8>, n_out> row_codes{};
    std::array<std::array<std::uint32_t, 8>, n_out> row_scales{};
    std::array<std::array<std::uint32_t, 8>, n_out> row_mins{};
    for (std::uint32_t row = 0; row < n_out; ++row) {
        block_q5_K& weight = weights[row];
        std::fill_n(weight.scales, 12, std::uint8_t{0});
        std::fill_n(weight.qh, 32, std::uint8_t{0});
        std::fill_n(weight.qs, 128, std::uint8_t{0});
        weight.dm =
            static_cast<std::uint32_t>(
                strata::kernels::f16_from_f32(weight_scales[row])) |
            (static_cast<std::uint32_t>(
                 strata::kernels::f16_from_f32(minimum_scales[row])) << 16u);
        for (std::uint32_t group = 0; group < 8u; ++group) {
            row_scales[row][group] = (scales[group] + 3u * row) % 64u;
            row_mins[row][group] = (mins[group] + 5u * row) % 64u;
            row_codes[row][group] = (row * 11u + group * 13u + 3u) % 32u;
            set_q5_k_scale_min(weight, group, row_scales[row][group],
                               row_mins[row][group]);
            for (std::uint32_t local = 0; local < 32u; ++local) {
                set_q5_k_code(
                    weight, group * 32u + local,
                    (row_codes[row][group] + 5u * local) % 32u);
            }
        }
    }
    for (std::uint32_t column = 0; column < nc; ++column) {
        for (std::uint32_t group = 0; group < 8u; ++group) {
            inputs[static_cast<std::size_t>(column) * n_in + group * 32u] =
                127.0f * static_cast<float>(column + 1u);
        }
    }

    std::array<float, n_out * nc> known_answer{};
    for (std::uint32_t column = 0; column < nc; ++column) {
        const std::uint32_t input_column = column < n_out ? column : 0u;
        for (std::uint32_t row = 0; row < n_out; ++row) {
            float sum = 0.0f;
            for (std::uint32_t group = 0; group < 8u; ++group) {
                const float dot = static_cast<float>(
                    127 * static_cast<int>(row_codes[row][group]) *
                    static_cast<int>(row_scales[row][group]));
                const float min_dot = static_cast<float>(
                    127 * static_cast<int>(row_mins[row][group]));
                const float d8 = static_cast<float>(input_column + 1u);
                const float scaled_dot = std::fma(d8, dot, 0.0f);
                const float scaled_min = std::fma(d8, min_dot, 0.0f);
                sum += std::fma(
                    -minimum_scales[row], scaled_min,
                    weight_scales[row] * scaled_dot);
            }
            known_answer[column * n_out + row] = sum;
        }
    }
    return test_q5_k_shape(
        context, n_in, n_out, nc, 3, twin::InputSet::fixture, rng, options,
        inputs, weights, known_answer);
}

bool run_q5_k_raw_decoder_golden(Context& context, std::mt19937_64& rng,
                                 const twin::Options& options) {
    constexpr std::array<std::uint32_t, 8> raw_scales = {
        1, 8, 16, 31, 32, 47, 63, 15};
    constexpr std::array<std::uint32_t, 8> raw_mins = {
        0, 1, 16, 31, 63, 48, 17, 2};
    constexpr std::array<std::uint8_t, 12> packed_scales = {
        0x81, 0x88, 0xd0, 0x1f, 0xc0, 0xc1,
        0x50, 0x1f, 0xf0, 0x0f, 0x1f, 0x2f};
    constexpr std::array<std::uint8_t, 32> raw_qh = {
        0xe0, 0xf8, 0x3f, 0x07, 0xc1, 0xf0, 0x7c, 0x1f,
        0x03, 0xe0, 0xf8, 0x3e, 0x0f, 0x81, 0xf0, 0x7c,
        0x1f, 0x07, 0xc0, 0xf8, 0x3e, 0x0f, 0x83, 0xe0,
        0xfc, 0x1f, 0x07, 0xc1, 0xf0, 0x7e, 0x0f, 0x83};
    constexpr std::array<std::uint8_t, 128> raw_qs = {
        0x52, 0xc9, 0x30, 0xa7, 0x1e, 0x85, 0xfc, 0x63,
        0xda, 0x41, 0xb8, 0x2f, 0x96, 0x0d, 0x74, 0xeb,
        0x52, 0xc9, 0x30, 0xa7, 0x1e, 0x85, 0xfc, 0x63,
        0xda, 0x41, 0xb8, 0x2f, 0x96, 0x0d, 0x74, 0xeb,
        0xb8, 0x2f, 0x96, 0x0d, 0x74, 0xeb, 0x52, 0xc9,
        0x30, 0xa7, 0x1e, 0x85, 0xfc, 0x63, 0xda, 0x41,
        0xb8, 0x2f, 0x96, 0x0d, 0x74, 0xeb, 0x52, 0xc9,
        0x30, 0xa7, 0x1e, 0x85, 0xfc, 0x63, 0xda, 0x41,
        0x1e, 0x85, 0xfc, 0x63, 0xda, 0x41, 0xb8, 0x2f,
        0x96, 0x0d, 0x74, 0xeb, 0x52, 0xc9, 0x30, 0xa7,
        0x1e, 0x85, 0xfc, 0x63, 0xda, 0x41, 0xb8, 0x2f,
        0x96, 0x0d, 0x74, 0xeb, 0x52, 0xc9, 0x30, 0xa7,
        0x74, 0xeb, 0x52, 0xc9, 0x30, 0xa7, 0x1e, 0x85,
        0xfc, 0x63, 0xda, 0x41, 0xb8, 0x2f, 0x96, 0x0d,
        0x74, 0xeb, 0x52, 0xc9, 0x30, 0xa7, 0x1e, 0x85,
        0xfc, 0x63, 0xda, 0x41, 0xb8, 0x2f, 0x96, 0x0d};
    block_q5_K weight{};
    weight.dm =
        static_cast<std::uint32_t>(strata::kernels::f16_from_f32(1.0f)) |
        (static_cast<std::uint32_t>(
             strata::kernels::f16_from_f32(0.5f)) << 16u);
    std::copy(packed_scales.begin(), packed_scales.end(), weight.scales);
    std::copy(raw_qh.begin(), raw_qh.end(), weight.qh);
    std::copy(raw_qs.begin(), raw_qs.end(), weight.qs);
    const std::array<block_q5_K, 1> weights = {weight};

    std::array<std::uint32_t, 8> observed_codes{};
    std::array<std::uint8_t, 8> observed_high_bits{};
    for (std::uint32_t position = 0; position < 256u; ++position) {
        const std::uint32_t group = position / 32u;
        const std::uint32_t local = position % 32u;
        const std::uint32_t code =
            (position * 7u + group * 3u + 2u) % 32u;
        const std::uint32_t qs_index = 32u * (group / 2u) + local;
        const std::uint32_t nibble_shift = 4u * (group % 2u);
        const std::uint32_t low_code =
            (raw_qs[qs_index] >> nibble_shift) & 0x0fu;
        const std::uint32_t high_bit = (raw_qh[local] >> group) & 1u;
        const std::uint32_t raw_code = low_code | (high_bit << 4u);
        check(raw_code == code,
              "Q5_K literal packed bytes decode at every position");
        observed_codes[group] |= 1u << raw_code;
        observed_high_bits[group] |=
            static_cast<std::uint8_t>(1u << high_bit);
        if (position + 8u < 256u) {
            const std::uint32_t other = position + 8u;
            check(code !=
                      ((other * 7u + (other / 32u) * 3u + 2u) % 32u),
                  "Q5_K raw code changes across adjacent lane groups");
        }
        if (position % 32u < 16u) {
            const std::uint32_t other = position + 16u;
            check(code !=
                      ((other * 7u + (other / 32u) * 3u + 2u) % 32u),
                  "Q5_K raw low/high nibble codes are distinct");
        }
    }
    for (std::uint32_t group = 0; group < 8u; ++group) {
        check(observed_codes[group] == 0xffffffffu,
              "Q5_K raw group covers all 32 codes");
        check(observed_high_bits[group] == 0x03u,
              "Q5_K raw group covers both high-bit values");
    }

    bool completed = true;
    for (std::uint32_t batch = 0; batch < 256u; batch += 8u) {
        std::array<float, 256 * 8> inputs{};
        std::array<float, 8> known_answer{};
        for (std::uint32_t column = 0; column < 8u; ++column) {
            const std::uint32_t position = batch + column;
            const std::uint32_t group = position / 32u;
            const std::uint32_t code =
                (position * 7u + group * 3u + 2u) % 32u;
            inputs[static_cast<std::size_t>(column) * 256u + position] =
                127.0f;
            const float dot = static_cast<float>(
                127 * static_cast<int>(code) *
                static_cast<int>(raw_scales[group]));
            const float min_dot =
                static_cast<float>(127 * static_cast<int>(raw_mins[group]));
            known_answer[column] = std::fma(-0.5f, min_dot, dot);
        }
        completed &= test_q5_k_shape(
            context, 256, 1, 8, 8, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q5_k_long_k_golden(Context& context, std::mt19937_64& rng,
                            const twin::Options& options) {
    constexpr std::array<std::uint32_t, 3> block_counts = {63, 64, 65};
    constexpr std::array<float, 3> known_answers = {
        42501312.0f, 43175936.0f, 43850560.0f};
    bool completed = true;
    for (std::size_t shape = 0; shape < block_counts.size(); ++shape) {
        const std::uint32_t n_in = 256u * block_counts[shape];
        std::vector<float> inputs(n_in, 127.0f);
        std::vector<block_q5_K> weights(block_counts[shape]);
        for (block_q5_K& weight : weights) {
            weight.dm =
                static_cast<std::uint32_t>(
                    strata::kernels::f16_from_f32(0.5f)) |
                (static_cast<std::uint32_t>(
                     strata::kernels::f16_from_f32(0.25f)) << 16u);
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            std::fill_n(weight.qh, 32, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
            for (std::uint32_t group = 0; group < 8u; ++group) {
                set_q5_k_scale_min(weight, group, 2u, 1u);
                for (std::uint32_t local = 0; local < 32u; ++local) {
                    set_q5_k_code(weight, group * 32u + local, 21u);
                }
            }
        }
        const std::array<float, 1> known_answer = {known_answers[shape]};
        completed &= test_q5_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q5_k_multi_eq_single(Context& context, std::mt19937_64& rng,
                              const twin::Options& options) {
    constexpr std::array<std::uint32_t, 12> widths = {
        256, 512, 768, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 3> column_counts = {2, 4, 8};
    constexpr std::array<float, 8> input_pattern = {
        1.0f, -0.75f, 0.5f, -0.25f, 0.125f, 0.625f, -0.5f, 0.875f};
    constexpr std::array<float, 5> scales = {
        0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    constexpr std::array<float, 5> minimum_scales = {
        0.0625f, 0.125f, 0.25f, 0.5f, 1.0f};
    constexpr std::uint32_t n_out = 3;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        const std::uint32_t blocks_per_row = n_in / 256u;
        std::vector<block_q5_K> weights(
            static_cast<std::size_t>(n_out) * blocks_per_row);
        for (std::uint32_t row = 0; row < n_out; ++row) {
            for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
                block_q5_K& weight =
                    weights[static_cast<std::size_t>(row) * blocks_per_row + block];
                std::fill_n(weight.scales, 12, std::uint8_t{0});
                std::fill_n(weight.qh, 32, std::uint8_t{0});
                std::fill_n(weight.qs, 128, std::uint8_t{0});
                weight.dm =
                    static_cast<std::uint32_t>(strata::kernels::f16_from_f32(
                        scales[(row + block) % scales.size()])) |
                    (static_cast<std::uint32_t>(
                         strata::kernels::f16_from_f32(
                             minimum_scales[(row + 2u * block) %
                                            minimum_scales.size()])) << 16u);
                for (std::uint32_t group = 0; group < 8u; ++group) {
                    set_q5_k_scale_min(
                        weight, group,
                        (row * 7u + block * 11u + group * 13u + 1u) % 64u,
                        (row * 17u + block * 5u + group * 19u + 3u) % 64u);
                }
                for (std::uint32_t index = 0; index < 256u; ++index) {
                    set_q5_k_code(
                        weight, index,
                        (row * 3u + block * 7u + index * 11u + 1u) % 32u);
                }
            }
        }

        for (const std::uint32_t nc : column_counts) {
            std::vector<float> inputs(static_cast<std::size_t>(n_in) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                for (std::uint32_t k = 0; k < n_in; ++k) {
                    inputs[static_cast<std::size_t>(column) * n_in + k] =
                        127.0f * static_cast<float>(column + 1u) *
                        input_pattern[(k + column) % input_pattern.size()];
                }
            }
            std::vector<float> multi;
            completed &= test_q5_k_shape(
                context, n_in, n_out, nc, nc, twin::InputSet::fixture, rng,
                options, inputs, weights, {}, &multi);
            if (multi.size() != static_cast<std::size_t>(n_out) * nc) {
                check(false, "Q5_K multi-column result available for property");
                completed = false;
                continue;
            }

            std::vector<float> singles(static_cast<std::size_t>(n_out) * nc);
            for (std::uint32_t column = 0; column < nc; ++column) {
                const auto begin =
                    inputs.begin() + static_cast<std::size_t>(column) * n_in;
                const std::vector<float> single_input(begin, begin + n_in);
                std::vector<float> single;
                completed &= test_q5_k_shape(
                    context, n_in, n_out, 1, 1, twin::InputSet::fixture, rng,
                    options, single_input, weights, {}, &single);
                if (single.size() != n_out) {
                    check(false, "Q5_K single-column result available for property");
                    completed = false;
                    continue;
                }
                const auto multi_column = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(
                        multi.data() + static_cast<std::size_t>(column) * n_out),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto single_bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(single.data()),
                    static_cast<std::size_t>(n_out) * sizeof(float));
                const auto mismatch = twin::compare_bytes(multi_column, single_bytes);
                if (mismatch) {
                    twin::report_failure("Q5_K mv_multi_eq_single", options,
                                         column, *mismatch);
                }
                check(!mismatch, "Q5_K NC column equals single-column dispatch");
                std::copy(single.begin(), single.end(),
                          singles.begin() + static_cast<std::size_t>(column) * n_out);
            }

            if (n_in == 2560u && nc == 4u) {
                auto flipped_weights = weights;
                flipped_weights[0].qh[0] ^= 1u;
                std::vector<float> flipped_multi;
                completed &= test_q5_k_shape(
                    context, n_in, n_out, nc, nc, twin::InputSet::fixture,
                    rng, options, inputs, flipped_weights, {}, &flipped_multi);
                if (flipped_multi.size() == singles.size()) {
                    std::vector<float> flipped_singles(singles.size());
                    bool singles_available = true;
                    for (std::uint32_t column = 0; column < nc; ++column) {
                        const auto begin =
                            inputs.begin() + static_cast<std::size_t>(column) * n_in;
                        const std::vector<float> single_input(begin, begin + n_in);
                        std::vector<float> single;
                        completed &= test_q5_k_shape(
                            context, n_in, n_out, 1, 1,
                            twin::InputSet::fixture, rng, options,
                            single_input, flipped_weights, {}, &single);
                        if (single.size() != n_out) {
                            singles_available = false;
                            break;
                        }
                        std::copy(single.begin(), single.end(),
                                  flipped_singles.begin() +
                                      static_cast<std::size_t>(column) * n_out);
                    }
                    check(singles_available,
                          "Q5_K flipped-weight single-column results available");
                    if (singles_available) {
                        const auto multi_bytes = std::span<const std::uint8_t>(
                            reinterpret_cast<const std::uint8_t*>(
                                flipped_multi.data()),
                            flipped_multi.size() * sizeof(float));
                        const auto single_bytes = std::span<const std::uint8_t>(
                            reinterpret_cast<const std::uint8_t*>(
                                flipped_singles.data()),
                            flipped_singles.size() * sizeof(float));
                        check(!twin::compare_bytes(multi_bytes, single_bytes),
                              "Q5_K flipped weights preserve NC parity");
                        auto divergent_singles = flipped_singles;
                        const auto first_bits =
                            std::bit_cast<std::uint32_t>(divergent_singles[0]);
                        divergent_singles[0] =
                            std::bit_cast<float>(first_bits ^ 1u);
                        const auto divergent_bytes =
                            std::span<const std::uint8_t>(
                                reinterpret_cast<const std::uint8_t*>(
                                    divergent_singles.data()),
                                divergent_singles.size() * sizeof(float));
                        check(twin::compare_bytes(multi_bytes, divergent_bytes)
                                  .has_value(),
                              "Q5_K NC parity comparator detects one-bit divergence");
                    }
                } else {
                    check(false, "Q5_K flipped multi-column result available");
                    completed = false;
                }
            }
        }
    }
    return completed;
}

bool run_q5_k_fma_golden(Context& context, std::mt19937_64& rng,
                         const twin::Options& options) {
    constexpr std::array<std::uint32_t, 2> block_counts = {9, 13};
    // Independently derived with a scalar lane/warp calculator, not the Q5_K oracle:
    // 0x455ecdce for 9 blocks and 0x4597a411 for 13 blocks.
    constexpr std::array<float, 2> known_answers = {
        0x1.bd9b9cp+11f, 0x1.2f4822p+12f};
    bool completed = true;
    for (std::size_t shape = 0; shape < block_counts.size(); ++shape) {
        const std::uint32_t blocks = block_counts[shape];
        const std::uint32_t n_in = 256u * blocks;
        std::vector<float> inputs(n_in, 0.0f);
        std::vector<block_q5_K> weights(blocks);
        for (std::uint32_t block = 0; block < blocks; ++block) {
            block_q5_K& weight = weights[block];
            std::fill_n(weight.scales, 12, std::uint8_t{0});
            std::fill_n(weight.qh, 32, std::uint8_t{0});
            std::fill_n(weight.qs, 128, std::uint8_t{0});
            const bool active = block == 0u || block + 1u == blocks;
            const float d = block == 0u ? 0.13f : 0.27f;
            const float dmin = block == 0u ? 0.31f : 0.17f;
            weight.dm =
                static_cast<std::uint32_t>(strata::kernels::f16_from_f32(
                    active ? d : 0.0f)) |
                (static_cast<std::uint32_t>(
                     strata::kernels::f16_from_f32(active ? dmin : 0.0f))
                 << 16u);
            for (std::uint32_t group = 0; group < 8u; ++group) {
                set_q5_k_scale_min(
                    weight, group,
                    (group * 13u + block * 7u + 9u) % 64u,
                    (group * 23u + block * 11u + 5u) % 64u);
            }
            for (std::uint32_t index = 0; index < 256u; ++index) {
                set_q5_k_code(
                    weight, index,
                    (index * 11u + block * 3u + index / 32u * 5u + 2u) % 32u);
                if (active) {
                    const std::int32_t centered =
                        static_cast<std::int32_t>((index * 17u + block * 13u) %
                                                  97u) -
                        48;
                    inputs[static_cast<std::size_t>(block) * 256u + index] =
                        0.1f + static_cast<float>(centered) * 0.00037f;
                }
            }
        }
        const std::array<float, 1> known_answer = {known_answers[shape]};
        completed &= test_q5_k_shape(
            context, n_in, 1, 1, 1, twin::InputSet::fixture, rng, options,
            inputs, weights, known_answer);
    }
    return completed;
}

bool run_q5_k_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x5b5b5u);
    constexpr std::array<std::uint32_t, 12> widths = {
        256, 512, 768, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q5_k_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_q5_k_fixture(context, rng, options);
        completed &= run_q5_k_raw_decoder_golden(context, rng, options);
        completed &= run_q5_k_fma_golden(context, rng, options);
        completed &= run_q5_k_long_k_golden(context, rng, options);
        completed &= run_q5_k_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_q4_k_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x4b4b4u);
    constexpr std::array<std::uint32_t, 12> widths = {
        256, 512, 768, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q4_k_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_q4_k_fixture(context, rng, options);
        completed &= run_q4_k_raw_decoder_golden(context, rng, options);
        completed &= run_q4_k_fma_golden(context, rng, options);
        completed &= run_q4_k_long_k_golden(context, rng, options);
        completed &= run_q4_k_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_q3_k_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x3a3b3u);
    constexpr std::array<std::uint32_t, 12> widths = {
        256, 512, 768, 1792, 2048, 2304, 2560, 6144, 10240,
        16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q3_k_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_q3_k_fixture(context, rng, options);
        completed &= run_q3_k_raw_decoder_golden(context, rng, options);
        completed &= run_q3_k_fma_golden(context, rng, options);
        completed &= run_q3_k_long_k_golden(context, rng, options);
        completed &= run_q3_k_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_q6_k_set(Context& context, twin::InputSet set,
                  const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0x6b6b6u);
    constexpr std::array<std::uint32_t, 11> widths = {
        256, 512, 768, 1024, 1280, 2560, 6144, 10240, 16128, 16384, 16640};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    constexpr std::uint32_t n_out = 7;
    bool completed = true;
    for (const std::uint32_t n_in : widths) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_q6_k_shape(
                    context, n_in, n_out, nc, live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_q6_k_fixture(context, rng, options);
        completed &= run_q6_k_raw_decoder_golden(context, rng, options);
        completed &= run_q6_k_raw_layout_golden(context, rng, options);
        completed &= run_q6_k_fma_golden(context, rng, options);
        completed &= run_q6_k_long_k_golden(context, rng, options);
        completed &= run_q6_k_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool run_bf16_set(Context& context, twin::InputSet set, const twin::Options& options) {
    std::mt19937_64 rng(options.seed ^ 0xb16b16u);
    constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 3> shapes = {
        std::pair<std::uint32_t, std::uint32_t>{320, 160},
        {2560, 256}, {10240, 256}};
    constexpr std::array<std::uint32_t, 4> column_counts = {1, 2, 4, 8};
    bool completed = true;
    for (const auto [n_in, block_size] : shapes) {
        for (const std::uint32_t nc : column_counts) {
            for (std::uint32_t live_rows = 1; live_rows <= nc; ++live_rows) {
                completed &= test_bf16_shape(context, n_in, 7, block_size, nc,
                                             live_rows, set, rng, options);
            }
        }
    }
    if (set == twin::InputSet::fixture) {
        constexpr std::uint32_t n_in = 320;
        constexpr std::uint32_t n_out = 3;
        constexpr std::uint32_t nc = 4;
        std::vector<float> fixed_inputs(n_in * nc, 0.0f);
        std::vector<float> weight_values(static_cast<std::size_t>(n_in) * n_out, 0.0f);
        constexpr std::array<std::array<float, 4>, 3> row_weights = {{
            {{1.0f, 1.0f, -1.0f, 0.5f}},
            {{2.0f, -1.0f, 0.5f, 1.0f}},
            {{-1.0f, 0.5f, 1.0f, -2.0f}},
        }};
        constexpr std::array<std::array<float, 4>, 3> columns = {{
            {{2.0f, 3.0f, 4.0f, 6.0f}},
            {{1.0f, -2.0f, 8.0f, 4.0f}},
            {{-3.0f, 5.0f, 2.0f, -1.0f}},
        }};
        for (std::uint32_t row = 0; row < n_out; ++row) {
            std::copy(row_weights[row].begin(), row_weights[row].end(),
                      weight_values.begin() + static_cast<std::size_t>(row) * n_in);
        }
        for (std::uint32_t column = 0; column < columns.size(); ++column) {
            std::copy(columns[column].begin(), columns[column].end(),
                      fixed_inputs.begin() + static_cast<std::size_t>(column) * n_in);
        }
        std::vector<std::uint16_t> fixed_weights(weight_values.size());
        std::transform(weight_values.begin(), weight_values.end(), fixed_weights.begin(),
                       strata::kernels::bf16_from_f32);
        constexpr std::array<float, nc * n_out> known_answer = {
            4.0f, 9.0f, -8.5f, -7.0f, 12.0f, -2.0f,
            -0.5f, -11.0f, 9.5f, 4.0f, 9.0f, -8.5f};
        completed &= test_bf16_shape(context, n_in, n_out, 160, nc, 3,
                                     set, rng, options, fixed_inputs,
                                     fixed_weights, known_answer);
    }
    if (set == twin::InputSet::fixture) {
        completed &= run_bf16_multi_eq_single(context, rng, options);
    }
    return completed;
}

bool wait_for_engine_command(CommandBuffer& command, const std::string& label,
                             std::string& error) {
    if (!command.commit(error)) {
        error = label + " command commit failed: " + error;
        return false;
    }
    if (!command.wait_until_completed(error)) {
        error = label + " command completion failed: " + error;
        return false;
    }
    error.clear();
    return true;
}

BufferRef engine_test_subview(BufferRef backing, std::size_t relative_offset,
                              std::size_t length) {
    if (!backing || relative_offset > backing.length ||
        length > backing.length - relative_offset) {
        return {};
    }
    return backing.slice(relative_offset, length);
}

template <class WeightBlock, class InitializeWeights, class Reference>
bool run_engine_quantized_numeric_case(
    Context& context, const std::string& label, MatVecWeightFormat format,
    std::uint32_t n_in, std::uint32_t weight_block_values,
    std::uint32_t n_out, std::uint32_t compiled_columns,
    std::uint32_t active_columns, InitializeWeights initialize_weights,
    Reference reference, bool use_read_only_encoder = false) {
    if (weight_block_values == 0u || n_in % weight_block_values != 0u) {
        return false;
    }
    const std::size_t weight_blocks_per_row = n_in / weight_block_values;
    std::vector<WeightBlock> weights(
        static_cast<std::size_t>(n_out) * weight_blocks_per_row);
    const std::size_t activation_blocks_per_row = n_in / 32u;
    initialize_weights(weights);

    std::vector<float> input_values(
        static_cast<std::size_t>(compiled_columns) * n_in);
    for (std::size_t index = 0u; index < input_values.size(); ++index) {
        input_values[index] =
            1.0f + static_cast<float>(index % 7u) * 0.015625f;
    }
    std::vector<Q81> activations;
    std::string error;
    check(twin::quantize_q8_1_reference(input_values, n_in,
                                        compiled_columns, activations, error),
          label + " Q8_1 scalar oracle", error);
    if (activations.size() !=
        static_cast<std::size_t>(compiled_columns) *
            activation_blocks_per_row) {
        return false;
    }

    std::vector<float> expected;
    check(reference(weights, activations, n_in, n_out, compiled_columns,
                   active_columns, expected, error),
          label + " mat-vec scalar oracle", error);
    if (expected.size() !=
        static_cast<std::size_t>(compiled_columns) * n_out) {
        return false;
    }
    const std::size_t output_count =
        static_cast<std::size_t>(active_columns) * n_out;
    const std::size_t sink_count =
        static_cast<std::size_t>(compiled_columns - active_columns) * n_out;
    check(std::any_of(expected.begin(), expected.begin() + output_count,
                      [](float value) { return value != 0.0f; }),
          label + " exercises nonzero mat-vec results");

    const std::size_t activation_bytes = activations.size() * sizeof(Q81);
    const std::size_t weight_bytes = weights.size() * sizeof(WeightBlock);
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_written_bytes = sink_count * sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_written_bytes);
    twin::GuardedBuffer activation_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer active_rows_buffer;
    check(activation_buffer.allocate(context, label + " activations",
                                     activation_bytes, error),
          label + " activation allocation", error);
    check(weight_buffer.allocate(context, label + " weights",
                                 weight_bytes, error),
          label + " weight allocation", error);
    check(output_buffer.allocate(context, label + " output", output_bytes,
                                 error),
          label + " output allocation", error);
    check(sink_buffer.allocate(context, label + " sink", sink_bytes, error),
          label + " sink allocation", error);
    check(active_rows_buffer.allocate(context, label + " active rows",
                                      sizeof(active_columns), error),
          label + " active-row allocation", error);
    if (!activation_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !active_rows_buffer.view()) {
        return false;
    }

    std::memcpy(activation_buffer.payload().data(), activations.data(),
                activation_bytes);
    std::memcpy(weight_buffer.payload().data(), weights.data(), weight_bytes);
    std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
    std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
    std::memcpy(active_rows_buffer.payload().data(), &active_columns,
                sizeof(active_columns));
    activation_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    active_rows_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command-buffer allocation", error);
    if (!command) return false;
    bool encoded = false;
    if (use_read_only_encoder) {
        const ReadOnlyMatVecBuffers buffers{
            context.buffers().read_only(activation_buffer.view()),
            context.buffers().read_only(weight_buffer.view()),
            context.buffers().read_only(active_rows_buffer.view()),
            output_buffer.view(), sink_buffer.view()};
        if (format == MatVecWeightFormat::q6_k) {
            encoded = Encoder::encode(
                context, *command, label + " caller-owned read-only segment",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_read_only_matvec(
                        context, encoder, format, n_in, n_out,
                        active_columns, compiled_columns, buffers,
                        encode_error);
                },
                error);
        } else {
            encoded = encode_read_only_matvec(
                context, *command, format, n_in, n_out, active_columns,
                compiled_columns, buffers, error);
        }
    } else {
        const MatVecBuffers buffers{
            activation_buffer.view(), weight_buffer.view(),
            output_buffer.view(), sink_buffer.view(),
            active_rows_buffer.view()};
        encoded = encode_matvec(context, *command, format, n_in, n_out,
                                active_columns, compiled_columns, buffers,
                                error);
    }
    check(encoded, label + " production encoder", error);
    if (!encoded) return false;
    const bool completed =
        wait_for_engine_command(*command, label, error);
    check(completed, label + " command completes", error);
    if (!completed) return false;

    const auto expected_output = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_output);
    check(!output_mismatch, label + " exact nonzero output",
          output_mismatch ? output_mismatch->reason : std::string{});
    if (sink_written_bytes != 0u) {
        const auto expected_sink = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected.data() +
                                                  output_count),
            sink_written_bytes);
        const auto actual_sink = std::span<const std::uint8_t>(
            sink_buffer.payload().data(), sink_written_bytes);
        const auto sink_mismatch =
            twin::compare_bytes(actual_sink, expected_sink);
        check(!sink_mismatch, label + " exact nonzero padded output",
              sink_mismatch ? sink_mismatch->reason : std::string{});
    }
    const std::array<twin::ByteRange, 1> output_written =
        {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written =
        {{{0u, sink_written_bytes}}};
    check(!activation_buffer.first_unwritten_change({}),
          label + " activations and guards remain unchanged");
    check(!weight_buffer.first_unwritten_change({}),
          label + " weights and guards remain unchanged");
    check(!active_rows_buffer.first_unwritten_change({}),
          label + " active-row metadata remains unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " output guards remain unchanged");
    check(!sink_buffer.first_unwritten_change(sink_written),
          label + " sink guards remain unchanged");
    bool late_failure_discarded = true;
    if (use_read_only_encoder && format == MatVecWeightFormat::q6_k) {
        const ReadOnlyMatVecBuffers valid_buffers{
            context.buffers().read_only(activation_buffer.view()),
            context.buffers().read_only(weight_buffer.view()),
            context.buffers().read_only(active_rows_buffer.view()),
            output_buffer.view(), sink_buffer.view()};
        ReadOnlyMatVecBuffers aliased_output = valid_buffers;
        aliased_output.output =
            weight_buffer.view().slice(0u, output_bytes);
        auto alias_command =
            context.make_command_buffer(label + " read-only alias", error);
        check(alias_command != nullptr,
              label + " read-only alias command buffer", error);
        if (!alias_command) return false;
        const bool alias_rejected =
            !encode_read_only_matvec(
                context, *alias_command, format, n_in, n_out,
                active_columns, compiled_columns, aliased_output, error) &&
            error.find("overlap") != std::string::npos &&
            alias_command->unsafe_native_handle() == nullptr;
        check(alias_rejected,
              label + " rejects output alias with read-only weights", error);

        std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
        std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
        ReadOnlyMatVecBuffers short_weights = valid_buffers;
        short_weights.weights =
            valid_buffers.weights.slice(0u, weight_bytes - 1u);
        auto rollback =
            context.make_command_buffer(label + " late failure", error);
        check(rollback != nullptr, label + " late-failure command buffer",
              error);
        if (!rollback) return false;
        const bool accepted = Encoder::encode(
            context, *rollback, label + " late-failure segment",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encode_read_only_matvec(
                        context, encoder, format, n_in, n_out,
                        active_columns, compiled_columns, valid_buffers,
                        encode_error)) {
                    return false;
                }
                return encode_read_only_matvec(
                    context, encoder, format, n_in, n_out, active_columns,
                    compiled_columns, short_weights, encode_error);
            },
            error);
        const bool outputs_unchanged =
            std::all_of(output_buffer.payload().begin(),
                        output_buffer.payload().end(),
                        [](std::uint8_t value) { return value == 0xa5u; }) &&
            std::all_of(sink_buffer.payload().begin(),
                        sink_buffer.payload().end(),
                        [](std::uint8_t value) { return value == 0xa5u; });
        late_failure_discarded =
            !accepted &&
            rollback->unsafe_native_handle() == nullptr &&
            error.find("undersized or misaligned") != std::string::npos &&
            outputs_unchanged;
        check(late_failure_discarded,
              label + " late failure discards prior read-only dispatches",
              error);
        late_failure_discarded &= alias_rejected;
    }
    return !output_mismatch && late_failure_discarded;
}

bool run_engine_bf16_numeric_case(
    Context& context, std::uint32_t n_in, std::uint32_t n_out,
    std::uint32_t compiled_columns, std::uint32_t active_columns,
    std::uint32_t block_size, const std::string& label) {

    std::vector<float> input_values(
        static_cast<std::size_t>(compiled_columns) * n_in);
    for (std::size_t index = 0u; index < input_values.size(); ++index) {
        const int code = static_cast<int>((index * 29u + 11u) % 53u) - 26;
        input_values[index] = static_cast<float>(code) * 0.015625f;
    }
    std::vector<float> weight_values(static_cast<std::size_t>(n_out) * n_in);
    for (std::size_t index = 0u; index < weight_values.size(); ++index) {
        const int code = static_cast<int>((index * 13u + 7u) % 37u) - 18;
        weight_values[index] = static_cast<float>(code) * 0.0078125f;
    }
    std::vector<std::uint16_t> weights(weight_values.size());
    std::transform(weight_values.begin(), weight_values.end(), weights.begin(),
                   strata::kernels::bf16_from_f32);

    std::string error;
    std::vector<float> expected;
    check(twin::bf16_matvec_reference(
              weights, input_values, n_in, n_out, compiled_columns,
              active_columns, block_size, expected, error),
          label + " scalar oracle", error);
    if (expected.size() !=
        static_cast<std::size_t>(compiled_columns) * n_out) {
        return false;
    }
    const std::size_t output_count =
        static_cast<std::size_t>(active_columns) * n_out;
    const std::size_t output_bytes = output_count * sizeof(float);
    const std::size_t sink_written_bytes =
        static_cast<std::size_t>(compiled_columns - active_columns) * n_out *
        sizeof(float);
    const std::size_t sink_bytes =
        std::max<std::size_t>(sizeof(float), sink_written_bytes);
    check(std::any_of(expected.begin(), expected.begin() + output_count,
                      [](float value) { return value != 0.0f; }),
          label + " fixture produces nonzero results");

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer active_rows_buffer;
    check(input_buffer.allocate(context, label + " inputs",
                                input_values.size() * sizeof(float), error),
          label + " input allocation", error);
    check(weight_buffer.allocate(context, label + " weights",
                                 weights.size() * sizeof(std::uint16_t), error),
          label + " weight allocation", error);
    check(output_buffer.allocate(context, label + " output",
                                 output_bytes, error),
          label + " output allocation", error);
    check(sink_buffer.allocate(context, label + " sink", sink_bytes,
                               error),
          label + " sink allocation", error);
    check(active_rows_buffer.allocate(context, label + " active rows",
                                      sizeof(active_columns), error),
          label + " active-row allocation", error);
    if (!input_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !active_rows_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), input_values.data(),
                input_values.size() * sizeof(float));
    std::memcpy(weight_buffer.payload().data(), weights.data(),
                weights.size() * sizeof(std::uint16_t));
    std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
    std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
    std::memcpy(active_rows_buffer.payload().data(), &active_columns,
                sizeof(active_columns));
    input_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    active_rows_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command-buffer allocation", error);
    if (!command) return false;
    const MatVecBuffers buffers{
        input_buffer.view(), weight_buffer.view(), output_buffer.view(),
        sink_buffer.view(), active_rows_buffer.view()};
    const bool encoded =
        encode_matvec(context, *command, MatVecWeightFormat::bf16, n_in,
                      n_out, active_columns, compiled_columns, buffers, error);
    const std::string thread_label =
        "engine BF16 " + std::to_string(block_size) + "-thread";
    check(encoded, thread_label + " production encoder", error);
    if (!encoded) return false;
    const bool completed =
        wait_for_engine_command(*command, label, error);
    check(completed, label + " command completes", error);
    if (!completed) return false;

    const auto expected_output = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_output);
    check(!output_mismatch, thread_label + " exact output",
          output_mismatch ? output_mismatch->reason : std::string{});
    const auto expected_sink = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data() + output_count),
        sink_written_bytes);
    const auto actual_sink = std::span<const std::uint8_t>(
        sink_buffer.payload().data(), sink_written_bytes);
    const auto sink_mismatch =
        twin::compare_bytes(actual_sink, expected_sink);
    check(!sink_mismatch, label + " padded sink",
          sink_mismatch ? sink_mismatch->reason : std::string{});
    const std::array<twin::ByteRange, 1> output_written =
        {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written =
        {{{0u, sink_written_bytes}}};
    check(!input_buffer.first_unwritten_change({}),
          label + " inputs remain unchanged");
    check(!weight_buffer.first_unwritten_change({}),
          label + " weights remain unchanged");
    check(!active_rows_buffer.first_unwritten_change({}),
          label + " active-row metadata remains unchanged");
    check(!output_buffer.first_unwritten_change(output_written),
          label + " output guards remain unchanged");
    check(!sink_buffer.first_unwritten_change(sink_written),
          label + " sink guards remain unchanged");

    std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
    std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
    auto shared_command = context.make_command_buffer(
        label + " shared Encoder", error);
    check(shared_command != nullptr,
          label + " shared-segment command-buffer allocation", error);
    if (!shared_command) return false;
    const bool shared_encoded = Encoder::encode(
        context, *shared_command, label + " shared segment",
        [&](Encoder& segment, std::string& encode_error) {
            return encode_matvec(
                context, segment, MatVecWeightFormat::bf16, n_in, n_out,
                active_columns, compiled_columns, buffers, encode_error);
        },
        error);
    check(shared_encoded, label + " shared-segment encode", error);
    if (!shared_encoded) return false;
    const bool shared_completed =
        wait_for_engine_command(*shared_command, label + " shared Encoder",
                                error);
    check(shared_completed, label + " shared-segment command completes", error);
    if (!shared_completed) return false;
    const auto shared_output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_output);
    const auto shared_sink_mismatch =
        twin::compare_bytes(actual_sink, expected_sink);
    check(!shared_output_mismatch,
          label + " shared-segment output matches scalar oracle",
          shared_output_mismatch ? shared_output_mismatch->reason
                                 : std::string{});
    check(!shared_sink_mismatch,
          label + " shared-segment padded sink matches scalar oracle",
          shared_sink_mismatch ? shared_sink_mismatch->reason
                               : std::string{});
    check(!output_buffer.first_unwritten_change(output_written),
          label + " shared-segment output guards remain unchanged");
    check(!sink_buffer.first_unwritten_change(sink_written),
          label + " shared-segment sink guards remain unchanged");
    return !output_mismatch && !sink_mismatch && !shared_output_mismatch &&
           !shared_sink_mismatch &&
           !output_buffer.first_unwritten_change(output_written) &&
           !sink_buffer.first_unwritten_change(sink_written);
}

bool run_readonly_bf16_router_width(Context& context) {
    constexpr std::uint32_t n_in = 2'560u;
    constexpr std::uint32_t n_out = 512u;
    constexpr std::uint32_t active_columns = 1u;
    constexpr std::uint32_t compiled_columns = 1u;
    constexpr std::uint32_t block_size = 256u;
    constexpr std::size_t input_bytes = n_in * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(n_out) * n_in * sizeof(std::uint16_t);
    constexpr std::size_t output_bytes = n_out * sizeof(float);
    constexpr std::size_t sink_bytes = sizeof(float);
    const std::string label = "read-only BF16 router width";
    std::string error;

    std::vector<float> inputs(n_in);
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const int code = static_cast<int>((index * 23u + 3u) % 67u) - 33;
        inputs[index] = static_cast<float>(code) * 0.0078125f;
    }
    std::vector<std::uint16_t> weights(
        static_cast<std::size_t>(n_out) * n_in);
    for (std::size_t index = 0u; index < weights.size(); ++index) {
        const int code = static_cast<int>((index * 19u + 1u) % 43u) - 21;
        weights[index] = strata::kernels::bf16_from_f32(
            static_cast<float>(code) * 0.00390625f);
    }
    std::vector<float> expected;
    check(twin::bf16_matvec_reference(
              weights, inputs, n_in, n_out, compiled_columns,
              active_columns, block_size, expected, error),
          label + " scalar oracle", error);
    if (expected.size() != n_out) return false;

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer active_rows_buffer;
    check(input_buffer.allocate(context, label + " activations", input_bytes,
                                error),
          label + " activation allocation", error);
    check(weight_buffer.allocate(context, label + " weights", weight_bytes,
                                 error),
          label + " weight allocation", error);
    check(output_buffer.allocate(context, label + " output", output_bytes,
                                 error),
          label + " output allocation", error);
    check(sink_buffer.allocate(context, label + " sink", sink_bytes, error),
          label + " sink allocation", error);
    check(active_rows_buffer.allocate(context, label + " active rows",
                                      sizeof(active_columns), error),
          label + " active-row allocation", error);
    if (!input_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !active_rows_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), inputs.data(), input_bytes);
    std::memcpy(weight_buffer.payload().data(), weights.data(), weight_bytes);
    std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
    std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
    const std::uint32_t live_rows = active_columns;
    std::memcpy(active_rows_buffer.payload().data(), &live_rows,
                sizeof(live_rows));
    input_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    active_rows_buffer.capture_before_dispatch();

    const ReadOnlyBF16MatVecBuffers buffers{
        context.buffers().read_only(input_buffer.view()),
        context.buffers().read_only(weight_buffer.view()),
        context.buffers().read_only(active_rows_buffer.view()),
        output_buffer.view(), sink_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command-buffer allocation", error);
    if (!command) return false;
    const bool encoded = Encoder::encode(
        context, *command, label + " segment",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_bf16_matvec(
                context, encoder, n_in, n_out, active_columns,
                compiled_columns, buffers, encode_error);
        },
        error);
    check(encoded, label + " 256-thread production-shape encoding", error);
    if (!encoded) return false;
    const bool completed = wait_for_engine_command(*command, label, error);
    check(completed, label + " completion", error);
    if (!completed) return false;

    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_bytes);
    check(!mismatch, label + " exact 512-expert output",
          mismatch ? mismatch->reason : std::string{});
    const std::array<twin::ByteRange, 1> written = {{{0u, output_bytes}}};
    const bool inputs_unchanged =
        !input_buffer.first_unwritten_change({}) &&
        !weight_buffer.first_unwritten_change({}) &&
        !active_rows_buffer.first_unwritten_change({});
    const bool output_guards =
        !output_buffer.first_unwritten_change(written) &&
        !sink_buffer.first_unwritten_change({});
    check(inputs_unchanged,
          label + " const input and active-row bytes remain unchanged");
    check(output_guards, label + " output and sink guards remain unchanged");
    return !mismatch && inputs_unchanged && output_guards;
}

bool run_readonly_q8_0_matvec_encoder(Context& context) {
    constexpr std::uint32_t n_in = 640u;
    constexpr std::uint32_t n_out = 37u;
    constexpr std::uint32_t active_columns = 3u;
    constexpr std::uint32_t compiled_columns = 4u;
    constexpr std::uint32_t blocks_per_row = n_in / 32u;
    constexpr std::size_t input_bytes =
        static_cast<std::size_t>(active_columns) * blocks_per_row *
        sizeof(Q81);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(n_out) * blocks_per_row *
        sizeof(block_q8_0);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(active_columns) * n_out * sizeof(float);
    constexpr std::size_t sink_written_bytes =
        static_cast<std::size_t>(compiled_columns - active_columns) * n_out *
        sizeof(float);
    constexpr std::size_t sink_bytes = sink_written_bytes;
    const std::string label = "read-only Q8_0 mat-vec";
    std::string error;

    std::vector<float> source_inputs(
        static_cast<std::size_t>(compiled_columns) * n_in);
    for (std::size_t index = 0u; index < source_inputs.size(); ++index) {
        const int code = static_cast<int>((index * 29u + 7u) % 113u) - 56;
        source_inputs[index] = static_cast<float>(code) * 0.0078125f;
    }
    std::vector<Q81> all_activations;
    check(twin::quantize_q8_1_reference(
              source_inputs, n_in, compiled_columns, all_activations, error),
          label + " activation oracle", error);
    if (all_activations.size() !=
        static_cast<std::size_t>(compiled_columns) * blocks_per_row) {
        return false;
    }

    std::vector<block_q8_0> weights(
        static_cast<std::size_t>(n_out) * blocks_per_row);
    for (std::size_t block = 0u; block < weights.size(); ++block) {
        weights[block].d =
            strata::kernels::f16_from_f32(0.0625f + (block % 5u) * 0.03125f);
        for (std::size_t lane = 0u; lane < 32u; ++lane) {
            const int code =
                static_cast<int>((block * 13u + lane * 7u + 3u) % 255u) -
                127;
            weights[block].qs[lane] = static_cast<std::int8_t>(code);
        }
    }
    std::vector<float> expected;
    check(twin::q8_0_matvec_reference(
              weights, all_activations, n_in, n_out, compiled_columns,
              active_columns, expected, error),
          label + " scalar oracle", error);
    if (expected.size() !=
        static_cast<std::size_t>(compiled_columns) * n_out) {
        return false;
    }

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer active_rows_buffer;
    check(input_buffer.allocate(context, label + " activations",
                                input_bytes + sizeof(float), error),
          label + " activation allocation", error);
    check(weight_buffer.allocate(context, label + " weights", weight_bytes + 1u,
                                 error),
          label + " weight allocation", error);
    check(output_buffer.allocate(context, label + " output", output_bytes,
                                 error),
          label + " output allocation", error);
    check(sink_buffer.allocate(context, label + " sink", sink_bytes, error),
          label + " sink allocation", error);
    check(active_rows_buffer.allocate(
              context, label + " active rows",
              std::max(output_bytes, sizeof(active_columns)), error),
          label + " active-row allocation", error);
    if (!input_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !active_rows_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), all_activations.data(),
                input_bytes);
    std::memcpy(weight_buffer.payload().data(), weights.data(), weight_bytes);
    const std::uint32_t live_rows = active_columns;
    std::memcpy(active_rows_buffer.payload().data(), &live_rows,
                sizeof(live_rows));
    std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
    std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
    input_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    active_rows_buffer.capture_before_dispatch();

    const ReadOnlyQ8_0MatVecBuffers buffers{
        context.buffers().read_only(input_buffer.view()),
        context.buffers().read_only(weight_buffer.view()),
        context.buffers().read_only(
            active_rows_buffer.view().slice(0u, sizeof(live_rows))),
        output_buffer.view(), sink_buffer.view()};
    const auto expected_output = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto expected_sink = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()) +
            output_bytes,
        sink_written_bytes);
    const std::array<twin::ByteRange, 1> output_written =
        {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written =
        {{{0u, sink_written_bytes}}};

    auto standalone =
        context.make_command_buffer(label + " standalone", error);
    check(standalone != nullptr, label + " standalone command buffer", error);
    if (!standalone) return false;
    const bool standalone_encoded = encode_q8_0_matvec(
        context, *standalone, n_in, n_out, active_columns, compiled_columns,
        buffers, error);
    check(standalone_encoded, label + " standalone encoding", error);
    if (!standalone_encoded) return false;
    const bool standalone_completed =
        wait_for_engine_command(*standalone, label + " standalone", error);
    check(standalone_completed, label + " standalone completion", error);
    if (!standalone_completed) return false;
    const auto standalone_output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_output);
    const auto standalone_sink_mismatch = twin::compare_bytes(
        std::span<const std::uint8_t>(sink_buffer.payload().data(),
                                      sink_written_bytes),
        expected_sink);
    check(!standalone_output_mismatch,
          label + " standalone exact output",
          standalone_output_mismatch ? standalone_output_mismatch->reason
                                     : std::string{});
    check(!standalone_sink_mismatch,
          label + " standalone exact padded output",
          standalone_sink_mismatch ? standalone_sink_mismatch->reason
                                   : std::string{});

    const std::vector<std::uint8_t> output_poison(output_bytes, 0xa5u);
    const std::vector<std::uint8_t> sink_poison(sink_bytes, 0xa5u);
    const auto reset_outputs = [&] {
        std::memcpy(output_buffer.payload().data(), output_poison.data(),
                    output_bytes);
        std::memcpy(sink_buffer.payload().data(), sink_poison.data(),
                    sink_bytes);
    };
    reset_outputs();
    auto shared = context.make_command_buffer(label + " caller-owned", error);
    check(shared != nullptr, label + " caller-owned command buffer", error);
    if (!shared) return false;
    const bool shared_encoded = Encoder::encode(
        context, *shared, label + " caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_q8_0_matvec(
                context, encoder, n_in, n_out, active_columns,
                compiled_columns, buffers, encode_error);
        },
        error);
    check(shared_encoded, label + " caller-owned encoding", error);
    if (!shared_encoded) return false;
    const bool shared_completed =
        wait_for_engine_command(*shared, label + " caller-owned", error);
    check(shared_completed, label + " caller-owned completion", error);
    if (!shared_completed) return false;
    const auto shared_output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_output);
    const auto shared_sink_mismatch = twin::compare_bytes(
        std::span<const std::uint8_t>(sink_buffer.payload().data(),
                                      sink_written_bytes),
        expected_sink);
    check(!shared_output_mismatch, label + " caller-owned exact output",
          shared_output_mismatch ? shared_output_mismatch->reason
                                 : std::string{});
    check(!shared_sink_mismatch,
          label + " caller-owned exact padded output",
          shared_sink_mismatch ? shared_sink_mismatch->reason
                               : std::string{});

    const auto expect_rejected = [&](
        const std::string& case_label,
        const ReadOnlyQ8_0MatVecBuffers& candidate,
        std::string_view error_fragment) {
        auto command = context.make_command_buffer(case_label, error);
        check(command != nullptr, case_label + " command buffer", error);
        if (!command) return false;
        const bool encoded = encode_q8_0_matvec(
            context, *command, n_in, n_out, active_columns, compiled_columns,
            candidate, error);
        const bool rejected = !encoded &&
            error.find(error_fragment) != std::string::npos &&
            command->unsafe_native_handle() == nullptr;
        check(rejected, case_label + " rejects before submission", error);
        return rejected;
    };

    bool rejected_cases = true;
    ReadOnlyQ8_0MatVecBuffers short_weights = buffers;
    short_weights.weights = buffers.weights.slice(0u, weight_bytes - 1u);
    rejected_cases &= expect_rejected(
        label + " undersized weights", short_weights, "undersized");
    ReadOnlyQ8_0MatVecBuffers misaligned_weights = buffers;
    misaligned_weights.weights = context.buffers().read_only(
        weight_buffer.view().slice(1u, weight_bytes));
    rejected_cases &= expect_rejected(
        label + " misaligned weights", misaligned_weights, "misaligned");
    ReadOnlyQ8_0MatVecBuffers misaligned_activations = buffers;
    misaligned_activations.activations = context.buffers().read_only(
        input_buffer.view().slice(sizeof(float), input_bytes - sizeof(float)));
    rejected_cases &= expect_rejected(
        label + " misaligned activations", misaligned_activations,
        "misaligned");
    ReadOnlyQ8_0MatVecBuffers misaligned_rows = buffers;
    misaligned_rows.active_rows = context.buffers().read_only(
        active_rows_buffer.view().slice(1u, sizeof(live_rows)));
    rejected_cases &= expect_rejected(
        label + " misaligned active rows", misaligned_rows, "misaligned");

    const std::uint32_t wrong_rows = active_columns - 1u;
    std::memcpy(active_rows_buffer.payload().data(), &wrong_rows,
                sizeof(wrong_rows));
    rejected_cases &= expect_rejected(
        label + " wrong active-row value", buffers,
        "does not match active columns");
    std::memcpy(active_rows_buffer.payload().data(), &live_rows,
                sizeof(live_rows));

    ReadOnlyQ8_0MatVecBuffers activation_output_alias = buffers;
    activation_output_alias.output =
        input_buffer.view().slice(0u, output_bytes);
    rejected_cases &= expect_rejected(
        label + " activation/output alias", activation_output_alias,
        "overlap");
    ReadOnlyQ8_0MatVecBuffers weight_sink_alias = buffers;
    weight_sink_alias.sink = weight_buffer.view().slice(0u, sink_bytes);
    rejected_cases &= expect_rejected(
        label + " weight/sink alias", weight_sink_alias, "overlap");
    ReadOnlyQ8_0MatVecBuffers output_sink_alias = buffers;
    output_sink_alias.sink = output_buffer.view().slice(0u, sink_bytes);
    rejected_cases &= expect_rejected(
        label + " output/sink alias", output_sink_alias, "overlap");

    // Overlap between read-only inputs is safe and is useful when views come
    // from a shared full-mode arena. Zero bytes form valid Q8_0 and Q8_1 rows.
    twin::GuardedBuffer aliased_inputs_buffer;
    check(aliased_inputs_buffer.allocate(
              context, label + " aliased read-only inputs", weight_bytes,
              error),
          label + " aliased-input allocation", error);
    if (!aliased_inputs_buffer.view()) return false;
    std::memset(aliased_inputs_buffer.payload().data(), 0, weight_bytes);
    const ReadOnlyBufferRef aliased_view =
        context.buffers().read_only(aliased_inputs_buffer.view());
    ReadOnlyQ8_0MatVecBuffers aliased_inputs = buffers;
    aliased_inputs.activations = aliased_view;
    aliased_inputs.weights = aliased_view;
    std::vector<Q81> zero_activations(
        static_cast<std::size_t>(compiled_columns) * blocks_per_row);
    std::vector<block_q8_0> zero_weights(
        static_cast<std::size_t>(n_out) * blocks_per_row);
    std::vector<float> zero_expected;
    check(twin::q8_0_matvec_reference(
              zero_weights, zero_activations, n_in, n_out, compiled_columns,
              active_columns, zero_expected, error),
          label + " aliased-input oracle", error);
    reset_outputs();
    auto alias_command =
        context.make_command_buffer(label + " read-only input alias", error);
    check(alias_command != nullptr,
          label + " read-only alias command buffer", error);
    if (!alias_command) return false;
    const bool alias_encoded = encode_q8_0_matvec(
        context, *alias_command, n_in, n_out, active_columns,
        compiled_columns, aliased_inputs, error);
    check(alias_encoded, label + " permits read-only input overlap", error);
    if (!alias_encoded) return false;
    const bool alias_completed = wait_for_engine_command(
        *alias_command, label + " read-only input alias", error);
    check(alias_completed, label + " read-only alias completion", error);
    if (!alias_completed) return false;
    const auto zero_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(zero_expected.data()),
        output_bytes);
    const auto zero_sink_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(zero_expected.data()) +
            output_bytes,
        sink_written_bytes);
    const auto alias_output_mismatch =
        twin::compare_bytes(output_buffer.payload(), zero_output_bytes);
    const auto alias_sink_mismatch = twin::compare_bytes(
        std::span<const std::uint8_t>(sink_buffer.payload().data(),
                                      sink_written_bytes),
        zero_sink_bytes);
    check(!alias_output_mismatch,
          label + " overlapping inputs match the zero oracle",
          alias_output_mismatch ? alias_output_mismatch->reason
                                : std::string{});
    check(!alias_sink_mismatch,
          label + " overlapping-input sink matches the zero oracle",
          alias_sink_mismatch ? alias_sink_mismatch->reason
                              : std::string{});

    // The 2560-wide shared-expert input uses one row per threadgroup instead
    // of the small-K four-row grouping tested above. Exercise NC=2 and NC=8
    // through the read-only Encoder with one live row and padded sinks.
    constexpr std::uint32_t long_n_in = 2'560u;
    constexpr std::uint32_t long_n_out = 17u;
    constexpr std::uint32_t long_blocks = long_n_in / 32u;
    constexpr std::size_t long_activation_bytes =
        static_cast<std::size_t>(long_blocks) * sizeof(Q81);
    constexpr std::size_t long_weight_bytes =
        static_cast<std::size_t>(long_n_out) * long_blocks *
        sizeof(block_q8_0);
    constexpr std::size_t long_output_bytes =
        static_cast<std::size_t>(long_n_out) * sizeof(float);
    constexpr std::uint32_t long_oracle_columns = 8u;
    constexpr std::size_t long_max_sink_bytes =
        static_cast<std::size_t>(long_oracle_columns - 1u) * long_n_out *
        sizeof(float);
    std::vector<float> long_source(
        static_cast<std::size_t>(long_oracle_columns) * long_n_in);
    for (std::size_t index = 0u; index < long_source.size(); ++index) {
        const int code = static_cast<int>((index * 31u + 5u) % 97u) - 48;
        long_source[index] = static_cast<float>(code) * 0.0078125f;
    }
    std::vector<Q81> long_activations;
    check(twin::quantize_q8_1_reference(
              long_source, long_n_in, long_oracle_columns, long_activations,
              error),
          label + " long-K activation oracle", error);
    if (long_activations.size() !=
        static_cast<std::size_t>(long_oracle_columns) * long_blocks) {
        return false;
    }
    std::vector<block_q8_0> long_weights(
        static_cast<std::size_t>(long_n_out) * long_blocks);
    for (std::size_t block = 0u; block < long_weights.size(); ++block) {
        long_weights[block].d =
            strata::kernels::f16_from_f32(0.125f + (block % 3u) * 0.0625f);
        for (std::size_t lane = 0u; lane < 32u; ++lane) {
            const int code =
                static_cast<int>((block * 17u + lane * 11u + 1u) % 255u) -
                127;
            long_weights[block].qs[lane] = static_cast<std::int8_t>(code);
        }
    }
    twin::GuardedBuffer long_input_buffer;
    twin::GuardedBuffer long_weight_buffer;
    twin::GuardedBuffer long_output_buffer;
    twin::GuardedBuffer long_sink_buffer;
    twin::GuardedBuffer long_rows_buffer;
    check(long_input_buffer.allocate(context, label + " long-K activations",
                                     long_activation_bytes, error),
          label + " long-K activation allocation", error);
    check(long_weight_buffer.allocate(context, label + " long-K weights",
                                      long_weight_bytes, error),
          label + " long-K weight allocation", error);
    check(long_output_buffer.allocate(context, label + " long-K output",
                                      long_output_bytes, error),
          label + " long-K output allocation", error);
    check(long_sink_buffer.allocate(context, label + " long-K sink",
                                    long_max_sink_bytes, error),
          label + " long-K sink allocation", error);
    check(long_rows_buffer.allocate(context, label + " long-K active rows",
                                    sizeof(std::uint32_t), error),
          label + " long-K active-row allocation", error);
    if (!long_input_buffer.view() || !long_weight_buffer.view() ||
        !long_output_buffer.view() || !long_sink_buffer.view() ||
        !long_rows_buffer.view()) {
        return false;
    }
    std::memcpy(long_input_buffer.payload().data(), long_activations.data(),
                long_activation_bytes);
    std::memcpy(long_weight_buffer.payload().data(), long_weights.data(),
                long_weight_bytes);
    const std::uint32_t long_live_rows = 1u;
    std::memcpy(long_rows_buffer.payload().data(), &long_live_rows,
                sizeof(long_live_rows));
    std::memset(long_output_buffer.payload().data(), 0xa5,
                long_output_bytes);
    std::memset(long_sink_buffer.payload().data(), 0xa5, long_max_sink_bytes);
    long_input_buffer.capture_before_dispatch();
    long_weight_buffer.capture_before_dispatch();
    long_output_buffer.capture_before_dispatch();
    long_sink_buffer.capture_before_dispatch();
    long_rows_buffer.capture_before_dispatch();
    const ReadOnlyQ8_0MatVecBuffers long_buffers{
        context.buffers().read_only(long_input_buffer.view()),
        context.buffers().read_only(long_weight_buffer.view()),
        context.buffers().read_only(long_rows_buffer.view()),
        long_output_buffer.view(), long_sink_buffer.view()};
    bool long_cases_pass = true;
    for (const std::uint32_t columns :
         std::array<std::uint32_t, 3>{1u, 2u, 8u}) {
        std::vector<float> long_expected;
        const auto oracle_activations = std::span<const Q81>(
            long_activations.data(),
            static_cast<std::size_t>(columns) * long_blocks);
        const bool long_oracle_ready = twin::q8_0_matvec_reference(
            long_weights, oracle_activations, long_n_in, long_n_out,
            columns, 1u, long_expected, error);
        check(long_oracle_ready, label + " long-K scalar oracle", error);
        if (!long_oracle_ready ||
            long_expected.size() !=
                static_cast<std::size_t>(columns) * long_n_out) {
            return false;
        }

        std::memset(long_output_buffer.payload().data(), 0xa5,
                    long_output_bytes);
        std::memset(long_sink_buffer.payload().data(), 0xa5,
                    long_max_sink_bytes);
        auto long_command = context.make_command_buffer(
            label + " long-K NC=" + std::to_string(columns), error);
        check(long_command != nullptr,
              label + " long-K command buffer", error);
        if (!long_command) return false;
        const bool long_encoded = encode_q8_0_matvec(
            context, *long_command, long_n_in, long_n_out, 1u, columns,
            long_buffers, error);
        check(long_encoded,
              label + " long-K NC=" + std::to_string(columns) +
                  " production encoder",
              error);
        if (!long_encoded) return false;
        const bool long_completed = wait_for_engine_command(
            *long_command, label + " long-K", error);
        check(long_completed,
              label + " long-K NC=" + std::to_string(columns) + " completion",
              error);
        if (!long_completed) return false;

        const auto long_expected_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(long_expected.data()),
            long_output_bytes);
        const std::size_t long_sink_written_bytes =
            static_cast<std::size_t>(columns - 1u) * long_n_out *
            sizeof(float);
        const auto long_expected_sink = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(long_expected.data()) +
                long_output_bytes,
            long_sink_written_bytes);
        const auto long_output_mismatch = twin::compare_bytes(
            long_output_buffer.payload(), long_expected_bytes);
        const auto long_sink_mismatch = twin::compare_bytes(
            std::span<const std::uint8_t>(long_sink_buffer.payload().data(),
                                          long_sink_written_bytes),
            long_expected_sink);
        check(!long_output_mismatch,
              label + " long-K NC=" + std::to_string(columns) +
                  " output matches scalar oracle",
              long_output_mismatch ? long_output_mismatch->reason
                                   : std::string{});
        check(!long_sink_mismatch,
              label + " long-K NC=" + std::to_string(columns) +
                  " sink matches scalar oracle",
              long_sink_mismatch ? long_sink_mismatch->reason
                                 : std::string{});
        const std::array<twin::ByteRange, 1> case_output_written =
            {{{0u, long_output_bytes}}};
        const std::array<twin::ByteRange, 1> case_sink_written =
            {{{0u, long_sink_written_bytes}}};
        const bool case_guards =
            !long_input_buffer.first_unwritten_change({}) &&
            !long_weight_buffer.first_unwritten_change({}) &&
            !long_rows_buffer.first_unwritten_change({}) &&
            !long_output_buffer.first_unwritten_change(case_output_written) &&
            !long_sink_buffer.first_unwritten_change(case_sink_written);
        check(case_guards,
              label + " long-K NC=" + std::to_string(columns) +
                  " guards remain unchanged");
        long_cases_pass &= !long_output_mismatch && !long_sink_mismatch &&
                           case_guards;
    }

    const std::array<twin::ByteRange, 1> long_output_written =
        {{{0u, long_output_bytes}}};
    const std::array<twin::ByteRange, 1> long_sink_written =
        {{{0u, long_max_sink_bytes}}};
    const bool long_guards =
        !long_input_buffer.first_unwritten_change({}) &&
        !long_weight_buffer.first_unwritten_change({}) &&
        !long_rows_buffer.first_unwritten_change({}) &&
        !long_output_buffer.first_unwritten_change(long_output_written) &&
        !long_sink_buffer.first_unwritten_change(long_sink_written);
    check(long_guards, label + " long-K input/output guards remain unchanged");

    // A rejected second dispatch must discard the first one in a shared
    // Encoder segment, leaving the prefilled outputs untouched.
    reset_outputs();
    ReadOnlyQ8_0MatVecBuffers late_failure = buffers;
    late_failure.weights = short_weights.weights;
    auto rollback =
        context.make_command_buffer(label + " late failure", error);
    check(rollback != nullptr, label + " rollback command buffer", error);
    if (!rollback) return false;
    const bool rollback_encoded = Encoder::encode(
        context, *rollback, label + " rollback segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_q8_0_matvec(
                    context, encoder, n_in, n_out, active_columns,
                    compiled_columns, buffers, encode_error)) {
                return false;
            }
            return encode_q8_0_matvec(
                context, encoder, n_in, n_out, active_columns,
                compiled_columns, late_failure, encode_error);
        },
        error);
    const bool rollback_discarded =
        !rollback_encoded &&
        rollback->unsafe_native_handle() == nullptr &&
        error.find("undersized") != std::string::npos;
    check(rollback_discarded,
          label + " caller-owned late failure discards earlier dispatches",
          error);
    const bool rollback_outputs_untouched =
        std::equal(output_poison.begin(), output_poison.end(),
                   output_buffer.payload().begin()) &&
        std::equal(sink_poison.begin(), sink_poison.end(),
                   sink_buffer.payload().begin());
    check(rollback_outputs_untouched,
          label + " rolled-back output and sink remain untouched");

    const bool inputs_unchanged =
        !input_buffer.first_unwritten_change({}) &&
        !weight_buffer.first_unwritten_change({}) &&
        !active_rows_buffer.first_unwritten_change({});
    const bool output_guards =
        !output_buffer.first_unwritten_change(output_written) &&
        !sink_buffer.first_unwritten_change(sink_written);
    check(inputs_unchanged, label + " read-only bytes remain unchanged");
    check(output_guards, label + " output and sink guards remain unchanged");

    return !standalone_output_mismatch && !standalone_sink_mismatch &&
           !shared_output_mismatch && !shared_sink_mismatch &&
           rejected_cases && !alias_output_mismatch && !alias_sink_mismatch &&
           long_cases_pass && long_guards &&
           rollback_discarded && rollback_outputs_untouched &&
           inputs_unchanged && output_guards;
}

bool run_readonly_q2_0_matvec_encoder(Context& context) {
    constexpr std::uint32_t n_in = 256u;
    constexpr std::uint32_t n_out = 7u;
    constexpr std::uint32_t active_columns = 3u;
    constexpr std::uint32_t compiled_columns = 4u;
    constexpr std::uint32_t q2_blocks_per_row = n_in / 64u;
    constexpr std::uint32_t q8_blocks_per_row = n_in / 32u;
    constexpr std::uint32_t max_columns = 8u;
    constexpr std::size_t activation_capacity_bytes =
        static_cast<std::size_t>(max_columns) * q8_blocks_per_row *
        sizeof(Q81);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(n_out) * q2_blocks_per_row *
        sizeof(block_q2_0);
    constexpr std::size_t output_capacity_bytes =
        static_cast<std::size_t>(max_columns) * n_out * sizeof(float);
    constexpr std::size_t sink_capacity_bytes =
        static_cast<std::size_t>(max_columns - 1u) * n_out * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(active_columns) * n_out * sizeof(float);
    constexpr std::size_t sink_written_bytes =
        static_cast<std::size_t>(compiled_columns - active_columns) * n_out *
        sizeof(float);
    constexpr std::size_t minimum_sink_bytes = sizeof(float);
    const std::string label = "read-only Q2_0 mat-vec";
    std::string error;

    std::vector<float> source_inputs(
        static_cast<std::size_t>(max_columns) * n_in);
    for (std::size_t index = 0u; index < source_inputs.size(); ++index) {
        const int code = static_cast<int>((index * 23u + 9u) % 109u) - 54;
        source_inputs[index] = static_cast<float>(code) * 0.0078125f;
    }
    std::vector<Q81> activations;
    check(twin::quantize_q8_1_reference(
              source_inputs, n_in, max_columns, activations, error),
          label + " activation oracle", error);
    if (activations.size() !=
        static_cast<std::size_t>(max_columns) * q8_blocks_per_row) {
        return false;
    }

    std::vector<block_q2_0> weights(
        static_cast<std::size_t>(n_out) * q2_blocks_per_row);
    for (std::size_t block = 0u; block < weights.size(); ++block) {
        weights[block].d =
            strata::kernels::f16_from_f32(0.0625f + (block % 5u) * 0.03125f);
        for (std::size_t byte = 0u; byte < sizeof(weights[block].qs); ++byte) {
            weights[block].qs[byte] =
                static_cast<std::uint8_t>((block * 37u + byte * 53u + 11u) &
                                          0xffu);
        }
    }

    std::vector<float> expected;
    const auto active_width_activations = std::span<const Q81>(
        activations.data(),
        static_cast<std::size_t>(compiled_columns) * q8_blocks_per_row);
    check(twin::q2_0_matvec_reference(
              weights, active_width_activations, n_in, n_out, compiled_columns,
              active_columns, expected, error),
          label + " scalar oracle", error);
    if (expected.size() !=
        static_cast<std::size_t>(compiled_columns) * n_out) {
        return false;
    }

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer active_rows_buffer;
    check(input_buffer.allocate(context, label + " activations",
                                activation_capacity_bytes + sizeof(float),
                                error),
          label + " activation allocation", error);
    check(weight_buffer.allocate(context, label + " weights",
                                 weight_bytes + sizeof(std::uint16_t), error),
          label + " weight allocation", error);
    check(output_buffer.allocate(context, label + " output",
                                 output_capacity_bytes, error),
          label + " output allocation", error);
    check(sink_buffer.allocate(context, label + " sink",
                               sink_capacity_bytes, error),
          label + " sink allocation", error);
    check(active_rows_buffer.allocate(context, label + " active rows",
                                      output_capacity_bytes, error),
          label + " active-row allocation", error);
    if (!input_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !active_rows_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), activations.data(),
                activation_capacity_bytes);
    std::memcpy(weight_buffer.payload().data(), weights.data(), weight_bytes);
    const std::uint32_t live_rows = active_columns;
    std::memcpy(active_rows_buffer.payload().data(), &live_rows,
                sizeof(live_rows));
    const std::vector<std::uint8_t> output_poison(output_capacity_bytes,
                                                  0xa5u);
    const std::vector<std::uint8_t> sink_poison(sink_capacity_bytes, 0xa5u);
    const auto reset_outputs = [&] {
        std::memcpy(output_buffer.payload().data(), output_poison.data(),
                    output_capacity_bytes);
        std::memcpy(sink_buffer.payload().data(), sink_poison.data(),
                    sink_capacity_bytes);
    };
    reset_outputs();
    input_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    active_rows_buffer.capture_before_dispatch();

    const ReadOnlyQ2_0MatVecBuffers buffers{
        context.buffers().read_only(input_buffer.view()),
        context.buffers().read_only(weight_buffer.view()),
        context.buffers().read_only(
            active_rows_buffer.view().slice(0u, sizeof(live_rows))),
        output_buffer.view().slice(0u, output_bytes),
        sink_buffer.view().slice(0u, sink_written_bytes)};
    const auto expected_output = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto expected_sink = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()) + output_bytes,
        sink_written_bytes);

    auto standalone =
        context.make_command_buffer(label + " standalone", error);
    check(standalone != nullptr, label + " standalone command buffer", error);
    if (!standalone) return false;
    const bool standalone_encoded = encode_q2_0_matvec(
        context, *standalone, n_in, n_out, active_columns, compiled_columns,
        buffers, error);
    check(standalone_encoded, label + " standalone encoding", error);
    if (!standalone_encoded) return false;
    const bool standalone_completed =
        wait_for_engine_command(*standalone, label + " standalone", error);
    check(standalone_completed, label + " standalone completion", error);
    if (!standalone_completed) return false;
    const auto standalone_output_mismatch =
        twin::compare_bytes(output_buffer.payload().first(output_bytes),
                            expected_output);
    const auto standalone_sink_mismatch =
        twin::compare_bytes(sink_buffer.payload().first(sink_written_bytes),
                            expected_sink);
    check(!standalone_output_mismatch, label + " standalone exact output",
          standalone_output_mismatch ? standalone_output_mismatch->reason
                                     : std::string{});
    check(!standalone_sink_mismatch,
          label + " standalone exact padded output",
          standalone_sink_mismatch ? standalone_sink_mismatch->reason
                                   : std::string{});

    reset_outputs();
    auto shared = context.make_command_buffer(label + " caller-owned", error);
    check(shared != nullptr, label + " caller-owned command buffer", error);
    if (!shared) return false;
    const bool shared_encoded = Encoder::encode(
        context, *shared, label + " caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_q2_0_matvec(
                context, encoder, n_in, n_out, active_columns,
                compiled_columns, buffers, encode_error);
        },
        error);
    check(shared_encoded, label + " caller-owned encoding", error);
    if (!shared_encoded) return false;
    const bool shared_completed =
        wait_for_engine_command(*shared, label + " caller-owned", error);
    check(shared_completed, label + " caller-owned completion", error);
    if (!shared_completed) return false;
    const auto shared_output_mismatch =
        twin::compare_bytes(output_buffer.payload().first(output_bytes),
                            expected_output);
    const auto shared_sink_mismatch =
        twin::compare_bytes(sink_buffer.payload().first(sink_written_bytes),
                            expected_sink);
    check(!shared_output_mismatch, label + " caller-owned exact output",
          shared_output_mismatch ? shared_output_mismatch->reason
                                 : std::string{});
    check(!shared_sink_mismatch, label + " caller-owned exact padded output",
          shared_sink_mismatch ? shared_sink_mismatch->reason
                               : std::string{});

    const auto expect_rejected = [&](
        const std::string& case_label,
        const ReadOnlyQ2_0MatVecBuffers& candidate, std::uint32_t case_n_in,
        std::uint32_t case_active_columns,
        std::uint32_t case_compiled_columns, std::string_view error_fragment) {
        auto command = context.make_command_buffer(case_label, error);
        check(command != nullptr, case_label + " command buffer", error);
        if (!command) return false;
        const bool encoded = encode_q2_0_matvec(
            context, *command, case_n_in, n_out, case_active_columns,
            case_compiled_columns, candidate, error);
        const bool rejected =
            !encoded && error.find(error_fragment) != std::string::npos &&
            command->unsafe_native_handle() == nullptr;
        check(rejected, case_label + " rejects and discards command buffer",
              error);
        return rejected;
    };

    bool rejected_cases = true;
    ReadOnlyQ2_0MatVecBuffers short_weights = buffers;
    short_weights.weights = buffers.weights.slice(0u, weight_bytes - 1u);
    rejected_cases &= expect_rejected(label + " undersized weights",
                                      short_weights, n_in, active_columns,
                                      compiled_columns, "undersized");
    ReadOnlyQ2_0MatVecBuffers misaligned_weights = buffers;
    misaligned_weights.weights = context.buffers().read_only(
        weight_buffer.view().slice(1u, weight_bytes));
    rejected_cases &= expect_rejected(label + " misaligned weights",
                                      misaligned_weights, n_in, active_columns,
                                      compiled_columns, "misaligned");
    ReadOnlyQ2_0MatVecBuffers misaligned_activations = buffers;
    misaligned_activations.activations = context.buffers().read_only(
        input_buffer.view().slice(sizeof(float), activation_capacity_bytes));
    rejected_cases &= expect_rejected(
        label + " misaligned activations", misaligned_activations, n_in,
        active_columns, compiled_columns, "misaligned");
    ReadOnlyQ2_0MatVecBuffers misaligned_rows = buffers;
    misaligned_rows.active_rows = context.buffers().read_only(
        active_rows_buffer.view().slice(1u, sizeof(live_rows)));
    rejected_cases &= expect_rejected(label + " misaligned active rows",
                                      misaligned_rows, n_in, active_columns,
                                      compiled_columns, "misaligned");

    const std::uint32_t wrong_rows = active_columns - 1u;
    std::memcpy(active_rows_buffer.payload().data(), &wrong_rows,
                sizeof(wrong_rows));
    rejected_cases &= expect_rejected(
        label + " wrong active-row value", buffers, n_in, active_columns,
        compiled_columns, "does not match active columns");
    std::memcpy(active_rows_buffer.payload().data(), &live_rows,
                sizeof(live_rows));
    active_rows_buffer.capture_before_dispatch();
    rejected_cases &= expect_rejected(label + " invalid input width", buffers,
                                      65u, active_columns, compiled_columns,
                                      "divisible by 64");
    rejected_cases &= expect_rejected(
        label + " unsupported compiled width", buffers, n_in, active_columns,
        3u, "compiled column width");

    ReadOnlyQ2_0MatVecBuffers activation_output_alias = buffers;
    activation_output_alias.output =
        input_buffer.view().slice(0u, output_bytes);
    rejected_cases &= expect_rejected(label + " activation/output alias",
                                      activation_output_alias, n_in,
                                      active_columns, compiled_columns,
                                      "overlap");
    ReadOnlyQ2_0MatVecBuffers weight_sink_alias = buffers;
    weight_sink_alias.sink = weight_buffer.view().slice(0u, sink_written_bytes);
    rejected_cases &= expect_rejected(label + " weight/sink alias",
                                      weight_sink_alias, n_in, active_columns,
                                      compiled_columns, "overlap");
    ReadOnlyQ2_0MatVecBuffers rows_output_alias = buffers;
    rows_output_alias.output =
        active_rows_buffer.view().slice(0u, output_bytes);
    rejected_cases &= expect_rejected(label + " active-row/output alias",
                                      rows_output_alias, n_in, active_columns,
                                      compiled_columns, "overlap");
    ReadOnlyQ2_0MatVecBuffers output_sink_alias = buffers;
    output_sink_alias.sink =
        output_buffer.view().slice(0u, sink_written_bytes);
    rejected_cases &= expect_rejected(label + " output/sink alias",
                                      output_sink_alias, n_in, active_columns,
                                      compiled_columns, "overlap");

    twin::GuardedBuffer aliased_inputs_buffer;
    check(aliased_inputs_buffer.allocate(
              context, label + " aliased read-only inputs",
              activation_capacity_bytes, error),
          label + " aliased-input allocation", error);
    if (!aliased_inputs_buffer.view()) return false;
    std::memset(aliased_inputs_buffer.payload().data(), 0,
                activation_capacity_bytes);
    const ReadOnlyBufferRef aliased_view =
        context.buffers().read_only(aliased_inputs_buffer.view());
    ReadOnlyQ2_0MatVecBuffers aliased_inputs = buffers;
    aliased_inputs.activations = aliased_view;
    aliased_inputs.weights = aliased_view;
    std::vector<Q81> zero_activations(
        static_cast<std::size_t>(compiled_columns) * q8_blocks_per_row);
    std::vector<block_q2_0> zero_weights(
        static_cast<std::size_t>(n_out) * q2_blocks_per_row);
    std::vector<float> zero_expected;
    check(twin::q2_0_matvec_reference(
              zero_weights, zero_activations, n_in, n_out, compiled_columns,
              active_columns, zero_expected, error),
          label + " aliased-input oracle", error);
    reset_outputs();
    auto alias_command =
        context.make_command_buffer(label + " read-only input alias", error);
    check(alias_command != nullptr, label + " alias command buffer", error);
    if (!alias_command) return false;
    const bool alias_encoded = encode_q2_0_matvec(
        context, *alias_command, n_in, n_out, active_columns,
        compiled_columns, aliased_inputs, error);
    check(alias_encoded, label + " permits read-only input overlap", error);
    if (!alias_encoded) return false;
    const bool alias_completed =
        wait_for_engine_command(*alias_command, label + " input alias", error);
    check(alias_completed, label + " input-alias completion", error);
    if (!alias_completed) return false;
    const std::size_t zero_output_bytes = output_bytes;
    const std::size_t zero_sink_bytes = sink_written_bytes;
    const auto zero_expected_output = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(zero_expected.data()),
        zero_output_bytes);
    const auto zero_expected_sink = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(zero_expected.data()) +
            zero_output_bytes,
        zero_sink_bytes);
    const auto alias_output_mismatch =
        twin::compare_bytes(output_buffer.payload().first(zero_output_bytes),
                            zero_expected_output);
    const auto alias_sink_mismatch =
        twin::compare_bytes(sink_buffer.payload().first(zero_sink_bytes),
                            zero_expected_sink);
    check(!alias_output_mismatch,
          label + " overlapping inputs match the zero oracle",
          alias_output_mismatch ? alias_output_mismatch->reason
                                : std::string{});
    check(!alias_sink_mismatch,
          label + " overlapping-input sink matches the zero oracle",
          alias_sink_mismatch ? alias_sink_mismatch->reason
                              : std::string{});

    bool widths_pass = true;
    for (const std::uint32_t width :
         std::array<std::uint32_t, 4>{1u, 2u, 4u, 8u}) {
        const std::uint32_t one_live_row = 1u;
        std::memcpy(active_rows_buffer.payload().data(), &one_live_row,
                    sizeof(one_live_row));
        active_rows_buffer.capture_before_dispatch();
        ReadOnlyQ2_0MatVecBuffers width_buffers = buffers;
        width_buffers.active_rows = context.buffers().read_only(
            active_rows_buffer.view().slice(0u, sizeof(one_live_row)));
        width_buffers.output = output_buffer.view().slice(
            0u, static_cast<std::size_t>(n_out) * sizeof(float));
        width_buffers.sink = sink_buffer.view().slice(
            0u, std::max(minimum_sink_bytes,
                         static_cast<std::size_t>(width - 1u) * n_out *
                             sizeof(float)));
        std::vector<float> width_expected;
        const auto width_activations = std::span<const Q81>(
            activations.data(),
            static_cast<std::size_t>(width) * q8_blocks_per_row);
        const bool oracle_ready = twin::q2_0_matvec_reference(
            weights, width_activations, n_in, n_out, width, 1u,
            width_expected, error);
        check(oracle_ready, label + " NC=" + std::to_string(width) +
                                " scalar oracle",
              error);
        if (!oracle_ready) return false;
        reset_outputs();
        auto width_command = context.make_command_buffer(
            label + " NC=" + std::to_string(width), error);
        check(width_command != nullptr,
              label + " NC=" + std::to_string(width) + " command buffer",
              error);
        if (!width_command) return false;
        const bool encoded = encode_q2_0_matvec(
            context, *width_command, n_in, n_out, 1u, width, width_buffers,
            error);
        check(encoded, label + " NC=" + std::to_string(width) + " encoding",
              error);
        if (!encoded) return false;
        const bool completed = wait_for_engine_command(
            *width_command, label + " NC=" + std::to_string(width), error);
        check(completed,
              label + " NC=" + std::to_string(width) + " completion", error);
        if (!completed) return false;
        const std::size_t width_output_bytes =
            static_cast<std::size_t>(n_out) * sizeof(float);
        const std::size_t width_sink_bytes =
            static_cast<std::size_t>(width - 1u) * n_out * sizeof(float);
        const auto width_expected_output = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(width_expected.data()),
            width_output_bytes);
        const auto width_expected_sink = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(width_expected.data()) +
                width_output_bytes,
            width_sink_bytes);
        const auto width_output_mismatch = twin::compare_bytes(
            output_buffer.payload().first(width_output_bytes),
            width_expected_output);
        const auto width_sink_mismatch = twin::compare_bytes(
            sink_buffer.payload().first(width_sink_bytes),
            width_expected_sink);
        check(!width_output_mismatch,
              label + " NC=" + std::to_string(width) +
                  " output matches scalar oracle",
              width_output_mismatch ? width_output_mismatch->reason
                                    : std::string{});
        check(!width_sink_mismatch,
              label + " NC=" + std::to_string(width) +
                  " sink matches scalar oracle",
              width_sink_mismatch ? width_sink_mismatch->reason
                                  : std::string{});
        widths_pass &= !width_output_mismatch && !width_sink_mismatch;
    }

    constexpr std::uint32_t large_n_in = 4'160u;
    constexpr std::uint32_t large_n_out = 3u;
    constexpr std::uint32_t large_columns = 2u;
    constexpr std::uint32_t large_q2_blocks = large_n_in / 64u;
    constexpr std::uint32_t large_q8_blocks = large_n_in / 32u;
    constexpr std::size_t large_activation_bytes =
        static_cast<std::size_t>(large_columns) * large_q8_blocks *
        sizeof(Q81);
    constexpr std::size_t large_weight_bytes =
        static_cast<std::size_t>(large_n_out) * large_q2_blocks *
        sizeof(block_q2_0);
    constexpr std::size_t large_output_bytes =
        static_cast<std::size_t>(large_n_out) * sizeof(float);
    constexpr std::size_t large_sink_bytes =
        static_cast<std::size_t>(large_n_out) * sizeof(float);
    std::vector<float> large_source(
        static_cast<std::size_t>(large_columns) * large_n_in);
    for (std::size_t index = 0u; index < large_source.size(); ++index) {
        const int code = static_cast<int>((index * 19u + 5u) % 101u) - 50;
        large_source[index] = static_cast<float>(code) * 0.0078125f;
    }
    std::vector<Q81> large_activations;
    check(twin::quantize_q8_1_reference(
              large_source, large_n_in, large_columns, large_activations,
              error),
          label + " large-K activation oracle", error);
    if (large_activations.size() !=
        static_cast<std::size_t>(large_columns) * large_q8_blocks) {
        return false;
    }
    std::vector<block_q2_0> large_weights(
        static_cast<std::size_t>(large_n_out) * large_q2_blocks);
    for (std::size_t block = 0u; block < large_weights.size(); ++block) {
        large_weights[block].d =
            strata::kernels::f16_from_f32(0.03125f + (block % 7u) * 0.015625f);
        for (std::size_t byte = 0u;
             byte < sizeof(large_weights[block].qs); ++byte) {
            large_weights[block].qs[byte] =
                static_cast<std::uint8_t>((block * 13u + byte * 31u + 7u) &
                                          0xffu);
        }
    }
    std::vector<float> large_expected;
    check(twin::q2_0_matvec_reference(
              large_weights, large_activations, large_n_in, large_n_out,
              large_columns, 1u, large_expected, error),
          label + " large-K scalar oracle", error);
    twin::GuardedBuffer large_input_buffer;
    twin::GuardedBuffer large_weight_buffer;
    twin::GuardedBuffer large_output_buffer;
    twin::GuardedBuffer large_sink_buffer;
    twin::GuardedBuffer large_rows_buffer;
    check(large_input_buffer.allocate(context, label + " large-K activations",
                                      large_activation_bytes, error),
          label + " large-K activation allocation", error);
    check(large_weight_buffer.allocate(context, label + " large-K weights",
                                       large_weight_bytes, error),
          label + " large-K weight allocation", error);
    check(large_output_buffer.allocate(context, label + " large-K output",
                                       large_output_bytes, error),
          label + " large-K output allocation", error);
    check(large_sink_buffer.allocate(context, label + " large-K sink",
                                     large_sink_bytes, error),
          label + " large-K sink allocation", error);
    check(large_rows_buffer.allocate(context, label + " large-K active rows",
                                     sizeof(std::uint32_t), error),
          label + " large-K active-row allocation", error);
    if (!large_input_buffer.view() || !large_weight_buffer.view() ||
        !large_output_buffer.view() || !large_sink_buffer.view() ||
        !large_rows_buffer.view()) {
        return false;
    }
    std::memcpy(large_input_buffer.payload().data(), large_activations.data(),
                large_activation_bytes);
    std::memcpy(large_weight_buffer.payload().data(), large_weights.data(),
                large_weight_bytes);
    const std::uint32_t large_live_rows = 1u;
    std::memcpy(large_rows_buffer.payload().data(), &large_live_rows,
                sizeof(large_live_rows));
    large_input_buffer.capture_before_dispatch();
    large_weight_buffer.capture_before_dispatch();
    large_output_buffer.capture_before_dispatch();
    large_sink_buffer.capture_before_dispatch();
    large_rows_buffer.capture_before_dispatch();
    const ReadOnlyQ2_0MatVecBuffers large_buffers{
        context.buffers().read_only(large_input_buffer.view()),
        context.buffers().read_only(large_weight_buffer.view()),
        context.buffers().read_only(large_rows_buffer.view()),
        large_output_buffer.view(), large_sink_buffer.view()};
    auto large_command =
        context.make_command_buffer(label + " large-K", error);
    check(large_command != nullptr, label + " large-K command buffer", error);
    if (!large_command) return false;
    const bool large_encoded = encode_q2_0_matvec(
        context, *large_command, large_n_in, large_n_out, 1u, large_columns,
        large_buffers, error);
    check(large_encoded, label + " 65-block row encoding", error);
    if (!large_encoded) return false;
    const bool large_completed =
        wait_for_engine_command(*large_command, label + " large-K", error);
    check(large_completed, label + " large-K completion", error);
    if (!large_completed) return false;
    const std::size_t large_sink_written_bytes =
        static_cast<std::size_t>(large_columns - 1u) * large_n_out *
        sizeof(float);
    const auto large_expected_output = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(large_expected.data()),
        large_output_bytes);
    const auto large_expected_sink = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(large_expected.data()) +
            large_output_bytes,
        large_sink_written_bytes);
    const auto large_output_mismatch = twin::compare_bytes(
        large_output_buffer.payload(), large_expected_output);
    const auto large_sink_mismatch = twin::compare_bytes(
        large_sink_buffer.payload().first(large_sink_written_bytes),
        large_expected_sink);
    check(!large_output_mismatch,
          label + " large-K NC=2 output matches scalar oracle",
          large_output_mismatch ? large_output_mismatch->reason
                                : std::string{});
    check(!large_sink_mismatch,
          label + " large-K NC=2 sink matches scalar oracle",
          large_sink_mismatch ? large_sink_mismatch->reason
                              : std::string{});
    const bool large_inputs_unchanged =
        !large_input_buffer.first_unwritten_change({}) &&
        !large_weight_buffer.first_unwritten_change({}) &&
        !large_rows_buffer.first_unwritten_change({});
    check(large_inputs_unchanged,
          label + " large-K read-only bytes remain unchanged");

    reset_outputs();
    const std::uint32_t one_live_row = 1u;
    std::memcpy(active_rows_buffer.payload().data(), &one_live_row,
                sizeof(one_live_row));
    active_rows_buffer.capture_before_dispatch();
    ReadOnlyQ2_0MatVecBuffers late_failure = buffers;
    late_failure.active_rows = context.buffers().read_only(
        active_rows_buffer.view().slice(0u, sizeof(one_live_row)));
    late_failure.output =
        output_buffer.view().slice(0u, static_cast<std::size_t>(n_out) *
                                          sizeof(float));
    late_failure.sink =
        sink_buffer.view().slice(0u, minimum_sink_bytes);
    late_failure.weights = short_weights.weights;
    ReadOnlyQ2_0MatVecBuffers rollback_valid = late_failure;
    rollback_valid.weights = buffers.weights;
    auto rollback = context.make_command_buffer(label + " late failure", error);
    check(rollback != nullptr, label + " rollback command buffer", error);
    if (!rollback) return false;
    const bool rollback_encoded = Encoder::encode(
        context, *rollback, label + " rollback segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_q2_0_matvec(context, encoder, n_in, n_out, 1u, 1u,
                                    rollback_valid, encode_error)) {
                return false;
            }
            return encode_q2_0_matvec(context, encoder, n_in, n_out, 1u, 1u,
                                      late_failure, encode_error);
        },
        error);
    const bool rollback_discarded =
        !rollback_encoded && rollback->unsafe_native_handle() == nullptr &&
        error.find("undersized") != std::string::npos;
    check(rollback_discarded,
          label + " caller-owned late failure discards prior dispatch",
          error);
    const bool rollback_outputs_untouched =
        std::equal(output_poison.begin(), output_poison.end(),
                   output_buffer.payload().begin()) &&
        std::equal(sink_poison.begin(), sink_poison.end(),
                   sink_buffer.payload().begin());
    check(rollback_outputs_untouched,
          label + " rolled-back output and sink remain untouched");

    const bool inputs_unchanged =
        !input_buffer.first_unwritten_change({}) &&
        !weight_buffer.first_unwritten_change({}) &&
        !active_rows_buffer.first_unwritten_change({});
    const std::array<twin::ByteRange, 1> output_written =
        {{{0u, static_cast<std::size_t>(n_out) * sizeof(float)}}};
    const std::array<twin::ByteRange, 1> sink_written =
        {{{0u, minimum_sink_bytes}}};
    const bool output_guards =
        !output_buffer.first_unwritten_change(output_written) &&
        !sink_buffer.first_unwritten_change(sink_written);
    check(inputs_unchanged, label + " read-only bytes remain unchanged");
    check(output_guards, label + " output and sink guards remain unchanged");
    return !standalone_output_mismatch && !standalone_sink_mismatch &&
           !shared_output_mismatch && !shared_sink_mismatch && rejected_cases &&
           !alias_output_mismatch && !alias_sink_mismatch && widths_pass &&
           !large_output_mismatch && !large_sink_mismatch &&
           large_inputs_unchanged && rollback_discarded &&
           rollback_outputs_untouched &&
           inputs_unchanged && output_guards;
}

bool run_readonly_bf16_matvec_encoder(Context& context) {
    constexpr std::uint32_t n_in = 320u;
    constexpr std::uint32_t n_out = 7u;
    constexpr std::uint32_t active_columns = 3u;
    constexpr std::uint32_t compiled_columns = 4u;
    constexpr std::uint32_t block_size = 160u;
    constexpr std::size_t input_count =
        static_cast<std::size_t>(compiled_columns) * n_in;
    constexpr std::size_t output_count =
        static_cast<std::size_t>(active_columns) * n_out;
    constexpr std::size_t output_bytes = output_count * sizeof(float);
    constexpr std::size_t sink_written_bytes =
        static_cast<std::size_t>(compiled_columns - active_columns) * n_out *
        sizeof(float);
    constexpr std::size_t sink_bytes = sink_written_bytes;

    const std::string label = "read-only BF16 mat-vec";
    std::string error;
    std::vector<float> input_values(input_count);
    for (std::size_t index = 0u; index < input_values.size(); ++index) {
        const int code = static_cast<int>((index * 17u + 9u) % 47u) - 23;
        input_values[index] = static_cast<float>(code) * 0.015625f;
    }
    std::vector<float> weight_values(
        static_cast<std::size_t>(n_out) * n_in);
    for (std::size_t index = 0u; index < weight_values.size(); ++index) {
        const int code = static_cast<int>((index * 11u + 5u) % 41u) - 20;
        weight_values[index] = static_cast<float>(code) * 0.0078125f;
    }
    std::vector<std::uint16_t> weights(weight_values.size());
    std::transform(weight_values.begin(), weight_values.end(), weights.begin(),
                   strata::kernels::bf16_from_f32);
    std::vector<float> expected;
    check(twin::bf16_matvec_reference(
              weights, input_values, n_in, n_out, compiled_columns,
              active_columns, block_size, expected, error),
          label + " scalar oracle", error);
    if (expected.size() !=
        static_cast<std::size_t>(compiled_columns) * n_out) {
        return false;
    }

    const std::size_t activation_bytes =
        input_values.size() * sizeof(float);
    const std::size_t weight_bytes = weights.size() * sizeof(std::uint16_t);
    const std::size_t active_rows_storage_bytes =
        std::max({output_bytes, sink_bytes, sizeof(active_columns)}) + 4u;
    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer active_rows_buffer;
    check(input_buffer.allocate(context, label + " activations",
                                activation_bytes, error),
          label + " activation allocation", error);
    check(weight_buffer.allocate(context, label + " weights",
                                 weight_bytes + 4u, error),
          label + " weight allocation", error);
    check(output_buffer.allocate(context, label + " output", output_bytes,
                                 error),
          label + " output allocation", error);
    check(sink_buffer.allocate(context, label + " sink", sink_bytes, error),
          label + " sink allocation", error);
    check(active_rows_buffer.allocate(context, label + " active rows",
                                      active_rows_storage_bytes, error),
          label + " active-row allocation", error);
    if (!input_buffer.view() || !weight_buffer.view() ||
        !output_buffer.view() || !sink_buffer.view() ||
        !active_rows_buffer.view()) {
        return false;
    }

    std::memcpy(input_buffer.payload().data(), input_values.data(),
                activation_bytes);
    std::memcpy(weight_buffer.payload().data(), weights.data(), weight_bytes);
    std::memset(weight_buffer.payload().data() + weight_bytes, 0xa5, 4u);
    const std::uint32_t live_rows = active_columns;
    std::memcpy(active_rows_buffer.payload().data(), &live_rows,
                sizeof(live_rows));
    std::memset(active_rows_buffer.payload().data() + sizeof(live_rows), 0xa5,
                active_rows_storage_bytes - sizeof(live_rows));

    const std::vector<std::uint8_t> output_poison(output_bytes, 0xa5u);
    const std::vector<std::uint8_t> sink_poison(sink_bytes, 0xa5u);
    const auto reset_outputs = [&] {
        std::memcpy(output_buffer.payload().data(), output_poison.data(),
                    output_bytes);
        std::memcpy(sink_buffer.payload().data(), sink_poison.data(),
                    sink_bytes);
    };
    reset_outputs();
    input_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    active_rows_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();

    const ReadOnlyBF16MatVecBuffers buffers{
        context.buffers().read_only(input_buffer.view()),
        context.buffers().read_only(weight_buffer.view()),
        context.buffers().read_only(active_rows_buffer.view()),
        output_buffer.view(), sink_buffer.view()};
    const auto expected_output = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto expected_sink = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data() + output_count),
        sink_written_bytes);
    const std::array<twin::ByteRange, 1> output_written =
        {{{0u, output_bytes}}};
    const std::array<twin::ByteRange, 1> sink_written =
        {{{0u, sink_written_bytes}}};

    auto standalone = context.make_command_buffer(label + " standalone", error);
    check(standalone != nullptr, label + " standalone command buffer", error);
    if (!standalone) return false;
    const bool standalone_encoded = encode_bf16_matvec(
        context, *standalone, n_in, n_out, active_columns, compiled_columns,
        buffers, error);
    check(standalone_encoded, label + " standalone encoding", error);
    if (!standalone_encoded) return false;
    const bool standalone_completed =
        wait_for_engine_command(*standalone, label + " standalone", error);
    check(standalone_completed, label + " standalone completion", error);
    if (!standalone_completed) return false;
    const auto standalone_output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_output);
    const auto standalone_sink_mismatch =
        twin::compare_bytes(
            std::span<const std::uint8_t>(sink_buffer.payload().data(),
                                          sink_written_bytes),
            expected_sink);
    check(!standalone_output_mismatch,
          label + " standalone exact output",
          standalone_output_mismatch ? standalone_output_mismatch->reason
                                     : std::string{});
    check(!standalone_sink_mismatch,
          label + " standalone exact padded output",
          standalone_sink_mismatch ? standalone_sink_mismatch->reason
                                   : std::string{});

    reset_outputs();
    auto shared = context.make_command_buffer(label + " shared", error);
    check(shared != nullptr, label + " shared command buffer", error);
    if (!shared) return false;
    const bool shared_encoded = Encoder::encode(
        context, *shared, label + " shared segment",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_bf16_matvec(
                context, encoder, n_in, n_out, active_columns,
                compiled_columns, buffers, encode_error);
        },
        error);
    check(shared_encoded, label + " caller-owned Encoder encoding", error);
    if (!shared_encoded) return false;
    const bool shared_completed =
        wait_for_engine_command(*shared, label + " shared", error);
    check(shared_completed, label + " caller-owned completion", error);
    if (!shared_completed) return false;
    const auto shared_output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_output);
    const auto shared_sink_mismatch = twin::compare_bytes(
        std::span<const std::uint8_t>(sink_buffer.payload().data(),
                                      sink_written_bytes),
        expected_sink);
    check(!shared_output_mismatch,
          label + " caller-owned exact output",
          shared_output_mismatch ? shared_output_mismatch->reason
                                 : std::string{});
    check(!shared_sink_mismatch,
          label + " caller-owned exact padded output",
          shared_sink_mismatch ? shared_sink_mismatch->reason
                               : std::string{});

    // Full-mode inputs may share one read-only arena; the encoder must allow
    // read-only/read-only overlap while still separating outputs.
    std::vector<std::uint16_t> aliased_weights(weight_bytes / sizeof(std::uint16_t));
    std::memcpy(aliased_weights.data(), input_values.data(), weight_bytes);
    std::vector<float> aliased_expected;
    check(twin::bf16_matvec_reference(
              aliased_weights, input_values, n_in, n_out, compiled_columns,
              active_columns, block_size, aliased_expected, error),
          label + " aliased-read-only scalar oracle", error);
    reset_outputs();
    auto aliased_command =
        context.make_command_buffer(label + " read-only input alias", error);
    check(aliased_command != nullptr,
          label + " read-only input alias command buffer", error);
    if (!aliased_command) return false;
    ReadOnlyBF16MatVecBuffers aliased_inputs = buffers;
    aliased_inputs.weights = aliased_inputs.activations;
    const bool aliased_encoded = encode_bf16_matvec(
        context, *aliased_command, n_in, n_out, active_columns,
        compiled_columns, aliased_inputs, error);
    check(aliased_encoded,
          label + " permits overlapping read-only inputs", error);
    if (!aliased_encoded) return false;
    const bool aliased_completed = wait_for_engine_command(
        *aliased_command, label + " read-only input alias", error);
    check(aliased_completed,
          label + " read-only input alias completion", error);
    if (!aliased_completed) return false;
    const auto aliased_output_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(aliased_expected.data()),
        output_bytes);
    const auto aliased_sink_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(aliased_expected.data() +
                                              output_count),
        sink_written_bytes);
    const auto aliased_output_mismatch =
        twin::compare_bytes(output_buffer.payload(), aliased_output_bytes);
    const auto aliased_sink_mismatch = twin::compare_bytes(
        std::span<const std::uint8_t>(sink_buffer.payload().data(),
                                      sink_written_bytes),
        aliased_sink_bytes);
    check(!aliased_output_mismatch,
          label + " overlapping read-only output matches oracle",
          aliased_output_mismatch ? aliased_output_mismatch->reason
                                  : std::string{});
    check(!aliased_sink_mismatch,
          label + " overlapping read-only sink matches oracle",
          aliased_sink_mismatch ? aliased_sink_mismatch->reason
                                : std::string{});

    const auto expect_rejected = [&](
        const std::string& case_label,
        const ReadOnlyBF16MatVecBuffers& candidate,
        std::uint32_t candidate_n_in,
        std::string_view error_fragment) {
        auto command = context.make_command_buffer(case_label, error);
        check(command != nullptr, case_label + " command buffer", error);
        if (!command) return false;
        const bool encoded = encode_bf16_matvec(
            context, *command, candidate_n_in, n_out, active_columns,
            compiled_columns, candidate, error);
        const bool rejected = !encoded;
        const bool discarded = command->unsafe_native_handle() == nullptr;
        const bool expected_error = error.find(error_fragment) != std::string::npos;
        check(rejected && expected_error,
              case_label + " rejects at the intended validation", error);
        check(discarded, case_label + " discards its unsubmitted command");
        return rejected && discarded && expected_error;
    };

    bool rejected_cases = true;
    rejected_cases &= expect_rejected(
        label + " invalid width", buffers, n_in + 1u, "n_in must be");
    ReadOnlyBF16MatVecBuffers short_weights = buffers;
    short_weights.weights = buffers.weights.slice(0u, weight_bytes - 2u);
    rejected_cases &= expect_rejected(
        label + " undersized weights", short_weights, n_in, "undersized");
    ReadOnlyBF16MatVecBuffers misaligned_weights = buffers;
    misaligned_weights.weights = context.buffers().read_only(
        weight_buffer.view().slice(2u, weight_bytes));
    rejected_cases &= expect_rejected(
        label + " misaligned weights", misaligned_weights, n_in,
        "misaligned");
    ReadOnlyBF16MatVecBuffers misaligned_activations = buffers;
    misaligned_activations.activations = context.buffers().read_only(
        input_buffer.view().slice(sizeof(float),
                                  activation_bytes - sizeof(float)));
    rejected_cases &= expect_rejected(
        label + " misaligned activations", misaligned_activations, n_in,
        "misaligned");
    ReadOnlyBF16MatVecBuffers misaligned_rows = buffers;
    misaligned_rows.active_rows = context.buffers().read_only(
        active_rows_buffer.view().slice(1u,
                                        active_rows_storage_bytes - 1u));
    rejected_cases &= expect_rejected(
        label + " misaligned active rows", misaligned_rows, n_in,
        "misaligned");

    const std::uint32_t wrong_rows = active_columns - 1u;
    std::memcpy(active_rows_buffer.payload().data(), &wrong_rows,
                sizeof(wrong_rows));
    rejected_cases &= expect_rejected(
        label + " wrong active-row value", buffers, n_in,
        "does not match active columns");
    std::memcpy(active_rows_buffer.payload().data(), &live_rows,
                sizeof(live_rows));

    ReadOnlyBF16MatVecBuffers empty_weights = buffers;
    empty_weights.weights = {};
    rejected_cases &= expect_rejected(
        label + " unowned weights", empty_weights, n_in,
        "not owned by this Context");

    ReadOnlyBF16MatVecBuffers activation_output_alias = buffers;
    activation_output_alias.output = input_buffer.view().slice(0u, output_bytes);
    rejected_cases &= expect_rejected(
        label + " activation/output alias", activation_output_alias, n_in,
        "overlap");
    ReadOnlyBF16MatVecBuffers activation_sink_alias = buffers;
    activation_sink_alias.sink = input_buffer.view().slice(0u, sink_bytes);
    rejected_cases &= expect_rejected(
        label + " activation/sink alias", activation_sink_alias, n_in,
        "overlap");
    ReadOnlyBF16MatVecBuffers weight_output_alias = buffers;
    weight_output_alias.output = weight_buffer.view().slice(0u, output_bytes);
    rejected_cases &= expect_rejected(
        label + " weight/output alias", weight_output_alias, n_in, "overlap");
    ReadOnlyBF16MatVecBuffers weight_sink_alias = buffers;
    weight_sink_alias.sink = weight_buffer.view().slice(0u, sink_bytes);
    rejected_cases &= expect_rejected(
        label + " weight/sink alias", weight_sink_alias, n_in, "overlap");
    ReadOnlyBF16MatVecBuffers rows_output_alias = buffers;
    rows_output_alias.output =
        active_rows_buffer.view().slice(0u, output_bytes);
    rejected_cases &= expect_rejected(
        label + " active-row/output alias", rows_output_alias, n_in,
        "overlap");
    ReadOnlyBF16MatVecBuffers rows_sink_alias = buffers;
    rows_sink_alias.sink = active_rows_buffer.view().slice(0u, sink_bytes);
    rejected_cases &= expect_rejected(
        label + " active-row/sink alias", rows_sink_alias, n_in, "overlap");
    ReadOnlyBF16MatVecBuffers output_sink_alias = buffers;
    output_sink_alias.sink = output_buffer.view().slice(0u, sink_bytes);
    rejected_cases &= expect_rejected(
        label + " output/sink alias", output_sink_alias, n_in, "overlap");

    const long page_size = getpagesize();
    void* alias_memory = nullptr;
    if (page_size <= 0 ||
        posix_memalign(&alias_memory, static_cast<std::size_t>(page_size),
                       static_cast<std::size_t>(page_size)) != 0) {
        check(false, label + " cross-handle alias allocation");
        return false;
    }
    const std::size_t alias_bytes = static_cast<std::size_t>(page_size);
    std::shared_ptr<void> alias_lifetime(alias_memory, std::free);
    std::memset(alias_memory, 0xa5, alias_bytes);
    std::memcpy(alias_memory, input_values.data(), activation_bytes);
    const std::vector<std::uint8_t> alias_before(
        static_cast<const std::uint8_t*>(alias_memory),
        static_cast<const std::uint8_t*>(alias_memory) + alias_bytes);
    const BufferRef alias_input = context.buffers().wrap_no_copy(
        label + " cross-handle input", alias_memory, alias_bytes,
        alias_lifetime, error);
    const BufferRef alias_output = context.buffers().wrap_no_copy(
        label + " cross-handle output", alias_memory, alias_bytes,
        alias_lifetime, error);
    const bool distinct_handles = alias_input && alias_output &&
                                 alias_input.handle != alias_output.handle;
    check(distinct_handles,
          label + " uses distinct no-copy Metal handles", error);
    if (!distinct_handles) return false;
    ReadOnlyBF16MatVecBuffers cross_handle_alias = buffers;
    cross_handle_alias.activations = context.buffers()
                                        .read_only(alias_input)
                                        .slice(0u, activation_bytes);
    cross_handle_alias.output = alias_output.slice(64u, output_bytes);
    auto cross_handle_command =
        context.make_command_buffer(label + " cross-handle alias", error);
    check(cross_handle_command != nullptr,
          label + " cross-handle command buffer", error);
    if (!cross_handle_command) return false;
    const bool cross_handle_rejected =
        !encode_bf16_matvec(context, *cross_handle_command, n_in, n_out,
                            active_columns, compiled_columns,
                            cross_handle_alias, error) &&
        error.find("overlap") != std::string::npos &&
        cross_handle_command->unsafe_native_handle() == nullptr;
    check(cross_handle_rejected,
          label + " rejects partial CPU-backing overlap across handles",
          error);
    const bool alias_bytes_unchanged = std::equal(
        alias_before.begin(), alias_before.end(),
        static_cast<const std::uint8_t*>(alias_memory));
    check(alias_bytes_unchanged,
          label + " rejected cross-handle views leave backing bytes unchanged");

    // A valid first dispatch followed by a rejected one must discard the
    // caller-owned command buffer instead of allowing the first write through.
    reset_outputs();
    ReadOnlyBF16MatVecBuffers late_failure = buffers;
    late_failure.weights = buffers.weights.slice(0u, weight_bytes - 2u);
    auto rollback = context.make_command_buffer(label + " late failure", error);
    check(rollback != nullptr, label + " rollback command buffer", error);
    if (!rollback) return false;
    const bool rollback_accepted = Encoder::encode(
        context, *rollback, label + " rollback segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_bf16_matvec(context, encoder, n_in, n_out,
                                    active_columns, compiled_columns, buffers,
                                    encode_error)) {
                return false;
            }
            return encode_bf16_matvec(
                context, encoder, n_in, n_out, active_columns,
                compiled_columns, late_failure, encode_error);
        },
        error);
    const bool rollback_discarded =
        !rollback_accepted && rollback->unsafe_native_handle() == nullptr &&
        error.find("undersized") != std::string::npos;
    check(rollback_discarded,
          label + " caller-owned late failure discards its first dispatch",
          error);
    const bool outputs_unchanged =
        std::equal(output_poison.begin(), output_poison.end(),
                   output_buffer.payload().begin()) &&
        std::equal(sink_poison.begin(), sink_poison.end(),
                   sink_buffer.payload().begin());
    check(outputs_unchanged,
          label + " discarded caller segment leaves output bytes untouched");

    const bool inputs_unchanged =
        !input_buffer.first_unwritten_change({}) &&
        !weight_buffer.first_unwritten_change({}) &&
        !active_rows_buffer.first_unwritten_change({});
    const bool output_guards =
        !output_buffer.first_unwritten_change(output_written) &&
        !sink_buffer.first_unwritten_change(sink_written);
    check(inputs_unchanged,
          label + " read-only inputs and row metadata remain unchanged");
    check(output_guards, label + " output and sink guards remain unchanged");

    return !standalone_output_mismatch && !standalone_sink_mismatch &&
           !shared_output_mismatch && !shared_sink_mismatch &&
           !aliased_output_mismatch && !aliased_sink_mismatch &&
           rejected_cases && cross_handle_rejected && alias_bytes_unchanged &&
           rollback_discarded && outputs_unchanged &&
           inputs_unchanged && output_guards;
}

bool run_q8_1_engine_encoder(Context& context) {
    constexpr std::uint32_t width = 640u;
    constexpr std::uint32_t rows = 3u;
    std::vector<float> input(static_cast<std::size_t>(width) * rows);
    for (std::size_t index = 0u; index < input.size(); ++index) {
        const int code = static_cast<int>((index * 37u + 19u) % 129u) - 64;
        input[index] = static_cast<float>(code) * 0.015625f;
    }

    std::vector<Q81> expected;
    std::string error;
    check(twin::quantize_q8_1_reference(input, width, rows, expected, error),
          "engine Q8_1 scalar reference", error);
    if (expected.empty()) return false;

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer output_buffer;
    const std::size_t input_bytes = input.size() * sizeof(float);
    const std::size_t output_bytes = expected.size() * sizeof(Q81);
    check(input_buffer.allocate(context, "engine Q8_1 input", input_bytes, error),
          "engine Q8_1 input allocation", error);
    check(output_buffer.allocate(context, "engine Q8_1 output", output_bytes, error),
          "engine Q8_1 output allocation", error);
    if (!input_buffer.view() || !output_buffer.view()) return false;
    std::memcpy(input_buffer.payload().data(), input.data(), input_bytes);
    std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
    input_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer("Q8_1 encoder twin", error);
    check(command != nullptr, "create engine Q8_1 command buffer", error);
    if (!command) return false;

    const Q8_1Buffers buffers{input_buffer.view(), output_buffer.view()};
    const bool rejects_bad_width =
        !encode_q8_1(context, *command, rows, width + 1u, buffers, error);
    check(rejects_bad_width &&
              error == "Q8_1 shape requires positive rows and width in 32..16640 divisible by 32",
          "engine Q8_1 rejects an unsupported width before encoding", error);
    check(std::all_of(output_buffer.payload().begin(),
                      output_buffer.payload().end(),
                      [](std::uint8_t byte) { return byte == 0xa5u; }),
          "rejected engine Q8_1 call leaves output unchanged");

    const bool encoded =
        encode_q8_1(context, *command, rows, width, buffers, error);
    check(encoded, "engine Q8_1 production encoder", error);
    if (!encoded) return false;
    const bool completed =
        wait_for_engine_command(*command, "engine Q8_1", error);
    check(completed, "engine Q8_1 command completes", error);
    if (!completed) return false;

    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_bytes);
    check(!mismatch, "engine Q8_1 output matches scalar oracle",
          mismatch ? mismatch->reason : std::string{});
    check(!input_buffer.first_unwritten_change({}),
          "engine Q8_1 input and guards remain unchanged");
    const std::array<twin::ByteRange, 1> output_written = {{{0u, output_bytes}}};
    check(!output_buffer.first_unwritten_change(output_written),
          "engine Q8_1 output guards remain unchanged");
    return !mismatch;
}

bool run_q8_1_buffer_alias_contract(Context& context) {
    constexpr std::uint32_t width = 32u;
    constexpr std::size_t input_bytes = width * sizeof(float);
    constexpr std::size_t output_bytes = sizeof(Q81);
    constexpr std::size_t arena_bytes = 256u;
    std::array<float, width> input{};
    for (std::size_t index = 0u; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index) - 13) *
                       0.125f;
    }

    twin::GuardedBuffer arena;
    std::string error;
    check(arena.allocate(context, "engine Q8_1 alias views", arena_bytes,
                         error),
          "engine Q8_1 alias-view allocation", error);
    if (!arena.view()) return false;
    std::memcpy(arena.payload().data(), input.data(), input_bytes);
    arena.capture_before_dispatch();

    const BufferRef backing = arena.view();
    const BufferRef input_view =
        engine_test_subview(backing, 0u, input_bytes);
    const BufferRef overlapping_output =
        engine_test_subview(backing, 64u, output_bytes);
    const BufferRef adjacent_output =
        engine_test_subview(backing, input_bytes, output_bytes);
    if (!input_view || !overlapping_output || !adjacent_output) return false;

    auto command = context.make_command_buffer("Q8_1 alias contract", error);
    check(command != nullptr, "engine Q8_1 alias command-buffer allocation",
          error);
    if (!command) return false;
    const Q8_1Buffers overlapping_buffers{input_view, overlapping_output};
    const bool rejects_overlap =
        !encode_q8_1(context, *command, 1u, width, overlapping_buffers,
                     error) &&
        error == "Q8_1 output must not overlap its input";
    check(rejects_overlap, "engine Q8_1 rejects partially overlapping views",
          error);

    const Q8_1Buffers adjacent_buffers{input_view, adjacent_output};
    const bool accepts_adjacent =
        encode_q8_1(context, *command, 1u, width, adjacent_buffers, error);
    check(accepts_adjacent,
          "engine Q8_1 accepts adjacent non-overlapping views", error);
    if (!accepts_adjacent) return false;
    const bool completed =
        wait_for_engine_command(*command, "Q8_1 adjacent views", error);
    check(completed, "engine Q8_1 adjacent-view command completes", error);
    if (!completed) return false;

    std::vector<Q81> expected;
    check(twin::quantize_q8_1_reference(input, width, 1u, expected, error),
          "engine Q8_1 adjacent-view scalar oracle", error);
    if (expected.size() != 1u) return false;
    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto actual_bytes = std::span<const std::uint8_t>(
        static_cast<const std::uint8_t*>(adjacent_output.contents),
        output_bytes);
    const auto mismatch = twin::compare_bytes(actual_bytes, expected_bytes);
    check(!mismatch, "engine Q8_1 adjacent output matches scalar oracle",
          mismatch ? mismatch->reason : std::string{});
    const std::array<twin::ByteRange, 1> written =
        {{{input_bytes, output_bytes}}};
    check(!arena.first_unwritten_change(written),
          "engine Q8_1 adjacent views preserve all other bytes and guards");
    return !mismatch;
}

bool run_engine_cross_handle_alias_contract(Context& context) {
    constexpr std::uint32_t q8_width = 32u;
    constexpr std::uint32_t mv_width = 256u;
    const long page_size = getpagesize();
    void* alias_memory = nullptr;
    if (page_size <= 0 ||
        posix_memalign(&alias_memory, static_cast<std::size_t>(page_size),
                       static_cast<std::size_t>(page_size)) != 0) {
        check(false, "allocate engine cross-handle alias fixture");
        return false;
    }
    const std::size_t alias_bytes = static_cast<std::size_t>(page_size);
    std::memset(alias_memory, 0, alias_bytes);
    std::shared_ptr<void> alias_lifetime(alias_memory, std::free);
    std::string error;
    const BufferRef alias_input = context.buffers().wrap_no_copy(
        "engine cross-handle alias input", alias_memory, alias_bytes,
        alias_lifetime, error);
    const BufferRef alias_output = context.buffers().wrap_no_copy(
        "engine cross-handle alias output", alias_memory, alias_bytes,
        alias_lifetime, error);
    if (!alias_input || !alias_output) {
        check(false, "create engine cross-handle no-copy aliases", error);
        return false;
    }
    check(alias_input.handle != alias_output.handle,
          "engine cross-handle alias fixtures use distinct Metal handles");

    auto q8_command =
        context.make_command_buffer("Q8_1 cross-handle alias", error);
    check(q8_command != nullptr,
          "engine Q8_1 cross-handle command-buffer allocation", error);
    if (!q8_command) return false;
    const bool rejects_q8_cross_handle =
        alias_input.handle != alias_output.handle &&
        !encode_q8_1(context, *q8_command, 1u, q8_width,
                     {alias_input, alias_output}, error) &&
        error == "Q8_1 output must not overlap its input";
    check(rejects_q8_cross_handle,
          "engine Q8_1 rejects cross-handle CPU-backing overlap", error);
    q8_command->discard();

    const std::size_t weight_bytes =
        (mv_width / 32u) * sizeof(block_q8_0);
    const BufferRef weights = context.buffers().allocate(
        "engine cross-handle mat-vec weights", weight_bytes, error);
    const BufferRef sink = context.buffers().allocate(
        "engine cross-handle mat-vec sink", sizeof(float), error);
    const BufferRef active_rows = context.buffers().allocate(
        "engine cross-handle mat-vec rows", sizeof(std::uint32_t), error);
    check(weights && sink && active_rows,
          "engine cross-handle mat-vec allocations", error);
    if (!weights || !sink || !active_rows) return false;
    auto* weight_blocks = static_cast<block_q8_0*>(weights.contents);
    for (std::size_t index = 0u; index < mv_width / 32u; ++index) {
        weight_blocks[index].d = strata::kernels::f16_from_f32(0.25f);
        std::fill_n(weight_blocks[index].qs, 32u, std::int8_t{1});
    }
    const std::uint32_t live_rows = 1u;
    std::memcpy(active_rows.contents, &live_rows, sizeof(live_rows));

    auto matvec_command =
        context.make_command_buffer("mat-vec cross-handle alias", error);
    check(matvec_command != nullptr,
          "engine mat-vec cross-handle command-buffer allocation", error);
    if (!matvec_command) return false;
    const MatVecBuffers buffers{
        alias_input, weights, alias_output, sink, active_rows};
    const bool rejects_matvec_cross_handle =
        !encode_matvec(context, *matvec_command,
                       MatVecWeightFormat::q8_0, mv_width, 1u, live_rows, 1u,
                       buffers, error) &&
        error == "mat-vec outputs must not overlap an input or each other";
    check(rejects_matvec_cross_handle,
          "engine mat-vec rejects cross-handle CPU-backing overlap", error);
    matvec_command->discard();
    return rejects_q8_cross_handle && rejects_matvec_cross_handle;
}

bool run_matvec_engine_encoders(Context& context) {
    struct Case {
        const char* name;
        MatVecWeightFormat format;
        std::uint32_t n_in;
        std::size_t row_bytes;
        bool bf16;
    };
    const std::array cases{
        Case{"Q8_0", MatVecWeightFormat::q8_0, 256u,
             8u * sizeof(block_q8_0), false},
        Case{"Q4_0", MatVecWeightFormat::q4_0, 256u,
             8u * sizeof(block_q4_0), false},
        Case{"Q5_0", MatVecWeightFormat::q5_0, 256u,
             8u * sizeof(block_q5_0), false},
        Case{"Q2_0", MatVecWeightFormat::q2_0, 256u,
             4u * sizeof(block_q2_0), false},
        Case{"IQ4_NL", MatVecWeightFormat::iq4_nl, 256u,
             8u * sizeof(block_iq4_nl), false},
        Case{"IQ4_XS", MatVecWeightFormat::iq4_xs, 256u,
             sizeof(block_iq4_xs), false},
        Case{"IQ3_S", MatVecWeightFormat::iq3_s, 256u,
             sizeof(block_iq3_s), false},
        Case{"Q4_K", MatVecWeightFormat::q4_k, 256u,
             sizeof(block_q4_K), false},
        Case{"Q5_K", MatVecWeightFormat::q5_k, 256u,
             sizeof(block_q5_K), false},
        Case{"Q3_K", MatVecWeightFormat::q3_k, 256u,
             sizeof(block_q3_K), false},
        Case{"Q6_K", MatVecWeightFormat::q6_k, 256u,
             sizeof(block_q6_K), false},
        Case{"BF16", MatVecWeightFormat::bf16, 320u,
             320u * sizeof(std::uint16_t), true},
    };

    constexpr std::uint32_t n_out = 7u;
    constexpr std::uint32_t active_columns = 3u;
    constexpr std::uint32_t compiled_columns = 4u;
    bool passed = true;
    for (std::size_t case_index = 0u; case_index < cases.size(); ++case_index) {
        const Case& test_case = cases[case_index];
        const std::string label =
            std::string("engine ") + test_case.name + " mat-vec";
        std::string error;
        const std::size_t input_bytes = test_case.bf16
            ? static_cast<std::size_t>(active_columns) * test_case.n_in *
                  sizeof(float)
            : static_cast<std::size_t>(active_columns) *
                  (test_case.n_in / 32u) * sizeof(Q81);
        const std::size_t weight_bytes =
            static_cast<std::size_t>(n_out) * test_case.row_bytes;
        const std::size_t output_bytes =
            static_cast<std::size_t>(active_columns) * n_out * sizeof(float);
        const std::size_t sink_bytes =
            static_cast<std::size_t>(compiled_columns - active_columns) *
            n_out * sizeof(float);

        twin::GuardedBuffer input_buffer;
        twin::GuardedBuffer weight_buffer;
        twin::GuardedBuffer output_buffer;
        twin::GuardedBuffer sink_buffer;
        twin::GuardedBuffer row_count_buffer;
        check(input_buffer.allocate(context, label + " inputs", input_bytes, error),
              label + " input allocation", error);
        check(weight_buffer.allocate(context, label + " weights", weight_bytes, error),
              label + " weight allocation", error);
        check(output_buffer.allocate(context, label + " output", output_bytes, error),
              label + " output allocation", error);
        check(sink_buffer.allocate(context, label + " sink", sink_bytes, error),
              label + " sink allocation", error);
        check(row_count_buffer.allocate(context, label + " active rows",
                                        sizeof(active_columns), error),
              label + " active-row allocation", error);
        if (!input_buffer.view() || !weight_buffer.view() ||
            !output_buffer.view() || !sink_buffer.view() ||
            !row_count_buffer.view()) {
            passed = false;
            continue;
        }

        std::memset(input_buffer.payload().data(), 0, input_bytes);
        std::memset(weight_buffer.payload().data(), 0, weight_bytes);
        std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
        std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
        std::memcpy(row_count_buffer.payload().data(), &active_columns,
                    sizeof(active_columns));
        input_buffer.capture_before_dispatch();
        weight_buffer.capture_before_dispatch();
        output_buffer.capture_before_dispatch();
        sink_buffer.capture_before_dispatch();
        row_count_buffer.capture_before_dispatch();

        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command-buffer allocation", error);
        if (!command) {
            passed = false;
            continue;
        }
        MatVecBuffers buffers{
            input_buffer.view(), weight_buffer.view(), output_buffer.view(),
            sink_buffer.view(), row_count_buffer.view()};

        if (case_index == 0u) {
            BufferRef forged_activation = buffers.activations;
            forged_activation.gpu_address += sizeof(std::uint32_t);
            MatVecBuffers forged_buffers = buffers;
            forged_buffers.activations = forged_activation;
            const bool rejects_forged_view =
                !encode_matvec(context, *command, test_case.format,
                               test_case.n_in, n_out, active_columns,
                               compiled_columns, forged_buffers, error) &&
                error.find("not owned by this Context") != std::string::npos;
            check(rejects_forged_view,
                  "engine mat-vec rejects forged GPU view metadata", error);

            BufferRef forged_cpu_view = buffers.activations;
            forged_cpu_view.contents =
                static_cast<std::uint8_t*>(forged_cpu_view.contents) +
                sizeof(std::uint32_t);
            MatVecBuffers forged_cpu_buffers = buffers;
            forged_cpu_buffers.activations = forged_cpu_view;
            const bool rejects_forged_cpu_view =
                !encode_matvec(context, *command, test_case.format,
                               test_case.n_in, n_out, active_columns,
                               compiled_columns, forged_cpu_buffers, error) &&
                error.find("not owned by this Context") != std::string::npos;
            check(rejects_forged_cpu_view,
                  "engine mat-vec rejects mismatched CPU view metadata",
                  error);

            const std::uint32_t wrong_rows = active_columns - 1u;
            std::memcpy(row_count_buffer.payload().data(), &wrong_rows,
                        sizeof(wrong_rows));
            const bool rejects_mismatched_rows =
                !encode_matvec(context, *command, test_case.format,
                               test_case.n_in, n_out, active_columns,
                               compiled_columns, buffers, error) &&
                error == "mat-vec active-row buffer does not match active_columns";
            check(rejects_mismatched_rows,
                  "engine mat-vec rejects a mismatched live-row value", error);
            std::memcpy(row_count_buffer.payload().data(), &active_columns,
                        sizeof(active_columns));

            const bool rejects_bad_width =
                !encode_matvec(context, *command, test_case.format,
                               test_case.n_in, n_out, active_columns, 3u,
                               buffers, error) &&
                error == "mat-vec compiled column width must be 1, 2, 4 or 8";
            check(rejects_bad_width,
                  "engine mat-vec rejects an unavailable NC specialization", error);

            twin::GuardedBuffer input_overlap_storage;
            check(input_overlap_storage.allocate(
                      context, "engine mat-vec input/output overlap",
                      input_bytes + 512u, error),
                  "engine mat-vec input/output overlap allocation", error);
            const BufferRef overlap_backing = input_overlap_storage.view();
            MatVecBuffers overlapping_input_output = buffers;
            overlapping_input_output.activations =
                engine_test_subview(overlap_backing, 0u, input_bytes);
            overlapping_input_output.output =
                engine_test_subview(overlap_backing, 256u, output_bytes);
            const bool rejects_output_input_overlap =
                !encode_matvec(context, *command, test_case.format,
                               test_case.n_in, n_out, active_columns,
                               compiled_columns, overlapping_input_output,
                               error) &&
                error == "mat-vec outputs must not overlap an input or each other";
            check(rejects_output_input_overlap,
                  "engine mat-vec rejects output/input overlap", error);

            twin::GuardedBuffer output_sink_overlap_storage;
            check(output_sink_overlap_storage.allocate(
                      context, "engine mat-vec output/sink overlap",
                      output_bytes + sink_bytes, error),
                  "engine mat-vec output/sink overlap allocation", error);
            const BufferRef output_sink_backing =
                output_sink_overlap_storage.view();
            MatVecBuffers overlapping_output_sink = buffers;
            overlapping_output_sink.output =
                engine_test_subview(output_sink_backing, 0u, output_bytes);
            overlapping_output_sink.sink =
                engine_test_subview(output_sink_backing, output_bytes -
                                        sizeof(float),
                                    sink_bytes);
            const bool rejects_output_sink_overlap =
                !encode_matvec(context, *command, test_case.format,
                               test_case.n_in, n_out, active_columns,
                               compiled_columns, overlapping_output_sink,
                               error) &&
                error == "mat-vec outputs must not overlap an input or each other";
            check(rejects_output_sink_overlap,
                  "engine mat-vec rejects output/sink overlap", error);

            const auto rejects_alias = [&](const char* alias_label,
                                            const MatVecBuffers& alias_buffers) {
                const bool rejected =
                    !encode_matvec(context, *command, test_case.format,
                                   test_case.n_in, n_out, active_columns,
                                   compiled_columns, alias_buffers, error) &&
                    error ==
                        "mat-vec outputs must not overlap an input or each other";
                check(rejected, alias_label, error);
            };

            twin::GuardedBuffer output_weight_overlap_storage;
            check(output_weight_overlap_storage.allocate(
                      context, "engine mat-vec output/weight overlap",
                      weight_bytes + 512u, error),
                  "engine mat-vec output/weight overlap allocation", error);
            MatVecBuffers overlapping_output_weights = buffers;
            const BufferRef output_weight_backing =
                output_weight_overlap_storage.view();
            overlapping_output_weights.output =
                engine_test_subview(output_weight_backing, 0u, output_bytes);
            overlapping_output_weights.weights =
                engine_test_subview(output_weight_backing, 0u, weight_bytes);
            rejects_alias("engine mat-vec rejects output/weight overlap",
                          overlapping_output_weights);

            twin::GuardedBuffer output_active_overlap_storage;
            check(output_active_overlap_storage.allocate(
                      context, "engine mat-vec output/active-row overlap",
                      output_bytes, error),
                  "engine mat-vec output/active-row overlap allocation",
                  error);
            MatVecBuffers overlapping_output_active = buffers;
            const BufferRef output_active_backing =
                output_active_overlap_storage.view();
            std::memcpy(output_active_backing.contents, &active_columns,
                        sizeof(active_columns));
            overlapping_output_active.output =
                engine_test_subview(output_active_backing, 0u, output_bytes);
            overlapping_output_active.active_rows =
                engine_test_subview(output_active_backing, 0u,
                                    sizeof(active_columns));
            rejects_alias("engine mat-vec rejects output/active-row overlap",
                          overlapping_output_active);

            twin::GuardedBuffer sink_input_overlap_storage;
            check(sink_input_overlap_storage.allocate(
                      context, "engine mat-vec sink/input overlap",
                      input_bytes + 512u, error),
                  "engine mat-vec sink/input overlap allocation", error);
            MatVecBuffers overlapping_sink_input = buffers;
            const BufferRef sink_input_backing =
                sink_input_overlap_storage.view();
            overlapping_sink_input.activations =
                engine_test_subview(sink_input_backing, 0u, input_bytes);
            overlapping_sink_input.sink =
                engine_test_subview(sink_input_backing, 256u, sink_bytes);
            rejects_alias("engine mat-vec rejects sink/input overlap",
                          overlapping_sink_input);

            twin::GuardedBuffer sink_weight_overlap_storage;
            check(sink_weight_overlap_storage.allocate(
                      context, "engine mat-vec sink/weight overlap",
                      weight_bytes + 512u, error),
                  "engine mat-vec sink/weight overlap allocation", error);
            MatVecBuffers overlapping_sink_weights = buffers;
            const BufferRef sink_weight_backing =
                sink_weight_overlap_storage.view();
            overlapping_sink_weights.weights =
                engine_test_subview(sink_weight_backing, 0u, weight_bytes);
            overlapping_sink_weights.sink =
                engine_test_subview(sink_weight_backing, 256u, sink_bytes);
            rejects_alias("engine mat-vec rejects sink/weight overlap",
                          overlapping_sink_weights);

            twin::GuardedBuffer sink_active_overlap_storage;
            check(sink_active_overlap_storage.allocate(
                      context, "engine mat-vec sink/active-row overlap",
                      sink_bytes, error),
                  "engine mat-vec sink/active-row overlap allocation", error);
            MatVecBuffers overlapping_sink_active = buffers;
            const BufferRef sink_active_backing =
                sink_active_overlap_storage.view();
            std::memcpy(sink_active_backing.contents, &active_columns,
                        sizeof(active_columns));
            overlapping_sink_active.sink =
                engine_test_subview(sink_active_backing, 0u, sink_bytes);
            overlapping_sink_active.active_rows =
                engine_test_subview(sink_active_backing, 0u,
                                    sizeof(active_columns));
            rejects_alias("engine mat-vec rejects sink/active-row overlap",
                          overlapping_sink_active);
        }

        const bool encoded =
            encode_matvec(context, *command, test_case.format,
                          test_case.n_in, n_out, active_columns,
                          compiled_columns, buffers, error);
        check(encoded, label + " production encoder", error);
        if (!encoded) {
            passed = false;
            continue;
        }
        const bool completed =
            wait_for_engine_command(*command, label, error);
        check(completed, label + " command completes", error);
        if (!completed) {
            passed = false;
            continue;
        }

        const std::vector<std::uint8_t> expected_output(output_bytes, 0u);
        const std::vector<std::uint8_t> expected_sink(sink_bytes, 0u);
        const auto output_mismatch =
            twin::compare_bytes(output_buffer.payload(), expected_output);
        const auto sink_mismatch =
            twin::compare_bytes(sink_buffer.payload(), expected_sink);
        check(!output_mismatch, label + " writes active output columns",
              output_mismatch ? output_mismatch->reason : std::string{});
        check(!sink_mismatch, label + " writes padded columns to sink",
              sink_mismatch ? sink_mismatch->reason : std::string{});
        check(!input_buffer.first_unwritten_change({}),
              label + " leaves activations unchanged");
        check(!weight_buffer.first_unwritten_change({}),
              label + " leaves weights unchanged");
        check(!row_count_buffer.first_unwritten_change({}),
              label + " leaves live-row metadata unchanged");
        const std::array<twin::ByteRange, 1> output_written =
            {{{0u, output_bytes}}};
        const std::array<twin::ByteRange, 1> sink_written =
            {{{0u, sink_bytes}}};
        check(!output_buffer.first_unwritten_change(output_written),
              label + " output guards remain unchanged");
        check(!sink_buffer.first_unwritten_change(sink_written),
              label + " sink guards remain unchanged");
        passed &= !output_mismatch && !sink_mismatch;
    }

    const std::uint32_t numeric_n_out = 7u;
    const bool q8_nc4 = run_engine_quantized_numeric_case<block_q8_0>(
        context, "engine Q8_0 NC=4 numeric", MatVecWeightFormat::q8_0,
        640u, 32u, numeric_n_out, 4u, 3u,
        [](std::vector<block_q8_0>& weights) {
            for (std::size_t block = 0u; block < weights.size(); ++block) {
                weights[block].d = strata::kernels::f16_from_f32(
                    0.125f + static_cast<float>(block % 3u) * 0.0625f);
                std::fill_n(weights[block].qs, 32u, std::int8_t{7});
            }
        },
        [](const auto& weights, const auto& activations, std::uint32_t n_in,
           std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
           auto& expected, std::string& error) {
            return twin::q8_0_matvec_reference(
                weights, activations, n_in, n_out, nc, live_rows, expected,
                error);
        });
    const bool q4_nc2_large_k =
        run_engine_quantized_numeric_case<block_q4_0>(
            context, "engine Q4_0 R=1 NC=2 numeric",
            MatVecWeightFormat::q4_0, 2'048u, 32u, numeric_n_out, 2u, 2u,
            [](std::vector<block_q4_0>& weights) {
                for (std::size_t block = 0u; block < weights.size(); ++block) {
                    weights[block].d = strata::kernels::f16_from_f32(
                        0.125f + static_cast<float>(block % 5u) * 0.03125f);
                    std::fill_n(weights[block].qs, 16u, std::uint8_t{0xff});
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::q4_0_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            });
    const bool q5_nc4 = run_engine_quantized_numeric_case<block_q5_0>(
        context, "engine Q5_0 NC=4 numeric", MatVecWeightFormat::q5_0,
        256u, 32u, numeric_n_out, 4u, 3u,
        [](std::vector<block_q5_0>& weights) {
            for (block_q5_0& weight : weights) {
                weight.d = strata::kernels::f16_from_f32(0.25f);
                std::fill_n(weight.qh, 4u, std::uint8_t{0xff});
                std::fill_n(weight.qs, 16u, std::uint8_t{0xff});
            }
        },
        [](const auto& weights, const auto& activations, std::uint32_t n_in,
           std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
           auto& expected, std::string& error) {
            return twin::q5_0_matvec_reference(
                weights, activations, n_in, n_out, nc, live_rows, expected,
                error);
        });
    const bool iq4_nl_nc4 =
        run_engine_quantized_numeric_case<block_iq4_nl>(
            context, "engine IQ4_NL NC=4 numeric",
            MatVecWeightFormat::iq4_nl, 256u, 32u, numeric_n_out, 4u, 3u,
            [](std::vector<block_iq4_nl>& weights) {
                for (block_iq4_nl& weight : weights) {
                    weight.d = strata::kernels::f16_from_f32(0.25f);
                    std::fill_n(weight.qs, 16u, std::uint8_t{0xff});
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::iq4_nl_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            });
    const bool iq4_xs_nc4 =
        run_engine_quantized_numeric_case<block_iq4_xs>(
            context, "engine IQ4_XS NC=4 numeric",
            MatVecWeightFormat::iq4_xs, 256u, 256u, numeric_n_out, 4u, 3u,
            [](std::vector<block_iq4_xs>& weights) {
                for (block_iq4_xs& weight : weights) {
                    weight.d = strata::kernels::f16_from_f32(0.25f);
                    weight.scales_h = 0xffffu;
                    std::fill_n(weight.scales_l, 4u, std::uint8_t{0xff});
                    std::fill_n(weight.qs, 128u, std::uint8_t{0xff});
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::iq4_xs_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            },
            true);
    const bool iq3_s_nc4 =
        run_engine_quantized_numeric_case<block_iq3_s>(
            context, "engine IQ3_S NC=4 numeric",
            MatVecWeightFormat::iq3_s, 256u, 256u, numeric_n_out, 4u, 3u,
            [](std::vector<block_iq3_s>& weights) {
                for (block_iq3_s& weight : weights) {
                    weight.d = strata::kernels::f16_from_f32(0.25f);
                    std::fill_n(weight.qs, 64u, std::uint8_t{0xff});
                    std::fill_n(weight.qh, 8u, std::uint8_t{0xff});
                    std::fill_n(weight.signs, 32u, std::uint8_t{0});
                    std::fill_n(weight.scales, 4u, std::uint8_t{0xff});
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::iq3_s_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            });
    const bool q4_k_nc4 =
        run_engine_quantized_numeric_case<block_q4_K>(
            context, "engine Q4_K NC=4 numeric", MatVecWeightFormat::q4_k,
            256u, 256u, numeric_n_out, 4u, 3u,
            [](std::vector<block_q4_K>& weights) {
                const std::uint32_t d =
                    strata::kernels::f16_from_f32(0.25f);
                const std::uint32_t dmin =
                    strata::kernels::f16_from_f32(0.125f);
                for (block_q4_K& weight : weights) {
                    weight.dm = d | (dmin << 16u);
                    std::fill_n(weight.scales, 12u, std::uint8_t{0x11});
                    std::fill_n(weight.qs, 128u, std::uint8_t{0xff});
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::q4_k_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            });
    const bool q5_k_nc4 =
        run_engine_quantized_numeric_case<block_q5_K>(
            context, "engine Q5_K NC=4 numeric", MatVecWeightFormat::q5_k,
            256u, 256u, numeric_n_out, 4u, 3u,
            [](std::vector<block_q5_K>& weights) {
                const std::uint32_t d =
                    strata::kernels::f16_from_f32(0.25f);
                const std::uint32_t dmin =
                    strata::kernels::f16_from_f32(0.125f);
                for (block_q5_K& weight : weights) {
                    weight.dm = d | (dmin << 16u);
                    std::fill_n(weight.scales, 12u, std::uint8_t{0x11});
                    std::fill_n(weight.qh, 32u, std::uint8_t{0xff});
                    std::fill_n(weight.qs, 128u, std::uint8_t{0xff});
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::q5_k_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            },
            true);
    const bool q3_k_nc4 =
        run_engine_quantized_numeric_case<block_q3_K>(
            context, "engine Q3_K NC=4 numeric", MatVecWeightFormat::q3_k,
            256u, 256u, numeric_n_out, 4u, 3u,
            [](std::vector<block_q3_K>& weights) {
                for (block_q3_K& weight : weights) {
                    std::fill_n(weight.hmask, 32u, std::uint8_t{0});
                    std::fill_n(weight.qs, 64u, std::uint8_t{0xff});
                    std::fill_n(weight.scales, 12u, std::uint8_t{0});
                    weight.d = strata::kernels::f16_from_f32(0.25f);
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::q3_k_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            });
    const bool q6_k_nc4 =
        run_engine_quantized_numeric_case<block_q6_K>(
            context, "engine Q6_K NC=4 numeric", MatVecWeightFormat::q6_k,
            256u, 256u, numeric_n_out, 4u, 3u,
            [](std::vector<block_q6_K>& weights) {
                for (block_q6_K& weight : weights) {
                    std::fill_n(weight.ql, 128u, std::uint8_t{0xff});
                    std::fill_n(weight.qh, 64u, std::uint8_t{0xff});
                    std::fill_n(weight.scales, 16u, std::int8_t{1});
                    weight.d = strata::kernels::f16_from_f32(0.25f);
                }
            },
            [](const auto& weights, const auto& activations,
               std::uint32_t n_in, std::uint32_t n_out, std::uint32_t nc,
               std::uint32_t live_rows, auto& expected,
               std::string& error) {
                return twin::q6_k_matvec_reference(
                    weights, activations, n_in, n_out, nc, live_rows,
                    expected, error);
            },
            true);
    const bool bf16_nc8_large_k = run_engine_bf16_numeric_case(
        context, 2'560u, 7u, 8u, 5u, 256u, "engine BF16 large-K NC=8");
    const bool bf16_160_thread = run_engine_bf16_numeric_case(
        context, 320u, 7u, 4u, 3u, 160u, "engine BF16 NC=4 160-thread");
    passed &= q8_nc4 && q4_nc2_large_k && q5_nc4 && iq4_nl_nc4 &&
              iq4_xs_nc4 && iq3_s_nc4 && q4_k_nc4 && q5_k_nc4 &&
              q3_k_nc4 && q6_k_nc4 && bf16_nc8_large_k && bf16_160_thread;
    return passed;
}

template <class WeightBlock, class InitializeWeights, class Reference>
bool run_indexed_head_engine_case(
    Context& context, Context& foreign_context, const std::string& name,
    MatVecWeightFormat format,
    InitializeWeights initialize_weights, Reference reference,
    bool test_rejections) {
    constexpr std::uint32_t n_in = 2'560u;
    constexpr std::uint32_t weight_rows = 9u;
    constexpr std::uint32_t output_rows = 7u;
    constexpr std::uint32_t blocks_per_row = n_in / 256u;
    constexpr std::size_t input_bytes = n_in * sizeof(float);
    constexpr std::size_t activation_bytes =
        (n_in / 32u) * sizeof(Q81);
    constexpr std::size_t output_bytes = output_rows * sizeof(float);
    constexpr std::size_t sink_bytes = sizeof(float);
    constexpr std::array<std::int32_t, output_rows> row_indices{
        8, 2, 5, 0, 7, 1, 4};
    const std::size_t weight_bytes =
        static_cast<std::size_t>(weight_rows) * blocks_per_row *
        sizeof(WeightBlock);
    const std::size_t row_map_bytes =
        output_rows * sizeof(std::int32_t);
    const std::string label = "engine indexed " + name + " head";
    std::string error;

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer activation_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer active_rows_buffer;
    twin::GuardedBuffer row_map_buffer;
    check(input_buffer.allocate(context, label + " F32 input",
                                input_bytes, error),
          label + " F32 input allocation", error);
    check(activation_buffer.allocate(context, label + " Q8_1 input",
                                     activation_bytes, error),
          label + " Q8_1 input allocation", error);
    check(weight_buffer.allocate(context, label + " full head",
                                 weight_bytes, error),
          label + " full-head allocation", error);
    check(output_buffer.allocate(context, label + " output",
                                 output_bytes, error),
          label + " output allocation", error);
    check(sink_buffer.allocate(context, label + " sink",
                               sink_bytes, error),
          label + " sink allocation", error);
    check(active_rows_buffer.allocate(context, label + " active rows",
                                      sizeof(std::uint32_t), error),
          label + " active-row allocation", error);
    check(row_map_buffer.allocate(context, label + " row map",
                                  row_map_bytes, error),
          label + " row-map allocation", error);
    if (!input_buffer.view() || !activation_buffer.view() ||
        !weight_buffer.view() || !output_buffer.view() || !sink_buffer.view() ||
        !active_rows_buffer.view() || !row_map_buffer.view()) {
        return false;
    }

    auto* const input =
        reinterpret_cast<float*>(input_buffer.payload().data());
    for (std::uint32_t index = 0u; index < n_in; ++index) {
        input[index] =
            (static_cast<float>(index % 19u) - 9.0f) * 0.0625f + 0.015625f;
    }
    auto* const all_weights =
        reinterpret_cast<WeightBlock*>(weight_buffer.payload().data());
    initialize_weights(std::span<WeightBlock>(
        all_weights, static_cast<std::size_t>(weight_rows) * blocks_per_row));
    std::memset(activation_buffer.payload().data(), 0xa5, activation_bytes);
    std::memset(output_buffer.payload().data(), 0xa5, output_bytes);
    std::memset(sink_buffer.payload().data(), 0xa5, sink_bytes);
    constexpr std::uint32_t active_columns = 1u;
    std::memcpy(active_rows_buffer.payload().data(), &active_columns,
                sizeof(active_columns));
    std::memcpy(row_map_buffer.payload().data(), row_indices.data(),
                row_map_bytes);

    input_buffer.capture_before_dispatch();
    activation_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    active_rows_buffer.capture_before_dispatch();
    row_map_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command-buffer allocation", error);
    if (!command) return false;

    const MatVecBuffers matvec_buffers{
        activation_buffer.view(), weight_buffer.view(), output_buffer.view(),
        sink_buffer.view(), active_rows_buffer.view()};
    const IndexedHeadMatVecBuffers buffers{matvec_buffers,
                                           row_map_buffer.view()};
    bool passed = true;
    if (test_rejections) {
        const auto expect_rejected = [&](const std::string& case_label,
                                         std::uint32_t full_rows,
                                         std::uint32_t rows,
                                         const IndexedHeadMatVecBuffers& test_buffers,
                                         const std::string& expected_error) {
            const bool rejected =
                !encode_indexed_head_matvec(context, *command, format, n_in,
                                            full_rows, rows, test_buffers,
                                            error) &&
                error.find(expected_error) != std::string::npos;
            check(rejected, case_label, error);
            passed &= rejected;
        };

        std::int32_t invalid_index = -1;
        std::memcpy(row_map_buffer.payload().data(), &invalid_index,
                    sizeof(invalid_index));
        expect_rejected(label + " rejects negative row index", weight_rows,
                        output_rows, buffers,
                        "row map contains an index outside");
        invalid_index = static_cast<std::int32_t>(weight_rows);
        std::memcpy(row_map_buffer.payload().data(), &invalid_index,
                    sizeof(invalid_index));
        expect_rejected(label + " rejects out-of-range row index", weight_rows,
                        output_rows, buffers,
                        "row map contains an index outside");
        std::memcpy(row_map_buffer.payload().data(), row_indices.data(),
                    row_map_bytes);
        expect_rejected(label + " rejects output count above full head",
                        output_rows - 1u, output_rows, buffers,
                        "row counts must fit signed indices");
        expect_rejected(label + " rejects signed-index overflow",
                        0x80000001u, output_rows, buffers,
                        "row counts must fit signed indices");

        const bool rejects_format =
            !encode_indexed_head_matvec(
                context, *command, MatVecWeightFormat::q4_0, n_in,
                weight_rows, output_rows, buffers, error) &&
            error == "indexed head format must be Q5_K, Q6_K or IQ4_XS";
        check(rejects_format, label + " rejects unsupported indexed format",
              error);
        passed &= rejects_format;

        std::uint32_t wrong_active_columns = 2u;
        std::memcpy(active_rows_buffer.payload().data(), &wrong_active_columns,
                    sizeof(wrong_active_columns));
        const bool rejects_live_rows =
            !encode_indexed_head_matvec(context, *command, format, n_in,
                                        weight_rows, output_rows, buffers,
                                        error) &&
            error ==
                "mat-vec active-row buffer does not match active_columns";
        check(rejects_live_rows, label + " requires one live column", error);
        passed &= rejects_live_rows;
        std::memcpy(active_rows_buffer.payload().data(), &active_columns,
                    sizeof(active_columns));

        twin::GuardedBuffer alias_buffer;
        check(alias_buffer.allocate(context, label + " alias storage",
                                    row_map_bytes, error),
              label + " alias-storage allocation", error);
        if (!alias_buffer.view()) return false;
        std::memcpy(alias_buffer.payload().data(), row_indices.data(),
                    row_map_bytes);
        IndexedHeadMatVecBuffers output_map_alias = buffers;
        output_map_alias.matvec.output = alias_buffer.view();
        output_map_alias.row_map = alias_buffer.view();
        expect_rejected(label + " rejects output/row-map overlap",
                        weight_rows, output_rows, output_map_alias,
                        "outputs must not overlap an input");

        IndexedHeadMatVecBuffers sink_map_alias = buffers;
        sink_map_alias.matvec.sink =
            engine_test_subview(alias_buffer.view(), 0u, sizeof(float));
        sink_map_alias.row_map = alias_buffer.view();
        expect_rejected(label + " rejects sink/row-map overlap", weight_rows,
                        output_rows, sink_map_alias,
                        "outputs must not overlap an input");

        twin::GuardedBuffer short_map_buffer;
        check(short_map_buffer.allocate(
                  context, label + " short row map",
                  (output_rows - 1u) * sizeof(std::int32_t), error),
              label + " short row-map allocation", error);
        if (!short_map_buffer.view()) return false;
        IndexedHeadMatVecBuffers short_map = buffers;
        short_map.row_map = short_map_buffer.view();
        expect_rejected(label + " rejects undersized row map", weight_rows,
                        output_rows, short_map,
                        "row map buffer is undersized or misaligned");

        twin::GuardedBuffer misaligned_map_storage;
        check(misaligned_map_storage.allocate(
                  context, label + " misaligned row map",
                  row_map_bytes + 1u, error),
              label + " misaligned row-map allocation", error);
        if (!misaligned_map_storage.view()) return false;
        const BufferRef misaligned_view =
            engine_test_subview(misaligned_map_storage.view(), 1u,
                                row_map_bytes);
        std::memcpy(misaligned_view.contents, row_indices.data(), row_map_bytes);
        IndexedHeadMatVecBuffers misaligned_map = buffers;
        misaligned_map.row_map = misaligned_view;
        expect_rejected(label + " rejects misaligned row map", weight_rows,
                        output_rows, misaligned_map,
                        "row map buffer is undersized or misaligned");

        BufferRef forged_cpu_map = buffers.row_map;
        forged_cpu_map.contents =
            static_cast<std::uint8_t*>(forged_cpu_map.contents) +
            sizeof(std::int32_t);
        IndexedHeadMatVecBuffers forged_cpu_metadata = buffers;
        forged_cpu_metadata.row_map = forged_cpu_map;
        expect_rejected(label + " rejects forged row-map CPU metadata",
                        weight_rows, output_rows, forged_cpu_metadata,
                        "not owned by this Context");

        BufferRef forged_gpu_map = buffers.row_map;
        forged_gpu_map.gpu_address += sizeof(std::uint32_t);
        IndexedHeadMatVecBuffers forged_gpu_metadata = buffers;
        forged_gpu_metadata.row_map = forged_gpu_map;
        expect_rejected(label + " rejects forged row-map GPU metadata",
                        weight_rows, output_rows, forged_gpu_metadata,
                        "not owned by this Context");

        BufferRef foreign_row_map = foreign_context.buffers().allocate(
            label + " foreign row map", row_map_bytes, error);
        check(static_cast<bool>(foreign_row_map),
              label + " foreign row-map allocation", error);
        if (!foreign_row_map) return false;
        std::memcpy(foreign_row_map.contents, row_indices.data(),
                    row_map_bytes);
        IndexedHeadMatVecBuffers foreign_map = buffers;
        foreign_map.row_map = foreign_row_map;
        expect_rejected(label + " rejects foreign-context row map",
                        weight_rows, output_rows, foreign_map,
                        "not owned by this Context");

        auto foreign_command = foreign_context.make_command_buffer(
            label + " foreign command", error);
        check(foreign_command != nullptr,
              label + " foreign command-buffer allocation", error);
        if (!foreign_command) return false;
        std::string foreign_command_error;
        const bool rejects_foreign_command =
            !encode_indexed_head_matvec(
                context, *foreign_command, format, n_in, weight_rows,
                output_rows, buffers, foreign_command_error) &&
            foreign_command_error.find("different Context") !=
                std::string::npos;
        check(rejects_foreign_command,
              label + " rejects foreign command buffer",
              foreign_command_error);
        passed &= rejects_foreign_command;
        foreign_command->discard();

        auto enqueued_command = context.make_command_buffer(
            label + " enqueued command", error);
        check(enqueued_command != nullptr,
              label + " enqueued command-buffer allocation", error);
        if (!enqueued_command) return false;
        const bool enqueued = enqueued_command->enqueue(error);
        check(enqueued, label + " enqueues validation command", error);
        if (!enqueued) return false;
        std::string enqueued_error;
        const bool rejects_enqueued =
            !encode_indexed_head_matvec(
                context, *enqueued_command, format, n_in, weight_rows,
                output_rows, buffers, enqueued_error) &&
            enqueued_error.find("before command-buffer submission") !=
                std::string::npos;
        check(rejects_enqueued, label + " rejects enqueued command buffer",
              enqueued_error);
        passed &= rejects_enqueued;
        const bool enqueued_completed =
            wait_for_engine_command(*enqueued_command,
                                    label + " enqueued validation",
                                    error);
        check(enqueued_completed,
              label + " enqueued validation command completes", error);
        if (!enqueued_completed) return false;

        auto committed_command = context.make_command_buffer(
            label + " committed command", error);
        check(committed_command != nullptr,
              label + " committed command-buffer allocation", error);
        if (!committed_command) return false;
        const bool committed = committed_command->commit(error);
        check(committed, label + " commits validation command", error);
        if (!committed) return false;
        std::string committed_error;
        const bool rejects_committed =
            !encode_indexed_head_matvec(
                context, *committed_command, format, n_in, weight_rows,
                output_rows, buffers, committed_error) &&
            committed_error.find("before command-buffer submission") !=
                std::string::npos;
        check(rejects_committed, label + " rejects committed command buffer",
              committed_error);
        passed &= rejects_committed;
        const bool committed_completed =
            committed_command->wait_until_completed(error);
        check(committed_completed,
              label + " committed validation command completes", error);
        if (!committed_completed) return false;

        const std::uint64_t shader_weight_count =
            std::uint64_t{0x7fffffffu} * blocks_per_row;
        expect_rejected(label + " rejects 32-bit weight indexing overflow",
                        0x7fffffffu, output_rows, buffers,
                        "flattened index exceeds the 32-bit shader range");
        check(shader_weight_count > (std::uint64_t{1u} << 32u),
              label + " overflow fixture exceeds shader index capacity");
    }

    const bool quantized = encode_q8_1(
        context, *command, 1u, n_in,
        {input_buffer.view(), activation_buffer.view()}, error);
    check(quantized, label + " Q8_1 encoding", error);
    if (!quantized) return false;
    const bool encoded = encode_indexed_head_matvec(
        context, *command, format, n_in, weight_rows, output_rows, buffers,
        error);
    check(encoded, label + " production encoder", error);
    if (!encoded) return false;
    const bool completed =
        wait_for_engine_command(*command, label, error);
    check(completed, label + " command completes", error);
    if (!completed) return false;

    std::vector<WeightBlock> selected_weights(
        static_cast<std::size_t>(output_rows) * blocks_per_row);
    for (std::uint32_t output_row = 0u; output_row < output_rows;
         ++output_row) {
        const std::size_t source_offset =
            static_cast<std::size_t>(row_indices[output_row]) *
            blocks_per_row;
        std::copy_n(all_weights + source_offset, blocks_per_row,
                    selected_weights.data() +
                        static_cast<std::size_t>(output_row) * blocks_per_row);
    }
    const auto* const activations =
        reinterpret_cast<const Q81*>(activation_buffer.payload().data());
    std::vector<float> expected;
    check(reference(std::span<const WeightBlock>(selected_weights),
                    std::span<const Q81>(activations, n_in / 32u), n_in,
                    output_rows, 1u, 1u, expected, error),
          label + " scalar oracle", error);
    if (expected.size() != output_rows) {
        check(false, label + " scalar oracle result size");
        return false;
    }
    const std::span<const std::uint8_t> expected_bytes(
        reinterpret_cast<const std::uint8_t*>(expected.data()),
        expected.size() * sizeof(float));
    const auto output_mismatch =
        twin::compare_bytes(output_buffer.payload(), expected_bytes);
    check(!output_mismatch, label + " indexed output matches scalar oracle",
          output_mismatch ? output_mismatch->reason : std::string{});
    const bool nonzero = std::any_of(
        expected.begin(), expected.end(), [](float value) { return value != 0.0f; });
    check(nonzero, label + " produces nonzero output");
    check(!input_buffer.first_unwritten_change({}),
          label + " leaves F32 input unchanged");
    check(!weight_buffer.first_unwritten_change({}),
          label + " leaves full-head weights unchanged");
    check(!active_rows_buffer.first_unwritten_change({}),
          label + " leaves active-row metadata unchanged");
    check(!row_map_buffer.first_unwritten_change({}),
          label + " leaves row map unchanged");
    check(!sink_buffer.first_unwritten_change({}),
          label + " leaves unused sink unchanged");
    const std::array<twin::ByteRange, 1> activation_written =
        {{{0u, activation_bytes}}};
    check(!activation_buffer.first_unwritten_change(activation_written),
          label + " Q8_1 activation guards remain unchanged");
    const std::array<twin::ByteRange, 1> output_written =
        {{{0u, output_bytes}}};
    check(!output_buffer.first_unwritten_change(output_written),
          label + " output guards remain unchanged");
    return passed && !output_mismatch && nonzero;
}

bool run_indexed_head_engine_encoders(Context& context) {
    std::string error;
    ContextOptions foreign_options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr,
          "open Context for indexed-head ownership tests", error);
    if (!foreign_context) return false;

    constexpr std::uint32_t read_only_n_in = 256u;
    constexpr std::size_t read_only_activation_bytes =
        (read_only_n_in / 32u) * sizeof(Q81);
    const BufferRef foreign_activations = foreign_context->buffers().allocate(
        "foreign read-only mat-vec activations",
        read_only_activation_bytes, error);
    const BufferRef foreign_weights = foreign_context->buffers().allocate(
        "foreign read-only mat-vec weights", sizeof(block_q5_K), error);
    const BufferRef foreign_active_rows = foreign_context->buffers().allocate(
        "foreign read-only mat-vec active rows", sizeof(std::uint32_t),
        error);
    const BufferRef local_output = context.buffers().allocate(
        "read-only mat-vec foreign output", sizeof(float), error);
    const BufferRef local_sink = context.buffers().allocate(
        "read-only mat-vec foreign sink", sizeof(float), error);
    check(foreign_activations && foreign_weights && foreign_active_rows &&
              local_output && local_sink,
          "read-only mat-vec ownership-test allocations", error);
    if (!foreign_activations || !foreign_weights || !foreign_active_rows ||
        !local_output || !local_sink) {
        return false;
    }
    const std::uint32_t one_live_row = 1u;
    std::memcpy(foreign_active_rows.contents, &one_live_row,
                sizeof(one_live_row));
    const ReadOnlyMatVecBuffers foreign_buffers{
        foreign_context->buffers().read_only(foreign_activations),
        foreign_context->buffers().read_only(foreign_weights),
        foreign_context->buffers().read_only(foreign_active_rows),
        local_output, local_sink};
    auto foreign_command =
        context.make_command_buffer("read-only mat-vec foreign input", error);
    check(foreign_command != nullptr,
          "read-only mat-vec foreign command buffer", error);
    if (!foreign_command) return false;
    const bool foreign_rejected =
        !encode_read_only_matvec(
            context, *foreign_command, MatVecWeightFormat::q5_k,
            read_only_n_in, 1u, 1u, 1u, foreign_buffers, error) &&
        error.find("not owned by this Context") != std::string::npos &&
        foreign_command->unsafe_native_handle() == nullptr;
    check(foreign_rejected,
          "read-only mat-vec rejects foreign Context model views", error);

    const bool q5 = run_indexed_head_engine_case<block_q5_K>(
        context, *foreign_context, "Q5_K", MatVecWeightFormat::q5_k,
        [](std::span<block_q5_K> weights) {
            for (std::size_t index = 0u; index < weights.size(); ++index) {
                block_q5_K& weight = weights[index];
                const std::size_t row = index / 10u;
                weight.dm =
                    strata::kernels::f16_from_f32(
                        0.125f + static_cast<float>(row) * 0.03125f) |
                    (strata::kernels::f16_from_f32(0.0625f) << 16u);
                for (std::size_t i = 0u; i < 12u; ++i) {
                    weight.scales[i] =
                        static_cast<std::uint8_t>(0x11u + ((row + i) % 7u));
                }
                for (std::size_t i = 0u; i < 32u; ++i) {
                    weight.qh[i] =
                        static_cast<std::uint8_t>(row * 37u + i * 3u);
                }
                for (std::size_t i = 0u; i < 128u; ++i) {
                    weight.qs[i] =
                        static_cast<std::uint8_t>(row * 19u + i * 11u);
                }
            }
        },
        [](auto weights, auto activations, std::uint32_t n_in,
           std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
           auto& output, std::string& error) {
            return twin::q5_k_matvec_reference(
                weights, activations, n_in, n_out, nc, live_rows, output,
                error);
        },
        true);
    const bool q6 = run_indexed_head_engine_case<block_q6_K>(
        context, *foreign_context, "Q6_K", MatVecWeightFormat::q6_k,
        [](std::span<block_q6_K> weights) {
            for (std::size_t index = 0u; index < weights.size(); ++index) {
                block_q6_K& weight = weights[index];
                const std::size_t row = index / 10u;
                weight.d = strata::kernels::f16_from_f32(
                    0.125f + static_cast<float>(row) * 0.03125f);
                for (std::size_t i = 0u; i < 128u; ++i) {
                    weight.ql[i] =
                        static_cast<std::uint8_t>(row * 17u + i * 5u);
                }
                for (std::size_t i = 0u; i < 64u; ++i) {
                    weight.qh[i] =
                        static_cast<std::uint8_t>(row * 29u + i * 7u);
                }
                for (std::size_t i = 0u; i < 16u; ++i) {
                    weight.scales[i] = static_cast<std::int8_t>(
                        static_cast<int>((row + i) % 11u) - 5);
                }
            }
        },
        [](auto weights, auto activations, std::uint32_t n_in,
           std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
           auto& output, std::string& error) {
            return twin::q6_k_matvec_reference(
                weights, activations, n_in, n_out, nc, live_rows, output,
                error);
        },
        false);
    const bool iq4_xs = run_indexed_head_engine_case<block_iq4_xs>(
        context, *foreign_context, "IQ4_XS", MatVecWeightFormat::iq4_xs,
        [](std::span<block_iq4_xs> weights) {
            for (std::size_t index = 0u; index < weights.size(); ++index) {
                block_iq4_xs& weight = weights[index];
                const std::size_t row = index / 10u;
                weight.d = strata::kernels::f16_from_f32(
                    0.125f + static_cast<float>(row) * 0.03125f);
                weight.scales_h =
                    static_cast<std::uint16_t>(0x1249u * (row + 1u));
                for (std::size_t i = 0u; i < 4u; ++i) {
                    weight.scales_l[i] =
                        static_cast<std::uint8_t>(0x13u + row * 7u + i * 9u);
                }
                for (std::size_t i = 0u; i < 128u; ++i) {
                    weight.qs[i] =
                        static_cast<std::uint8_t>(row * 23u + i * 13u);
                }
            }
        },
        [](auto weights, auto activations, std::uint32_t n_in,
           std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
           auto& output, std::string& error) {
            return twin::iq4_xs_matvec_reference(
                weights, activations, n_in, n_out, nc, live_rows, output,
                error);
        },
        false);
    return q5 && q6 && iq4_xs;
}

bool run_q8_1_to_q2_0_engine_chain(Context& context) {
    constexpr std::uint32_t width = 64u;
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t n_out = 1u;
    constexpr std::uint32_t active_columns = 1u;
    constexpr std::uint32_t compiled_columns = 1u;
    std::array<float, width> input{};
    input.fill(1.0f);
    block_q2_0 weight{};
    weight.d = strata::kernels::f16_from_f32(1.0f);

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer activation_buffer;
    twin::GuardedBuffer weight_buffer;
    twin::GuardedBuffer output_buffer;
    twin::GuardedBuffer sink_buffer;
    twin::GuardedBuffer row_count_buffer;
    std::string error;
    check(input_buffer.allocate(context, "engine chain F32 input",
                                sizeof(input), error),
          "engine chain input allocation", error);
    check(activation_buffer.allocate(context, "engine chain Q8_1",
                                     2u * sizeof(Q81), error),
          "engine chain Q8_1 allocation", error);
    check(weight_buffer.allocate(context, "engine chain Q2_0",
                                 sizeof(weight), error),
          "engine chain weight allocation", error);
    check(output_buffer.allocate(context, "engine chain output",
                                 sizeof(float), error),
          "engine chain output allocation", error);
    check(sink_buffer.allocate(context, "engine chain sink",
                               sizeof(float), error),
          "engine chain sink allocation", error);
    check(row_count_buffer.allocate(context, "engine chain rows",
                                    sizeof(active_columns), error),
          "engine chain row allocation", error);
    if (!input_buffer.view() || !activation_buffer.view() ||
        !weight_buffer.view() || !output_buffer.view() ||
        !sink_buffer.view() || !row_count_buffer.view()) {
        return false;
    }

    std::memcpy(input_buffer.payload().data(), input.data(), sizeof(input));
    std::memcpy(weight_buffer.payload().data(), &weight, sizeof(weight));
    std::memset(activation_buffer.payload().data(), 0xa5,
                activation_buffer.payload().size());
    std::memset(output_buffer.payload().data(), 0xa5, sizeof(float));
    std::memset(sink_buffer.payload().data(), 0xa5, sizeof(float));
    std::memcpy(row_count_buffer.payload().data(), &active_columns,
                sizeof(active_columns));
    input_buffer.capture_before_dispatch();
    activation_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    sink_buffer.capture_before_dispatch();
    row_count_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer("Q8_1 then Q2_0", error);
    check(command != nullptr, "engine chain command-buffer allocation", error);
    if (!command) return false;

    const bool quantized = encode_q8_1(
        context, *command, rows, width,
        {input_buffer.view(), activation_buffer.view()}, error);
    check(quantized, "engine chain Q8_1 encode", error);
    if (!quantized) return false;
    const bool multiplied = encode_matvec(
        context, *command, MatVecWeightFormat::q2_0, width, n_out,
        active_columns, compiled_columns,
        {activation_buffer.view(), weight_buffer.view(), output_buffer.view(),
         sink_buffer.view(), row_count_buffer.view()},
        error);
    check(multiplied, "engine chain Q2_0 encode", error);
    if (!multiplied) return false;
    const bool completed =
        wait_for_engine_command(*command, "Q8_1 to Q2_0 chain", error);
    check(completed, "engine Q8_1 to Q2_0 queue-order chain completes", error);
    if (!completed) return false;

    const auto* const activations =
        reinterpret_cast<const Q81*>(activation_buffer.payload().data());
    float expected = 0.0f;
    for (std::uint32_t block = 0u; block < width / 32u; ++block) {
        std::int32_t integer_dot = 0;
        for (const std::int8_t code : activations[block].qs) {
            integer_dot -= code;
        }
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(activations[block].ds & 0xffffu);
        const float activation_scale =
            strata::kernels::f32_from_f16(scale_bits);
        const float scale =
            strata::kernels::f32_from_f16(weight.d) * activation_scale;
        expected = std::fma(scale, static_cast<float>(integer_dot), expected);
    }
    float actual = 0.0f;
    std::memcpy(&actual, output_buffer.payload().data(), sizeof(actual));
    check(std::bit_cast<std::uint32_t>(actual) ==
              std::bit_cast<std::uint32_t>(expected),
          "same-command-buffer quantize and mat-vec result is exact");
    check(!input_buffer.first_unwritten_change({}),
          "engine chain input remains unchanged");
    check(!weight_buffer.first_unwritten_change({}),
          "engine chain weight remains unchanged");
    const std::array<twin::ByteRange, 1> activation_written =
        {{{0u, 2u * sizeof(Q81)}}};
    check(!activation_buffer.first_unwritten_change(activation_written),
          "engine chain Q8_1 guards remain unchanged");

    std::memset(activation_buffer.payload().data(), 0xa5,
                activation_buffer.payload().size());
    std::memset(output_buffer.payload().data(), 0xa5, sizeof(float));
    std::memset(sink_buffer.payload().data(), 0xa5, sizeof(float));
    const MatVecBuffers matvec_buffers{
        activation_buffer.view(), weight_buffer.view(), output_buffer.view(),
        sink_buffer.view(), row_count_buffer.view()};
    auto shared_command =
        context.make_command_buffer("Q8_1 to Q2_0 shared Encoder", error);
    check(shared_command != nullptr,
          "engine shared-chain command-buffer allocation", error);
    if (!shared_command) return false;
    const bool shared_encoded = Encoder::encode(
        context, *shared_command, "Q8_1 to Q2_0 shared segment",
        [&](Encoder& segment, std::string& encode_error) {
            if (!encode_q8_1(context, segment, rows, width,
                             {input_buffer.view(), activation_buffer.view()},
                             encode_error)) {
                return false;
            }
            return encode_matvec(
                context, segment, MatVecWeightFormat::q2_0, width, n_out,
                active_columns, compiled_columns, matvec_buffers,
                encode_error);
        },
        error);
    check(shared_encoded, "engine Q8_1 to Q2_0 shared-segment encode",
          error);
    if (!shared_encoded) return false;
    const bool shared_completed =
        wait_for_engine_command(*shared_command,
                                "Q8_1 to Q2_0 shared Encoder", error);
    check(shared_completed, "engine shared-chain command completes", error);
    if (!shared_completed) return false;
    std::memcpy(&actual, output_buffer.payload().data(), sizeof(actual));
    const bool shared_output_exact =
        std::bit_cast<std::uint32_t>(actual) ==
        std::bit_cast<std::uint32_t>(expected);
    check(shared_output_exact,
          "shared Encoder Q8_1 then Q2_0 result is exact");

    std::memset(activation_buffer.payload().data(), 0xa5,
                activation_buffer.payload().size());
    std::memset(output_buffer.payload().data(), 0xa5, sizeof(float));
    std::memset(sink_buffer.payload().data(), 0xa5, sizeof(float));
    const std::uint32_t invalid_active_rows = 2u;
    std::memcpy(row_count_buffer.payload().data(), &invalid_active_rows,
                sizeof(invalid_active_rows));
    std::vector<std::uint8_t> activation_poison(
        activation_buffer.payload().size(), 0xa5u);
    std::array<std::uint8_t, sizeof(float)> output_poison{};
    output_poison.fill(0xa5u);
    auto rejected_command =
        context.make_command_buffer("Q8_1 then invalid Q2_0", error);
    check(rejected_command != nullptr,
          "engine rejected shared-chain command-buffer allocation", error);
    if (!rejected_command) return false;
    const bool rejected_segment_accepted = Encoder::encode(
        context, *rejected_command, "Q8_1 then invalid Q2_0",
        [&](Encoder& segment, std::string& encode_error) {
            if (!encode_q8_1(context, segment, rows, width,
                             {input_buffer.view(), activation_buffer.view()},
                             encode_error)) {
                return false;
            }
            return encode_matvec(
                context, segment, MatVecWeightFormat::q2_0, width, n_out,
                active_columns, compiled_columns, matvec_buffers,
                encode_error);
        },
        error);
    const bool rejected_segment_discarded =
        !rejected_segment_accepted &&
        rejected_command->unsafe_native_handle() == nullptr &&
        error.find("active-row buffer") != std::string::npos;
    check(rejected_segment_discarded,
          "late mat-vec validation failure discards the Q8_1 segment",
          error);
    const bool rejected_segment_outputs_unchanged =
        std::equal(activation_poison.begin(), activation_poison.end(),
                   activation_buffer.payload().begin()) &&
        std::equal(output_poison.begin(), output_poison.end(),
                   output_buffer.payload().begin()) &&
        std::equal(output_poison.begin(), output_poison.end(),
                   sink_buffer.payload().begin());
    check(rejected_segment_outputs_unchanged,
          "discarded shared segment leaves Q8_1 and mat-vec outputs unchanged");
    std::uint32_t active_rows_after_failure = 0u;
    std::memcpy(&active_rows_after_failure, row_count_buffer.payload().data(),
                sizeof(active_rows_after_failure));
    const bool active_rows_unchanged =
        active_rows_after_failure == invalid_active_rows;
    check(active_rows_unchanged,
          "discarded shared segment leaves active-row metadata unchanged");
    return std::bit_cast<std::uint32_t>(actual) ==
               std::bit_cast<std::uint32_t>(expected) &&
           shared_output_exact && rejected_segment_discarded &&
           rejected_segment_outputs_unchanged && active_rows_unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_mv_test strata.metallib [--seed N] [--input-set random|edge|fixture|real]\n");
        return 2;
    }

    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;
    if (!run_window_output_weight_mapping_tests()) ++g_failures;

    std::vector<twin::InputSet> sets;
    if (options.input_set_explicit) {
        sets.push_back(options.input_set);
    } else {
        sets = {twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
    }
    for (const twin::InputSet set : sets) {
        options.input_set = set;
        if (!run_set(*context, set, options)) ++g_failures;
        if (!run_q8_0_set(*context, set, options)) ++g_failures;
        if (!run_q4_0_set(*context, set, options)) ++g_failures;
        if (!run_q5_0_set(*context, set, options)) ++g_failures;
        if (!run_q2_0_set(*context, set, options)) ++g_failures;
        if (!run_iq4_nl_set(*context, set, options)) ++g_failures;
        if (!run_iq4_xs_set(*context, set, options)) ++g_failures;
        if (!run_iq3_s_set(*context, set, options)) ++g_failures;
        if (!run_iq2_xxs_set(*context, set, options)) ++g_failures;
        if (!run_iq2_xs_set(*context, set, options)) ++g_failures;
        if (!run_iq2_s_set(*context, set, options)) ++g_failures;
        if (!run_iq3_xxs_set(*context, set, options)) ++g_failures;
        if (!run_iq1_m_set(*context, set, options)) ++g_failures;
        if (!run_q4_k_set(*context, set, options)) ++g_failures;
        if (!run_q5_k_set(*context, set, options)) ++g_failures;
        if (!run_q3_k_set(*context, set, options)) ++g_failures;
        if (!run_q6_k_set(*context, set, options)) ++g_failures;
        if (!run_bf16_set(*context, set, options)) ++g_failures;
    }
    if (!run_q8_1_engine_encoder(*context)) ++g_failures;
    if (!run_q8_1_buffer_alias_contract(*context)) ++g_failures;
    if (!run_engine_cross_handle_alias_contract(*context)) ++g_failures;
    if (!run_matvec_engine_encoders(*context)) ++g_failures;
    if (!run_readonly_bf16_matvec_encoder(*context)) ++g_failures;
    if (!run_readonly_bf16_router_width(*context)) ++g_failures;
    if (!run_readonly_q8_0_matvec_encoder(*context)) ++g_failures;
    if (!run_readonly_q2_0_matvec_encoder(*context)) ++g_failures;
    if (!run_indexed_head_engine_encoders(*context)) ++g_failures;
    if (!run_q8_1_to_q2_0_engine_chain(*context)) ++g_failures;
    std::printf("metal_mv_test: %s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
