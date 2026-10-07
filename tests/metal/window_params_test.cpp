// Device-free tests for Metal decode-window parameter construction.
#include "strata/metal/window_params.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

strata::metal::WindowSamplerSettings sampler(std::uint32_t row) {
    return {
        0.25f + static_cast<float>(row),
        0.8f,
        0.05f,
        1.1f,
        0.02f,
        0.03f,
        20,
        2,
        0,
        row * 13u,
        row + 1u,
    };
}

void builds_decode_and_verify_rows() {
    using namespace strata::metal;
    std::string error;
    for (std::uint32_t rows = 1; rows <= kWindowMaxRows; ++rows) {
        std::vector<std::uint32_t> tokens(rows);
        std::vector<WindowSamplerSettings> samplers(rows);
        for (std::uint32_t row = 0; row < rows; ++row) {
            tokens[row] = 100u + row;
            samplers[row] = sampler(row);
        }

        WindowParams params{};
        check(build_window_params(2048u, tokens, samplers, 0u,
                                  0x12345678u, 0x9abcdef0u, params, error),
              "builds a window for each supported T");
        if (params.T != rows) continue;
        std::uint32_t expected_nc = 1u;
        while (expected_nc < rows) expected_nc <<= 1u;
        check(params.nc == expected_nc && params.pos0 == 2048u &&
                  params.n_keep == 0u &&
                  params.seed_lo == 0x12345678u &&
                  params.seed_hi == 0x9abcdef0u,
              "sets the batch width, position, keep count and seed");
        for (std::uint32_t row = 0; row < rows; ++row) {
            const std::int32_t n_kv = static_cast<std::int32_t>(2049u + row);
            check(params.tokens[row] == tokens[row],
                  "copies each token into its matching parameter row");
            check(params.step[row].pos == static_cast<std::int32_t>(2048u + row) &&
                      params.step[row].n_kv == n_kv &&
                      params.step[row].n_bid == n_kv / 4 &&
                      params.step[row].width ==
                          std::min(n_kv, static_cast<std::int32_t>(kWindowMaxSelectedCells)),
                  "derives a consistent StepRow for each live row");
            check(params.samp[row].temperature == samplers[row].temperature &&
                      params.samp[row].top_p == samplers[row].top_p &&
                      params.samp[row].min_p == samplers[row].min_p &&
                      params.samp[row].penalty_repeat == samplers[row].penalty_repeat &&
                      params.samp[row].penalty_freq == samplers[row].penalty_freq &&
                      params.samp[row].penalty_present == samplers[row].penalty_present &&
                      params.samp[row].top_k == samplers[row].top_k &&
                      params.samp[row].min_keep == samplers[row].min_keep &&
                      params.samp[row].greedy == samplers[row].greedy &&
                      params.samp[row].counter == 2048u + row &&
                      params.samp[row].hist_off == samplers[row].hist_off &&
                      params.samp[row].hist_len == samplers[row].hist_len,
                  "copies resolved sampler settings and derives the Philox counter");
        }
        for (std::uint32_t row = rows; row < kWindowMaxRows; ++row) {
            const WindowSampRow& samp = params.samp[row];
            check(params.tokens[row] == 0u && params.step[row].pos == 0 &&
                      params.step[row].n_kv == 0 && params.step[row].n_bid == 0 &&
                      params.step[row].width == 0 && samp.temperature == 0.0f &&
                      samp.top_p == 0.0f && samp.min_p == 0.0f &&
                      samp.penalty_repeat == 0.0f && samp.penalty_freq == 0.0f &&
                      samp.penalty_present == 0.0f && samp.top_k == 0 &&
                      samp.min_keep == 0 && samp.greedy == 0 && samp.counter == 0u &&
                      samp.hist_off == 0u && samp.hist_len == 0u,
                  "zeros unused fixed-capacity rows");
        }
    }
}

void builds_write_through_and_width_clamp() {
    using namespace strata::metal;
    const std::array<std::uint32_t, 1> tokens{7u};
    const std::array<WindowSamplerSettings, 1> samplers{sampler(4u)};
    WindowParams params{};
    std::string error;
    check(build_window_params(2050u, tokens, samplers, 1u, 7u, 9u, params, error),
          "builds a one-row write-through parameter block");
    check(params.T == 1u && params.nc == 1u && params.n_keep == 1u,
          "T=1 decode uses one mat-vec column and keeps its sole row");
    check(params.step[0].pos == 2050 && params.step[0].n_kv == 2051 &&
              params.step[0].n_bid == 512 && params.step[0].width == 2051,
          "StepRow reaches the documented indexer boundary");

    check(build_window_params(2051u, tokens, samplers, 0u, 0u, 0u, params, error),
          "builds a one-row window beyond the indexer boundary");
    check(params.step[0].n_bid == 513 && params.step[0].width == 2051,
          "selected width clamps while complete indexer blocks continue growing");
}

void rejects_invalid_windows_transactionally() {
    using namespace strata::metal;
    WindowParams params{};
    params.T = 7u;
    params.nc = 8u;
    params.pos0 = 91u;
    params.n_keep = 3u;
    params.tokens[0] = 77u;
    params.step[0] = {11, 12, 13, 14};
    params.samp[0].counter = 15u;
    std::string error;
    const auto unchanged = [&] {
        return params.T == 7u && params.nc == 8u && params.pos0 == 91u &&
               params.n_keep == 3u && params.tokens[0] == 77u &&
               params.step[0].pos == 11 && params.samp[0].counter == 15u;
    };
    const std::array<std::uint32_t, 1> one_token{1u};
    const std::array<WindowSamplerSettings, 1> one_sampler{sampler(0u)};
    const std::array<WindowSamplerSettings, 2> two_samplers{sampler(0u), sampler(1u)};
    const std::array<std::uint32_t, 9> nine_tokens{};
    const std::array<WindowSamplerSettings, 9> nine_samplers{};

    check(!build_window_params(0u, {}, {}, 0u, 0u, 0u, params, error) &&
              error.find("between one and eight") != std::string::npos,
          "rejects an empty window");
    check(unchanged(), "empty-window failure preserves the output");
    check(!build_window_params(0u, nine_tokens, nine_samplers, 0u, 0u, 0u, params, error) &&
              error.find("between one and eight") != std::string::npos,
          "rejects more than eight rows");
    check(unchanged(), "oversized-window failure preserves the output");
    check(!build_window_params(0u, one_token, two_samplers, 0u, 0u, 0u, params, error) &&
              error.find("sampler row count") != std::string::npos,
          "rejects sampler and token row-count mismatch");
    check(unchanged(), "sampler-count failure preserves the output");
    check(!build_window_params(0u, one_token, one_sampler, 2u, 0u, 0u, params, error) &&
              error.find("keep count") != std::string::npos,
          "rejects a keep count larger than T");
    check(unchanged(), "keep-count failure preserves the output");
    const std::array<std::uint32_t, 2> two_tokens{1u, 2u};
    const std::array<WindowSamplerSettings, 2> two_rows{sampler(0u), sampler(1u)};
    check(!build_window_params(
              static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() - 1),
              two_tokens, two_rows, 0u, 0u, 0u, params, error) &&
              error.find("signed StepRow") != std::string::npos,
          "rejects a multi-row position range that overflows signed StepRow");
    check(unchanged(), "multi-row position-overflow failure preserves the output");

    const std::uint32_t last_position =
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() - 1);
    check(build_window_params(last_position, one_token, one_sampler, 0u,
                              0u, 0u, params, error),
          "accepts the last position whose n_kv fits signed StepRow");
    check(params.step[0].pos == std::numeric_limits<std::int32_t>::max() - 1 &&
              params.step[0].n_kv == std::numeric_limits<std::int32_t>::max(),
          "the last valid position does not overflow StepRow");

    check(!build_window_params(
              static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()),
              one_token, one_sampler, 0u, 0u, 0u, params, error) &&
              error.find("signed StepRow") != std::string::npos,
          "rejects n_kv overflow");
    check(params.step[0].pos == std::numeric_limits<std::int32_t>::max() - 1 &&
              params.step[0].n_kv == std::numeric_limits<std::int32_t>::max(),
          "overflow rejection preserves the last valid parameter block");
}

}  // namespace

int main() {
    builds_decode_and_verify_rows();
    builds_write_through_and_width_clamp();
    rejects_invalid_windows_transactionally();
    if (failures != 0) {
        std::cerr << "window_params_test: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "window_params_test: all checks passed\n";
    return 0;
}
