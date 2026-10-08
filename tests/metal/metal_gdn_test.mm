// Metal twins for the GDN block and its validated encoders (docs/mac/04 §5.5).
#include "metallib_sha256.inc"
#include "oracles/bf16_mv.hpp"
#include "oracles/gdn.hpp"
#include "oracles/moe_plan.hpp"
#include "oracles/moe_route.hpp"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gdn_dispatch.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/tiered.hpp"
#include "strata/metal/window_dispatch.hpp"
#include "strata/metal/window_gdn_dispatch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "twin.hpp"

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kStateElements = GDN_STATE_ELEMENTS;
constexpr std::uint32_t kHistoryElements = GDN_HISTORY_ELEMENTS;
constexpr std::uint32_t kChannels = GDN_CONV_CHANNELS;
constexpr std::uint32_t kValueDim = GDN_VALUE_DIM;
constexpr std::uint32_t kHeads = GDN_V_HEADS;
constexpr std::uint32_t kHeadDim = GDN_HEAD_DIM;

enum class EdgeProfile { regular, all_zero, one_outlier };

int g_failures = 0;

const char* edge_profile_name(EdgeProfile profile) {
    switch (profile) {
        case EdgeProfile::regular: return "";
        case EdgeProfile::all_zero: return " edge=all-zero";
        case EdgeProfile::one_outlier: return " edge=2^10-outlier";
    }
    return "";
}

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
    auto bytes = buffer.payload();
    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
}

template <class T>
std::span<const T> typed(const twin::GuardedBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<const T*>(bytes.data()),
            bytes.size() / sizeof(T)};
}

float sample(twin::InputSet set, std::mt19937_64& rng,
             std::uint32_t index, float scale) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 12> edge{
            0.0f, -0.0f, 0.125f, -0.125f, 0.25f, -0.25f,
            0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f};
        return edge[index % edge.size()] * scale;
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t value =
            static_cast<std::int32_t>((index * 37u + 19u) % 1021u) - 510;
        return static_cast<float>(value) * (scale / 512.0f);
    }
    const std::int32_t value =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(value) * (scale / 32768.0f);
}

template <class F>
bool run_command(Context& context, const std::string& label, F&& encode,
                 std::string& error) {
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    if (!encode(*command, error)) {
        check(false, label + " encode", error);
        command->discard();
        return false;
    }
    if (!command->commit(error)) {
        check(false, label + " submit", error);
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        check(false, label + " native command buffer",
              "command buffer became unavailable");
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        check(false, label + " GPU completion", error);
        return false;
    }
    error.clear();
    return true;
}

bool compare_units(std::span<const float> actual,
                   std::span<const twin::tf> expected, double units,
                   const std::string& label, const twin::Options& options,
                   std::uint32_t rows) {
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, units, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    check(!mismatch, label + " U(" + std::to_string(units) + ")");
    return !mismatch;
}

bool run_conv_l2_case(Context& context, std::uint32_t rows,
                      twin::InputSet set, const twin::Options& options,
                      EdgeProfile edge_profile = EdgeProfile::regular,
                      bool shared_encoder = false) {
    const std::string label =
        "GDN conv_l2 T=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) + edge_profile_name(edge_profile) +
        (shared_encoder ? " shared-encoder" : "");
    std::mt19937_64 rng(options.seed ^ (std::uint64_t{rows} << 32u) ^
                        static_cast<std::uint64_t>(set));
    std::vector<float> history(kHistoryElements);
    std::vector<float> qkv(static_cast<std::size_t>(rows) * kChannels);
    std::vector<float> weights(
        static_cast<std::size_t>(kChannels) * GDN_CONV_KERNEL);
    for (std::uint32_t i = 0u; i < history.size(); ++i) {
        history[i] = sample(set, rng, i, 0.2f);
    }
    for (std::uint32_t i = 0u; i < qkv.size(); ++i) {
        qkv[i] = sample(set, rng, i + 71u, 0.5f);
    }
    for (std::uint32_t i = 0u; i < weights.size(); ++i) {
        weights[i] = sample(set, rng, i + 113u, 0.15f);
    }
    if (edge_profile == EdgeProfile::all_zero) {
        std::fill(history.begin(), history.end(), 0.0f);
        std::fill(qkv.begin(), qkv.end(), 0.0f);
        std::fill(weights.begin(), weights.end(), 0.0f);
    } else if (edge_profile == EdgeProfile::one_outlier) {
        constexpr float base = 1.0f / 1024.0f;
        std::fill(history.begin(), history.end(), base);
        std::fill(qkv.begin(), qkv.end(), base);
        std::fill(weights.begin(), weights.end(), base);
        std::fill_n(qkv.begin() + kChannels, kChannels, 1.0f);
    }
    std::vector<twin::tf> expected;
    oracle::gdn_conv_l2(history, qkv, weights, rows, expected);

    std::string error;
    twin::GuardedBuffer history_buffer, qkv_buffer, weight_buffer, output_buffer;
    check(history_buffer.allocate(context, label + " history",
                                  history.size() * sizeof(float), error),
          label + " history allocation", error);
    check(qkv_buffer.allocate(context, label + " qkv",
                              qkv.size() * sizeof(float), error),
          label + " qkv allocation", error);
    check(weight_buffer.allocate(context, label + " weights",
                                 weights.size() * sizeof(float), error),
          label + " weights allocation", error);
    check(output_buffer.allocate(context, label + " output",
                                 qkv.size() * sizeof(float), error),
          label + " output allocation", error);
    if (!history_buffer.view() || !qkv_buffer.view() ||
        !weight_buffer.view() || !output_buffer.view()) {
        return false;
    }
    std::memcpy(history_buffer.payload().data(), history.data(),
                history.size() * sizeof(float));
    std::memcpy(qkv_buffer.payload().data(), qkv.data(),
                qkv.size() * sizeof(float));
    std::memcpy(weight_buffer.payload().data(), weights.data(),
                weights.size() * sizeof(float));
    const bool poisoned = output_buffer.poison_floats_against(
        expected, twin::Comparison::units, error);
    check(poisoned, label + " output poison", error);
    history_buffer.capture_before_dispatch();
    qkv_buffer.capture_before_dispatch();
    weight_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    const GDNConvL2Buffers buffers{
        history_buffer.view(), qkv_buffer.view(), weight_buffer.view(),
        output_buffer.view()};
    const bool dispatched = run_command(
        context, label, [&](CommandBuffer& command, std::string& err) {
            if (!shared_encoder) {
                return encode_gdn_conv_l2(context, command, rows, buffers,
                                          err);
            }
            return Encoder::encode(
                context, command, "shared GDN convolution and q/k L2",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gdn_conv_l2(context, encoder, rows, buffers,
                                              encode_error);
                },
                err);
        }, error);
    if (!dispatched) return false;
    const auto actual = typed<float>(output_buffer);
    bool passed = compare_units(actual, expected, 128.0, label, options, rows);
    passed &= !history_buffer.first_unwritten_change({});
    passed &= !qkv_buffer.first_unwritten_change({});
    passed &= !weight_buffer.first_unwritten_change({});
    const std::array<twin::ByteRange, 1> output_written{{
        {0u, qkv.size() * sizeof(float)}}};
    const auto output_guard =
        output_buffer.first_unwritten_change(output_written);
    check(!output_guard, label + " output guards",
          output_guard ? output_guard->reason : "");
    check(passed, label + " inputs remain unchanged");
    return passed;
}

bool run_conv_commit_case(Context& context, std::uint32_t rows,
                          const twin::Options& options,
                          EdgeProfile edge_profile = EdgeProfile::regular,
                          bool shared_encoder = false) {
    const std::string label =
        "GDN conv_commit n_keep=" + std::to_string(rows) +
        edge_profile_name(edge_profile) +
        (shared_encoder ? " shared-encoder" : "");
    std::mt19937_64 rng(options.seed ^ (std::uint64_t{rows} << 40u) ^
                        0x434f4e56434f4d4dull);
    std::vector<float> qkv(8u * kChannels);
    std::vector<float> initial(kHistoryElements);
    for (std::uint32_t i = 0u; i < qkv.size(); ++i) {
        qkv[i] = sample(options.input_set, rng, i, 0.4f);
    }
    for (std::uint32_t i = 0u; i < initial.size(); ++i) {
        initial[i] = sample(options.input_set, rng, i + 17u, 0.3f);
    }
    if (edge_profile == EdgeProfile::all_zero) {
        std::fill(qkv.begin(), qkv.end(), 0.0f);
        std::fill(initial.begin(), initial.end(), 0.0f);
    } else if (edge_profile == EdgeProfile::one_outlier) {
        constexpr float base = 1.0f / 1024.0f;
        std::fill(qkv.begin(), qkv.end(), base);
        std::fill(initial.begin(), initial.end(), base);
        std::fill_n(qkv.end() - kChannels, kChannels, 1.0f);
    }
    std::vector<float> expected = initial;
    oracle::gdn_conv_commit(expected, qkv, rows);

    std::string error;
    twin::GuardedBuffer history_buffer, qkv_buffer;
    check(history_buffer.allocate(context, label + " history",
                                  initial.size() * sizeof(float), error),
          label + " history allocation", error);
    check(qkv_buffer.allocate(context, label + " qkv",
                              qkv.size() * sizeof(float), error),
          label + " qkv allocation", error);
    if (!history_buffer.view() || !qkv_buffer.view()) return false;
    std::memcpy(history_buffer.payload().data(), initial.data(),
                initial.size() * sizeof(float));
    std::memcpy(qkv_buffer.payload().data(), qkv.data(),
                qkv.size() * sizeof(float));
    history_buffer.capture_before_dispatch();
    qkv_buffer.capture_before_dispatch();

    const GDNConvCommitBuffers buffers{history_buffer.view(),
                                       qkv_buffer.view()};
    const bool dispatched = run_command(
        context, label, [&](CommandBuffer& command, std::string& err) {
            if (!shared_encoder) {
                return encode_gdn_conv_commit(context, command, rows, buffers,
                                              err);
            }
            return Encoder::encode(
                context, command, "shared GDN convolution history commit",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gdn_conv_commit(
                        context, encoder, rows, buffers, encode_error);
                },
                err);
        }, error);
    if (!dispatched) return false;
    const auto actual = typed<float>(history_buffer);
    const bool exact =
        std::memcmp(actual.data(), expected.data(),
                    expected.size() * sizeof(float)) == 0;
    check(exact, label + " exact three-tap history");
    const std::array<twin::ByteRange, 1> written{{
        {0u, expected.size() * sizeof(float)}}};
    const auto changed = history_buffer.first_unwritten_change(written);
    check(!changed, label + " history guards",
          changed ? changed->reason : "");
    check(!qkv_buffer.first_unwritten_change({}),
          label + " qkv remains read-only");
    return exact && !changed;
}

bool run_ab_case(Context& context, std::uint32_t rows,
                 twin::InputSet set, const twin::Options& options,
                 EdgeProfile edge_profile = EdgeProfile::regular,
                 bool shared_encoder = false) {
    const std::string label =
        "GDN alpha/beta T=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) + edge_profile_name(edge_profile) +
        (shared_encoder ? " shared-encoder" : "");
    std::mt19937_64 rng(options.seed ^ (std::uint64_t{rows} << 36u) ^
                        (static_cast<std::uint64_t>(set) << 20u) ^
                        0x414c504841424554ull);
    std::vector<float> input(static_cast<std::size_t>(rows) * N_EMBD);
    std::vector<std::uint16_t> alpha(
        static_cast<std::size_t>(kHeads) * N_EMBD);
    std::vector<std::uint16_t> beta_weights(alpha.size());
    std::vector<float> dt(kHeads), a(kHeads);
    for (std::uint32_t i = 0u; i < input.size(); ++i) {
        input[i] = sample(set, rng, i, 0.25f);
    }
    for (std::uint32_t i = 0u; i < alpha.size(); ++i) {
        alpha[i] = strata::kernels::bf16_from_f32(
            sample(set, rng, i + 31u, 0.06f));
        beta_weights[i] = strata::kernels::bf16_from_f32(
            sample(set, rng, i + 97u, 0.06f));
    }
    for (std::uint32_t i = 0u; i < kHeads; ++i) {
        dt[i] = sample(set, rng, i + 7u, 0.3f);
        a[i] = -0.02f - std::fabs(sample(set, rng, i + 9u, 0.1f));
    }
    if (edge_profile == EdgeProfile::all_zero) {
        std::fill(input.begin(), input.end(), 0.0f);
        std::fill(alpha.begin(), alpha.end(), 0u);
        std::fill(beta_weights.begin(), beta_weights.end(), 0u);
        std::fill(dt.begin(), dt.end(), 0.0f);
        std::fill(a.begin(), a.end(), 0.0f);
    } else if (edge_profile == EdgeProfile::one_outlier) {
        constexpr float base = 1.0f / 1024.0f;
        std::fill(input.begin(), input.end(), base);
        std::fill_n(input.begin() + N_EMBD, N_EMBD, 1.0f);
        std::fill(alpha.begin(), alpha.end(),
                  strata::kernels::bf16_from_f32(1.0f / 64.0f));
        std::fill(beta_weights.begin(), beta_weights.end(),
                  strata::kernels::bf16_from_f32(1.0f / 64.0f));
        std::fill(dt.begin(), dt.end(), 0.0f);
        std::fill(a.begin(), a.end(), -0.02f);
    }
    std::vector<twin::tf> expected_gate, expected_beta;
    oracle::gdn_ab(input, alpha, beta_weights, dt, a, rows,
                   expected_gate, expected_beta);

    std::string error;
    twin::GuardedBuffer input_buffer, alpha_buffer, beta_weight_buffer;
    twin::GuardedBuffer dt_buffer, a_buffer, gate_buffer, beta_buffer;
    check(input_buffer.allocate(context, label + " input",
                                input.size() * sizeof(float), error),
          label + " input allocation", error);
    check(alpha_buffer.allocate(context, label + " alpha",
                                alpha.size() * sizeof(std::uint16_t), error),
          label + " alpha allocation", error);
    check(beta_weight_buffer.allocate(
              context, label + " beta weights",
              beta_weights.size() * sizeof(std::uint16_t), error),
          label + " beta weight allocation", error);
    check(dt_buffer.allocate(context, label + " dt", dt.size() * sizeof(float),
                             error),
          label + " dt allocation", error);
    check(a_buffer.allocate(context, label + " a", a.size() * sizeof(float),
                            error),
          label + " a allocation", error);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * kHeads * sizeof(float);
    check(gate_buffer.allocate(context, label + " gate", output_bytes, error),
          label + " gate allocation", error);
    check(beta_buffer.allocate(context, label + " beta", output_bytes, error),
          label + " beta allocation", error);
    if (!input_buffer.view() || !alpha_buffer.view() ||
        !beta_weight_buffer.view() || !dt_buffer.view() || !a_buffer.view() ||
        !gate_buffer.view() || !beta_buffer.view()) {
        return false;
    }
    std::memcpy(input_buffer.payload().data(), input.data(),
                input.size() * sizeof(float));
    std::memcpy(alpha_buffer.payload().data(), alpha.data(),
                alpha.size() * sizeof(std::uint16_t));
    std::memcpy(beta_weight_buffer.payload().data(), beta_weights.data(),
                beta_weights.size() * sizeof(std::uint16_t));
    std::memcpy(dt_buffer.payload().data(), dt.data(), dt.size() * sizeof(float));
    std::memcpy(a_buffer.payload().data(), a.data(), a.size() * sizeof(float));
    check(gate_buffer.poison_floats_against(
              expected_gate, twin::Comparison::units, error),
          label + " gate poison", error);
    check(beta_buffer.poison_floats_against(
              expected_beta, twin::Comparison::units, error),
          label + " beta poison", error);
    input_buffer.capture_before_dispatch();
    alpha_buffer.capture_before_dispatch();
    beta_weight_buffer.capture_before_dispatch();
    dt_buffer.capture_before_dispatch();
    a_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    beta_buffer.capture_before_dispatch();

    const GDNABBuffers buffers{
        input_buffer.view(), alpha_buffer.view(), beta_weight_buffer.view(),
        dt_buffer.view(), a_buffer.view(), gate_buffer.view(), beta_buffer.view()};
    const bool dispatched = run_command(
        context, label, [&](CommandBuffer& command, std::string& err) {
            if (!shared_encoder) {
                return encode_gdn_ab(context, command, rows, buffers, err);
            }
            return Encoder::encode(
                context, command, "shared GDN alpha/beta gates",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gdn_ab(context, encoder, rows, buffers,
                                         encode_error);
                },
                err);
        }, error);
    if (!dispatched) return false;
    bool passed = compare_units(typed<float>(gate_buffer), expected_gate, 256.0,
                                label + " softplus gate", options, rows);
    passed &= compare_units(typed<float>(beta_buffer), expected_beta, 128.0,
                            label + " sigmoid beta", options, rows);
    passed &= !input_buffer.first_unwritten_change({});
    passed &= !alpha_buffer.first_unwritten_change({});
    passed &= !beta_weight_buffer.first_unwritten_change({});
    passed &= !dt_buffer.first_unwritten_change({});
    passed &= !a_buffer.first_unwritten_change({});
    const std::array<twin::ByteRange, 1> written{{{0u, output_bytes}}};
    const auto gate_guard = gate_buffer.first_unwritten_change(written);
    const auto beta_guard = beta_buffer.first_unwritten_change(written);
    check(!gate_guard && !beta_guard, label + " output guards",
          gate_guard ? gate_guard->reason
                     : (beta_guard ? beta_guard->reason : ""));
    return passed && !gate_guard && !beta_guard;
}

bool run_step_norm_set(Context& context, twin::InputSet set,
                       const twin::Options& options,
                       EdgeProfile edge_profile = EdgeProfile::regular,
                       bool shared_encoder = false) {
    const std::string set_label =
        "GDN recurrence set=" + std::string(twin::input_set_name(set)) +
        edge_profile_name(edge_profile) +
        (shared_encoder ? " shared-encoder" : "");
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kStateElements) * sizeof(float);
    constexpr std::size_t h_max_bytes =
        static_cast<std::size_t>(8u) * kChannels * sizeof(float);
    constexpr std::size_t heads_max_bytes =
        static_cast<std::size_t>(8u) * kHeads * sizeof(float);
    constexpr std::size_t values_max_bytes =
        static_cast<std::size_t>(8u) * kValueDim * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kHeadDim) * sizeof(float);
    std::string error;
    std::array<twin::GuardedBuffer, 3> state_buffers;
    twin::GuardedBuffer h_buffer, gate_buffer, beta_input_buffer, z_buffer;
    twin::GuardedBuffer gamma_buffer, output_buffer;
    for (std::uint32_t mode = 0u; mode < state_buffers.size(); ++mode) {
        check(state_buffers[mode].allocate(
                  context, set_label + " state " + std::to_string(mode),
                  state_bytes, error),
              set_label + " state allocation", error);
    }
    check(h_buffer.allocate(context, set_label + " h", h_max_bytes, error),
          set_label + " h allocation", error);
    check(gate_buffer.allocate(context, set_label + " gate", heads_max_bytes,
                               error),
          set_label + " gate allocation", error);
    check(beta_input_buffer.allocate(context, set_label + " beta",
                                     heads_max_bytes, error),
          set_label + " beta allocation", error);
    check(z_buffer.allocate(context, set_label + " z", values_max_bytes, error),
          set_label + " z allocation", error);
    check(gamma_buffer.allocate(context, set_label + " gamma", gamma_bytes,
                                error),
          set_label + " gamma allocation", error);
    check(output_buffer.allocate(context, set_label + " output",
                                 values_max_bytes, error),
          set_label + " output allocation", error);
    if (!state_buffers[0].view() || !state_buffers[1].view() ||
        !state_buffers[2].view() || !h_buffer.view() || !gate_buffer.view() ||
        !beta_input_buffer.view() || !z_buffer.view() ||
        !gamma_buffer.view() || !output_buffer.view()) {
        return false;
    }

    std::mt19937_64 rng(options.seed ^ 0x535445504e4f524dull ^
                        (static_cast<std::uint64_t>(set) << 44u));
    std::vector<float> initial_state(kStateElements);
    for (std::uint32_t i = 0u; i < initial_state.size(); ++i) {
        initial_state[i] = sample(set, rng, i, 0.02f);
    }
    std::vector<float> h(8u * kChannels);
    std::vector<float> gates(8u * kHeads);
    std::vector<float> betas(8u * kHeads);
    std::vector<float> z(8u * kValueDim);
    std::vector<float> gamma(kHeadDim);
    for (std::uint32_t i = 0u; i < h.size(); ++i) {
        h[i] = sample(set, rng, i + 31u, 0.3f);
    }
    for (std::uint32_t i = 0u; i < gates.size(); ++i) {
        gates[i] = -2.5f + sample(set, rng, i + 53u, 0.6f);
        betas[i] = 0.2f + 0.6f * std::fabs(sample(set, rng, i + 73u, 1.0f));
        betas[i] = std::min(betas[i], 1.0f);
    }
    for (std::uint32_t i = 0u; i < z.size(); ++i) {
        z[i] = sample(set, rng, i + 101u, 1.0f);
    }
    for (std::uint32_t i = 0u; i < gamma.size(); ++i) {
        gamma[i] = 0.5f + sample(set, rng, i + 7u, 0.5f);
    }
    if (edge_profile == EdgeProfile::all_zero) {
        std::fill(initial_state.begin(), initial_state.end(), 0.0f);
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(gates.begin(), gates.end(), 0.0f);
        std::fill(betas.begin(), betas.end(), 0.0f);
        std::fill(z.begin(), z.end(), 0.0f);
        std::fill(gamma.begin(), gamma.end(), 0.0f);
    } else if (edge_profile == EdgeProfile::one_outlier) {
        constexpr float base = 1.0f / 1024.0f;
        std::fill(initial_state.begin(), initial_state.end(), base);
        std::fill(h.begin(), h.end(), base);
        std::fill_n(h.begin() + kChannels, kChannels, 1.0f);
        std::fill(gates.begin(), gates.end(), 0.0f);
        std::fill(betas.begin(), betas.end(), 1.0f);
        std::fill(z.begin(), z.end(), 0.0f);
        std::fill(gamma.begin(), gamma.end(), 1.0f);
    }
    std::memcpy(h_buffer.payload().data(), h.data(), h.size() * sizeof(float));
    std::memcpy(gate_buffer.payload().data(), gates.data(),
                gates.size() * sizeof(float));
    std::memcpy(beta_input_buffer.payload().data(), betas.data(),
                betas.size() * sizeof(float));
    std::memcpy(z_buffer.payload().data(), z.data(), z.size() * sizeof(float));
    std::memcpy(gamma_buffer.payload().data(), gamma.data(),
                gamma.size() * sizeof(float));

    bool all_passed = true;
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        if (shared_encoder && rows != 1u && rows != 8u) continue;
        std::vector<float> oracle_window_state = initial_state;
        std::vector<float> oracle_commit_state = initial_state;
        std::vector<twin::tf> expected_output, unused_state;
        std::vector<twin::tf> expected_state;
        const auto h_rows = std::span<const float>(h).first(
            static_cast<std::size_t>(rows) * kChannels);
        const auto gate_rows = std::span<const float>(gates).first(
            static_cast<std::size_t>(rows) * kHeads);
        const auto beta_rows = std::span<const float>(betas).first(
            static_cast<std::size_t>(rows) * kHeads);
        const auto z_rows = std::span<const float>(z).first(
            static_cast<std::size_t>(rows) * kValueDim);
        oracle::gdn_step_norm(
            oracle_window_state, h_rows, gate_rows, beta_rows, z_rows, gamma,
            rows, GDNStateMode::window, expected_output, unused_state);
        oracle::gdn_step_norm(
            oracle_commit_state, h_rows, gate_rows, beta_rows, z_rows, gamma,
            rows, GDNStateMode::commit, unused_state, expected_state);

        std::vector<float> mode_states[3];
        for (std::uint32_t mode_index = 0u; mode_index < 3u; ++mode_index) {
            const auto mode = static_cast<GDNStateMode>(mode_index);
            const std::string label =
                set_label + " T=" + std::to_string(rows) + " mode=" +
                std::to_string(mode_index);
            std::memcpy(state_buffers[mode_index].payload().data(),
                        initial_state.data(), state_bytes);
            auto out = typed<float>(output_buffer);
            std::fill(out.begin(), out.end(), -123.25f);
            state_buffers[mode_index].capture_before_dispatch();
            h_buffer.capture_before_dispatch();
            gate_buffer.capture_before_dispatch();
            beta_input_buffer.capture_before_dispatch();
            z_buffer.capture_before_dispatch();
            gamma_buffer.capture_before_dispatch();
            output_buffer.capture_before_dispatch();

            const GDNStepNormBuffers buffers{
                state_buffers[mode_index].view(), h_buffer.view(),
                gate_buffer.view(), beta_input_buffer.view(), z_buffer.view(),
                gamma_buffer.view(), output_buffer.view()};
            const bool dispatched = run_command(
                context, label,
                [&](CommandBuffer& command, std::string& err) {
                    if (!shared_encoder) {
                        return encode_gdn_step_norm(context, command, rows,
                                                    mode, buffers, err);
                    }
                    return Encoder::encode(
                        context, command, "shared GDN recurrent step norm",
                        [&](Encoder& encoder, std::string& encode_error) {
                            return encode_gdn_step_norm(
                                context, encoder, rows, mode, buffers,
                                encode_error);
                        },
                        err);
                },
                error);
            if (!dispatched) {
                all_passed = false;
                continue;
            }

            const std::size_t output_count =
                static_cast<std::size_t>(rows) * kValueDim;
            const std::size_t active_output_bytes =
                output_count * sizeof(float);
            bool passed = true;
            if (mode == GDNStateMode::commit) {
                const auto changed =
                    output_buffer.first_unwritten_change({});
                check(!changed, label + " commit leaves output untouched",
                      changed ? changed->reason : "");
                passed &= !changed;
            } else {
                const auto actual =
                    std::span<const float>(typed<float>(output_buffer))
                        .first(output_count);
                passed &= compare_units(actual, expected_output, 512.0,
                                        label + " output", options, rows);
                const std::array<twin::ByteRange, 1> written{{
                    {0u, active_output_bytes}}};
                const auto changed =
                    output_buffer.first_unwritten_change(written);
                check(!changed, label + " output guards",
                      changed ? changed->reason : "");
                passed &= !changed;
            }

            const auto actual_state = typed<float>(state_buffers[mode_index]);
            if (mode == GDNStateMode::window) {
                const bool unchanged =
                    std::memcmp(actual_state.data(), initial_state.data(),
                                state_bytes) == 0;
                check(unchanged, label + " window leaves state untouched");
                passed &= unchanged;
                check(!state_buffers[mode_index].first_unwritten_change({}),
                      label + " state guards");
                mode_states[mode_index].assign(actual_state.begin(),
                                               actual_state.end());
            } else {
                passed &= compare_units(actual_state, expected_state, 512.0,
                                        label + " recurrent state", options,
                                        rows);
                const std::array<twin::ByteRange, 1> state_written{{
                    {0u, state_bytes}}};
                const auto changed =
                    state_buffers[mode_index].first_unwritten_change(
                        state_written);
                check(!changed, label + " state guards",
                      changed ? changed->reason : "");
                passed &= !changed;
                mode_states[mode_index].assign(actual_state.begin(),
                                               actual_state.end());
            }

            passed &= !h_buffer.first_unwritten_change({});
            passed &= !gate_buffer.first_unwritten_change({});
            passed &= !beta_input_buffer.first_unwritten_change({});
            passed &= !z_buffer.first_unwritten_change({});
            passed &= !gamma_buffer.first_unwritten_change({});
            check(passed, label + " input immutability");
            all_passed &= passed;
        }
        if (mode_states[1].size() == kStateElements &&
            mode_states[2].size() == kStateElements) {
            const bool same_state =
                std::memcmp(mode_states[1].data(), mode_states[2].data(),
                            state_bytes) == 0;
            check(same_state,
                  set_label + " T=" + std::to_string(rows) +
                      " commit and write-through states are bitwise equal");
            all_passed &= same_state;
        }
    }
    return all_passed;
}

bool test_encoder_validation(Context& context, Context& foreign_context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t float_bytes = sizeof(float);
    const std::size_t hist_bytes =
        static_cast<std::size_t>(kHistoryElements) * float_bytes;
    const std::size_t qkv_bytes =
        static_cast<std::size_t>(rows) * kChannels * float_bytes;
    const std::size_t conv_weight_bytes =
        static_cast<std::size_t>(kChannels) * GDN_CONV_KERNEL * float_bytes;
    const std::size_t ab_weight_bytes =
        static_cast<std::size_t>(kHeads) * N_EMBD * sizeof(std::uint16_t);
    const std::size_t gate_bytes =
        static_cast<std::size_t>(rows) * kHeads * float_bytes;
    const std::size_t state_bytes =
        static_cast<std::size_t>(kStateElements) * float_bytes;
    const std::size_t value_bytes =
        static_cast<std::size_t>(rows) * kValueDim * float_bytes;
    const std::size_t gamma_bytes =
        static_cast<std::size_t>(kHeadDim) * float_bytes;
    std::string error;
    const auto alloc = [&](Context& ctx, const char* name, std::size_t bytes) {
        return ctx.buffers().allocate(name, bytes, error);
    };
    BufferRef hist = alloc(context, "GDN validation history", hist_bytes);
    BufferRef qkv = alloc(context, "GDN validation qkv", qkv_bytes);
    BufferRef conv_weights =
        alloc(context, "GDN validation convolution weights", conv_weight_bytes);
    BufferRef conv_out = alloc(context, "GDN validation convolution output",
                               qkv_bytes);
    BufferRef input = alloc(context, "GDN validation input",
                            rows * N_EMBD * float_bytes);
    BufferRef alpha = alloc(context, "GDN validation alpha", ab_weight_bytes);
    BufferRef beta_weights =
        alloc(context, "GDN validation beta weights", ab_weight_bytes);
    BufferRef dt = alloc(context, "GDN validation dt", kHeads * float_bytes);
    BufferRef a = alloc(context, "GDN validation a", kHeads * float_bytes);
    BufferRef gate = alloc(context, "GDN validation gate", gate_bytes);
    BufferRef beta = alloc(context, "GDN validation beta", gate_bytes);
    BufferRef state = alloc(context, "GDN validation state", state_bytes);
    BufferRef z = alloc(context, "GDN validation z", value_bytes);
    BufferRef gamma = alloc(context, "GDN validation gamma", gamma_bytes);
    BufferRef output = alloc(context, "GDN validation output", value_bytes);
    BufferRef h = conv_out;
    BufferRef foreign_qkv =
        alloc(foreign_context, "GDN foreign qkv", qkv_bytes);
    if (!hist || !qkv || !conv_weights || !conv_out || !input || !alpha ||
        !beta_weights || !dt || !a || !gate || !beta || !state || !z ||
        !gamma || !output || !foreign_qkv) {
        check(false, "GDN validation allocations", error);
        return false;
    }
    auto command = context.make_command_buffer("GDN validation", error);
    if (!command) {
        check(false, "GDN validation command buffer", error);
        return false;
    }
    bool passed = true;
    const auto reject = [&](const std::string& label, auto&& call,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !call() && error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    const GDNConvL2Buffers valid_conv{hist, qkv, conv_weights, conv_out};
    passed &= reject("GDN conv_l2 rejects zero rows",
                     [&] { return encode_gdn_conv_l2(
                               context, *command, 0u, valid_conv, error); },
                     "row count");
    passed &= reject("GDN conv_l2 rejects rows above eight",
                     [&] { return encode_gdn_conv_l2(
                               context, *command, 9u, valid_conv, error); },
                     "row count");
    GDNConvL2Buffers short_qkv = valid_conv;
    --short_qkv.qkv.length;
    passed &= reject("GDN conv_l2 rejects a short qkv view",
                     [&] { return encode_gdn_conv_l2(
                               context, *command, rows, short_qkv, error); },
                     "undersized");
    GDNConvL2Buffers alias_conv = valid_conv;
    alias_conv.output = qkv;
    passed &= reject("GDN conv_l2 rejects output/qkv alias",
                     [&] { return encode_gdn_conv_l2(
                               context, *command, rows, alias_conv, error); },
                     "overlap");
    GDNConvL2Buffers foreign_conv = valid_conv;
    foreign_conv.qkv = foreign_qkv;
    passed &= reject("GDN conv_l2 rejects foreign buffers",
                     [&] { return encode_gdn_conv_l2(
                               context, *command, rows, foreign_conv, error); },
                     "not owned");
    GDNConvL2Buffers forged_cpu = valid_conv;
    forged_cpu.conv_weights.contents =
        static_cast<char*>(forged_cpu.conv_weights.contents) + sizeof(float);
    passed &= reject("GDN conv_l2 rejects forged CPU metadata",
                     [&] { return encode_gdn_conv_l2(
                               context, *command, rows, forged_cpu, error); },
                     "not owned by this Context");
    GDNConvL2Buffers forged_gpu = valid_conv;
    ++forged_gpu.history.gpu_address;
    passed &= reject("GDN conv_l2 rejects forged GPU metadata",
                     [&] { return encode_gdn_conv_l2(
                               context, *command, rows, forged_gpu, error); },
                     "not owned by this Context");

    const GDNABBuffers valid_ab{
        input, alpha, beta_weights, dt, a, gate, beta};
    GDNABBuffers misaligned_alpha = valid_ab;
    misaligned_alpha.alpha =
        alpha.slice(1u, alpha.length - 1u);
    passed &= reject("GDN ab rejects misaligned BF16 weights",
                     [&] { return encode_gdn_ab(
                               context, *command, rows, misaligned_alpha,
                               error); },
                     "misaligned");
    GDNABBuffers alias_ab = valid_ab;
    alias_ab.beta = gate;
    passed &= reject("GDN ab rejects overlapping outputs",
                     [&] { return encode_gdn_ab(
                               context, *command, rows, alias_ab, error); },
                     "overlap");
    GDNABBuffers short_ab = valid_ab;
    --short_ab.input.length;
    passed &= reject("GDN ab rejects a short projection input",
                     [&] { return encode_gdn_ab(
                               context, *command, rows, short_ab, error); },
                     "undersized");
    passed &= reject("GDN ab rejects zero rows",
                     [&] { return encode_gdn_ab(
                               context, *command, 0u, valid_ab, error); },
                     "row count");

    const GDNStepNormBuffers valid_step{
        state, h, gate, beta, z, gamma, output};
    passed &= reject("GDN step rejects zero window rows",
                     [&] { return encode_gdn_step_norm(
                               context, *command, 0u, GDNStateMode::window,
                               valid_step, error); },
                     "row count");
    passed &= reject("GDN step rejects invalid mode",
                     [&] { return encode_gdn_step_norm(
                               context, *command, 1u,
                               static_cast<GDNStateMode>(7u), valid_step,
                               error); },
                     "state mode");
    GDNStepNormBuffers alias_step = valid_step;
    alias_step.output = state;
    passed &= reject("GDN step rejects output/state alias",
                     [&] { return encode_gdn_step_norm(
                               context, *command, rows, GDNStateMode::window,
                               alias_step, error); },
                     "overlap");
    GDNStepNormBuffers short_state = valid_step;
    --short_state.state.length;
    passed &= reject("GDN step rejects a short state",
                     [&] { return encode_gdn_step_norm(
                               context, *command, rows, GDNStateMode::window,
                               short_state, error); },
                     "undersized");
    error.clear();
    const bool zero_commit = encode_gdn_step_norm(
        context, *command, 0u, GDNStateMode::commit, {}, error);
    check(zero_commit, "GDN step accepts an empty commit as a no-op", error);
    passed &= zero_commit;

    const GDNConvCommitBuffers valid_commit{hist, qkv};
    error.clear();
    const bool zero_conv_commit =
        encode_gdn_conv_commit(context, *command, 0u, {}, error);
    check(zero_conv_commit,
          "GDN conv_commit accepts zero rows as a no-op", error);
    passed &= zero_conv_commit;
    GDNConvCommitBuffers alias_commit{hist, hist};
    passed &= reject("GDN conv_commit rejects history/qkv alias",
                     [&] { return encode_gdn_conv_commit(
                               context, *command, rows, alias_commit, error); },
                     "overlap");
    passed &= reject("GDN conv_commit rejects rows above eight",
                     [&] { return encode_gdn_conv_commit(
                               context, *command, 9u, valid_commit, error); },
                     "row count");

    auto foreign_command = foreign_context.make_command_buffer(
        "GDN foreign command buffer", error);
    if (!foreign_command) {
        check(false, "GDN foreign command buffer allocation", error);
        command->discard();
        return false;
    }
    const bool rejected_foreign_command =
        reject("GDN rejects a foreign command buffer",
               [&] { return encode_gdn_conv_l2(
                         context, *foreign_command, rows, valid_conv, error); },
               "different Context");
    passed &= rejected_foreign_command;
    foreign_command->discard();

    auto committed = context.make_command_buffer(
        "GDN committed command buffer", error);
    if (!committed || !committed->commit(error)) {
        check(false, "GDN committed command buffer setup", error);
        command->discard();
        return false;
    }
    id<MTLCommandBuffer> committed_native =
        (__bridge id<MTLCommandBuffer>)committed->unsafe_native_handle();
    [committed_native waitUntilCompleted];
    passed &= reject("GDN rejects a committed command buffer",
                     [&] { return encode_gdn_conv_l2(
                               context, *committed, rows, valid_conv, error); },
                     "before command-buffer submission");

    auto enqueued = context.make_command_buffer("GDN enqueued command buffer",
                                                error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "GDN enqueued command buffer setup", error);
        command->discard();
        return false;
    }
    passed &= reject("GDN rejects an enqueued command buffer",
                     [&] { return encode_gdn_conv_l2(
                               context, *enqueued, rows, valid_conv, error); },
                     "before command-buffer submission");
    if (!enqueued->commit(error)) {
        check(false, "GDN enqueued command buffer drain", error);
        command->discard();
        return false;
    }
    id<MTLCommandBuffer> enqueued_native =
        (__bridge id<MTLCommandBuffer>)enqueued->unsafe_native_handle();
    [enqueued_native waitUntilCompleted];
    passed &= enqueued_native.status == MTLCommandBufferStatusCompleted;
    command->discard();
    return passed;
}

bool test_conv_l2_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GDN conv_l2 shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    constexpr std::size_t float_bytes = sizeof(float);
    const std::size_t history_bytes =
        static_cast<std::size_t>(kHistoryElements) * float_bytes;
    const std::size_t qkv_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * kChannels * float_bytes;
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kChannels) * GDN_CONV_KERNEL * float_bytes;
    const std::size_t output_bytes = qkv_bytes;
    const auto alloc = [&](Context& owner, const std::string& label,
                           std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef history = alloc(context, "GDN shared validation history",
                              history_bytes);
    BufferRef qkv_storage = alloc(context, "GDN shared validation qkv",
                                  qkv_bytes + 2u * float_bytes);
    BufferRef weight_storage = alloc(
        context, "GDN shared validation convolution weights",
        weight_bytes + 2u * float_bytes);
    BufferRef output = alloc(context, "GDN shared validation output",
                             output_bytes);
    BufferRef foreign_qkv = alloc(
        foreign_context, "GDN shared validation foreign qkv", qkv_bytes);
    if (!history || !qkv_storage || !weight_storage || !output ||
        !foreign_qkv) {
        check(false, "GDN conv_l2 shared-validation allocations", error);
        return false;
    }
    const BufferRef qkv = qkv_storage.slice(0u, qkv_bytes);
    const BufferRef weights = weight_storage.slice(0u, weight_bytes);
    const GDNConvL2Buffers valid{history, qkv, weights, output};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GDNConvL2Buffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gdn_conv_l2(
                        context, encoder, rows, buffers, encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const GDNConvL2Buffers empty{};
    passed &= reject("GDN conv_l2 shared Encoder rejects zero rows", 0u,
                     empty, "row count");
    passed &= reject("GDN conv_l2 shared Encoder rejects rows above eight",
                     9u, empty, "row count");
    GDNConvL2Buffers foreign = valid;
    foreign.qkv = foreign_qkv;
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects a foreign Context view", 1u,
        foreign, "not owned by this Context");

    const auto reject_short = [&](const std::string& label,
                                  GDNConvL2Buffers buffers,
                                  BufferRef GDNConvL2Buffers::*member) {
        (buffers.*member).length = float_bytes;
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GDN conv_l2 shared Encoder rejects short history",
                           valid, &GDNConvL2Buffers::history);
    passed &= reject_short("GDN conv_l2 shared Encoder rejects short qkv",
                           valid, &GDNConvL2Buffers::qkv);
    passed &= reject_short(
        "GDN conv_l2 shared Encoder rejects short convolution weights",
        valid, &GDNConvL2Buffers::conv_weights);
    passed &= reject_short("GDN conv_l2 shared Encoder rejects short output",
                           valid, &GDNConvL2Buffers::output);

    GDNConvL2Buffers one_row_for_two = valid;
    one_row_for_two.qkv.length =
        static_cast<std::size_t>(kChannels) * float_bytes;
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects one-row qkv for T=2", 2u,
        one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.output.length =
        static_cast<std::size_t>(kChannels) * float_bytes;
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects one-row output for T=2", 2u,
        one_row_for_two, "undersized");

    GDNConvL2Buffers misaligned = valid;
    misaligned.qkv = qkv_storage.slice(float_bytes / 2u, qkv_bytes);
    passed &= reject("GDN conv_l2 shared Encoder rejects misaligned qkv", 1u,
                     misaligned, "misaligned");
    misaligned = valid;
    misaligned.conv_weights =
        weight_storage.slice(float_bytes / 2u, weight_bytes);
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects misaligned convolution weights",
        1u, misaligned, "misaligned");

    GDNConvL2Buffers out_of_range = valid;
    out_of_range.qkv =
        qkv_storage.slice(qkv_storage.length - float_bytes, 2u * float_bytes);
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects an out-of-range qkv view", 1u,
        out_of_range, "not owned by this Context");

    GDNConvL2Buffers alias = valid;
    alias.output = alias.history;
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects history/output overlap", 1u,
        alias, "overlap");
    alias = valid;
    alias.output = alias.qkv;
    passed &= reject("GDN conv_l2 shared Encoder rejects qkv/output overlap",
                     1u, alias, "overlap");
    alias = valid;
    alias.output = alias.conv_weights;
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects weight/output overlap", 1u,
        alias, "overlap");

    BufferRef partial_storage = alloc(
        context, "GDN shared validation partial-alias backing",
        qkv_bytes + float_bytes);
    if (!partial_storage) {
        check(false, "GDN conv_l2 shared partial-alias allocation", error);
        return false;
    }
    GDNConvL2Buffers partial = valid;
    partial.qkv = partial_storage.slice(0u, qkv_bytes);
    partial.output = partial_storage.slice(float_bytes, qkv_bytes);
    passed &= reject(
        "GDN conv_l2 shared Encoder rejects partial qkv/output overlap", 1u,
        partial, "overlap");

    GDNConvL2Buffers readonly_overlap = valid;
    readonly_overlap.history = readonly_overlap.qkv;
    error.clear();
    auto command = context.make_command_buffer(
        "GDN conv_l2 shared read-only overlap", error);
    if (!command) {
        check(false, "GDN conv_l2 shared read-only overlap command buffer",
              error);
        return false;
    }
    const bool accepts_readonly_overlap = Encoder::encode(
        context, *command, "GDN conv_l2 shared read-only overlap",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_gdn_conv_l2(context, encoder, 1u, readonly_overlap,
                                      encode_error);
        },
        error);
    check(accepts_readonly_overlap,
          "GDN conv_l2 shared Encoder accepts overlapping read-only inputs",
          error);
    passed &= accepts_readonly_overlap;
    if (accepts_readonly_overlap) command->discard();
    return passed;
}

bool test_ab_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GDN alpha/beta shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    constexpr std::size_t float_bytes = sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kHeads) * N_EMBD * sizeof(std::uint16_t);
    constexpr std::size_t head_bytes =
        static_cast<std::size_t>(kHeads) * float_bytes;
    const std::size_t input_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * N_EMBD * float_bytes;
    const std::size_t output_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * kHeads * float_bytes;
    const auto alloc = [&](Context& owner, const std::string& label,
                           std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef input_storage = alloc(
        context, "GDN alpha/beta shared validation input",
        input_bytes + 2u * float_bytes);
    BufferRef alpha_storage = alloc(
        context, "GDN alpha/beta shared validation alpha",
        weight_bytes + 32u);
    BufferRef beta_weight_storage = alloc(
        context, "GDN alpha/beta shared validation beta weights",
        weight_bytes + 32u);
    BufferRef dt_bias = alloc(context, "GDN alpha/beta shared validation dt",
                              head_bytes);
    BufferRef a = alloc(context, "GDN alpha/beta shared validation a",
                        head_bytes);
    BufferRef gate_storage = alloc(
        context, "GDN alpha/beta shared validation gate",
        output_bytes + 2u * float_bytes);
    BufferRef beta_storage = alloc(
        context, "GDN alpha/beta shared validation beta",
        output_bytes + 2u * float_bytes);
    BufferRef foreign_input = alloc(
        foreign_context, "GDN alpha/beta shared foreign input", input_bytes);
    if (!input_storage || !alpha_storage || !beta_weight_storage ||
        !dt_bias || !a || !gate_storage || !beta_storage || !foreign_input) {
        check(false, "GDN alpha/beta shared-validation allocations", error);
        return false;
    }
    const BufferRef input = input_storage.slice(0u, input_bytes);
    const BufferRef alpha = alpha_storage.slice(0u, weight_bytes);
    const BufferRef beta_weights =
        beta_weight_storage.slice(0u, weight_bytes);
    const BufferRef gate = gate_storage.slice(0u, output_bytes);
    const BufferRef beta = beta_storage.slice(0u, output_bytes);
    const GDNABBuffers valid{
        input, alpha, beta_weights, dt_bias, a, gate, beta};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GDNABBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gdn_ab(context, encoder, rows, buffers,
                                         encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    bool passed = true;
    const GDNABBuffers empty{};
    passed &= reject("GDN alpha/beta shared Encoder rejects zero rows", 0u,
                     empty, "row count");
    passed &= reject("GDN alpha/beta shared Encoder rejects rows above eight",
                     9u, empty, "row count");
    GDNABBuffers foreign = valid;
    foreign.input = foreign_input;
    passed &= reject(
        "GDN alpha/beta shared Encoder rejects a foreign Context view", 1u,
        foreign, "not owned by this Context");

    const auto reject_short = [&](const std::string& label,
                                  GDNABBuffers buffers,
                                  BufferRef GDNABBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, buffers, "undersized");
    };
    passed &= reject_short("GDN alpha/beta shared rejects short input",
                           valid, &GDNABBuffers::input);
    passed &= reject_short("GDN alpha/beta shared rejects short alpha",
                           valid, &GDNABBuffers::alpha);
    passed &= reject_short(
        "GDN alpha/beta shared rejects short beta weights",
        valid, &GDNABBuffers::beta_weights);
    passed &= reject_short("GDN alpha/beta shared rejects short dt bias",
                           valid, &GDNABBuffers::dt_bias);
    passed &= reject_short("GDN alpha/beta shared rejects short a",
                           valid, &GDNABBuffers::a);
    passed &= reject_short("GDN alpha/beta shared rejects short gate output",
                           valid, &GDNABBuffers::gate);
    passed &= reject_short("GDN alpha/beta shared rejects short beta output",
                           valid, &GDNABBuffers::beta);

    GDNABBuffers one_row_for_two = valid;
    one_row_for_two.input.length =
        static_cast<std::size_t>(N_EMBD) * float_bytes;
    passed &= reject(
        "GDN alpha/beta shared rejects one-row input for T=2", 2u,
        one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.gate.length = head_bytes;
    passed &= reject(
        "GDN alpha/beta shared rejects one-row gate output for T=2", 2u,
        one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.beta.length = head_bytes;
    passed &= reject(
        "GDN alpha/beta shared rejects one-row beta output for T=2", 2u,
        one_row_for_two, "undersized");

    GDNABBuffers misaligned = valid;
    misaligned.input = input_storage.slice(1u, input_bytes);
    passed &= reject("GDN alpha/beta shared rejects misaligned input", 1u,
                     misaligned, "misaligned");
    misaligned = valid;
    misaligned.alpha = alpha_storage.slice(sizeof(std::uint16_t), weight_bytes);
    passed &= reject("GDN alpha/beta shared rejects misaligned alpha", 1u,
                     misaligned, "misaligned");
    misaligned = valid;
    misaligned.beta_weights =
        beta_weight_storage.slice(sizeof(std::uint16_t), weight_bytes);
    passed &= reject(
        "GDN alpha/beta shared rejects misaligned beta weights", 1u,
        misaligned, "misaligned");
    misaligned = valid;
    misaligned.gate = gate_storage.slice(1u, output_bytes);
    passed &= reject("GDN alpha/beta shared rejects misaligned gate output",
                     1u, misaligned, "misaligned");

    GDNABBuffers out_of_range = valid;
    out_of_range.input = input_storage.slice(
        input_storage.length - sizeof(std::uint16_t),
        2u * sizeof(std::uint16_t));
    passed &= reject(
        "GDN alpha/beta shared rejects an out-of-range input view", 1u,
        out_of_range, "not owned by this Context");

    const std::array gate_aliases{
        std::pair{"input", valid.input},
        std::pair{"alpha", valid.alpha},
        std::pair{"beta weights", valid.beta_weights},
        std::pair{"dt bias", valid.dt_bias},
        std::pair{"a", valid.a},
        std::pair{"beta output", valid.beta},
    };
    for (const auto& [name, view] : gate_aliases) {
        GDNABBuffers alias = valid;
        alias.gate = view;
        passed &= reject(
            "GDN alpha/beta shared rejects gate/" + std::string(name) +
                " overlap",
            1u, alias, "overlap");
    }
    const std::array beta_aliases{
        std::pair{"input", valid.input},
        std::pair{"alpha", valid.alpha},
        std::pair{"beta weights", valid.beta_weights},
        std::pair{"dt bias", valid.dt_bias},
        std::pair{"a", valid.a},
    };
    for (const auto& [name, view] : beta_aliases) {
        GDNABBuffers alias = valid;
        alias.beta = view;
        passed &= reject(
            "GDN alpha/beta shared rejects beta/" + std::string(name) +
                " overlap",
            1u, alias, "overlap");
    }

    BufferRef partial_storage = alloc(
        context, "GDN alpha/beta shared partial-alias backing",
        input_bytes + output_bytes);
    if (!partial_storage) {
        check(false, "GDN alpha/beta shared partial-alias allocation", error);
        return false;
    }
    GDNABBuffers partial = valid;
    partial.input = partial_storage.slice(0u, input_bytes);
    partial.gate = partial_storage.slice(float_bytes, output_bytes);
    passed &= reject(
        "GDN alpha/beta shared rejects partial input/gate overlap", 1u,
        partial, "overlap");

    GDNABBuffers readonly_overlap = valid;
    readonly_overlap.alpha = readonly_overlap.beta_weights;
    error.clear();
    auto command = context.make_command_buffer(
        "GDN alpha/beta shared read-only overlap", error);
    if (!command) {
        check(false, "GDN alpha/beta shared read-only overlap command buffer",
              error);
        return false;
    }
    const bool accepts_readonly_overlap = Encoder::encode(
        context, *command, "GDN alpha/beta shared read-only overlap",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_gdn_ab(context, encoder, 1u, readonly_overlap,
                                 encode_error);
        },
        error);
    check(accepts_readonly_overlap,
          "GDN alpha/beta shared accepts overlapping read-only inputs",
          error);
    passed &= accepts_readonly_overlap;
    if (accepts_readonly_overlap) command->discard();
    return passed;
}

bool test_step_norm_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GDN step-norm shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    constexpr std::size_t float_bytes = sizeof(float);
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kStateElements) * float_bytes;
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * kChannels * float_bytes;
    constexpr std::size_t heads_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * kHeads * float_bytes;
    constexpr std::size_t values_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * kValueDim * float_bytes;
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kHeadDim) * float_bytes;
    const auto alloc = [&](Context& owner, const std::string& label,
                           std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef state_storage = alloc(
        context, "GDN step shared validation state",
        state_bytes + 2u * float_bytes);
    BufferRef h_storage = alloc(
        context, "GDN step shared validation h",
        h_bytes + 2u * float_bytes);
    BufferRef gate_storage = alloc(
        context, "GDN step shared validation gate",
        heads_bytes + 2u * float_bytes);
    BufferRef beta_storage = alloc(
        context, "GDN step shared validation beta",
        heads_bytes + 2u * float_bytes);
    BufferRef z_storage = alloc(
        context, "GDN step shared validation z",
        values_bytes + 2u * float_bytes);
    BufferRef gamma_storage = alloc(
        context, "GDN step shared validation gamma",
        gamma_bytes + 2u * float_bytes);
    BufferRef output_storage = alloc(
        context, "GDN step shared validation output",
        values_bytes + 2u * float_bytes);
    BufferRef foreign_h =
        alloc(foreign_context, "GDN step shared foreign h", h_bytes);
    if (!state_storage || !h_storage || !gate_storage || !beta_storage ||
        !z_storage || !gamma_storage || !output_storage || !foreign_h) {
        check(false, "GDN step shared-validation allocations", error);
        return false;
    }
    const BufferRef state = state_storage.slice(0u, state_bytes);
    const BufferRef h = h_storage.slice(0u, h_bytes);
    const BufferRef gate = gate_storage.slice(0u, heads_bytes);
    const BufferRef beta = beta_storage.slice(0u, heads_bytes);
    const BufferRef z = z_storage.slice(0u, values_bytes);
    const BufferRef gamma = gamma_storage.slice(0u, gamma_bytes);
    const BufferRef output = output_storage.slice(0u, values_bytes);
    const GDNStepNormBuffers valid{state, h, gate, beta, z, gamma, output};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            GDNStateMode mode,
                            const GDNStepNormBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gdn_step_norm(
                        context, encoder, rows, mode, buffers, encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };
    bool passed = true;
    const GDNStepNormBuffers empty{};
    passed &= reject("GDN step shared Encoder rejects invalid mode", 1u,
                     static_cast<GDNStateMode>(7u), empty, "state mode");
    passed &= reject("GDN step shared Encoder rejects zero window rows", 0u,
                     GDNStateMode::window, empty, "1..8");
    passed &= reject("GDN step shared Encoder rejects rows above eight", 9u,
                     GDNStateMode::commit, empty, "0..8");

    error.clear();
    auto zero_commit = context.make_command_buffer(
        "GDN step shared zero-row commit", error);
    if (!zero_commit) {
        check(false, "GDN step shared zero-row commit command buffer", error);
        return false;
    }
    const bool accepts_zero_commit = Encoder::encode(
        context, *zero_commit, "GDN step shared zero-row commit",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_gdn_step_norm(
                context, encoder, 0u, GDNStateMode::commit, {}, encode_error);
        },
        error);
    check(accepts_zero_commit,
          "GDN step shared Encoder accepts empty commit without buffers",
          error);
    passed &= accepts_zero_commit;
    if (accepts_zero_commit) zero_commit->discard();

    GDNStateMode mode = GDNStateMode::window;
    const auto reject_short = [&](const std::string& label,
                                  GDNStepNormBuffers buffers,
                                  BufferRef GDNStepNormBuffers::*member) {
        (buffers.*member).length = 1u;
        return reject(label, 1u, mode, buffers, "undersized");
    };
    passed &= reject_short("GDN step shared rejects short state", valid,
                           &GDNStepNormBuffers::state);
    passed &= reject_short("GDN step shared rejects short h", valid,
                           &GDNStepNormBuffers::h);
    passed &= reject_short("GDN step shared rejects short gate", valid,
                           &GDNStepNormBuffers::gate);
    passed &= reject_short("GDN step shared rejects short beta", valid,
                           &GDNStepNormBuffers::beta);
    passed &= reject_short("GDN step shared rejects short z", valid,
                           &GDNStepNormBuffers::z);
    passed &= reject_short("GDN step shared rejects short gamma", valid,
                           &GDNStepNormBuffers::gamma);
    passed &= reject_short("GDN step shared rejects short output", valid,
                           &GDNStepNormBuffers::output);
    GDNStepNormBuffers short_h = valid;
    short_h.h.length = 1u;
    passed &= reject("GDN step shared identifies the rejected h view", 1u,
                     GDNStateMode::window, short_h, "GDN h buffer");

    GDNStepNormBuffers foreign = valid;
    foreign.h = foreign_h;
    passed &= reject("GDN step shared rejects a foreign Context view", 1u,
                     GDNStateMode::window, foreign,
                     "not owned by this Context");

    GDNStepNormBuffers one_row_for_two = valid;
    one_row_for_two.h.length =
        static_cast<std::size_t>(kChannels) * float_bytes;
    passed &= reject("GDN step shared rejects one-row h for T=2", 2u,
                     GDNStateMode::window, one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.gate.length =
        static_cast<std::size_t>(kHeads) * float_bytes;
    passed &= reject("GDN step shared rejects one-row gate for T=2", 2u,
                     GDNStateMode::window, one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.beta.length =
        static_cast<std::size_t>(kHeads) * float_bytes;
    passed &= reject("GDN step shared rejects one-row beta for T=2", 2u,
                     GDNStateMode::window, one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.z.length =
        static_cast<std::size_t>(kValueDim) * float_bytes;
    passed &= reject("GDN step shared rejects one-row z for T=2", 2u,
                     GDNStateMode::window, one_row_for_two, "undersized");
    one_row_for_two = valid;
    one_row_for_two.output.length =
        static_cast<std::size_t>(kValueDim) * float_bytes;
    passed &= reject("GDN step shared rejects one-row output for T=2", 2u,
                     GDNStateMode::window, one_row_for_two, "undersized");

    GDNStepNormBuffers misaligned = valid;
    misaligned.state = state_storage.slice(1u, state_bytes);
    passed &= reject("GDN step shared rejects misaligned state", 1u,
                     GDNStateMode::window, misaligned, "misaligned");
    misaligned = valid;
    misaligned.h = h_storage.slice(1u, h_bytes);
    passed &= reject("GDN step shared rejects misaligned h", 1u,
                     GDNStateMode::window, misaligned, "misaligned");
    misaligned = valid;
    misaligned.output = output_storage.slice(1u, values_bytes);
    passed &= reject("GDN step shared rejects misaligned output", 1u,
                     GDNStateMode::window, misaligned, "misaligned");

    GDNStepNormBuffers out_of_range = valid;
    out_of_range.h =
        h_storage.slice(h_storage.length - sizeof(std::uint16_t),
                        2u * sizeof(std::uint16_t));
    passed &= reject("GDN step shared rejects an out-of-range h view", 1u,
                     GDNStateMode::window, out_of_range,
                     "not owned by this Context");

    GDNStepNormBuffers alias = valid;
    alias.output = alias.state;
    passed &= reject("GDN step shared rejects output/state overlap", 1u,
                     GDNStateMode::window, alias, "overlap");
    alias = valid;
    alias.output = alias.h;
    passed &= reject("GDN step shared rejects output/h overlap", 1u,
                     GDNStateMode::window, alias, "overlap");
    alias = valid;
    alias.output = alias.z;
    passed &= reject("GDN step shared rejects output/z overlap", 1u,
                     GDNStateMode::window, alias, "overlap");
    alias = valid;
    alias.gate = alias.output;
    passed &= reject("GDN step shared rejects gate/output overlap", 1u,
                     GDNStateMode::window, alias, "overlap");
    alias = valid;
    alias.beta = alias.output;
    passed &= reject("GDN step shared rejects beta/output overlap", 1u,
                     GDNStateMode::window, alias, "overlap");
    alias = valid;
    alias.gamma = alias.output;
    passed &= reject("GDN step shared rejects gamma/output overlap", 1u,
                     GDNStateMode::window, alias, "overlap");

    BufferRef state_alias_storage = alloc(
        context, "GDN step shared partial state-alias backing",
        state_bytes + float_bytes + h_bytes);
    if (!state_alias_storage) {
        check(false, "GDN step shared state-alias allocation", error);
        return false;
    }
    GDNStepNormBuffers state_alias = valid;
    state_alias.state = state_alias_storage.slice(0u, state_bytes);
    state_alias.h = state_alias_storage.slice(
        float_bytes, h_bytes);
    passed &= reject("GDN step shared rejects state/h overlap", 1u,
                     GDNStateMode::window, state_alias, "state must not overlap");

    GDNStepNormBuffers readonly_overlap = valid;
    readonly_overlap.gate = readonly_overlap.h;
    error.clear();
    auto command = context.make_command_buffer(
        "GDN step shared read-only overlap", error);
    if (!command) {
        check(false, "GDN step shared read-only overlap command buffer", error);
        return false;
    }
    const bool accepts_readonly_overlap = Encoder::encode(
        context, *command, "GDN step shared read-only overlap",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_gdn_step_norm(
                context, encoder, 1u, GDNStateMode::window,
                readonly_overlap, encode_error);
        },
        error);
    check(accepts_readonly_overlap,
          "GDN step shared Encoder accepts overlapping read-only inputs",
          error);
    passed &= accepts_readonly_overlap;
    if (accepts_readonly_overlap) command->discard();
    return passed;
}

bool test_conv_commit_shared_encoder_validation(
    const ContextOptions& context_options, Context& foreign_context) {
    std::string error;
    auto validation_context = Context::open(context_options, error);
    check(validation_context != nullptr,
          "open isolated GDN conv_commit shared-validation context", error);
    if (!validation_context) return false;
    Context& context = *validation_context;

    constexpr std::size_t float_bytes = sizeof(float);
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kHistoryElements) * float_bytes;
    constexpr std::size_t qkv_bytes =
        static_cast<std::size_t>(VERIFY_MAX_T) * kChannels * float_bytes;
    const auto alloc = [&](Context& owner, const std::string& label,
                           std::size_t bytes) {
        return owner.buffers().allocate(label, bytes, error);
    };
    BufferRef history = alloc(context, "GDN conv_commit shared history",
                              history_bytes);
    BufferRef qkv_storage = alloc(
        context, "GDN conv_commit shared qkv",
        qkv_bytes + 2u * float_bytes);
    BufferRef foreign_qkv = alloc(
        foreign_context, "GDN conv_commit shared foreign qkv", qkv_bytes);
    if (!history || !qkv_storage || !foreign_qkv) {
        check(false, "GDN conv_commit shared-validation allocations", error);
        return false;
    }
    const BufferRef qkv = qkv_storage.slice(0u, qkv_bytes);
    const GDNConvCommitBuffers valid{history, qkv};
    const auto reject = [&](const std::string& label, std::uint32_t rows,
                            const GDNConvCommitBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool rejected =
            !Encoder::encode(
                context, *command, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_gdn_conv_commit(
                        context, encoder, rows, buffers, encode_error);
                },
                error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    passed &= reject("GDN conv_commit shared rejects rows above eight", 9u,
                     {}, "row count");
    GDNConvCommitBuffers foreign = valid;
    foreign.qkv = foreign_qkv;
    passed &= reject("GDN conv_commit shared rejects a foreign Context view",
                     1u, foreign, "not owned by this Context");

    GDNConvCommitBuffers short_view = valid;
    short_view.history.length = float_bytes;
    passed &= reject("GDN conv_commit shared rejects short history", 1u,
                     short_view, "undersized");
    short_view = valid;
    short_view.qkv.length = float_bytes;
    passed &= reject("GDN conv_commit shared rejects short qkv", 1u,
                     short_view, "undersized");
    GDNConvCommitBuffers one_row_for_two = valid;
    one_row_for_two.qkv.length =
        static_cast<std::size_t>(kChannels) * float_bytes;
    passed &= reject("GDN conv_commit shared rejects one-row qkv for T=2",
                     2u, one_row_for_two, "undersized");

    GDNConvCommitBuffers misaligned = valid;
    misaligned.qkv = qkv_storage.slice(1u, qkv_bytes);
    passed &= reject("GDN conv_commit shared rejects misaligned qkv", 1u,
                     misaligned, "misaligned");
    GDNConvCommitBuffers out_of_range = valid;
    out_of_range.qkv = qkv_storage.slice(
        qkv_storage.length - float_bytes, 2u * float_bytes);
    passed &= reject("GDN conv_commit shared rejects out-of-range qkv", 1u,
                     out_of_range, "not owned by this Context");

    const GDNConvCommitBuffers alias{qkv, qkv};
    passed &= reject("GDN conv_commit shared rejects history/qkv alias", 1u,
                     alias, "overlap");
    BufferRef partial_storage = alloc(
        context, "GDN conv_commit shared partial alias backing",
        qkv_bytes + float_bytes);
    if (!partial_storage) {
        check(false, "GDN conv_commit shared partial-alias allocation", error);
        return false;
    }
    const GDNConvCommitBuffers partial{
        partial_storage.slice(0u, history_bytes),
        partial_storage.slice(float_bytes, qkv_bytes)};
    passed &= reject("GDN conv_commit shared rejects partial history/qkv alias",
                     1u, partial, "overlap");

    error.clear();
    auto zero_commit = context.make_command_buffer(
        "GDN conv_commit shared zero-row no-op", error);
    if (!zero_commit) {
        check(false, "GDN conv_commit shared zero-row command buffer", error);
        return false;
    }
    const bool accepts_zero_rows = Encoder::encode(
        context, *zero_commit, "GDN conv_commit shared zero-row no-op",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_gdn_conv_commit(
                context, encoder, 0u, {}, encode_error);
        },
        error);
    check(accepts_zero_rows,
          "GDN conv_commit shared accepts empty zero-row no-op", error);
    passed &= accepts_zero_rows;
    if (accepts_zero_rows) zero_commit->discard();
    return passed;
}

bool compare_float_bytes(std::span<const float> actual,
                         std::span<const float> expected,
                         const std::string& label,
                         const twin::Options& options,
                         std::uint32_t rows) {
    const auto actual_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(actual.data()),
        actual.size_bytes());
    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()),
        expected.size_bytes());
    const auto mismatch = twin::compare_bytes(actual_bytes, expected_bytes);
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    check(!mismatch, label + " exact bytes");
    return !mismatch;
}

std::vector<Q81> pad_q8_1_rows(std::span<const Q81> active,
                                std::uint32_t width, std::uint32_t rows,
                                std::uint32_t compiled_columns) {
    const std::size_t blocks_per_row = width / 32u;
    std::vector<Q81> padded(
        static_cast<std::size_t>(compiled_columns) * blocks_per_row);
    std::copy(active.begin(), active.end(), padded.begin());
    for (std::uint32_t row = rows; row < compiled_columns; ++row) {
        std::copy_n(active.begin(), blocks_per_row,
                    padded.begin() + static_cast<std::size_t>(row) *
                                         blocks_per_row);
    }
    return padded;
}

void fill_q8_0_weights(std::span<block_q8_0> weights,
                       std::uint32_t n_in, std::uint32_t salt) {
    const std::uint32_t blocks_per_row = n_in / 32u;
    for (std::size_t index = 0u; index < weights.size(); ++index) {
        block_q8_0& block = weights[index];
        block.d = strata::kernels::f16_from_f32(1.0f / 128.0f);
        for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
            const std::uint32_t output_row =
                static_cast<std::uint32_t>(index / blocks_per_row);
            const int code = static_cast<int>(
                (output_row * 11u + static_cast<std::uint32_t>(index) * 3u +
                 lane * 5u + salt) % 9u) - 4;
            block.qs[lane] = static_cast<std::int8_t>(code);
        }
    }
}

bool run_window_gdn_block(Context& context, std::uint32_t rows,
                          bool shared_encoder,
                          const twin::Options& options) {
    const std::string label = "Window GDN block T=" + std::to_string(rows) +
                              (shared_encoder ? " shared-encoder" : "");
    if (rows == 0u || rows > 8u) return false;
    const std::size_t mixed_count =
        static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t qkv_count =
        static_cast<std::size_t>(rows) * kChannels;
    const std::size_t gate_count =
        static_cast<std::size_t>(rows) * kHeads;
    const std::size_t value_count =
        static_cast<std::size_t>(rows) * kValueDim;
    const std::size_t q8_mixed_count =
        mixed_count / 32u;
    const std::size_t q8_value_count =
        value_count / 32u;
    const std::size_t qkv_weight_count =
        static_cast<std::size_t>(kChannels) * (N_EMBD / 32u);
    const std::size_t gate_weight_count =
        static_cast<std::size_t>(kValueDim) * (N_EMBD / 32u);
    const std::size_t output_weight_count =
        static_cast<std::size_t>(N_EMBD) * (kValueDim / 32u);
    const std::size_t ab_weight_count =
        static_cast<std::size_t>(N_EMBD) * kHeads;
    const std::uint32_t active_columns =
        rows <= 1u ? 1u : rows <= 2u ? 2u : rows <= 4u ? 4u : 8u;
    const std::size_t sink_count = std::max<std::size_t>(
        1u, static_cast<std::size_t>(kChannels) *
                (active_columns - rows));

    std::mt19937_64 rng(options.seed ^ (std::uint64_t{rows} << 40u) ^
                        0x57494e444f574744ull);
    std::vector<float> mixed(mixed_count);
    std::vector<float> history(kHistoryElements);
    std::vector<float> conv_weights(
        static_cast<std::size_t>(kChannels) * GDN_CONV_KERNEL);
    std::vector<float> initial_state(kStateElements);
    std::vector<std::uint16_t> alpha(ab_weight_count);
    std::vector<std::uint16_t> beta_weights(ab_weight_count);
    std::vector<float> dt_bias(kHeads);
    std::vector<float> a_values(kHeads);
    std::vector<float> gamma(kHeadDim);
    for (std::uint32_t i = 0u; i < mixed.size(); ++i) {
        mixed[i] = sample(twin::InputSet::fixture, rng, i, 0.2f);
    }
    for (std::uint32_t i = 0u; i < history.size(); ++i) {
        history[i] = sample(twin::InputSet::fixture, rng, i + 17u, 0.04f);
    }
    for (std::uint32_t i = 0u; i < conv_weights.size(); ++i) {
        conv_weights[i] = sample(twin::InputSet::fixture, rng, i + 31u, 0.05f);
    }
    for (std::uint32_t i = 0u; i < initial_state.size(); ++i) {
        initial_state[i] = sample(twin::InputSet::fixture, rng, i + 47u, 0.01f);
    }
    for (std::uint32_t i = 0u; i < alpha.size(); ++i) {
        const int code = static_cast<int>((i * 7u + 3u) % 9u) - 4;
        alpha[i] = strata::kernels::bf16_from_f32(
            static_cast<float>(code) * (1.0f / 8192.0f));
        beta_weights[i] = strata::kernels::bf16_from_f32(
            static_cast<float>(4 - code) * (1.0f / 8192.0f));
    }
    for (std::uint32_t i = 0u; i < kHeads; ++i) {
        dt_bias[i] = static_cast<float>(static_cast<int>(i % 5u) - 2) * 0.01f;
        a_values[i] = -0.04f - static_cast<float>(i % 3u) * 0.01f;
    }
    for (std::uint32_t i = 0u; i < gamma.size(); ++i) {
        gamma[i] = 0.75f + static_cast<float>(i % 11u) * (1.0f / 32.0f);
    }

    twin::GuardedBuffer mixed_buffer, history_buffer, state_buffer;
    twin::GuardedBuffer qkv_weight_buffer, conv_weight_buffer;
    twin::GuardedBuffer alpha_buffer, beta_weight_buffer, dt_buffer, a_buffer;
    twin::GuardedBuffer gate_weight_buffer, gamma_buffer, output_weight_buffer;
    twin::GuardedBuffer xq_buffer, qkv_buffer, h_buffer;
    twin::GuardedBuffer gate_buffer, beta_buffer, z_buffer, y_buffer, yq_buffer;
    twin::GuardedBuffer block_out_buffer, sink_buffer, active_rows_buffer;
    std::string error;
    const auto alloc = [&](twin::GuardedBuffer& buffer, const char* suffix,
                           std::size_t bytes) {
        const bool ok = buffer.allocate(context, label + " " + suffix,
                                        bytes, error);
        check(ok, label + " " + suffix + " allocation", error);
        return ok;
    };
    if (!alloc(mixed_buffer, "mixed", mixed_count * sizeof(float)) ||
        !alloc(history_buffer, "history", history.size() * sizeof(float)) ||
        !alloc(state_buffer, "state", initial_state.size() * sizeof(float)) ||
        !alloc(qkv_weight_buffer, "qkv weights",
               qkv_weight_count * sizeof(block_q8_0)) ||
        !alloc(conv_weight_buffer, "convolution weights",
               conv_weights.size() * sizeof(float)) ||
        !alloc(alpha_buffer, "alpha weights",
               ab_weight_count * sizeof(std::uint16_t)) ||
        !alloc(beta_weight_buffer, "beta weights",
               ab_weight_count * sizeof(std::uint16_t)) ||
        !alloc(dt_buffer, "dt bias", dt_bias.size() * sizeof(float)) ||
        !alloc(a_buffer, "a", a_values.size() * sizeof(float)) ||
        !alloc(gate_weight_buffer, "gate weights",
               gate_weight_count * sizeof(block_q8_0)) ||
        !alloc(gamma_buffer, "gamma", gamma.size() * sizeof(float)) ||
        !alloc(output_weight_buffer, "output weights",
               output_weight_count * sizeof(block_q8_0)) ||
        !alloc(xq_buffer, "xq", q8_mixed_count * sizeof(Q81)) ||
        !alloc(qkv_buffer, "qkv", qkv_count * sizeof(float)) ||
        !alloc(h_buffer, "h", qkv_count * sizeof(float)) ||
        !alloc(gate_buffer, "gate", gate_count * sizeof(float)) ||
        !alloc(beta_buffer, "beta", gate_count * sizeof(float)) ||
        !alloc(z_buffer, "z", value_count * sizeof(float)) ||
        !alloc(y_buffer, "y", value_count * sizeof(float)) ||
        !alloc(yq_buffer, "yq", q8_value_count * sizeof(Q81)) ||
        !alloc(block_out_buffer, "block output",
               mixed_count * sizeof(float)) ||
        !alloc(sink_buffer, "matvec sink", sink_count * sizeof(float)) ||
        !alloc(active_rows_buffer, "active rows", sizeof(std::uint32_t))) {
        return false;
    }

    std::memcpy(mixed_buffer.payload().data(), mixed.data(),
                mixed.size() * sizeof(float));
    std::memcpy(history_buffer.payload().data(), history.data(),
                history.size() * sizeof(float));
    std::memcpy(state_buffer.payload().data(), initial_state.data(),
                initial_state.size() * sizeof(float));
    std::memcpy(conv_weight_buffer.payload().data(), conv_weights.data(),
                conv_weights.size() * sizeof(float));
    std::memcpy(alpha_buffer.payload().data(), alpha.data(),
                alpha.size() * sizeof(std::uint16_t));
    std::memcpy(beta_weight_buffer.payload().data(), beta_weights.data(),
                beta_weights.size() * sizeof(std::uint16_t));
    std::memcpy(dt_buffer.payload().data(), dt_bias.data(),
                dt_bias.size() * sizeof(float));
    std::memcpy(a_buffer.payload().data(), a_values.data(),
                a_values.size() * sizeof(float));
    std::memcpy(gamma_buffer.payload().data(), gamma.data(),
                gamma.size() * sizeof(float));
    fill_q8_0_weights(typed<block_q8_0>(qkv_weight_buffer), N_EMBD, 1u);
    fill_q8_0_weights(typed<block_q8_0>(gate_weight_buffer), N_EMBD, 4u);
    fill_q8_0_weights(typed<block_q8_0>(output_weight_buffer), kValueDim, 7u);
    std::memcpy(active_rows_buffer.payload().data(), &rows, sizeof(rows));

    const std::array output_buffers{
        &xq_buffer, &qkv_buffer, &h_buffer, &gate_buffer, &beta_buffer,
        &z_buffer, &y_buffer, &yq_buffer, &block_out_buffer, &sink_buffer};
    for (twin::GuardedBuffer* buffer : output_buffers) {
        std::fill(buffer->payload().begin(), buffer->payload().end(), 0xa5u);
    }
    const std::array all_buffers{
        &mixed_buffer, &history_buffer, &state_buffer, &qkv_weight_buffer,
        &conv_weight_buffer, &alpha_buffer, &beta_weight_buffer, &dt_buffer,
        &a_buffer, &gate_weight_buffer, &gamma_buffer, &output_weight_buffer,
        &xq_buffer, &qkv_buffer, &h_buffer, &gate_buffer, &beta_buffer,
        &z_buffer, &y_buffer, &yq_buffer, &block_out_buffer, &sink_buffer,
        &active_rows_buffer};
    for (twin::GuardedBuffer* buffer : all_buffers) {
        buffer->capture_before_dispatch();
    }

    const WindowGDNBlockBuffers buffers{
        mixed_buffer.view(), history_buffer.view(), state_buffer.view(),
        MatVecWeightFormat::q8_0, qkv_weight_buffer.view(),
        conv_weight_buffer.view(), alpha_buffer.view(), beta_weight_buffer.view(),
        dt_buffer.view(), a_buffer.view(), MatVecWeightFormat::q8_0,
        gate_weight_buffer.view(), gamma_buffer.view(),
        MatVecWeightFormat::q8_0, output_weight_buffer.view(), xq_buffer.view(),
        qkv_buffer.view(), h_buffer.view(), gate_buffer.view(), beta_buffer.view(),
        z_buffer.view(), y_buffer.view(), yq_buffer.view(), block_out_buffer.view(),
        sink_buffer.view(), active_rows_buffer.view()};

    if (rows == 1u && !shared_encoder) {
        WindowGDNBlockBuffers aliased = buffers;
        aliased.history = aliased.mixed;
        std::string alias_error;
        auto alias_command = context.make_command_buffer(
            "Window GDN cross-stage alias", alias_error);
        check(alias_command != nullptr,
              label + " cross-stage alias command buffer", alias_error);
        if (!alias_command) return false;
        const bool accepted_alias = encode_window_gdn_block(
            context, *alias_command, rows, aliased, alias_error);
        const bool alias_rejected =
            !accepted_alias &&
            alias_command->unsafe_native_handle() == nullptr &&
            alias_error.find("history buffer overlaps mixed") !=
                std::string::npos;
        check(alias_rejected,
              label + " rejects mixed/history cross-stage alias", alias_error);
        if (!alias_rejected) return false;
    }

    const bool dispatched = run_command(
        context, label,
        [&](CommandBuffer& command, std::string& encode_error) {
            if (!shared_encoder) {
                return encode_window_gdn_block(context, command, rows, buffers,
                                               encode_error);
            }
            return Encoder::encode(
                context, command, "shared Window GDN block",
                [&](Encoder& encoder, std::string& body_error) {
                    return encode_window_gdn_block(context, encoder, rows,
                                                   buffers, body_error);
                },
                encode_error);
        },
        error);
    if (!dispatched) return false;

    bool passed = true;
    const std::size_t xq_bytes = q8_mixed_count * sizeof(Q81);
    std::vector<Q81> expected_xq;
    passed &= twin::quantize_q8_1_reference(mixed, N_EMBD, rows,
                                            expected_xq, error);
    check(passed, label + " CPU mixed Q8_1 oracle", error);
    const auto actual_xq_bytes = xq_buffer.payload().first(xq_bytes);
    const auto expected_xq_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected_xq.data()), xq_bytes);
    const auto xq_mismatch = twin::compare_bytes(actual_xq_bytes,
                                                 expected_xq_bytes);
    if (xq_mismatch) {
        twin::report_failure(label.c_str(), options, rows, *xq_mismatch);
    }
    check(!xq_mismatch, label + " mixed Q8_1 matches scalar oracle");
    passed &= !xq_mismatch;

    const auto actual_xq = typed<const Q81>(xq_buffer).first(q8_mixed_count);
    const auto padded_xq = pad_q8_1_rows(actual_xq, N_EMBD, rows,
                                         active_columns);
    const auto qkv_weights = typed<const block_q8_0>(qkv_weight_buffer);
    std::vector<float> expected_qkv;
    const bool qkv_oracle = twin::q8_0_matvec_reference(
        qkv_weights, padded_xq, N_EMBD, kChannels, active_columns, rows,
        expected_qkv, error);
    check(qkv_oracle, label + " qkv scalar matvec oracle", error);
    const auto actual_qkv =
        typed<const float>(qkv_buffer).first(qkv_count);
    const bool qkv_match = qkv_oracle && compare_float_bytes(
        actual_qkv, std::span<const float>(expected_qkv).first(qkv_count),
        label + " qkv projection", options, rows);
    passed &= qkv_match;

    std::vector<twin::tf> expected_h;
    oracle::gdn_conv_l2(history, actual_qkv, conv_weights, rows, expected_h);
    const auto actual_h = typed<const float>(h_buffer).first(qkv_count);
    passed &= compare_units(actual_h, expected_h, 128.0,
                            label + " convolution and q/k L2", options, rows);

    std::vector<twin::tf> expected_gate, expected_beta;
    oracle::gdn_ab(mixed, alpha, beta_weights, dt_bias, a_values, rows,
                   expected_gate, expected_beta);
    const auto actual_gate = typed<const float>(gate_buffer).first(gate_count);
    const auto actual_beta = typed<const float>(beta_buffer).first(gate_count);
    passed &= compare_units(actual_gate, expected_gate, 256.0,
                            label + " alpha/beta gate", options, rows);
    passed &= compare_units(actual_beta, expected_beta, 128.0,
                            label + " beta projection", options, rows);

    std::vector<float> expected_z;
    const auto gate_weights = typed<const block_q8_0>(gate_weight_buffer);
    const bool z_oracle = twin::q8_0_matvec_reference(
        gate_weights, padded_xq, N_EMBD, kValueDim, active_columns, rows,
        expected_z, error);
    check(z_oracle, label + " gate scalar matvec oracle", error);
    const auto actual_z = typed<const float>(z_buffer).first(value_count);
    const bool z_match = z_oracle && compare_float_bytes(
        actual_z, std::span<const float>(expected_z).first(value_count),
        label + " gate projection", options, rows);
    passed &= z_match;

    std::vector<float> oracle_state = initial_state;
    std::vector<twin::tf> expected_y, expected_final_state;
    oracle::gdn_step_norm(oracle_state, actual_h, actual_gate, actual_beta,
                          actual_z, gamma, rows, GDNStateMode::window,
                          expected_y, expected_final_state);
    const auto actual_y = typed<const float>(y_buffer).first(value_count);
    passed &= compare_units(actual_y, expected_y, 512.0,
                            label + " window recurrence and norm", options,
                            rows);
    const bool state_unchanged =
        std::memcmp(state_buffer.payload().data(), initial_state.data(),
                    initial_state.size() * sizeof(float)) == 0;
    check(state_unchanged, label + " window keeps recurrent state unchanged");
    passed &= state_unchanged;

    std::vector<Q81> expected_yq;
    const bool yq_oracle = twin::quantize_q8_1_reference(
        actual_y, kValueDim, rows, expected_yq, error);
    check(yq_oracle, label + " output Q8_1 scalar oracle", error);
    const std::size_t yq_bytes = q8_value_count * sizeof(Q81);
    const auto yq_mismatch = twin::compare_bytes(
        yq_buffer.payload().first(yq_bytes),
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_yq.data()),
            yq_bytes));
    if (yq_mismatch) {
        twin::report_failure(label.c_str(), options, rows, *yq_mismatch);
    }
    check(yq_oracle && !yq_mismatch,
          label + " output Q8_1 matches scalar oracle");
    passed &= yq_oracle && !yq_mismatch;

    std::vector<float> expected_block_out;
    const auto output_weights =
        typed<const block_q8_0>(output_weight_buffer);
    const auto actual_yq = typed<const Q81>(yq_buffer).first(q8_value_count);
    const auto padded_yq = pad_q8_1_rows(actual_yq, kValueDim, rows,
                                         active_columns);
    const bool output_oracle = twin::q8_0_matvec_reference(
        output_weights, padded_yq, kValueDim, N_EMBD, active_columns, rows,
        expected_block_out, error);
    check(output_oracle, label + " output scalar matvec oracle", error);
    const auto actual_block_out =
        typed<const float>(block_out_buffer).first(mixed_count);
    const bool output_match = output_oracle && compare_float_bytes(
        actual_block_out,
        std::span<const float>(expected_block_out).first(mixed_count),
        label + " output projection", options, rows);
    passed &= output_match;
    const std::size_t padded_output_count =
        static_cast<std::size_t>(active_columns - rows) * N_EMBD;
    bool sink_match = output_oracle;
    if (output_oracle && padded_output_count != 0u) {
        sink_match = compare_float_bytes(
            typed<const float>(sink_buffer).first(padded_output_count),
            std::span<const float>(expected_block_out).subspan(
                mixed_count, padded_output_count),
            label + " padded output projection", options, rows);
    }
    passed &= sink_match;

    const std::array<twin::ByteRange, 1> xq_written{{{0u, xq_bytes}}};
    const std::array<twin::ByteRange, 1> qkv_written{{
        {0u, qkv_count * sizeof(float)}}};
    const std::array<twin::ByteRange, 1> gate_written{{
        {0u, gate_count * sizeof(float)}}};
    const std::array<twin::ByteRange, 1> value_written{{
        {0u, value_count * sizeof(float)}}};
    const std::array<twin::ByteRange, 1> yq_written{{{0u, yq_bytes}}};
    const std::array<twin::ByteRange, 1> out_written{{
        {0u, mixed_count * sizeof(float)}}};
    passed &= !xq_buffer.first_unwritten_change(xq_written);
    passed &= !qkv_buffer.first_unwritten_change(qkv_written);
    passed &= !h_buffer.first_unwritten_change(qkv_written);
    passed &= !gate_buffer.first_unwritten_change(gate_written);
    passed &= !beta_buffer.first_unwritten_change(gate_written);
    passed &= !z_buffer.first_unwritten_change(value_written);
    passed &= !y_buffer.first_unwritten_change(value_written);
    passed &= !yq_buffer.first_unwritten_change(yq_written);
    passed &= !block_out_buffer.first_unwritten_change(out_written);
    const std::array<twin::ByteRange, 1> sink_written{{
        {0u, static_cast<std::size_t>(active_columns - rows) * kChannels *
                 sizeof(float)}}};
    passed &= !sink_buffer.first_unwritten_change(sink_written);
    for (const auto* input_buffer : {
             &mixed_buffer, &history_buffer, &state_buffer, &qkv_weight_buffer,
             &conv_weight_buffer, &alpha_buffer, &beta_weight_buffer,
             &dt_buffer, &a_buffer, &gate_weight_buffer, &gamma_buffer,
             &output_weight_buffer, &active_rows_buffer}) {
        passed &= !input_buffer->first_unwritten_change({});
    }
    check(passed, label + " guards and read-only inputs");

    if (rows == 2u) {
        const auto short_gamma = gamma_buffer.view().slice(
            0u, gamma.size() * sizeof(float) - sizeof(float));
        WindowGDNBlockBuffers late_failure = buffers;
        late_failure.gamma = short_gamma;
        for (twin::GuardedBuffer* buffer : output_buffers) {
            std::fill(buffer->payload().begin(), buffer->payload().end(), 0xa5u);
            buffer->capture_before_dispatch();
        }
        auto command = context.make_command_buffer(
            "Window GDN command-buffer late failure", error);
        check(command != nullptr, label + " late-failure command buffer", error);
        if (!command) return false;
        const bool accepted_command = encode_window_gdn_block(
            context, *command, rows, late_failure, error);
        const bool command_discarded =
            !accepted_command && command->unsafe_native_handle() == nullptr &&
            error.find("gamma") != std::string::npos;
        check(command_discarded,
              label + " command-buffer late failure discards prior stages",
              error);
        passed &= command_discarded;
        bool command_outputs_untouched = true;
        for (twin::GuardedBuffer* buffer : output_buffers) {
            command_outputs_untouched &=
                !buffer->first_unwritten_change({});
        }
        check(command_outputs_untouched,
              label + " command-buffer rollback leaves all outputs untouched");
        passed &= command_outputs_untouched;

        for (twin::GuardedBuffer* buffer : output_buffers) {
            std::fill(buffer->payload().begin(), buffer->payload().end(), 0xa5u);
            buffer->capture_before_dispatch();
        }
        auto shared_command = context.make_command_buffer(
            "Window GDN caller-owned late failure", error);
        check(shared_command != nullptr,
              label + " caller-owned late-failure command buffer", error);
        if (!shared_command) return false;
        const bool accepted_shared = Encoder::encode(
            context, *shared_command, "Window GDN caller-owned rollback",
            [&](Encoder& encoder, std::string& body_error) {
                return encode_window_gdn_block(context, encoder, rows,
                                               late_failure, body_error);
            },
            error);
        const bool shared_discarded =
            !accepted_shared &&
            shared_command->unsafe_native_handle() == nullptr &&
            error.find("gamma") != std::string::npos;
        check(shared_discarded,
              label + " caller-owned late failure discards prior stages",
              error);
        passed &= shared_discarded;
        bool shared_outputs_untouched = true;
        for (twin::GuardedBuffer* buffer : output_buffers) {
            shared_outputs_untouched &=
                !buffer->first_unwritten_change({});
        }
        check(shared_outputs_untouched,
              label + " caller-owned rollback leaves all outputs untouched");
        passed &= shared_outputs_untouched;

        const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
        if (dispatch_mode != nullptr &&
            std::string_view(dispatch_mode) == "concurrent") {
            auto group_command = context.make_command_buffer(
                "Window GDN concurrent-group boundary", error);
            check(group_command != nullptr,
                  label + " concurrent-group command buffer", error);
            if (!group_command) return false;
            bool saw_open_group = false;
            bool rejected = false;
            bool commit_rejected = false;
            const WindowGDNCommitBuffers commit_buffers{
                history_buffer.view(), qkv_buffer.view(), state_buffer.view(),
                h_buffer.view(), gate_buffer.view(), beta_buffer.view(),
                z_buffer.view(), gamma_buffer.view(), y_buffer.view()};
            const bool group_encoded = Encoder::encode(
                context, *group_command, "Window GDN group boundary",
                [&](Encoder& encoder, std::string& body_error) {
                    if (!encoder.group_begin(body_error)) return false;
                    saw_open_group = encoder.concurrent_group_open();
                    if (saw_open_group) {
                        std::string block_error;
                        rejected = !encode_window_gdn_block(
                            context, encoder, rows, buffers, block_error) &&
                            block_error.find("ordered concurrent-group boundary") !=
                                std::string::npos;
                        std::string commit_error;
                        commit_rejected = !encode_window_gdn_commit(
                            context, encoder, rows, commit_buffers, commit_error) &&
                            commit_error.find("ordered concurrent-group boundary") !=
                                std::string::npos;
                    }
                    if (!encoder.group_end(body_error)) return false;
                    if (!saw_open_group || !rejected || !commit_rejected) {
                        body_error = "Window GDN coordinator did not reject an open concurrent group";
                        return false;
                    }
                    return true;
                },
                error);
            const bool group_contract = group_encoded && saw_open_group &&
                                        rejected && commit_rejected;
            check(group_contract,
                  label + " block and commit reject open concurrent group",
                  error);
            if (group_command) group_command->discard();
            passed &= group_contract;
        }
    }

    const std::uint32_t accepted_rows = rows == 1u ? 1u : rows - 1u;
    const WindowGDNCommitBuffers commit_buffers{
        history_buffer.view(), qkv_buffer.view(), state_buffer.view(),
        h_buffer.view(), gate_buffer.view(), beta_buffer.view(),
        z_buffer.view(), gamma_buffer.view(), y_buffer.view()};

    if (rows == 1u && !shared_encoder) {
        WindowGDNCommitBuffers aliased = commit_buffers;
        aliased.state = aliased.history;
        std::string alias_error;
        auto alias_command = context.make_command_buffer(
            "Window GDN commit cross-stage alias", alias_error);
        check(alias_command != nullptr,
              label + " commit alias command buffer", alias_error);
        if (!alias_command) return false;
        const bool accepted_alias = encode_window_gdn_commit(
            context, *alias_command, accepted_rows, aliased, alias_error);
        const bool alias_rejected = !accepted_alias &&
            alias_command->unsafe_native_handle() == nullptr &&
            alias_error.find("state buffer overlaps history") !=
                std::string::npos;
        check(alias_rejected,
              label + " commit rejects history/state cross-stage alias",
              alias_error);
        passed &= alias_rejected;
    }

    if (rows == 2u) {
        WindowGDNCommitBuffers late_failure = commit_buffers;
        late_failure.gamma = gamma_buffer.view().slice(
            0u, gamma.size() * sizeof(float) - sizeof(float));
        const std::vector<std::uint8_t> history_before(
            history_buffer.payload().begin(), history_buffer.payload().end());
        const std::vector<std::uint8_t> state_before(
            state_buffer.payload().begin(), state_buffer.payload().end());
        auto command = context.make_command_buffer(
            "Window GDN commit command-buffer late failure", error);
        check(command != nullptr,
              label + " commit late-failure command buffer", error);
        if (!command) return false;
        const bool accepted_command = encode_window_gdn_commit(
            context, *command, accepted_rows, late_failure, error);
        const bool command_discarded = !accepted_command &&
            command->unsafe_native_handle() == nullptr &&
            error.find("gamma") != std::string::npos;
        const bool command_untouched =
            history_before == std::vector<std::uint8_t>(
                                  history_buffer.payload().begin(),
                                  history_buffer.payload().end()) &&
            state_before == std::vector<std::uint8_t>(
                                state_buffer.payload().begin(),
                                state_buffer.payload().end());
        check(command_discarded && command_untouched,
              label + " command-buffer commit failure rolls back history and state",
              error);
        passed &= command_discarded && command_untouched;

        auto shared_command = context.make_command_buffer(
            "Window GDN shared commit late failure", error);
        check(shared_command != nullptr,
              label + " shared commit late-failure command buffer", error);
        if (!shared_command) return false;
        const bool accepted_shared = Encoder::encode(
            context, *shared_command, "Window GDN shared commit rollback",
            [&](Encoder& encoder, std::string& body_error) {
                return encode_window_gdn_commit(
                    context, encoder, accepted_rows, late_failure, body_error);
            },
            error);
        const bool shared_discarded = !accepted_shared &&
            shared_command->unsafe_native_handle() == nullptr &&
            error.find("gamma") != std::string::npos;
        const bool shared_untouched =
            history_before == std::vector<std::uint8_t>(
                                  history_buffer.payload().begin(),
                                  history_buffer.payload().end()) &&
            state_before == std::vector<std::uint8_t>(
                                state_buffer.payload().begin(),
                                state_buffer.payload().end());
        check(shared_discarded && shared_untouched,
              label + " caller-owned commit failure rolls back history and state",
              error);
        passed &= shared_discarded && shared_untouched;
    }

    std::vector<float> expected_history = history;
    oracle::gdn_conv_commit(
        expected_history,
        actual_qkv.first(static_cast<std::size_t>(accepted_rows) * kChannels),
        accepted_rows);
    std::vector<float> expected_replayed_state = initial_state;
    std::vector<twin::tf> unused_commit_output, expected_replayed_state_tf;
    oracle::gdn_step_norm(
        expected_replayed_state,
        actual_h.first(static_cast<std::size_t>(accepted_rows) * kChannels),
        actual_gate.first(static_cast<std::size_t>(accepted_rows) * kHeads),
        actual_beta.first(static_cast<std::size_t>(accepted_rows) * kHeads),
        actual_z.first(static_cast<std::size_t>(accepted_rows) * kValueDim),
        gamma, accepted_rows, GDNStateMode::commit, unused_commit_output,
        expected_replayed_state_tf);

    history_buffer.capture_before_dispatch();
    state_buffer.capture_before_dispatch();
    qkv_buffer.capture_before_dispatch();
    h_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    beta_buffer.capture_before_dispatch();
    z_buffer.capture_before_dispatch();
    gamma_buffer.capture_before_dispatch();
    y_buffer.capture_before_dispatch();
    const bool commit_dispatched = run_command(
        context, label + " commit n_keep=" + std::to_string(accepted_rows),
        [&](CommandBuffer& command, std::string& commit_error) {
            if (!shared_encoder) {
                return encode_window_gdn_commit(
                    context, command, accepted_rows, commit_buffers,
                    commit_error);
            }
            return Encoder::encode(
                context, command, "shared Window GDN accepted-prefix commit",
                [&](Encoder& encoder, std::string& body_error) {
                    return encode_window_gdn_commit(
                        context, encoder, accepted_rows, commit_buffers,
                        body_error);
                },
                commit_error);
        },
        error);
    if (!commit_dispatched) return false;

    const bool history_matches = std::memcmp(
        history_buffer.payload().data(), expected_history.data(),
        expected_history.size() * sizeof(float)) == 0;
    check(history_matches,
          label + " accepted-prefix commit advances convolution history exactly");
    passed &= history_matches;
    const bool state_matches = compare_units(
        typed<const float>(state_buffer).first(kStateElements),
        expected_replayed_state_tf, 512.0,
        label + " accepted-prefix commit replays recurrent state", options,
        accepted_rows);
    passed &= state_matches;
    const std::array<twin::ByteRange, 1> history_written{{
        {0u, expected_history.size() * sizeof(float)}}};
    const std::array<twin::ByteRange, 1> state_written{{
        {0u, initial_state.size() * sizeof(float)}}};
    const bool commit_guards =
        !history_buffer.first_unwritten_change(history_written) &&
        !state_buffer.first_unwritten_change(state_written) &&
        !qkv_buffer.first_unwritten_change({}) &&
        !h_buffer.first_unwritten_change({}) &&
        !gate_buffer.first_unwritten_change({}) &&
        !beta_buffer.first_unwritten_change({}) &&
        !z_buffer.first_unwritten_change({}) &&
        !gamma_buffer.first_unwritten_change({}) &&
        !y_buffer.first_unwritten_change({});
    check(commit_guards,
          label + " commit guards and replay inputs remain unchanged");
    passed &= commit_guards;
    return passed;
}

bool test_window_t1_attention_gdn(Context& context) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t vocabulary_size = 8u;
    constexpr std::uint32_t position = 17u;
    constexpr std::size_t kGrResidual = 0u;
    constexpr std::size_t kGrBlockOut = 1u;
    constexpr std::size_t kGrInjectPrev = 2u;
    constexpr std::size_t kGrNorm = 3u;
    constexpr std::size_t kGrNormalized = 4u;
    constexpr std::size_t kGrRs = 5u;
    constexpr std::size_t kGrDown = 6u;
    constexpr std::size_t kGrInject = 7u;
    constexpr std::size_t kGrLo = 8u;
    constexpr std::size_t kGrInjectOut = 9u;
    constexpr std::size_t kGrUp = 10u;
    constexpr std::size_t kMixed = 11u;
    constexpr std::size_t kHistory = 12u;
    constexpr std::size_t kState = 13u;
    constexpr std::size_t kQkvWeights = 14u;
    constexpr std::size_t kConvWeights = 15u;
    constexpr std::size_t kAlpha = 16u;
    constexpr std::size_t kBetaWeights = 17u;
    constexpr std::size_t kDtBias = 18u;
    constexpr std::size_t kA = 19u;
    constexpr std::size_t kGateWeights = 20u;
    constexpr std::size_t kGamma = 21u;
    constexpr std::size_t kOutputWeights = 22u;
    constexpr std::size_t kXq = 23u;
    constexpr std::size_t kQkv = 24u;
    constexpr std::size_t kH = 25u;
    constexpr std::size_t kGate = 26u;
    constexpr std::size_t kBeta = 27u;
    constexpr std::size_t kZ = 28u;
    constexpr std::size_t kY = 29u;
    constexpr std::size_t kYq = 30u;
    constexpr std::size_t kBlockOut = 31u;
    constexpr std::size_t kSink = 32u;
    constexpr std::size_t kActiveRows = 33u;
    constexpr std::size_t kViewCount = 34u;
    constexpr std::uint8_t poison = 0xa5u;
    const std::string label = "Window T=1 attention and GDN";
    std::string error;

    const std::size_t gr_rows = HC_DIM;
    const std::size_t gr_inject_rows = HC;
    const std::size_t gr_low_rank_rows = HC_LOW_RANK;
    const std::size_t gdn_mixed_count = N_EMBD;
    const std::size_t qkv_count = kChannels;
    const std::size_t gate_count = kHeads;
    const std::size_t value_count = kValueDim;
    const std::size_t qkv_weight_count =
        static_cast<std::size_t>(kChannels) * (N_EMBD / 32u);
    const std::size_t gate_weight_count =
        static_cast<std::size_t>(kValueDim) * (N_EMBD / 32u);
    const std::size_t output_weight_count =
        static_cast<std::size_t>(N_EMBD) * (kValueDim / 32u);
    const std::size_t ab_weight_count =
        static_cast<std::size_t>(N_EMBD) * kHeads;

    std::array<std::size_t, kViewCount> byte_sizes{};
    byte_sizes[kGrResidual] = gr_rows * sizeof(float);
    byte_sizes[kGrBlockOut] = N_EMBD * sizeof(float);
    byte_sizes[kGrInjectPrev] = gr_inject_rows * sizeof(float);
    byte_sizes[kGrNorm] = gr_rows * sizeof(float);
    byte_sizes[kGrNormalized] = gr_rows * sizeof(float);
    byte_sizes[kGrRs] = gr_inject_rows * sizeof(float);
    byte_sizes[kGrDown] = gr_low_rank_rows * gr_rows * sizeof(std::uint16_t);
    byte_sizes[kGrInject] = gr_inject_rows * gr_rows * sizeof(std::uint16_t);
    byte_sizes[kGrLo] = gr_low_rank_rows * sizeof(float);
    byte_sizes[kGrInjectOut] = gr_inject_rows * sizeof(float);
    byte_sizes[kGrUp] = gr_rows * gr_low_rank_rows * sizeof(std::uint16_t);
    byte_sizes[kMixed] = gdn_mixed_count * sizeof(float);
    byte_sizes[kHistory] = kHistoryElements * sizeof(float);
    byte_sizes[kState] = kStateElements * sizeof(float);
    byte_sizes[kQkvWeights] = qkv_weight_count * sizeof(block_q8_0);
    byte_sizes[kConvWeights] =
        static_cast<std::size_t>(kChannels) * GDN_CONV_KERNEL * sizeof(float);
    byte_sizes[kAlpha] = ab_weight_count * sizeof(std::uint16_t);
    byte_sizes[kBetaWeights] = ab_weight_count * sizeof(std::uint16_t);
    byte_sizes[kDtBias] = kHeads * sizeof(float);
    byte_sizes[kA] = kHeads * sizeof(float);
    byte_sizes[kGateWeights] = gate_weight_count * sizeof(block_q8_0);
    byte_sizes[kGamma] = kHeadDim * sizeof(float);
    byte_sizes[kOutputWeights] = output_weight_count * sizeof(block_q8_0);
    byte_sizes[kXq] = (gdn_mixed_count / 32u) * sizeof(Q81);
    byte_sizes[kQkv] = qkv_count * sizeof(float);
    byte_sizes[kH] = qkv_count * sizeof(float);
    byte_sizes[kGate] = gate_count * sizeof(float);
    byte_sizes[kBeta] = gate_count * sizeof(float);
    byte_sizes[kZ] = value_count * sizeof(float);
    byte_sizes[kY] = value_count * sizeof(float);
    byte_sizes[kYq] = (value_count / 32u) * sizeof(Q81);
    byte_sizes[kBlockOut] = gdn_mixed_count * sizeof(float);
    byte_sizes[kSink] = sizeof(float);
    byte_sizes[kActiveRows] = sizeof(std::uint32_t);

    const std::array<const char*, kViewCount> names{
        "GR residual", "GR block output", "GR injection input", "GR norm",
        "GR normalized", "GR rs", "GR down weights", "GR injection weights",
        "GR low rank", "GR injection output", "GR up weights", "shared mixed",
        "GDN history", "GDN state", "GDN QKV weights", "GDN convolution weights",
        "GDN alpha", "GDN beta weights", "GDN dt bias", "GDN A",
        "GDN gate weights", "GDN gamma", "GDN output weights", "GDN mixed Q8_1",
        "GDN QKV", "GDN H", "GDN gate", "GDN beta", "GDN Z", "GDN Y",
        "GDN Y Q8_1", "GDN block output", "GDN matvec sink", "GDN active rows",
    };
    std::array<twin::GuardedBuffer, kViewCount> storage;
    for (std::size_t index = 0u; index < storage.size(); ++index) {
        if (!storage[index].allocate(context, label + " " + names[index],
                                     byte_sizes[index], error)) {
            check(false, label + " allocate " + names[index], error);
            return false;
        }
        std::fill(storage[index].payload().begin(),
                  storage[index].payload().end(), 0u);
    }

    twin::GuardedBuffer ffn_norm_storage;
    twin::GuardedBuffer ffn_down_storage;
    twin::GuardedBuffer ffn_inject_storage;
    twin::GuardedBuffer ffn_up_storage;
    const std::array<std::pair<twin::GuardedBuffer*, std::size_t>, 4u>
        ffn_weight_allocations{{
            {&ffn_norm_storage, gr_rows * sizeof(float)},
            {&ffn_down_storage,
             gr_low_rank_rows * gr_rows * sizeof(std::uint16_t)},
            {&ffn_inject_storage,
             gr_inject_rows * gr_rows * sizeof(std::uint16_t)},
            {&ffn_up_storage,
             gr_rows * gr_low_rank_rows * sizeof(std::uint16_t)},
        }};
    const std::array<const char*, 4u> ffn_weight_names{
        "FFN GR norm", "FFN GR down weights", "FFN GR injection weights",
        "FFN GR up weights"};
    for (std::size_t index = 0u; index < ffn_weight_allocations.size(); ++index) {
        if (!ffn_weight_allocations[index].first->allocate(
                context, label + " " + ffn_weight_names[index],
                ffn_weight_allocations[index].second, error)) {
            check(false, label + " allocate " + ffn_weight_names[index], error);
            return false;
        }
    }

    std::mt19937_64 rng(0x57494e444f574731ull);
    auto residual = typed<float>(storage[kGrResidual]);
    for (std::size_t index = 0u; index < residual.size(); ++index) {
        residual[index] = 0.01f + static_cast<float>(index % 13u) * 0.0001f;
    }
    std::fill(typed<float>(storage[kGrNorm]).begin(),
              typed<float>(storage[kGrNorm]).end(), 1.0f);
    auto gr_block_out = typed<float>(storage[kGrBlockOut]);
    for (std::size_t index = 0u; index < gr_block_out.size(); ++index) {
        gr_block_out[index] =
            0.02f + static_cast<float>(index % 17u) * 0.0003f;
    }
    const std::vector<std::uint8_t> gr_residual_before(
        storage[kGrResidual].payload().begin(),
        storage[kGrResidual].payload().end());
    const std::uint16_t gr_weight =
        strata::kernels::bf16_from_f32(0.0005f);
    for (const std::size_t index : {kGrDown, kGrInject, kGrUp}) {
        std::fill(typed<std::uint16_t>(storage[index]).begin(),
                  typed<std::uint16_t>(storage[index]).end(), gr_weight);
    }
    std::fill(typed<float>(ffn_norm_storage).begin(),
              typed<float>(ffn_norm_storage).end(), 0.875f);
    const std::uint16_t ffn_weight =
        strata::kernels::bf16_from_f32(0.000375f);
    for (twin::GuardedBuffer* weight :
         {&ffn_down_storage, &ffn_inject_storage, &ffn_up_storage}) {
        std::fill(typed<std::uint16_t>(*weight).begin(),
                  typed<std::uint16_t>(*weight).end(), ffn_weight);
    }
    for (std::size_t index = 0u; index < typed<float>(storage[kHistory]).size();
         ++index) {
        typed<float>(storage[kHistory])[index] =
            sample(twin::InputSet::fixture, rng,
                   static_cast<std::uint32_t>(index + 11u), 0.02f);
    }
    for (std::size_t index = 0u; index < typed<float>(storage[kState]).size();
         ++index) {
        typed<float>(storage[kState])[index] =
            sample(twin::InputSet::fixture, rng,
                   static_cast<std::uint32_t>(index + 29u), 0.01f);
    }
    for (std::size_t index = 0u;
         index < typed<float>(storage[kConvWeights]).size(); ++index) {
        typed<float>(storage[kConvWeights])[index] =
            sample(twin::InputSet::fixture, rng,
                   static_cast<std::uint32_t>(index + 37u), 0.03f);
    }
    for (std::size_t index = 0u;
         index < typed<std::uint16_t>(storage[kAlpha]).size(); ++index) {
        const int code = static_cast<int>((index * 7u + 3u) % 9u) - 4;
        typed<std::uint16_t>(storage[kAlpha])[index] =
            strata::kernels::bf16_from_f32(
                static_cast<float>(code) * (1.0f / 8192.0f));
        typed<std::uint16_t>(storage[kBetaWeights])[index] =
            strata::kernels::bf16_from_f32(
                static_cast<float>(4 - code) * (1.0f / 8192.0f));
    }
    for (std::size_t index = 0u; index < typed<float>(storage[kDtBias]).size();
         ++index) {
        typed<float>(storage[kDtBias])[index] =
            static_cast<float>(static_cast<int>(index % 5u) - 2) * 0.01f;
        typed<float>(storage[kA])[index] =
            -0.04f - static_cast<float>(index % 3u) * 0.01f;
    }
    for (std::size_t index = 0u; index < typed<float>(storage[kGamma]).size();
         ++index) {
        typed<float>(storage[kGamma])[index] =
            0.75f + static_cast<float>(index % 11u) * (1.0f / 32.0f);
    }
    fill_q8_0_weights(typed<block_q8_0>(storage[kQkvWeights]), N_EMBD, 1u);
    fill_q8_0_weights(typed<block_q8_0>(storage[kGateWeights]), N_EMBD, 4u);
    fill_q8_0_weights(typed<block_q8_0>(storage[kOutputWeights]), kValueDim, 7u);
    std::memcpy(storage[kActiveRows].payload().data(), &rows, sizeof(rows));

    const std::array<std::size_t, 15u> output_indices{
        kGrNormalized, kGrRs, kGrLo, kGrInjectOut, kMixed, kXq, kQkv, kH,
        kGate, kBeta, kZ, kY, kYq, kBlockOut, kSink};
    const auto poison_outputs = [&] {
        for (const std::size_t index : output_indices) {
            std::fill(storage[index].payload().begin(),
                      storage[index].payload().end(), poison);
        }
    };
    poison_outputs();
    for (auto& buffer : storage) buffer.capture_before_dispatch();
    ffn_norm_storage.capture_before_dispatch();
    ffn_down_storage.capture_before_dispatch();
    ffn_inject_storage.capture_before_dispatch();
    ffn_up_storage.capture_before_dispatch();

    auto params_ring = WindowParamsRing::open(context, error);
    check(params_ring != nullptr, label + " open parameter ring", error);
    if (!params_ring) return false;
    const std::array<std::uint32_t, rows> tokens{1u};
    const std::array<WindowSamplerSettings, rows> samplers{};
    const BufferRef params = params_ring->build_slot(
        0u, position, tokens, samplers, 0u, 0x1234u, 0x5678u, error);
    check(static_cast<bool>(params), label + " build T=1 parameters", error);
    if (!params) return false;

    const GRReadBuffers attention{
        storage[kGrResidual].view(), storage[kGrBlockOut].view(),
        storage[kGrInjectPrev].view(), storage[kGrNorm].view(),
        storage[kGrNormalized].view(), storage[kGrRs].view(),
        storage[kGrDown].view(), storage[kGrInject].view(),
        storage[kGrLo].view(), storage[kGrInjectOut].view(),
        storage[kGrUp].view(), storage[kMixed].view()};
    const WindowGDNBlockBuffers gdn{
        storage[kMixed].view(), storage[kHistory].view(), storage[kState].view(),
        MatVecWeightFormat::q8_0, storage[kQkvWeights].view(),
        storage[kConvWeights].view(), storage[kAlpha].view(),
        storage[kBetaWeights].view(), storage[kDtBias].view(), storage[kA].view(),
        MatVecWeightFormat::q8_0, storage[kGateWeights].view(),
        storage[kGamma].view(), MatVecWeightFormat::q8_0,
        storage[kOutputWeights].view(), storage[kXq].view(), storage[kQkv].view(),
        storage[kH].view(), storage[kGate].view(), storage[kBeta].view(),
        storage[kZ].view(), storage[kY].view(), storage[kYq].view(),
        storage[kBlockOut].view(), storage[kSink].view(),
        storage[kActiveRows].view()};
    const WindowT1AttentionGDNBuffers buffers{params, attention, gdn};
    const GRReadBuffers ffn{
        storage[kGrResidual].view(), storage[kBlockOut].view(),
        storage[kGrInjectOut].view(), ffn_norm_storage.view(),
        storage[kGrNormalized].view(), storage[kGrRs].view(),
        ffn_down_storage.view(), ffn_inject_storage.view(),
        storage[kGrLo].view(), storage[kGrInjectPrev].view(),
        ffn_up_storage.view(), storage[kMixed].view()};
    const WindowT1AttentionGDNFFNBuffers layer0_buffers{
        params, attention, gdn, ffn};
    const std::vector<std::uint8_t> gr_inject2_before(
        storage[kGrInjectPrev].payload().begin(),
        storage[kGrInjectPrev].payload().end());
    const std::vector<std::uint8_t> history_before(
        storage[kHistory].payload().begin(), storage[kHistory].payload().end());
    const std::vector<std::uint8_t> state_before(
        storage[kState].payload().begin(), storage[kState].payload().end());

    const auto run_reference = [&] {
        return run_command(
            context, label + " component reference",
            [&](CommandBuffer& command, std::string& command_error) {
                return encode_gr_read(context, command, rows, false, attention,
                                      command_error) &&
                       encode_window_gdn_block(context, command, rows, gdn,
                                               command_error);
            },
            error);
    };
    const bool reference_completed = run_reference();
    check(reference_completed, label + " component reference executes", error);
    if (!reference_completed) return false;
    std::array<std::vector<std::uint8_t>, output_indices.size()> expected_outputs;
    for (std::size_t output = 0u; output < output_indices.size(); ++output) {
        const auto payload = storage[output_indices[output]].payload();
        expected_outputs[output].assign(payload.begin(), payload.end());
    }
    const auto mixed_payload = storage[kMixed].payload();
    const bool mixed_is_nonzero = std::any_of(
        mixed_payload.begin(), mixed_payload.end(),
        [](std::uint8_t value) { return value != 0u; });
    check(mixed_is_nonzero, label + " GR read produces a nonzero mixed row");

    poison_outputs();
    const bool window_completed = run_command(
        context, label + " Window coordinator",
        [&](CommandBuffer& command, std::string& command_error) {
            return encode_window_t1_attention_gdn(
                context, command, vocabulary_size, buffers, command_error);
        },
        error);
    check(window_completed, label + " Window coordinator executes", error);
    if (!window_completed) return false;

    bool outputs_match = true;
    for (std::size_t output = 0u; output < output_indices.size(); ++output) {
        const auto payload = storage[output_indices[output]].payload();
        const bool matches = std::equal(
            expected_outputs[output].begin(), expected_outputs[output].end(),
            payload.begin(), payload.end());
        check(matches, label + " ordered Window output " +
                           std::to_string(output) + " matches component chain");
        outputs_match &= matches;
        const std::array<twin::ByteRange, 1u> written{{
            {0u, payload.size()}}};
        const auto guard = storage[output_indices[output]].first_unwritten_change(
            written);
        check(!guard, label + " output guard " + std::to_string(output),
              guard ? guard->reason : "");
        outputs_match &= !guard;
    }
    const bool state_unchanged =
        std::equal(history_before.begin(), history_before.end(),
                   storage[kHistory].payload().begin(),
                   storage[kHistory].payload().end()) &&
        std::equal(state_before.begin(), state_before.end(),
                   storage[kState].payload().begin(),
                   storage[kState].payload().end());
    check(state_unchanged, label + " speculative GDN leaves state and history unchanged");
    const bool residual_unchanged = std::equal(
        gr_residual_before.begin(), gr_residual_before.end(),
        storage[kGrResidual].payload().begin(),
        storage[kGrResidual].payload().end());
    check(residual_unchanged, label + " attention read preserves the residual input");
    if (!outputs_match || !state_unchanged || !mixed_is_nonzero ||
        !residual_unchanged) {
        return false;
    }

    const bool apply_control_completed = run_command(
        context, label + " apply=true negative control",
        [&](CommandBuffer& command, std::string& command_error) {
            return encode_gr_read(context, command, rows, true, attention,
                                  command_error);
        },
        error);
    check(apply_control_completed,
          label + " apply=true negative control executes", error);
    if (!apply_control_completed) return false;
    const auto true_apply_mixed = storage[kMixed].payload();
    const bool apply_false_is_observable = !std::equal(
        expected_outputs[4].begin(), expected_outputs[4].end(),
        true_apply_mixed.begin(), true_apply_mixed.end());
    check(apply_false_is_observable,
          label + " nonzero block output distinguishes apply=false");
    if (!apply_false_is_observable) return false;

    const auto outputs_are_poisoned = [&] {
        for (const std::size_t index : output_indices) {
            if (!std::all_of(storage[index].payload().begin(),
                             storage[index].payload().end(),
                             [=](std::uint8_t value) { return value == poison; })) {
                return false;
            }
        }
        return true;
    };
    poison_outputs();
    auto separate_mixed_storage = twin::GuardedBuffer{};
    check(separate_mixed_storage.allocate(
              context, label + " wrong mixed handoff", byte_sizes[kMixed], error),
          label + " allocate wrong mixed handoff", error);
    if (!separate_mixed_storage.view()) return false;
    WindowT1AttentionGDNBuffers wrong_handoff = buffers;
    wrong_handoff.gdn.mixed = separate_mixed_storage.view();
    auto wrong_handoff_command = context.make_command_buffer(
        label + " wrong mixed handoff", error);
    check(wrong_handoff_command != nullptr,
          label + " wrong-handoff command buffer", error);
    if (!wrong_handoff_command) return false;
    const bool wrong_handoff_rejected = !encode_window_t1_attention_gdn(
        context, *wrong_handoff_command, vocabulary_size, wrong_handoff, error);
    const bool wrong_handoff_discarded =
        wrong_handoff_command->unsafe_native_handle() == nullptr;
    check(wrong_handoff_rejected && wrong_handoff_discarded &&
              error.find("must be the GDN mixed input") != std::string::npos &&
              outputs_are_poisoned(),
          label + " rejects a split mixed handoff", error);

    const std::array<std::uint32_t, 2u> two_tokens{1u, 2u};
    const std::array<WindowSamplerSettings, 2u> two_samplers{};
    const BufferRef two_row_params = params_ring->build_slot(
        1u, position, two_tokens, two_samplers, 0u, 0x1234u, 0x5678u, error);
    check(static_cast<bool>(two_row_params), label + " build T=2 rejection params",
          error);
    if (!two_row_params) return false;
    WindowT1AttentionGDNBuffers wrong_rows = buffers;
    wrong_rows.params = two_row_params;
    auto wrong_rows_command = context.make_command_buffer(
        label + " reject T=2", error);
    check(wrong_rows_command != nullptr, label + " T=2 command buffer", error);
    if (!wrong_rows_command) return false;
    const bool wrong_rows_rejected = !encode_window_t1_attention_gdn(
        context, *wrong_rows_command, vocabulary_size, wrong_rows, error);
    const bool wrong_rows_discarded =
        wrong_rows_command->unsafe_native_handle() == nullptr;
    check(wrong_rows_rejected && wrong_rows_discarded &&
              error.find("T=1") != std::string::npos && outputs_are_poisoned(),
          label + " rejects T=2 parameters", error);

    strata::metal::WindowParams valid_params{};
    std::memcpy(&valid_params, params.contents, sizeof(valid_params));
    const auto reject_mutated_params = [&](const std::string& case_name,
                                           const std::string& expected_error,
                                           auto mutate) {
        strata::metal::WindowParams malformed = valid_params;
        mutate(malformed);
        std::memcpy(params.contents, &malformed, sizeof(malformed));
        WindowT1AttentionGDNBuffers malformed_buffers = buffers;
        malformed_buffers.params = params;
        auto command = context.make_command_buffer(
            label + " reject " + case_name, error);
        if (!command) {
            std::memcpy(params.contents, &valid_params, sizeof(valid_params));
            check(false, label + " " + case_name + " command buffer", error);
            return false;
        }
        poison_outputs();
        const bool rejected = !encode_window_t1_attention_gdn(
            context, *command, vocabulary_size, malformed_buffers, error);
        const bool discarded = command->unsafe_native_handle() == nullptr;
        const bool unchanged = outputs_are_poisoned();
        std::memcpy(params.contents, &valid_params, sizeof(valid_params));
        const bool correct = rejected && discarded && unchanged &&
                             error.find(expected_error) != std::string::npos;
        check(correct, label + " rejects " + case_name, error);
        return correct;
    };
    const bool invalid_nc_rejected = reject_mutated_params(
        "T=1 with invalid nc", "T=1", [](strata::metal::WindowParams& malformed) {
            malformed.nc = 2u;
        });
    const bool nonzero_keep_rejected = reject_mutated_params(
        "nonzero n_keep", "n_keep=0", [](strata::metal::WindowParams& malformed) {
            malformed.n_keep = 1u;
        });
    const bool invalid_token_rejected = reject_mutated_params(
        "out-of-vocabulary token", "outside the text vocabulary",
        [&](strata::metal::WindowParams& malformed) {
            malformed.tokens[0] = vocabulary_size;
        });
    const bool invalid_step_rejected = reject_mutated_params(
        "inconsistent StepRow", "StepRow", [](strata::metal::WindowParams& malformed) {
            ++malformed.step[0].n_kv;
        });

    WindowT1AttentionGDNBuffers overlapping_stages = buffers;
    overlapping_stages.gdn.state = attention.residual;
    auto overlap_command = context.make_command_buffer(
        label + " reject cross-stage overlap", error);
    check(overlap_command != nullptr,
          label + " cross-stage overlap command buffer", error);
    if (!overlap_command) return false;
    poison_outputs();
    const bool overlap_rejected = !encode_window_t1_attention_gdn(
        context, *overlap_command, vocabulary_size, overlapping_stages, error);
    const bool overlap_discarded =
        overlap_command->unsafe_native_handle() == nullptr;
    const bool overlap_untouched = outputs_are_poisoned();
    check(overlap_rejected && overlap_discarded && overlap_untouched &&
              error.find("overlap outside the mixed handoff") !=
                  std::string::npos,
          label + " rejects writable cross-stage overlap", error);

    const BufferRef short_active_rows = storage[kActiveRows].view().slice(0u, 3u);
    WindowT1AttentionGDNBuffers late_failure = buffers;
    late_failure.gdn.active_rows = short_active_rows;
    auto late_failure_command = context.make_command_buffer(
        label + " late GDN validation failure", error);
    check(late_failure_command != nullptr,
          label + " late-failure command buffer", error);
    if (!late_failure_command) return false;
    poison_outputs();
    const bool late_failure_rejected = !encode_window_t1_attention_gdn(
        context, *late_failure_command, vocabulary_size, late_failure, error);
    const bool late_failure_discarded =
        late_failure_command->unsafe_native_handle() == nullptr;
    check(late_failure_rejected && late_failure_discarded &&
              error.find("undersized") != std::string::npos && outputs_are_poisoned(),
          label + " late GDN failure discards the earlier GR read", error);

    auto shared_failure_command = context.make_command_buffer(
        label + " caller-owned late failure", error);
    check(shared_failure_command != nullptr,
          label + " caller-owned failure command buffer", error);
    if (!shared_failure_command) return false;
    poison_outputs();
    const bool shared_failure_rejected = !Encoder::encode(
        context, *shared_failure_command, "Window attention/GDN shared failure",
        [&](Encoder& encoder, std::string& body_error) {
            return encode_window_t1_attention_gdn(
                context, encoder, vocabulary_size, late_failure, body_error);
        },
        error);
    const bool shared_failure_discarded =
        shared_failure_command->unsafe_native_handle() == nullptr;
    check(shared_failure_rejected && shared_failure_discarded &&
              error.find("undersized") != std::string::npos && outputs_are_poisoned(),
          label + " caller-owned late failure discards the segment", error);

    const auto prepare_layer0_chain = [&](bool poison_inject2) {
        std::memcpy(storage[kGrResidual].payload().data(),
                    gr_residual_before.data(), gr_residual_before.size());
        std::memcpy(storage[kGrInjectPrev].payload().data(),
                    gr_inject2_before.data(), gr_inject2_before.size());
        poison_outputs();
        if (poison_inject2) {
            std::fill(storage[kGrInjectPrev].payload().begin(),
                      storage[kGrInjectPrev].payload().end(), poison);
        }
    };

    prepare_layer0_chain(false);
    const bool layer0_apply_control_completed = run_command(
        context, label + " layer-0 apply=true control",
        [&](CommandBuffer& command, std::string& command_error) {
            return encode_gr_read(context, command, rows, true, attention,
                                  command_error);
        },
        error);
    check(layer0_apply_control_completed,
          label + " layer-0 apply=true control executes", error);
    if (!layer0_apply_control_completed) return false;
    const bool layer0_apply_control_writes_residual = !std::equal(
        gr_residual_before.begin(), gr_residual_before.end(),
        storage[kGrResidual].payload().begin(),
        storage[kGrResidual].payload().end());
    check(layer0_apply_control_writes_residual,
          label + " nonzero attention placeholder makes apply=true observable");

    prepare_layer0_chain(true);
    const bool layer0_reference_completed = run_command(
        context, label + " attention-GDN-FFN component chain",
        [&](CommandBuffer& command, std::string& command_error) {
            return Encoder::encode(
                context, command, "Window layer-0 component chain",
                [&](Encoder& encoder, std::string& body_error) {
                    return encode_gr_read(context, encoder, rows, false,
                                          attention, body_error) &&
                           encode_window_gdn_block(context, encoder, rows, gdn,
                                                   body_error) &&
                           encode_gr_read(context, encoder, rows, true, ffn,
                                          body_error);
                },
                command_error);
        },
        error);
    check(layer0_reference_completed,
          label + " attention-GDN-FFN component chain executes", error);
    if (!layer0_reference_completed) return false;
    std::array<std::vector<std::uint8_t>, output_indices.size()>
        layer0_expected_outputs;
    for (std::size_t output = 0u; output < output_indices.size(); ++output) {
        const auto payload = storage[output_indices[output]].payload();
        layer0_expected_outputs[output].assign(payload.begin(), payload.end());
    }
    const std::vector<std::uint8_t> layer0_expected_residual(
        storage[kGrResidual].payload().begin(),
        storage[kGrResidual].payload().end());
    const std::vector<std::uint8_t> layer0_expected_inject2(
        storage[kGrInjectPrev].payload().begin(),
        storage[kGrInjectPrev].payload().end());
    const bool layer0_ffn_applies_pending_write = !std::equal(
        layer0_expected_residual.begin(), layer0_expected_residual.end(),
        gr_residual_before.begin(), gr_residual_before.end());
    check(layer0_ffn_applies_pending_write,
          label + " FFN read applies the GDN block output to residual");
    const bool component_state_unchanged =
        std::equal(history_before.begin(), history_before.end(),
                   storage[kHistory].payload().begin(),
                   storage[kHistory].payload().end()) &&
        std::equal(state_before.begin(), state_before.end(),
                   storage[kState].payload().begin(),
                   storage[kState].payload().end());
    check(component_state_unchanged,
          label + " component chain keeps GDN state and history speculative");

    prepare_layer0_chain(true);
    const bool layer0_completed = run_command(
        context, label + " full layer-0 coordinator",
        [&](CommandBuffer& command, std::string& command_error) {
            return encode_window_t1_attention_gdn_ffn(
                context, command, vocabulary_size, layer0_buffers,
                command_error);
        },
        error);
    check(layer0_completed, label + " full layer-0 coordinator executes", error);
    if (!layer0_completed) return false;
    bool layer0_outputs_match = true;
    for (std::size_t output = 0u; output < output_indices.size(); ++output) {
        const auto payload = storage[output_indices[output]].payload();
        const bool matches = std::equal(
            layer0_expected_outputs[output].begin(),
            layer0_expected_outputs[output].end(), payload.begin(), payload.end());
        check(matches, label + " layer-0 output " + std::to_string(output) +
                           " matches component chain");
        layer0_outputs_match &= matches;
        const std::array<twin::ByteRange, 1u> written{{{0u, payload.size()}}};
        const auto guard = storage[output_indices[output]].first_unwritten_change(
            written);
        check(!guard, label + " layer-0 output guard " +
                           std::to_string(output),
              guard ? guard->reason : "");
        layer0_outputs_match &= !guard;
    }
    const bool layer0_residual_matches = std::equal(
        layer0_expected_residual.begin(), layer0_expected_residual.end(),
        storage[kGrResidual].payload().begin(),
        storage[kGrResidual].payload().end());
    const bool layer0_inject2_matches = std::equal(
        layer0_expected_inject2.begin(), layer0_expected_inject2.end(),
        storage[kGrInjectPrev].payload().begin(),
        storage[kGrInjectPrev].payload().end());
    check(layer0_residual_matches,
          label + " FFN residual fold matches component chain");
    check(layer0_inject2_matches,
          label + " FFN injection output matches component chain");
    const std::array<twin::ByteRange, 1u> residual_written{{{
        0u, storage[kGrResidual].payload().size()}}};
    const auto residual_guard =
        storage[kGrResidual].first_unwritten_change(residual_written);
    check(!residual_guard, label + " layer-0 residual guard",
          residual_guard ? residual_guard->reason : "");
    const std::array<twin::ByteRange, 1u> inject2_written{{{
        0u, storage[kGrInjectPrev].payload().size()}}};
    const auto inject2_guard =
        storage[kGrInjectPrev].first_unwritten_change(inject2_written);
    check(!inject2_guard, label + " layer-0 inject2 guard",
          inject2_guard ? inject2_guard->reason : "");
    const bool layer0_state_unchanged =
        std::equal(history_before.begin(), history_before.end(),
                   storage[kHistory].payload().begin(),
                   storage[kHistory].payload().end()) &&
        std::equal(state_before.begin(), state_before.end(),
                   storage[kState].payload().begin(),
                   storage[kState].payload().end());
    check(layer0_state_unchanged,
          label + " full layer-0 coordinator keeps GDN state and history speculative");

    constexpr std::uint32_t kRouterExperts = 512u;
    constexpr std::size_t kRouterInputWidth = N_EMBD;
    constexpr std::size_t kRouterWeightCount =
        static_cast<std::size_t>(kRouterExperts) * kRouterInputWidth;
    constexpr std::size_t kRouteTopK = 10u;
    const std::size_t router_weight_bytes =
        kRouterWeightCount * sizeof(std::uint16_t);
    twin::GuardedBuffer router_tensor_storage;
    twin::GuardedBuffer router_logits_storage;
    twin::GuardedBuffer route_ids_storage;
    twin::GuardedBuffer route_weights_storage;
    twin::GuardedBuffer router_matvec_sink_storage;
    twin::GuardedBuffer resident_table_storage;
    twin::GuardedBuffer usage_table_storage;
    twin::GuardedBuffer moe_plan_storage;
    const bool router_storage_allocated =
        router_tensor_storage.allocate(
            context, label + " BF16 router tensor", router_weight_bytes,
            error) &&
        router_logits_storage.allocate(
            context, label + " router logits",
            kRouterExperts * sizeof(float), error) &&
        route_ids_storage.allocate(
            context, label + " route IDs",
            kRouteTopK * sizeof(std::int32_t), error) &&
        route_weights_storage.allocate(
            context, label + " route weights", kRouteTopK * sizeof(float),
            error) &&
        router_matvec_sink_storage.allocate(
            context, label + " router mat-vec sink", sizeof(float), error) &&
        resident_table_storage.allocate(
            context, label + " resident table",
            kRouterExperts * sizeof(std::int32_t), error) &&
        usage_table_storage.allocate(
            context, label + " usage table",
            kRouterExperts * sizeof(float), error) &&
        moe_plan_storage.allocate(
            context, label + " MoE plan", sizeof(TieredMoePlan), error);
    check(router_storage_allocated, label + " router prefix scratch allocation",
          error);
    if (!router_storage_allocated) return false;

    std::vector<std::uint16_t> router_weight_values(kRouterWeightCount);
    for (std::size_t index = 0u; index < router_weight_values.size(); ++index) {
        const std::int32_t code =
            static_cast<std::int32_t>((index * 17u + 31u) % 251u) - 125;
        router_weight_values[index] = strata::kernels::bf16_from_f32(
            static_cast<float>(code) * 0.00003125f);
    }
    std::memcpy(router_tensor_storage.payload().data(),
                router_weight_values.data(), router_weight_bytes);
    std::vector<float> expected_mixed(kRouterInputWidth);
    std::memcpy(expected_mixed.data(), layer0_expected_outputs[4].data(),
                expected_mixed.size() * sizeof(float));

    const WindowT1MoeRouterPrefixBuffers router_prefix_buffers{
        layer0_buffers,
        context.buffers().read_only(router_tensor_storage.view()),
        router_logits_storage.view(), route_ids_storage.view(),
        route_weights_storage.view(), router_matvec_sink_storage.view()};
    const WindowT1MoePlanPrefixBuffers plan_prefix_buffers{
        router_prefix_buffers,
        context.buffers().read_only(resident_table_storage.view()),
        usage_table_storage.view(), moe_plan_storage.view()};
    const auto exercise_router_prefix = [&](std::uint32_t experts) {
        std::string prefix_error;
        std::vector<float> expected_logits;
        const auto expert_weights =
            std::span<const std::uint16_t>(router_weight_values).first(
                static_cast<std::size_t>(experts) * kRouterInputWidth);
        const bool logits_oracle = twin::bf16_matvec_reference(
            expert_weights, expected_mixed, N_EMBD, experts, 1u, 1u, 256u,
            expected_logits, prefix_error);
        check(logits_oracle,
              label + " router BF16 oracle experts=" +
                  std::to_string(experts),
              prefix_error);
        if (!logits_oracle) return false;

        std::vector<std::int32_t> expected_ids;
        std::vector<twin::tf> expected_route_weights;
        const bool route_oracle = twin::moe_route_reference(
            expected_logits, 1u, experts, false, expected_ids,
            expected_route_weights, prefix_error);
        check(route_oracle,
              label + " route oracle experts=" + std::to_string(experts),
              prefix_error);
        if (!route_oracle) return false;

        std::vector<Q81> expected_q8_1;
        const bool q8_oracle = twin::quantize_q8_1_reference(
            expected_mixed, N_EMBD, 1u, expected_q8_1, prefix_error);
        check(q8_oracle,
              label + " Q8_1 oracle experts=" + std::to_string(experts),
              prefix_error);
        if (!q8_oracle) return false;

        prepare_layer0_chain(true);
        const auto poison_route_outputs = [&] {
            for (twin::GuardedBuffer* output :
                 {&router_logits_storage, &route_ids_storage,
                  &route_weights_storage, &router_matvec_sink_storage}) {
                std::fill(output->payload().begin(), output->payload().end(),
                          poison);
                output->capture_before_dispatch();
            }
        };
        poison_route_outputs();
        router_tensor_storage.capture_before_dispatch();

        const bool completed = run_command(
            context, label + " router route Q8_1 experts=" +
                         std::to_string(experts),
            [&](CommandBuffer& command, std::string& command_error) {
                return encode_window_t1_moe_router_prefix(
                    context, command, vocabulary_size, experts,
                    router_prefix_buffers, command_error);
            },
            prefix_error);
        check(completed,
              label + " ordered router route Q8_1 prefix experts=" +
                  std::to_string(experts),
              prefix_error);
        if (!completed) return false;

        const auto expected_logit_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_logits.data()),
            expected_logits.size() * sizeof(float));
        const auto actual_logit_bytes =
            router_logits_storage.payload().first(expected_logit_bytes.size());
        const auto logit_mismatch =
            twin::compare_bytes(actual_logit_bytes, expected_logit_bytes);
        check(!logit_mismatch,
              label + " exact router logits experts=" + std::to_string(experts),
              logit_mismatch ? logit_mismatch->reason : std::string{});

        const auto expected_id_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_ids.data()),
            expected_ids.size() * sizeof(std::int32_t));
        const auto id_mismatch = twin::compare_bytes(
            route_ids_storage.payload(), expected_id_bytes);
        check(!id_mismatch,
              label + " exact top-10 route IDs experts=" +
                  std::to_string(experts),
              id_mismatch ? id_mismatch->reason : std::string{});

        const auto actual_route_weights =
            typed<const float>(route_weights_storage).first(kRouteTopK);
        const std::vector<float> prefix_route_weights(
            actual_route_weights.begin(), actual_route_weights.end());
        const auto actual_route_ids =
            typed<const std::int32_t>(route_ids_storage).first(kRouteTopK);
        const std::vector<std::int32_t> prefix_route_ids(
            actual_route_ids.begin(), actual_route_ids.end());
        const twin::Tolerance route_tolerance{
            twin::Comparison::units, 16.0, 0.0, 0u};
        const auto route_weight_mismatch = twin::compare_floats(
            actual_route_weights, expected_route_weights, route_tolerance);
        check(!route_weight_mismatch,
              label + " top-10 route weights U(16) experts=" +
                  std::to_string(experts),
              route_weight_mismatch ? route_weight_mismatch->reason
                                    : std::string{});

        const auto expected_q8_1_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_q8_1.data()),
            expected_q8_1.size() * sizeof(Q81));
        const auto actual_q8_1_bytes = storage[kXq].payload().first(
            expected_q8_1_bytes.size());
        const auto q8_1_mismatch =
            twin::compare_bytes(actual_q8_1_bytes, expected_q8_1_bytes);
        check(!q8_1_mismatch,
              label + " exact MoE input Q8_1 row experts=" +
                  std::to_string(experts),
              q8_1_mismatch ? q8_1_mismatch->reason : std::string{});

        const auto expected_mixed_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_mixed.data()),
            expected_mixed.size() * sizeof(float));
        const auto actual_mixed_mismatch = twin::compare_bytes(
            storage[kMixed].payload(), expected_mixed_bytes);
        check(!actual_mixed_mismatch,
              label + " router and Q8_1 preserve FFN mixed input experts=" +
                  std::to_string(experts),
              actual_mixed_mismatch ? actual_mixed_mismatch->reason
                                    : std::string{});

        const std::array<twin::ByteRange, 1u> logits_written{{{
            0u, expected_logit_bytes.size()}}};
        const auto logits_guard =
            router_logits_storage.first_unwritten_change(logits_written);
        const std::array<twin::ByteRange, 1u> ids_written{{{
            0u, expected_id_bytes.size()}}};
        const auto ids_guard =
            route_ids_storage.first_unwritten_change(ids_written);
        const std::array<twin::ByteRange, 1u> weights_written{{{
            0u, expected_route_weights.size() * sizeof(float)}}};
        const auto weights_guard =
            route_weights_storage.first_unwritten_change(weights_written);
        const std::array<twin::ByteRange, 1u> q8_1_written{{{
            0u, expected_q8_1_bytes.size()}}};
        const auto q8_1_guard = storage[kXq].first_unwritten_change(q8_1_written);
        const auto sink_guard = router_matvec_sink_storage.first_unwritten_change({});
        const auto router_tensor_guard =
            router_tensor_storage.first_unwritten_change({});
        const bool guards_ok = !logits_guard && !ids_guard && !weights_guard &&
            !q8_1_guard && !sink_guard && !router_tensor_guard;
        check(guards_ok,
              label + " router prefix guards and read-only weights experts=" +
                  std::to_string(experts),
              logits_guard ? logits_guard->reason :
                  ids_guard ? ids_guard->reason :
                  weights_guard ? weights_guard->reason :
                  q8_1_guard ? q8_1_guard->reason :
                  sink_guard ? sink_guard->reason :
                  router_tensor_guard ? router_tensor_guard->reason :
                  std::string{});

        const auto run_route_mode = [&](bool divide_weights,
                                        std::vector<std::int32_t>& output_ids,
                                        std::vector<float>& output_weights) {
            std::fill(route_ids_storage.payload().begin(),
                      route_ids_storage.payload().end(), poison);
            std::fill(route_weights_storage.payload().begin(),
                      route_weights_storage.payload().end(), poison);
            const bool dispatched = run_command(
                context, label + " route mode negative control experts=" +
                             std::to_string(experts) + " divide=" +
                             std::to_string(divide_weights ? 1 : 0),
                [&](CommandBuffer& command, std::string& command_error) {
                    return encode_moe_route(
                        context, command, 1u, experts, divide_weights,
                        {context.buffers().read_only(
                             router_logits_storage.view()),
                         route_ids_storage.view(), route_weights_storage.view()},
                        command_error);
                },
                prefix_error);
            if (dispatched) {
                const auto ids =
                    typed<const std::int32_t>(route_ids_storage).first(kRouteTopK);
                const auto weights =
                    typed<const float>(route_weights_storage).first(kRouteTopK);
                output_ids.assign(ids.begin(), ids.end());
                output_weights.assign(weights.begin(), weights.end());
            }
            return dispatched;
        };
        std::vector<std::int32_t> reciprocal_ids;
        std::vector<float> reciprocal_weights;
        std::vector<std::int32_t> division_ids;
        std::vector<float> division_weights;
        const bool reciprocal_completed =
            run_route_mode(false, reciprocal_ids, reciprocal_weights);
        const bool division_completed =
            run_route_mode(true, division_ids, division_weights);
        const bool prefix_matches_reciprocal = reciprocal_completed &&
            prefix_route_ids == reciprocal_ids &&
            prefix_route_weights == reciprocal_weights;
        bool mode_negative_control_differs = division_completed &&
            division_ids == reciprocal_ids &&
            division_weights.size() == reciprocal_weights.size();
        if (mode_negative_control_differs) {
            mode_negative_control_differs = false;
            for (std::size_t index = 0u; index < reciprocal_weights.size();
                 ++index) {
                if (std::bit_cast<std::uint32_t>(reciprocal_weights[index]) !=
                    std::bit_cast<std::uint32_t>(division_weights[index])) {
                    mode_negative_control_differs = true;
                    break;
                }
            }
        }
        const bool route_mode_ok = prefix_matches_reciprocal &&
                                   mode_negative_control_differs;
        check(route_mode_ok,
              label + " prefix selects reciprocal route mode experts=" +
                  std::to_string(experts),
              prefix_route_weights.empty() ? prefix_error : std::string{});
        return !logit_mismatch && !id_mismatch && !route_weight_mismatch &&
               !q8_1_mismatch && !actual_mixed_mismatch && guards_ok &&
               reciprocal_completed && division_completed && route_mode_ok;
    };
    const bool standard_router_prefix = exercise_router_prefix(512u);
    const bool coder_router_prefix = exercise_router_prefix(256u);

    const auto exercise_plan_prefix = [&](std::uint32_t experts, bool tiered,
                                          bool caller_owned) {
        std::string prefix_error;
        const auto expert_weights =
            std::span<const std::uint16_t>(router_weight_values).first(
                static_cast<std::size_t>(experts) * kRouterInputWidth);
        std::vector<float> expected_logits;
        const bool logits_oracle = twin::bf16_matvec_reference(
            expert_weights, expected_mixed, N_EMBD, experts, 1u, 1u, 256u,
            expected_logits, prefix_error);
        check(logits_oracle,
              label + " plan router oracle experts=" +
                  std::to_string(experts),
              prefix_error);
        if (!logits_oracle) return false;

        std::vector<std::int32_t> expected_ids;
        std::vector<twin::tf> expected_route_weights;
        const bool route_oracle = twin::moe_route_reference(
            expected_logits, 1u, experts, false, expected_ids,
            expected_route_weights, prefix_error);
        check(route_oracle,
              label + " plan route oracle experts=" +
                  std::to_string(experts),
              prefix_error);
        if (!route_oracle) return false;

        std::vector<std::int32_t> resident(experts);
        std::vector<float> initial_usage(experts);
        for (std::uint32_t expert = 0u; expert < experts; ++expert) {
            resident[expert] = -1;
            initial_usage[expert] =
                static_cast<float>(expert % 17u) * 0.125f;
        }
        if (tiered && !expected_ids.empty()) {
            resident[static_cast<std::uint32_t>(expected_ids.front())] = 0;
        }
        std::memcpy(resident_table_storage.payload().data(), resident.data(),
                    experts * sizeof(std::int32_t));
        std::memcpy(usage_table_storage.payload().data(), initial_usage.data(),
                    experts * sizeof(float));

        TieredMoePlan expected_plan{};
        std::vector<float> expected_usage;
        const bool plan_oracle = twin::moe_plan_reference(
            expected_ids, 1u, experts, tiered, 0u, resident, initial_usage,
            expected_plan, expected_usage, prefix_error);
        check(plan_oracle,
              label + " plan oracle experts=" + std::to_string(experts) +
                  " tiered=" + std::to_string(tiered ? 1 : 0),
              prefix_error);
        if (!plan_oracle) return false;
        const bool tiered_hit_and_miss =
            !tiered || (expected_plan.n_miss > 0u &&
                        expected_plan.n_miss < expected_plan.n_groups);
        check(tiered_hit_and_miss,
              label + " tiered plan contains both resident and missing groups",
              "groups=" + std::to_string(expected_plan.n_groups) +
                  " misses=" + std::to_string(expected_plan.n_miss));
        if (!tiered_hit_and_miss) return false;

        prepare_layer0_chain(true);
        for (twin::GuardedBuffer* output :
             {&router_logits_storage, &route_ids_storage,
              &route_weights_storage, &router_matvec_sink_storage,
              &moe_plan_storage}) {
            std::fill(output->payload().begin(), output->payload().end(),
                      poison);
            output->capture_before_dispatch();
        }
        usage_table_storage.capture_before_dispatch();
        resident_table_storage.capture_before_dispatch();
        router_tensor_storage.capture_before_dispatch();

        const bool completed = run_command(
            context, label + " router and MoE plan experts=" +
                         std::to_string(experts) + " tiered=" +
                         std::to_string(tiered ? 1 : 0) +
                         (caller_owned ? " caller-owned" : ""),
            [&](CommandBuffer& command, std::string& command_error) {
                if (!caller_owned) {
                    return encode_window_t1_moe_plan_prefix(
                        context, command, vocabulary_size, experts, tiered,
                        plan_prefix_buffers, command_error);
                }
                return Encoder::encode(
                    context, command, "Window T=1 caller-owned MoE plan",
                    [&](Encoder& encoder, std::string& encode_error) {
                        return encode_window_t1_moe_plan_prefix(
                            context, encoder, vocabulary_size, experts, tiered,
                            plan_prefix_buffers, encode_error);
                    },
                    command_error);
            },
            prefix_error);
        check(completed,
              label + " ordered router-to-plan prefix experts=" +
                  std::to_string(experts) + " tiered=" +
                  std::to_string(tiered ? 1 : 0),
              prefix_error);
        if (!completed) return false;

        const auto expected_id_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_ids.data()),
            expected_ids.size() * sizeof(std::int32_t));
        const auto ids_mismatch = twin::compare_bytes(
            route_ids_storage.payload(), expected_id_bytes);
        check(!ids_mismatch,
              label + " plan consumes routed expert IDs experts=" +
                  std::to_string(experts),
              ids_mismatch ? ids_mismatch->reason : std::string{});

        const auto expected_plan_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(&expected_plan),
            sizeof(expected_plan));
        const auto plan_mismatch = twin::compare_bytes(
            moe_plan_storage.payload(), expected_plan_bytes);
        check(!plan_mismatch,
              label + " exact GPU plan experts=" + std::to_string(experts) +
                  " tiered=" + std::to_string(tiered ? 1 : 0),
              plan_mismatch ? plan_mismatch->reason : std::string{});

        const auto expected_usage_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_usage.data()),
            expected_usage.size() * sizeof(float));
        const auto usage_mismatch = twin::compare_bytes(
            usage_table_storage.payload().first(expected_usage_bytes.size()),
            expected_usage_bytes);
        check(!usage_mismatch,
              label + " exact plan usage experts=" +
                  std::to_string(experts) + " tiered=" +
                  std::to_string(tiered ? 1 : 0),
              usage_mismatch ? usage_mismatch->reason : std::string{});

        const std::array<twin::ByteRange, 1u> plan_written{{
            {0u, sizeof(TieredMoePlan)}}};
        const auto plan_guard =
            moe_plan_storage.first_unwritten_change(plan_written);
        const std::array<twin::ByteRange, 1u> usage_written{{
            {0u, expected_usage_bytes.size()}}};
        const auto usage_guard =
            usage_table_storage.first_unwritten_change(usage_written);
        check(!plan_guard && !usage_guard,
              label + " plan and usage output guards",
              plan_guard ? plan_guard->reason
                         : usage_guard ? usage_guard->reason : std::string{});
        const bool immutable_inputs_unchanged =
            !resident_table_storage.first_unwritten_change({}) &&
            !router_tensor_storage.first_unwritten_change({});
        check(immutable_inputs_unchanged,
              label + " plan inputs remain unchanged");
        return !ids_mismatch && !plan_mismatch && !usage_mismatch &&
               !plan_guard && !usage_guard && immutable_inputs_unchanged;
    };
    const bool standard_tiered_plan =
        exercise_plan_prefix(512u, true, true);
    const bool coder_full_plan =
        exercise_plan_prefix(256u, false, false);

    WindowT1MoePlanPrefixBuffers short_moe_plan = plan_prefix_buffers;
    short_moe_plan.plans =
        moe_plan_storage.view().slice(0u, sizeof(TieredMoePlan) - 1u);
    prepare_layer0_chain(true);
    for (twin::GuardedBuffer* output :
         {&router_logits_storage, &route_ids_storage, &route_weights_storage,
          &router_matvec_sink_storage, &moe_plan_storage}) {
        std::fill(output->payload().begin(), output->payload().end(), poison);
        output->capture_before_dispatch();
    }
    std::fill(usage_table_storage.payload().begin(),
              usage_table_storage.payload().end(), poison);
    usage_table_storage.capture_before_dispatch();
    auto short_plan_command = context.make_command_buffer(
        label + " short MoE plan", error);
    check(short_plan_command != nullptr,
          label + " short-plan command buffer", error);
    if (!short_plan_command) return false;
    const bool short_plan_rejected = !encode_window_t1_moe_plan_prefix(
        context, *short_plan_command, vocabulary_size, 512u, true,
        short_moe_plan, error);
    const bool short_plan_discarded =
        short_plan_command->unsafe_native_handle() == nullptr;
    const bool short_plan_untouched =
        outputs_are_poisoned() &&
        std::all_of(router_logits_storage.payload().begin(),
                    router_logits_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(route_ids_storage.payload().begin(),
                    route_ids_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(route_weights_storage.payload().begin(),
                    route_weights_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(usage_table_storage.payload().begin(),
                    usage_table_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(moe_plan_storage.payload().begin(),
                    moe_plan_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; });
    check(short_plan_rejected && short_plan_discarded &&
              short_plan_untouched &&
              error.find("plan output") != std::string::npos &&
              error.find("undersized") != std::string::npos,
          label + " undersized plan rejects before prefix dispatch", error);

    prepare_layer0_chain(true);
    for (twin::GuardedBuffer* output :
         {&router_logits_storage, &route_ids_storage, &route_weights_storage,
          &router_matvec_sink_storage, &usage_table_storage,
          &moe_plan_storage}) {
        std::fill(output->payload().begin(), output->payload().end(), poison);
    }
    auto late_plan_failure_command = context.make_command_buffer(
        label + " caller-owned late plan-prefix failure", error);
    check(late_plan_failure_command != nullptr,
          label + " late-plan-prefix command buffer", error);
    if (!late_plan_failure_command) return false;
    bool plan_prefix_encoded_before_failure = false;
    constexpr std::string_view kInjectedLateFailure =
        "injected failure after Window T=1 MoE plan prefix";
    const bool late_plan_failure_rejected = !Encoder::encode(
        context, *late_plan_failure_command,
        "Window T=1 late MoE plan-prefix failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_window_t1_moe_plan_prefix(
                    context, encoder, vocabulary_size, 512u, true,
                    plan_prefix_buffers, encode_error)) {
                return false;
            }
            plan_prefix_encoded_before_failure = true;
            encode_error = kInjectedLateFailure;
            return false;
        },
        error);
    const bool late_plan_failure_discarded =
        late_plan_failure_command->unsafe_native_handle() == nullptr;
    const auto poisoned_bytes = [](const twin::GuardedBuffer& buffer) {
        return std::all_of(buffer.payload().begin(), buffer.payload().end(),
                           [=](std::uint8_t value) { return value == poison; });
    };
    const bool late_plan_failure_untouched = outputs_are_poisoned() &&
        poisoned_bytes(router_logits_storage) &&
        poisoned_bytes(route_ids_storage) &&
        poisoned_bytes(route_weights_storage) &&
        poisoned_bytes(router_matvec_sink_storage) &&
        poisoned_bytes(usage_table_storage) &&
        poisoned_bytes(moe_plan_storage);
    check(late_plan_failure_rejected &&
              plan_prefix_encoded_before_failure &&
              late_plan_failure_discarded && late_plan_failure_untouched &&
              error == kInjectedLateFailure,
          label + " late caller failure rolls back the encoded router and plan",
          error);

    WindowT1MoeRouterPrefixBuffers short_router_sink = router_prefix_buffers;
    short_router_sink.router_matvec_sink =
        router_matvec_sink_storage.view().slice(0u, sizeof(float) - 1u);
    prepare_layer0_chain(true);
    std::fill(router_logits_storage.payload().begin(),
              router_logits_storage.payload().end(), poison);
    std::fill(route_ids_storage.payload().begin(), route_ids_storage.payload().end(),
              poison);
    std::fill(route_weights_storage.payload().begin(),
              route_weights_storage.payload().end(), poison);
    std::fill(router_matvec_sink_storage.payload().begin(),
              router_matvec_sink_storage.payload().end(), poison);
    auto short_sink_command = context.make_command_buffer(
        label + " short router sink", error);
    check(short_sink_command != nullptr,
          label + " short-router-sink command buffer", error);
    if (!short_sink_command) return false;
    const bool short_sink_rejected = !encode_window_t1_moe_router_prefix(
        context, *short_sink_command, vocabulary_size, 512u,
        short_router_sink, error);
    const bool short_sink_discarded =
        short_sink_command->unsafe_native_handle() == nullptr;
    const bool short_sink_untouched = outputs_are_poisoned() &&
        std::equal(gr_residual_before.begin(), gr_residual_before.end(),
                   storage[kGrResidual].payload().begin(),
                   storage[kGrResidual].payload().end()) &&
        std::all_of(router_logits_storage.payload().begin(),
                    router_logits_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(route_ids_storage.payload().begin(),
                    route_ids_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(route_weights_storage.payload().begin(),
                    route_weights_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; });
    check(short_sink_rejected && short_sink_discarded && short_sink_untouched &&
              error.find("router mat-vec sink") != std::string::npos,
          label + " undersized router sink rejects before dispatch", error);

    WindowT1MoeRouterPrefixBuffers aliased_router_logits =
        router_prefix_buffers;
    aliased_router_logits.router_logits = storage[kGrResidual].view();
    prepare_layer0_chain(true);
    auto alias_command = context.make_command_buffer(
        label + " router logits alias", error);
    check(alias_command != nullptr,
          label + " router-logits-alias command buffer", error);
    if (!alias_command) return false;
    const bool alias_rejected = !Encoder::encode(
        context, *alias_command, "Window router-prefix alias rollback",
        [&](Encoder& encoder, std::string& body_error) {
            return encode_window_t1_moe_router_prefix(
                context, encoder, vocabulary_size, 512u,
                aliased_router_logits, body_error);
        },
        error);
    const bool alias_discarded =
        alias_command->unsafe_native_handle() == nullptr;
    const bool alias_untouched = outputs_are_poisoned() &&
        std::equal(gr_residual_before.begin(), gr_residual_before.end(),
                   storage[kGrResidual].payload().begin(),
                   storage[kGrResidual].payload().end()) &&
        std::all_of(route_ids_storage.payload().begin(),
                    route_ids_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(route_weights_storage.payload().begin(),
                    route_weights_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; });
    check(alias_rejected && alias_discarded && alias_untouched &&
              error.find("overlaps a layer-0 stage buffer") != std::string::npos,
          label + " caller-owned router alias failure discards the segment",
          error);

    twin::GuardedBuffer router_tensor_param_alias_storage;
    const bool router_tensor_param_alias_allocated =
        router_tensor_param_alias_storage.allocate(
            context, label + " router tensor and parameter overlap",
            router_weight_bytes, error);
    check(router_tensor_param_alias_allocated,
          label + " router-tensor/parameter alias allocation", error);
    if (!router_tensor_param_alias_allocated) return false;
    std::fill(router_tensor_param_alias_storage.payload().begin(),
              router_tensor_param_alias_storage.payload().end(), poison);
    std::memcpy(router_tensor_param_alias_storage.payload().data(),
                &valid_params, sizeof(valid_params));
    WindowT1MoeRouterPrefixBuffers aliased_router_tensor_params =
        router_prefix_buffers;
    aliased_router_tensor_params.layer0.params =
        router_tensor_param_alias_storage.view();
    aliased_router_tensor_params.router_tensor = context.buffers().read_only(
        router_tensor_param_alias_storage.view());
    prepare_layer0_chain(true);
    std::fill(router_logits_storage.payload().begin(),
              router_logits_storage.payload().end(), poison);
    std::fill(route_ids_storage.payload().begin(), route_ids_storage.payload().end(),
              poison);
    std::fill(route_weights_storage.payload().begin(),
              route_weights_storage.payload().end(), poison);
    auto router_param_alias_command = context.make_command_buffer(
        label + " router tensor parameter alias", error);
    check(router_param_alias_command != nullptr,
          label + " router-tensor/parameter-alias command buffer", error);
    if (!router_param_alias_command) return false;
    const bool router_param_alias_rejected =
        !encode_window_t1_moe_router_prefix(
            context, *router_param_alias_command, vocabulary_size, 512u,
            aliased_router_tensor_params, error);
    const bool router_param_alias_discarded =
        router_param_alias_command->unsafe_native_handle() == nullptr;
    const bool router_param_alias_untouched = outputs_are_poisoned() &&
        std::equal(gr_residual_before.begin(), gr_residual_before.end(),
                   storage[kGrResidual].payload().begin(),
                   storage[kGrResidual].payload().end()) &&
        std::all_of(router_logits_storage.payload().begin(),
                    router_logits_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(route_ids_storage.payload().begin(),
                    route_ids_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; }) &&
        std::all_of(route_weights_storage.payload().begin(),
                    route_weights_storage.payload().end(),
                    [=](std::uint8_t value) { return value == poison; });
    check(router_param_alias_rejected && router_param_alias_discarded &&
              router_param_alias_untouched &&
              error.find("router tensor overlaps its parameters") !=
                  std::string::npos,
          label + " router tensor overlapping WindowParams rejects before dispatch",
          error);

    const auto reject_layer0_params = [&](const std::string& case_name,
                                          const std::string& expected_error,
                                          auto mutate) {
        strata::metal::WindowParams malformed{};
        std::memcpy(&malformed, params.contents, sizeof(malformed));
        mutate(malformed);
        std::memcpy(params.contents, &malformed, sizeof(malformed));
        auto command = context.make_command_buffer(
            label + " reject layer-0 " + case_name, error);
        if (!command) {
            std::memcpy(params.contents, &valid_params, sizeof(valid_params));
            check(false, label + " layer-0 " + case_name + " command buffer",
                  error);
            return false;
        }
        prepare_layer0_chain(true);
        const bool rejected = !encode_window_t1_attention_gdn_ffn(
            context, *command, vocabulary_size, layer0_buffers, error);
        const bool discarded = command->unsafe_native_handle() == nullptr;
        const bool untouched = outputs_are_poisoned() &&
            std::equal(gr_residual_before.begin(), gr_residual_before.end(),
                       storage[kGrResidual].payload().begin(),
                       storage[kGrResidual].payload().end());
        std::memcpy(params.contents, &valid_params, sizeof(valid_params));
        const bool correct = rejected && discarded && untouched &&
            error.find(expected_error) != std::string::npos;
        check(correct, label + " layer-0 rejects " + case_name, error);
        return correct;
    };
    const bool layer0_n_keep_rejected = reject_layer0_params(
        "nonzero n_keep", "n_keep=0", [](strata::metal::WindowParams& value) {
            value.n_keep = 1u;
        });
    const bool layer0_bad_nc_rejected = reject_layer0_params(
        "T=1 with invalid nc", "T=1", [](strata::metal::WindowParams& value) {
            value.nc = 2u;
        });

    WindowT1AttentionGDNFFNBuffers split_injection = layer0_buffers;
    split_injection.ffn.inject_prev = separate_mixed_storage.view();
    auto split_injection_command = context.make_command_buffer(
        label + " reject split injection handoff", error);
    check(split_injection_command != nullptr,
          label + " split-injection command buffer", error);
    if (!split_injection_command) return false;
    prepare_layer0_chain(true);
    const bool split_injection_rejected =
        !encode_window_t1_attention_gdn_ffn(
            context, *split_injection_command, vocabulary_size,
            split_injection, error);
    const bool split_injection_discarded =
        split_injection_command->unsafe_native_handle() == nullptr;
    check(split_injection_rejected && split_injection_discarded &&
              outputs_are_poisoned() &&
              error.find("swapped injection views") != std::string::npos,
          label + " rejects a split injection handoff", error);

    WindowT1AttentionGDNFFNBuffers shared_attention_ffn_weights = layer0_buffers;
    shared_attention_ffn_weights.ffn.norm = attention.norm;
    auto shared_weights_command = context.make_command_buffer(
        label + " reject aliased attention and FFN weights", error);
    check(shared_weights_command != nullptr,
          label + " shared-weight command buffer", error);
    if (!shared_weights_command) return false;
    prepare_layer0_chain(true);
    const bool shared_weights_rejected =
        !encode_window_t1_attention_gdn_ffn(
            context, *shared_weights_command, vocabulary_size,
            shared_attention_ffn_weights, error);
    const bool shared_weights_discarded =
        shared_weights_command->unsafe_native_handle() == nullptr;
    check(shared_weights_rejected && shared_weights_discarded &&
              outputs_are_poisoned() &&
              error.find("outside an exact scratch handoff") !=
                  std::string::npos,
          label + " rejects reusing attention weights for FFN", error);

    WindowT1AttentionGDNFFNBuffers partial_attention_ffn_weights = layer0_buffers;
    partial_attention_ffn_weights.ffn.norm = attention.norm.slice(
        sizeof(float), attention.norm.length - sizeof(float));
    auto partial_weights_command = context.make_command_buffer(
        label + " reject partially aliased attention and FFN weights", error);
    check(partial_weights_command != nullptr,
          label + " partial-weight command buffer", error);
    if (!partial_weights_command) return false;
    prepare_layer0_chain(true);
    const bool partial_weights_rejected =
        !encode_window_t1_attention_gdn_ffn(
            context, *partial_weights_command, vocabulary_size,
            partial_attention_ffn_weights, error);
    const bool partial_weights_discarded =
        partial_weights_command->unsafe_native_handle() == nullptr;
    check(partial_weights_rejected && partial_weights_discarded &&
              outputs_are_poisoned() &&
              error.find("outside an exact scratch handoff") !=
                  std::string::npos,
          label + " rejects partial attention/FFN weight overlap", error);

    auto short_ffn_up = ffn_up_storage.view().slice(0u, 16u);
    WindowT1AttentionGDNFFNBuffers late_ffn_failure = layer0_buffers;
    late_ffn_failure.ffn.hc_up = short_ffn_up;
    auto late_ffn_command = context.make_command_buffer(
        label + " late FFN validation failure", error);
    check(late_ffn_command != nullptr,
          label + " late-FFN command buffer", error);
    if (!late_ffn_command) return false;
    prepare_layer0_chain(true);
    const bool late_ffn_rejected = !encode_window_t1_attention_gdn_ffn(
        context, *late_ffn_command, vocabulary_size, late_ffn_failure, error);
    const bool late_ffn_discarded =
        late_ffn_command->unsafe_native_handle() == nullptr;
    const bool late_ffn_untouched = outputs_are_poisoned() &&
        std::equal(gr_residual_before.begin(), gr_residual_before.end(),
                   storage[kGrResidual].payload().begin(),
                   storage[kGrResidual].payload().end());
    check(late_ffn_rejected && late_ffn_discarded && late_ffn_untouched &&
              error.find("undersized") != std::string::npos,
          label + " late FFN failure discards attention and GDN work", error);

    auto shared_late_ffn_command = context.make_command_buffer(
        label + " shared late FFN validation failure", error);
    check(shared_late_ffn_command != nullptr,
          label + " shared late-FFN command buffer", error);
    if (!shared_late_ffn_command) return false;
    prepare_layer0_chain(true);
    const bool shared_late_ffn_rejected = !Encoder::encode(
        context, *shared_late_ffn_command, "Window layer-0 shared late failure",
        [&](Encoder& encoder, std::string& body_error) {
            return encode_window_t1_attention_gdn_ffn(
                context, encoder, vocabulary_size, late_ffn_failure, body_error);
        },
        error);
    const bool shared_late_ffn_discarded =
        shared_late_ffn_command->unsafe_native_handle() == nullptr;
    const bool shared_late_ffn_untouched = outputs_are_poisoned() &&
        std::equal(gr_residual_before.begin(), gr_residual_before.end(),
                   storage[kGrResidual].payload().begin(),
                   storage[kGrResidual].payload().end());
    check(shared_late_ffn_rejected && shared_late_ffn_discarded &&
              shared_late_ffn_untouched &&
              error.find("undersized") != std::string::npos,
          label + " caller-owned late FFN failure discards the segment", error);

    return wrong_handoff_rejected && wrong_handoff_discarded &&
           wrong_rows_rejected && wrong_rows_discarded && invalid_nc_rejected &&
           nonzero_keep_rejected && invalid_token_rejected &&
           invalid_step_rejected && overlap_rejected && overlap_discarded &&
           late_failure_rejected && late_failure_discarded &&
           shared_failure_rejected && shared_failure_discarded &&
           layer0_apply_control_writes_residual &&
           layer0_reference_completed && layer0_ffn_applies_pending_write &&
           component_state_unchanged && layer0_completed &&
           layer0_outputs_match && layer0_residual_matches &&
           layer0_inject2_matches && !residual_guard && !inject2_guard &&
           layer0_state_unchanged && standard_router_prefix &&
           coder_router_prefix && standard_tiered_plan && coder_full_plan &&
           short_plan_rejected && short_plan_discarded &&
           short_plan_untouched && late_plan_failure_rejected &&
           plan_prefix_encoded_before_failure &&
           late_plan_failure_discarded && late_plan_failure_untouched &&
           short_sink_rejected &&
           short_sink_discarded && short_sink_untouched && alias_rejected &&
           alias_discarded && alias_untouched &&
           router_param_alias_rejected && router_param_alias_discarded &&
           router_param_alias_untouched && layer0_n_keep_rejected &&
           layer0_bad_nc_rejected && split_injection_rejected &&
           split_injection_discarded && shared_weights_rejected &&
           shared_weights_discarded && partial_weights_rejected &&
           partial_weights_discarded && late_ffn_rejected &&
           late_ffn_discarded && late_ffn_untouched &&
           shared_late_ffn_rejected && shared_late_ffn_discarded &&
           shared_late_ffn_untouched;
}

bool test_window_gdn_block_contract(Context& context) {
    bool passed = true;
    const WindowGDNBlockBuffers empty{};
    for (const std::uint32_t rows : {0u, 9u}) {
        std::string error;
        auto command = context.make_command_buffer(
            "Window GDN invalid row count", error);
        check(command != nullptr, "Window GDN invalid-row command buffer", error);
        if (!command) return false;
        const bool accepted = encode_window_gdn_block(
            context, *command, rows, empty, error);
        const bool rejected = !accepted &&
            command->unsafe_native_handle() == nullptr &&
            error.find("row count must be in 1..8") != std::string::npos;
        check(rejected, "Window GDN rejects T=" + std::to_string(rows), error);
        passed &= rejected;
    }
    const WindowGDNCommitBuffers empty_commit{};
    {
        std::string error;
        auto command = context.make_command_buffer(
            "Window GDN empty commit", error);
        check(command != nullptr, "Window GDN empty-commit command buffer", error);
        if (!command) return false;
        const bool accepted = encode_window_gdn_commit(
            context, *command, 0u, empty_commit, error);
        const bool no_op = accepted && error.empty();
        check(no_op, "Window GDN commit accepts n_keep=0 without buffers", error);
        passed &= no_op;
        command->discard();
    }
    {
        std::string error;
        auto command = context.make_command_buffer(
            "Window GDN oversized commit", error);
        check(command != nullptr,
              "Window GDN oversized-commit command buffer", error);
        if (!command) return false;
        const bool accepted = encode_window_gdn_commit(
            context, *command, 9u, empty_commit, error);
        const bool rejected = !accepted &&
            command->unsafe_native_handle() == nullptr &&
            error.find("row count must be in 0..8") != std::string::npos;
        check(rejected, "Window GDN commit rejects n_keep=9", error);
        passed &= rejected;
    }
    return passed;
}

bool run_set(Context& context, twin::InputSet set,
             const twin::Options& options) {
    bool passed = true;
    for (std::uint32_t rows = 1u; rows <= 8u; ++rows) {
        passed &= run_conv_l2_case(context, rows, set, options);
        passed &= run_ab_case(context, rows, set, options);
    }
    if (set == twin::InputSet::fixture) {
        passed &= run_conv_l2_case(context, 1u, set, options,
                                   EdgeProfile::regular, true);
        passed &= run_conv_l2_case(context, 8u, set, options,
                                   EdgeProfile::regular, true);
        passed &= run_ab_case(context, 1u, set, options,
                              EdgeProfile::regular, true);
        passed &= run_ab_case(context, 8u, set, options,
                              EdgeProfile::regular, true);
        passed &= run_conv_commit_case(context, 1u, options,
                                       EdgeProfile::regular, true);
        passed &= run_conv_commit_case(context, 8u, options,
                                       EdgeProfile::regular, true);
    }
    for (std::uint32_t rows = 0u; rows <= 8u; ++rows) {
        passed &= run_conv_commit_case(context, rows, options);
    }
    passed &= run_step_norm_set(context, set, options);
    if (set == twin::InputSet::edge) {
        passed &= run_conv_l2_case(context, 1u, set, options,
                                   EdgeProfile::all_zero);
        passed &= run_conv_l2_case(context, 2u, set, options,
                                   EdgeProfile::one_outlier);
        passed &= run_ab_case(context, 1u, set, options,
                              EdgeProfile::all_zero);
        passed &= run_ab_case(context, 2u, set, options,
                              EdgeProfile::one_outlier);
        passed &= run_conv_commit_case(context, 1u, options,
                                       EdgeProfile::all_zero);
        passed &= run_conv_commit_case(context, 8u, options,
                                       EdgeProfile::one_outlier);
        passed &= run_step_norm_set(context, set, options,
                                    EdgeProfile::all_zero);
        passed &= run_step_norm_set(context, set, options,
                                    EdgeProfile::one_outlier);
    }
    if (set == twin::InputSet::fixture) {
        passed &= run_step_norm_set(context, set, options,
                                    EdgeProfile::regular, true);
        passed &= run_window_gdn_block(context, 1u, false, options);
        passed &= run_window_gdn_block(context, 2u, true, options);
        passed &= run_window_gdn_block(context, 3u, false, options);
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_gdn_test strata.metallib "
                     "[--seed N] [--input-set random|edge|fixture]\n");
        return 2;
    }
    twin::Options options;
    std::string error;
    if (!twin::parse_options(argc, argv, 2, options, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    if (options.input_set == twin::InputSet::real) {
        std::fprintf(stderr,
                     "metal_gdn_test has no model-backed real input set\n");
        return 77;
    }

    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, error);
    check(context != nullptr, "open Metal context", error);
    if (!context) return 1;
    ContextOptions foreign_options = context_options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign validation context",
          error);
    if (!foreign_context) return 1;

    bool passed = test_encoder_validation(*context, *foreign_context);
    passed &= test_window_t1_attention_gdn(*context);
    passed &= test_window_gdn_block_contract(*context);
    passed &= test_conv_l2_shared_encoder_validation(
        context_options, *foreign_context);
    passed &= test_ab_shared_encoder_validation(context_options,
                                                *foreign_context);
    passed &= test_step_norm_shared_encoder_validation(
        context_options, *foreign_context);
    passed &= test_conv_commit_shared_encoder_validation(
        context_options, *foreign_context);
    if (options.input_set_explicit) {
        passed &= run_set(*context, options.input_set, options);
    } else {
        constexpr std::array sets{twin::InputSet::random,
                                  twin::InputSet::edge,
                                  twin::InputSet::fixture};
        for (const auto set : sets) {
            std::printf("GDN input set: %s\n", twin::input_set_name(set));
            passed &= run_set(*context, set, options);
        }
    }
    check(passed && g_failures == 0,
          "GDN conv_l2/ab/step_norm/conv_commit encoder twins");
    std::printf("metal_gdn_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
