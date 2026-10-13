// Property pf_gdn_eq_step_norm (docs/mac/04 sections 6.4 and 10.4).
//
// SYNTHETIC: every tensor is random data in the real GDN shapes (tests/metal/synth.cpp). No model
// weights, CUDA dumps or model files are read. This checks kernel equivalence only; it is not
// model-backed inference.
//
// For T = 1, 3, 64, 65 and 300 the Prefill chain (tiled convolution, history advance, q/k L2,
// recurrence, output norm) runs over all T rows in one ordered Encoder, with the same qkv, gate,
// beta, z, history and state as the reference paths:
//   * Class B, against the decode GPU kernels run step by step in windows of up to 8 rows
//     (write-through mode, history advanced after each window). h, the history, the state and
//     y_h must equal the decode outputs bit for bit, with y_h = half(y).
//   * Independent host reference (oracle::gdn_conv_l2, gdn_conv_commit, gdn_step_norm), which
//     runs the same recurrence row by row. The prefill recurrence and output norm are fed the
//     oracle's own h so the stage's bound is the only error term:
//       conv + L2 (h) against the oracle, U(128); history, B;
//       recurrence and output norm: state U(512); y_h within the F16 bound below.
//     The whole chain, prefill and decode, is checked against the oracle at U(640) for the state
//     and the F32 y, which is the sum of the conv and recurrence bounds.
//
// Bounds. U(128) for conv + L2 and U(512) for the recurrence are the bounds metal_gdn_test uses for
// the same arithmetic. The recurrence does not amplify error with T. The state map
// s -> exp(gate) (I - beta k k^T) s has operator norm at most exp(gate), since k has unit length
// after L2 and 0 <= beta <= 1. So an error on one row is damped by exp(gate) on the next, and the
// synthetic gates are below zero (about -2.5). For y_h, which is F16, the bound is 2^-10 |y| +
// 2^-24 + n 2^-24 m, where n is the U bound and m the oracle's magnitude. It covers one F16
// rounding (relative 2^-11, absolute 2^-25 in the subnormal range) plus the F32 error.
//
// Negative controls: one flipped state bit, one flipped y_h bit and a 1e-3 relative offset in y must
// each be rejected by the comparison that covers that output (B, the F16 bound and U(640)).
// Each run prints the observed n of every U comparison, so the margin to its bound is visible.

#include "metallib_sha256.inc"
#include "oracles/gdn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gdn_dispatch.hpp"
#include "strata/metal/prefill_gdn_conv_dispatch.hpp"
#include "strata/metal/prefill_gdn_l2_dispatch.hpp"
#include "strata/metal/prefill_gdn_out_norm_dispatch.hpp"
#include "strata/metal/prefill_gdn_rec_dispatch.hpp"
#include "synth.hpp"
#include "twin.hpp"

#include "../../src/metal/kernels/common/params.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kChannels = GDN_CONV_CHANNELS;
constexpr std::uint32_t kValueDim = GDN_VALUE_DIM;
constexpr std::uint32_t kHeads = GDN_V_HEADS;
constexpr std::uint32_t kHeadDim = GDN_HEAD_DIM;
constexpr std::uint32_t kTaps = GDN_CONV_KERNEL;
constexpr std::uint32_t kStateElements = GDN_STATE_ELEMENTS;
constexpr std::uint32_t kHistoryElements = GDN_HISTORY_ELEMENTS;
constexpr std::uint32_t kWindow = 8u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint64_t kSeed = 0x70666764u;
constexpr std::array<std::uint32_t, 5> kRowCounts{1u, 3u, 64u, 65u, 300u};

constexpr double kConvUnits = 128.0;
constexpr double kRecurrenceUnits = 512.0;
constexpr double kChainUnits = kConvUnits + kRecurrenceUnits;

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

std::string describe(const twin::Failure& failure) {
    char buffer[256];
    std::snprintf(buffer, sizeof buffer,
                  "index %zu, actual 0x%llx, expected 0x%llx: %s", failure.index,
                  static_cast<unsigned long long>(failure.actual),
                  static_cast<unsigned long long>(failure.expected),
                  failure.reason.c_str());
    return buffer;
}

template <class T>
std::span<const std::uint8_t> bytes_of(std::span<const T> values) {
    return {reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size_bytes()};
}

// The largest n such that the U(n) bound holds for every element.
double observed_units(std::span<const float> actual,
                      std::span<const twin::tf> expected) {
    constexpr double kFloor = 0x1.0p-126;
    double worst = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error =
            std::fabs(static_cast<double>(actual[i]) - expected[i].value);
        if (error <= kFloor) continue;
        const double unit = std::ldexp(expected[i].magnitude, -24);
        if (!(unit > 0.0)) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, (error - kFloor) / unit);
    }
    return worst;
}

bool units_within(std::span<const float> actual,
                  std::span<const twin::tf> expected, double units) {
    return !twin::compare_floats(actual, expected,
                                 {twin::Comparison::units, units, 0.0, 0u});
}

bool units_match(const std::string& label, std::span<const float> actual,
                 std::span<const twin::tf> expected, double units) {
    const auto failure = twin::compare_floats(
        actual, expected, {twin::Comparison::units, units, 0.0, 0u});
    char detail[96];
    std::snprintf(detail, sizeof detail, "observed n %.4g, bound n %.0f",
                  observed_units(actual, expected), units);
    std::string text = detail;
    if (failure) text += "; " + describe(*failure);
    check(!failure, label + " U(" + std::to_string(static_cast<int>(units)) + ")",
          text);
    return !failure;
}

bool bits_equal(std::span<const std::uint8_t> actual,
                std::span<const std::uint8_t> expected) {
    return actual.size() == expected.size() &&
           std::memcmp(actual.data(), expected.data(), actual.size()) == 0;
}

bool bits_match(const std::string& label, std::span<const std::uint8_t> actual,
                std::span<const std::uint8_t> expected) {
    const auto failure = twin::compare_bytes(actual, expected);
    check(!failure, label + " B", failure ? describe(*failure) : "");
    return !failure;
}

// The F16 value y_h is the F32 result rounded to F16, so it is within
// 2^-10 |y| + 2^-24 + units 2^-24 magnitude of the oracle's F32 value. Returns the worst
// error-to-bound ratio, and whether every element is within its bound.
bool half_within(std::span<const std::uint16_t> actual,
                 std::span<const twin::tf> expected, double units,
                 double& worst_ratio, std::size_t& worst_index) {
    worst_ratio = 0.0;
    worst_index = 0;
    bool finite = true;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double want = expected[i].value;
        const double got = strata::kernels::f32_from_f16(actual[i]);
        finite = finite && std::isfinite(got);
        const double bound = 0x1.0p-10 * std::fabs(want) + 0x1.0p-24 +
                             units * 0x1.0p-24 * expected[i].magnitude;
        const double error = std::fabs(got - want);
        const double ratio = error == 0.0 ? 0.0 : error / bound;
        if (!(ratio <= worst_ratio)) {
            worst_ratio = ratio;
            worst_index = i;
        }
    }
    return finite && worst_ratio <= 1.0;
}

bool half_match(const std::string& label, std::span<const std::uint16_t> actual,
                std::span<const twin::tf> expected, double units) {
    double worst_ratio = 0.0;
    std::size_t worst_index = 0;
    const bool passed =
        half_within(actual, expected, units, worst_ratio, worst_index);
    char detail[160];
    std::snprintf(detail, sizeof detail,
                  "worst error/bound %.3g at index %zu, bound n %.0f", worst_ratio,
                  worst_index, units);
    check(passed, label + " F16 bound", detail);
    return passed;
}

bool guards_hold(const std::string& label, const twin::GuardedBuffer& buffer,
                 std::size_t written_bytes) {
    const std::array<twin::ByteRange, 1> written{{{0u, written_bytes}}};
    const auto change = buffer.first_unwritten_change(written);
    check(!change, label + " guards", change ? change->reason : "");
    return !change;
}

template <class T>
std::span<const T> typed(const twin::GuardedBuffer& buffer) {
    const auto bytes = buffer.payload();
    return {reinterpret_cast<const T*>(bytes.data()), bytes.size() / sizeof(T)};
}

template <class F>
bool run_encoded(Context& context, const std::string& label, F&& body,
                 std::string& error) {
    auto command = context.make_command_buffer(label, error);
    if (!command) return false;
    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& encode_error) {
            return body(encoder, encode_error);
        },
        error);
    if (!encoded) return false;
    return command->commit(error) && command->wait_until_completed(error);
}

bool upload(Context& context, twin::GuardedBuffer& buffer,
            const std::string& label, std::span<const float> values,
            std::string& error) {
    if (!buffer.allocate(context, label, values.size_bytes(), error)) return false;
    std::memcpy(buffer.payload().data(), values.data(), values.size_bytes());
    buffer.capture_before_dispatch();
    return true;
}

bool allocate_poisoned(Context& context, twin::GuardedBuffer& buffer,
                       const std::string& label, std::size_t bytes,
                       std::string& error) {
    if (!buffer.allocate(context, label, bytes, error)) return false;
    std::fill(buffer.payload().begin(), buffer.payload().end(), kPoison);
    buffer.capture_before_dispatch();
    return true;
}

struct Inputs {
    std::uint32_t rows = 0;
    std::vector<float> history;
    std::vector<float> qkv;
    std::vector<float> weights;
    std::vector<float> state;
    std::vector<float> gate;
    std::vector<float> beta;
    std::vector<float> z;
    std::vector<float> gamma;
};

Inputs make_inputs(std::uint32_t rows) {
    std::mt19937_64 rng(kSeed ^ (std::uint64_t{rows} << 24u));
    std::string error;
    auto draw = [&](std::size_t count, float scale) {
        std::vector<float> values =
            twin::synthesize_floats(twin::InputSet::random, count, rng, error);
        for (float& value : values) value *= scale;
        return values;
    };
    Inputs in;
    in.rows = rows;
    in.history = draw(kHistoryElements, 0.2f);
    in.qkv = draw(static_cast<std::size_t>(rows) * kChannels, 0.5f);
    in.weights = draw(static_cast<std::size_t>(kChannels) * kTaps, 0.15f);
    in.state = draw(kStateElements, 0.02f);
    // exp(gate) < 1 keeps the recurrence contractive; beta is a sigmoid, in (0, 1).
    in.gate = draw(static_cast<std::size_t>(rows) * kHeads, 0.3f);
    for (float& value : in.gate) value -= 2.5f;
    in.beta = draw(static_cast<std::size_t>(rows) * kHeads, 1.0f);
    for (float& value : in.beta) value = 1.0f / (1.0f + std::exp(-value));
    in.z = draw(static_cast<std::size_t>(rows) * kValueDim, 1.0f);
    in.gamma = draw(kHeadDim, 0.1f);
    for (float& value : in.gamma) value += 1.0f;
    if (!error.empty()) {
        std::fprintf(stderr, "synthetic input generation failed: %s\n", error.c_str());
        std::abort();
    }
    return in;
}

struct HostReference {
    std::vector<twin::tf> h;
    std::vector<float> h_values;
    std::vector<float> history;
    std::vector<twin::tf> y;
    std::vector<twin::tf> state;
};

// The independent reference: the decode recurrence applied row by row. Windowing does not change
// the arithmetic, since each row reads only the state and the history left by the rows before it.
HostReference run_host_reference(const Inputs& in) {
    HostReference ref;
    oracle::gdn_conv_l2(in.history, in.qkv, in.weights, in.rows, ref.h);
    ref.h_values.resize(ref.h.size());
    for (std::size_t i = 0; i < ref.h.size(); ++i) ref.h_values[i] = ref.h[i].value;
    ref.history = in.history;
    oracle::gdn_conv_commit(ref.history, in.qkv, in.rows);
    std::vector<float> state = in.state;
    oracle::gdn_step_norm(state, ref.h_values, in.gate, in.beta, in.z, in.gamma,
                          in.rows, GDNStateMode::write_through, ref.y, ref.state);
    return ref;
}

struct Outputs {
    std::vector<float> h;
    std::vector<float> history;
    std::vector<float> state;
    std::vector<float> y;                 // decode: F32 output
    std::vector<std::uint16_t> y_h;       // prefill: F16 output
};

// Decode GPU kernels, one window of up to 8 rows at a time: conv + L2, the write-through recurrence
// and output norm, then the history advance. Each window reads the state and history the last one
// left behind.
bool run_decode(Context& context, const std::string& tag, const Inputs& in,
                Outputs& out, std::string& error) {
    const std::uint32_t rows = in.rows;
    const std::size_t row_channel_bytes = kChannels * sizeof(float);
    const std::size_t row_value_bytes = kValueDim * sizeof(float);
    const std::size_t row_head_bytes = kHeads * sizeof(float);
    twin::GuardedBuffer history, state, qkv, weights, gate, beta, z, gamma, h, y;
    const bool allocated =
        upload(context, history, tag + " history", in.history, error) &&
        upload(context, state, tag + " state", in.state, error) &&
        upload(context, qkv, tag + " qkv", in.qkv, error) &&
        upload(context, weights, tag + " weights", in.weights, error) &&
        upload(context, gate, tag + " gate", in.gate, error) &&
        upload(context, beta, tag + " beta", in.beta, error) &&
        upload(context, z, tag + " z", in.z, error) &&
        upload(context, gamma, tag + " gamma", in.gamma, error) &&
        allocate_poisoned(context, h, tag + " h", rows * row_channel_bytes, error) &&
        allocate_poisoned(context, y, tag + " y", rows * row_value_bytes, error);
    if (!allocated) return false;

    for (std::uint32_t start = 0u; start < rows; start += kWindow) {
        const std::uint32_t count = std::min(kWindow, rows - start);
        const std::string label = tag + " decode window at row " +
                                  std::to_string(start) + " rows " +
                                  std::to_string(count);
        const auto qkv_w = qkv.view().slice(start * row_channel_bytes,
                                            count * row_channel_bytes);
        const auto h_w = h.view().slice(start * row_channel_bytes,
                                        count * row_channel_bytes);
        const auto gate_w = gate.view().slice(start * row_head_bytes,
                                              count * row_head_bytes);
        const auto beta_w = beta.view().slice(start * row_head_bytes,
                                              count * row_head_bytes);
        const auto z_w = z.view().slice(start * row_value_bytes,
                                        count * row_value_bytes);
        const auto y_w = y.view().slice(start * row_value_bytes,
                                        count * row_value_bytes);
        if (!qkv_w || !h_w || !gate_w || !beta_w || !z_w || !y_w) {
            error = "decode window slice is outside its buffer";
            return false;
        }
        const bool encoded = run_encoded(
            context, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_gdn_conv_l2(
                           context, encoder, count,
                           GDNConvL2Buffers{history.view(), qkv_w, weights.view(), h_w},
                           encode_error) &&
                       encode_gdn_step_norm(
                           context, encoder, count, GDNStateMode::write_through,
                           GDNStepNormBuffers{state.view(), h_w, gate_w, beta_w, z_w,
                                              gamma.view(), y_w},
                           encode_error) &&
                       encode_gdn_conv_commit(
                           context, encoder, count,
                           GDNConvCommitBuffers{history.view(), qkv_w}, encode_error);
            },
            error);
        if (!encoded) return false;
    }

    const bool guards =
        guards_hold(tag + " decode history", history, kHistoryElements * sizeof(float)) &&
        guards_hold(tag + " decode state", state, kStateElements * sizeof(float)) &&
        guards_hold(tag + " decode h", h, rows * row_channel_bytes) &&
        guards_hold(tag + " decode y", y, rows * row_value_bytes) &&
        guards_hold(tag + " decode qkv (read only)", qkv, 0u) &&
        guards_hold(tag + " decode gate (read only)", gate, 0u) &&
        guards_hold(tag + " decode beta (read only)", beta, 0u) &&
        guards_hold(tag + " decode z (read only)", z, 0u);
    if (!guards) {
        error = "decode touched a byte outside its declared output";
        return false;
    }
    const auto h_values = typed<float>(h);
    const auto y_values = typed<float>(y);
    out.h.assign(h_values.begin(), h_values.end());
    out.y.assign(y_values.begin(), y_values.end());
    const auto history_values = typed<float>(history);
    const auto state_values = typed<float>(state);
    out.history.assign(history_values.begin(), history_values.end());
    out.state.assign(state_values.begin(), state_values.end());
    return true;
}

// The Prefill chain in one ordered Encoder. With h_input it runs only the recurrence and output
// norm, on the h the host reference produced.
bool run_prefill(Context& context, const std::string& tag, const Inputs& in,
                 const std::vector<float>* h_input, Outputs& out,
                 std::string& error) {
    const std::uint32_t rows = in.rows;
    const std::size_t row_channel_bytes = kChannels * sizeof(float);
    const std::size_t row_value_bytes = kValueDim * sizeof(float);
    const std::size_t y_h_bytes = rows * kValueDim * sizeof(std::uint16_t);
    twin::GuardedBuffer history, state, qkv, weights, gate, beta, z, gamma, h, oc, y_h;
    bool allocated =
        upload(context, history, tag + " history", in.history, error) &&
        upload(context, state, tag + " state", in.state, error) &&
        upload(context, qkv, tag + " qkv", in.qkv, error) &&
        upload(context, weights, tag + " weights", in.weights, error) &&
        upload(context, gate, tag + " gate", in.gate, error) &&
        upload(context, beta, tag + " beta", in.beta, error) &&
        upload(context, z, tag + " z", in.z, error) &&
        upload(context, gamma, tag + " gamma", in.gamma, error) &&
        allocate_poisoned(context, oc, tag + " oc", rows * row_value_bytes, error) &&
        allocate_poisoned(context, y_h, tag + " y_h", y_h_bytes, error);
    if (allocated && h_input != nullptr) {
        allocated = upload(context, h, tag + " h input", *h_input, error);
    } else if (allocated) {
        allocated = allocate_poisoned(context, h, tag + " h", rows * row_channel_bytes, error);
    }
    if (!allocated) return false;

    const auto ro = [&](const twin::GuardedBuffer& buffer) {
        return context.buffers().read_only(buffer.view());
    };
    const bool encoded = run_encoded(
        context, tag + " prefill",
        [&](Encoder& encoder, std::string& encode_error) {
            if (h_input == nullptr) {
                if (!encode_prefill_gdn_conv_tiled(
                        context, encoder, rows,
                        PrefillGDNConvTiledBuffers{ro(history), ro(qkv), ro(weights),
                                                   h.view()},
                        encode_error)) {
                    return false;
                }
                if (!encode_prefill_gdn_conv_history(
                        context, encoder, rows,
                        PrefillGDNConvHistoryBuffers{history.view(), ro(qkv)},
                        encode_error)) {
                    return false;
                }
                if (!encode_prefill_gdn_l2(context, encoder, rows, h.view(),
                                           encode_error)) {
                    return false;
                }
            }
            if (!encode_prefill_gdn_rec(
                    context, encoder, rows,
                    PrefillGDNRecBuffers{state.view(), ro(h), ro(gate), ro(beta),
                                         oc.view()},
                    encode_error)) {
                return false;
            }
            return encode_prefill_gdn_out_norm(
                context, encoder, rows,
                PrefillGDNOutNormBuffers{ro(oc), ro(z), ro(gamma), y_h.view()},
                encode_error);
        },
        error);
    if (!encoded) return false;

    const bool conv_written = h_input == nullptr;
    const bool guards =
        guards_hold(tag + " history", history,
                    conv_written ? kHistoryElements * sizeof(float) : 0u) &&
        guards_hold(tag + " state", state, kStateElements * sizeof(float)) &&
        guards_hold(tag + " h", h, conv_written ? rows * row_channel_bytes : 0u) &&
        guards_hold(tag + " oc", oc, rows * row_value_bytes) &&
        guards_hold(tag + " y_h", y_h, y_h_bytes) &&
        guards_hold(tag + " qkv (read only)", qkv, 0u) &&
        guards_hold(tag + " gate (read only)", gate, 0u) &&
        guards_hold(tag + " beta (read only)", beta, 0u) &&
        guards_hold(tag + " z (read only)", z, 0u);
    if (!guards) {
        error = "prefill touched a byte outside its declared output";
        return false;
    }
    const auto h_values = typed<float>(h);
    const auto history_values = typed<float>(history);
    const auto state_values = typed<float>(state);
    const auto y_h_values = typed<std::uint16_t>(y_h);
    out.h.assign(h_values.begin(), h_values.end());
    out.history.assign(history_values.begin(), history_values.end());
    out.state.assign(state_values.begin(), state_values.end());
    out.y_h.assign(y_h_values.begin(), y_h_values.end());
    return true;
}

// Index of the largest-magnitude value, where a one-element offset is most clearly out of bound.
std::size_t largest_index(std::span<const twin::tf> values) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (std::fabs(values[i].value) > std::fabs(values[best].value)) best = i;
    }
    return best;
}

// Flips the lowest mantissa bit: the smallest change a bit comparison must catch.
void flip_low_bit(float& value) {
    value = std::bit_cast<float>(std::bit_cast<std::uint32_t>(value) ^ 1u);
}

bool run_rows(Context& context, std::uint32_t rows) {
    const std::string tag = "pf_gdn_eq_step_norm T=" + std::to_string(rows);
    const Inputs in = make_inputs(rows);
    const HostReference ref = run_host_reference(in);

    std::string error;
    Outputs decode, prefill, recurrence;
    if (!run_decode(context, tag, in, decode, error)) {
        check(false, tag + " decode GPU chain", error);
        return false;
    }
    if (!run_prefill(context, tag, in, nullptr, prefill, error)) {
        check(false, tag + " prefill GPU chain", error);
        return false;
    }
    if (!run_prefill(context, tag + " from host h", in, &ref.h_values, recurrence,
                     error)) {
        check(false, tag + " prefill recurrence GPU", error);
        return false;
    }

    // The decode output rounded to F16 (RNE), which is what the output norm must store.
    std::vector<std::uint16_t> decode_y_h(decode.y.size());
    for (std::size_t i = 0; i < decode.y.size(); ++i) {
        decode_y_h[i] = strata::kernels::f16_from_f32(decode.y[i]);
    }

    bool passed = true;
    // Stage checks. The recurrence stage is fed the host reference's own h, so its bound is the
    // only error term; the conv stage is checked on the prefill output directly.
    passed &= units_match(tag + " prefill conv+L2 h vs host", prefill.h, ref.h,
                          kConvUnits);
    passed &= bits_match(tag + " prefill history vs host commit",
                         bytes_of(std::span<const float>(prefill.history)),
                         bytes_of(std::span<const float>(ref.history)));
    passed &= units_match(tag + " recurrence state vs host", recurrence.state,
                          ref.state, kRecurrenceUnits);
    passed &= half_match(tag + " recurrence y_h vs host", recurrence.y_h, ref.y,
                         kRecurrenceUnits);
    // The whole chain against the host, decode and prefill.
    passed &= units_match(tag + " decode h vs host", decode.h, ref.h, kConvUnits);
    passed &= units_match(tag + " decode state vs host", decode.state, ref.state,
                          kChainUnits);
    passed &= units_match(tag + " decode y vs host", decode.y, ref.y, kChainUnits);
    passed &= bits_match(tag + " decode history vs host commit",
                         bytes_of(std::span<const float>(decode.history)),
                         bytes_of(std::span<const float>(ref.history)));
    passed &= units_match(tag + " prefill state vs host", prefill.state, ref.state,
                          kChainUnits);
    passed &= half_match(tag + " prefill y_h vs host", prefill.y_h, ref.y,
                         kChainUnits);
    // Bit-for-bit agreement with the decode path over the same rows.
    passed &= bits_match(tag + " prefill h vs decode",
                         bytes_of(std::span<const float>(prefill.h)),
                         bytes_of(std::span<const float>(decode.h)));
    passed &= bits_match(tag + " prefill history vs decode",
                         bytes_of(std::span<const float>(prefill.history)),
                         bytes_of(std::span<const float>(decode.history)));
    passed &= bits_match(tag + " prefill state vs decode",
                         bytes_of(std::span<const float>(prefill.state)),
                         bytes_of(std::span<const float>(decode.state)));
    passed &= bits_match(tag + " prefill y_h vs half(decode y)",
                         bytes_of(std::span<const std::uint16_t>(prefill.y_h)),
                         bytes_of(std::span<const std::uint16_t>(decode_y_h)));

    // Negative controls: each comparison must reject a change of the size it is meant to catch.
    std::vector<float> wrong_state = prefill.state;
    flip_low_bit(wrong_state[wrong_state.size() / 2u]);
    check(!bits_equal(bytes_of(std::span<const float>(wrong_state)),
                      bytes_of(std::span<const float>(decode.state))),
          tag + " negative control: one flipped state bit fails B");
    std::vector<std::uint16_t> wrong_y_h = prefill.y_h;
    const std::size_t largest = largest_index(ref.y);
    wrong_y_h[largest] = static_cast<std::uint16_t>(wrong_y_h[largest] ^ 0x0200u);
    double ratio = 0.0;
    std::size_t ratio_index = 0;
    check(!half_within(wrong_y_h, ref.y, kChainUnits, ratio, ratio_index),
          tag + " negative control: a flipped y_h mantissa bit fails the F16 bound");
    std::vector<float> wrong_y = decode.y;
    wrong_y[largest] += 1e-3f * std::fabs(wrong_y[largest]);
    check(!units_within(wrong_y, ref.y, kChainUnits),
          tag + " negative control: a 1e-3 relative y offset fails U(640)");
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: pf_gdn_eq_step_norm strata.metallib\n");
        return 2;
    }
    std::string error;
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;

    for (const std::uint32_t rows : kRowCounts) {
        run_rows(*context, rows);
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill GDN equivalence check(s) failed\n",
                     g_failures);
        return 1;
    }
    std::puts("all Prefill GDN equivalence checks passed");
    return 0;
}
