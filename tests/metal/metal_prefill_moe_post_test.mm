// GPU twins for Prefill MoE SwiGLU and weighted combine (docs/mac/04 §6.6).
#include "metallib_sha256.inc"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_moe_post.hpp"
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

constexpr std::size_t kSwiGLUWidth = 640u;
constexpr std::size_t kWidth = 2560u;
constexpr std::size_t kTopK = 10u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint8_t kGuardPoison = 0xffu;
constexpr float kCombineTolerance = 2.0e-6f;

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
    const std::size_t payload_offset = twin::GuardedBuffer::kGuardBytes;
    const std::size_t suffix_offset = payload_offset + payload.size();
    return allocation.size() >= suffix_offset &&
           std::all_of(allocation.begin(), allocation.begin() + payload_offset,
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       }) &&
           std::all_of(allocation.begin() + suffix_offset, allocation.end(),
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

std::uint16_t expected_half(float value) {
    if (std::fpclassify(value) == FP_SUBNORMAL) value = 0.0f;
    value = std::clamp(value, -65504.0f, 65504.0f);
    return strata::kernels::f16_from_f32(value);
}

std::uint16_t ordered_half(std::uint16_t bits) {
    return (bits & 0x8000u) != 0u
               ? static_cast<std::uint16_t>(~bits)
               : static_cast<std::uint16_t>(bits | 0x8000u);
}

std::uint16_t half_ulp_distance(std::uint16_t a, std::uint16_t b) {
    const std::uint16_t oa = ordered_half(a);
    const std::uint16_t ob = ordered_half(b);
    return oa > ob ? static_cast<std::uint16_t>(oa - ob)
                   : static_cast<std::uint16_t>(ob - oa);
}

float fixture_gate(std::size_t index) {
    const int bucket = static_cast<int>((index * 13u) % 41u) - 20;
    return static_cast<float>(bucket) * 0.31f;
}

float fixture_up(std::size_t index) {
    const int bucket = static_cast<int>((index * 17u) % 29u) - 14;
    return static_cast<float>(bucket) * 0.12f;
}

bool run_swiglu_case(Context& context, std::uint32_t rows,
                     bool caller_owned) {
    const std::string label =
        "Prefill MoE SwiGLU T=" + std::to_string(rows) +
        (caller_owned ? " caller-owned" : " standalone");
    const std::size_t values = static_cast<std::size_t>(rows) * kSwiGLUWidth;
    const std::size_t input_bytes = values * sizeof(float);
    const std::size_t output_bytes = values * sizeof(std::uint16_t);
    constexpr std::size_t kTailBytes = 64u;

    twin::GuardedBuffer gate_buffer, up_buffer, output_buffer;
    std::string error;
    const bool allocated =
        gate_buffer.allocate(context, label + " gate", input_bytes, error) &&
        up_buffer.allocate(context, label + " up", input_bytes, error) &&
        output_buffer.allocate(context, label + " output",
                               output_bytes + kTailBytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::fill(output_buffer.payload().begin(), output_buffer.payload().end(),
              kPoison);

    auto gate = typed<float>(gate_buffer);
    auto up = typed<float>(up_buffer);
    for (std::size_t i = 0u; i < values; ++i) {
        gate[i] = fixture_gate(i);
        up[i] = fixture_up(i);
    }
    if (values >= 2u) {
        gate[0] = 20.0f;
        up[0] = 1.0e6f;
        gate[1] = 20.0f;
        up[1] = -1.0e6f;
    }
    const std::vector<float> gate_before(gate.begin(), gate.end());
    const std::vector<float> up_before(up.begin(), up.end());

    const PrefillMoeSwiGLUBuffers buffers{
        context.buffers().read_only(gate_buffer.view()),
        context.buffers().read_only(up_buffer.view()),
        output_buffer.view(),
    };
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool encoded =
        caller_owned
            ? Encoder::encode(
                  context, *command, label,
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_moe_swiglu(
                          context, encoder, rows, buffers, encode_error);
                  },
                  error)
            : encode_prefill_moe_swiglu(context, *command, rows, buffers,
                                         error);
    check(encoded, label + " encode", error);
    if (!encoded) return false;
    const bool completed =
        command->commit(error) && command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto output = typed<std::uint16_t>(output_buffer);
    bool matches = true;
    std::size_t worst_index = 0u;
    std::uint16_t worst_distance = 0u;
    for (std::size_t i = 0u; i < values; ++i) {
        const double x = static_cast<double>(gate[i]);
        const double expected =
            (x / (1.0 + std::exp(-x))) * static_cast<double>(up[i]);
        const std::uint16_t expected_bits =
            expected_half(static_cast<float>(expected));
        const std::uint16_t distance =
            half_ulp_distance(output[i], expected_bits);
        if (distance > worst_distance) {
            worst_distance = distance;
            worst_index = i;
        }
        matches &= distance <= 1u;
    }
    check(matches, label + " matches independent scalar SwiGLU oracle",
          "worst half ULP distance " + std::to_string(worst_distance) +
              " at " + std::to_string(worst_index));
    if (values >= 2u) {
        check(output[0] == 0x7bffu && output[1] == 0xfbffu,
              label + " saturates both signs at the finite F16 limit");
    }
    check(std::equal(gate_before.begin(), gate_before.end(), gate.begin()) &&
              std::equal(up_before.begin(), up_before.end(), up.begin()),
          label + " preserves read-only inputs");
    check(tail_is_poison(output_buffer, output_bytes),
          label + " leaves active output tail untouched");
    check(guards_intact(gate_buffer) && guards_intact(up_buffer) &&
              guards_intact(output_buffer),
          label + " preserves allocation guards");
    return matches;
}

float fixture_part(std::size_t row, std::size_t rank, std::size_t column) {
    if (row == 0u && column == 0u) {
        if (rank == 0u) return 1.0e20f;
        if (rank == 1u) return -1.0e20f;
        if (rank == 2u) return 1.0f;
        return 0.0f;
    }
    const int bucket =
        static_cast<int>((row * 7u + rank * 13u + column * 17u) % 59u) - 29;
    return static_cast<float>(bucket) * 0.013f;
}

float fixture_weight(std::size_t row, std::size_t rank) {
    if (row == 0u && rank < 3u) return 1.0f;
    const std::size_t bucket = (rank * 11u + row * 3u) % 17u + 1u;
    return static_cast<float>(bucket) * 0.037f;
}

float fixture_shared(std::size_t row, std::size_t column) {
    if (row == 0u && column == 0u) return 0.0f;
    const int bucket =
        static_cast<int>((row * 7u + column * 19u) % 101u) - 50;
    return static_cast<float>(bucket) * 0.003f;
}

float fixture_shared_gate(std::size_t row) {
    return (static_cast<float>(row % 7u) - 3.0f) * 1.1f;
}

float expected_combine_value(std::size_t row, std::size_t column,
                             std::span<const float> parts,
                             std::span<const float> weights,
                             std::span<const float> shared,
                             std::span<const float> gate,
                             bool reverse = false) {
    float sum = 0.0f;
    for (std::size_t step = 0u; step < kTopK; ++step) {
        const std::size_t rank =
            reverse ? kTopK - 1u - step : step;
        const std::size_t route = row * kTopK + rank;
        sum = std::fma(weights[route],
                       parts[route * kWidth + column], sum);
    }
    const float sigmoid =
        1.0f / (1.0f + std::exp(-gate[row]));
    return std::fma(shared[row * kWidth + column], sigmoid, sum);
}

bool run_combine_case(Context& context, std::uint32_t rows,
                      bool caller_owned) {
    const std::string label =
        "Prefill MoE combine T=" + std::to_string(rows) +
        (caller_owned ? " caller-owned" : " standalone");
    const std::size_t row_values = static_cast<std::size_t>(rows) * kWidth;
    const std::size_t route_values = static_cast<std::size_t>(rows) * kTopK;
    const std::size_t parts_values = route_values * kWidth;
    const std::size_t parts_bytes = parts_values * sizeof(float);
    const std::size_t weights_bytes = route_values * sizeof(float);
    const std::size_t row_bytes = row_values * sizeof(float);
    const std::size_t gate_bytes = static_cast<std::size_t>(rows) * sizeof(float);
    constexpr std::size_t kTailBytes = 64u;

    twin::GuardedBuffer parts_buffer, weights_buffer, shared_buffer;
    twin::GuardedBuffer gate_buffer, output_buffer;
    std::string error;
    const bool allocated =
        parts_buffer.allocate(context, label + " routed parts", parts_bytes,
                              error) &&
        weights_buffer.allocate(context, label + " weights", weights_bytes,
                                error) &&
        shared_buffer.allocate(context, label + " shared", row_bytes, error) &&
        gate_buffer.allocate(context, label + " gate", gate_bytes, error) &&
        output_buffer.allocate(context, label + " output",
                               row_bytes + kTailBytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::fill(output_buffer.payload().begin(), output_buffer.payload().end(),
              kPoison);

    auto parts = typed<float>(parts_buffer);
    auto weights = typed<float>(weights_buffer);
    auto shared = typed<float>(shared_buffer);
    auto gate = typed<float>(gate_buffer);
    for (std::size_t row = 0u; row < rows; ++row) {
        gate[row] = fixture_shared_gate(row);
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            weights[row * kTopK + rank] = fixture_weight(row, rank);
            for (std::size_t column = 0u; column < kWidth; ++column) {
                parts[(row * kTopK + rank) * kWidth + column] =
                    fixture_part(row, rank, column);
            }
        }
        for (std::size_t column = 0u; column < kWidth; ++column) {
            shared[row * kWidth + column] = fixture_shared(row, column);
        }
    }
    const std::vector<float> parts_before(parts.begin(), parts.end());
    const std::vector<float> weights_before(weights.begin(), weights.end());
    const std::vector<float> shared_before(shared.begin(), shared.end());
    const std::vector<float> gate_before(gate.begin(), gate.end());

    const PrefillMoeCombineBuffers buffers{
        context.buffers().read_only(parts_buffer.view()),
        context.buffers().read_only(weights_buffer.view()),
        context.buffers().read_only(shared_buffer.view()),
        context.buffers().read_only(gate_buffer.view()),
        output_buffer.view(),
    };
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool encoded =
        caller_owned
            ? Encoder::encode(
                  context, *command, label,
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_moe_combine(
                          context, encoder, rows, buffers, encode_error);
                  },
                  error)
            : encode_prefill_moe_combine(context, *command, rows, buffers,
                                         error);
    check(encoded, label + " encode", error);
    if (!encoded) return false;
    const bool completed =
        command->commit(error) && command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto output = typed<float>(output_buffer);
    float max_error = 0.0f;
    bool matches = true;
    for (std::size_t row = 0u; row < rows; ++row) {
        for (std::size_t column = 0u; column < kWidth; ++column) {
            const std::size_t index = row * kWidth + column;
            const float expected = expected_combine_value(
                row, column, parts, weights, shared, gate);
            const float tolerance =
                kCombineTolerance * std::max(1.0f, std::abs(expected));
            const float difference = std::abs(output[index] - expected);
            max_error = std::max(max_error, difference);
            matches &= std::isfinite(output[index]) &&
                       difference <= tolerance;
        }
    }
    check(matches, label + " matches ordered FMA combine oracle",
          "max absolute error " + std::to_string(max_error));
    if (rows > 0u) {
        const float ordered = expected_combine_value(
            0u, 0u, parts, weights, shared, gate);
        const float reversed = expected_combine_value(
            0u, 0u, parts, weights, shared, gate, true);
        check(ordered == 1.0f && reversed == 0.0f &&
                  output[0] == ordered,
              label + " preserves rank order under cancellation");
    }
    check(std::equal(parts_before.begin(), parts_before.end(), parts.begin()) &&
              std::equal(weights_before.begin(), weights_before.end(),
                         weights.begin()) &&
              std::equal(shared_before.begin(), shared_before.end(),
                         shared.begin()) &&
              std::equal(gate_before.begin(), gate_before.end(), gate.begin()),
          label + " preserves read-only inputs");
    check(tail_is_poison(output_buffer, row_bytes),
          label + " leaves active output tail untouched");
    check(guards_intact(parts_buffer) && guards_intact(weights_buffer) &&
              guards_intact(shared_buffer) && guards_intact(gate_buffer) &&
              guards_intact(output_buffer),
          label + " preserves allocation guards");
    return matches;
}

bool run_disjoint_active_slice_cases(Context& context) {
    bool passed = true;
    constexpr std::uint32_t kSwiRows = 3u;
    const std::size_t swi_values =
        static_cast<std::size_t>(kSwiRows) * kSwiGLUWidth;
    const std::size_t swi_input_bytes = swi_values * sizeof(float);
    const std::size_t swi_output_bytes =
        swi_values * sizeof(std::uint16_t);
    constexpr std::size_t kTailBytes = 64u;

    twin::GuardedBuffer swi_arena, swi_up;
    std::string error;
    const bool swi_allocated =
        swi_arena.allocate(context, "Prefill MoE SwiGLU active-slice arena",
                           swi_input_bytes + swi_output_bytes + kTailBytes,
                           error) &&
        swi_up.allocate(context, "Prefill MoE SwiGLU active-slice up",
                        swi_input_bytes, error);
    check(swi_allocated, "SwiGLU disjoint active-slice allocate", error);
    if (!swi_allocated) return false;

    auto arena_bytes = swi_arena.payload();
    auto gate = std::span<float>(
        reinterpret_cast<float*>(arena_bytes.data()),
        swi_input_bytes / sizeof(float));
    auto up = typed<float>(swi_up);
    for (std::size_t i = 0u; i < swi_values; ++i) {
        gate[i] = fixture_gate(i);
        up[i] = fixture_up(i);
    }
    std::fill(arena_bytes.begin() + swi_input_bytes, arena_bytes.end(),
              kPoison);
    const auto gate_before = std::vector<float>(gate.begin(), gate.end());
    const auto up_before = std::vector<float>(up.begin(), up.end());
    const BufferRef arena = swi_arena.view();
    const BufferRef wide_gate = arena.slice(
        0u, swi_input_bytes + swi_output_bytes + kTailBytes);
    const BufferRef output_view =
        arena.slice(swi_input_bytes, swi_output_bytes + kTailBytes);
    const PrefillMoeSwiGLUBuffers swi_buffers{
        context.buffers().read_only(wide_gate),
        context.buffers().read_only(swi_up.view()), output_view};
    auto swi_command =
        context.make_command_buffer("Prefill MoE SwiGLU active slices", error);
    check(swi_command != nullptr, "SwiGLU active-slice command", error);
    if (!swi_command) return false;
    const bool swi_encoded = encode_prefill_moe_swiglu(
        context, *swi_command, kSwiRows, swi_buffers, error);
    check(swi_encoded,
          "SwiGLU accepts overlapping views with disjoint active ranges",
          error);
    if (swi_encoded) {
        const bool completed =
            swi_command->commit(error) &&
            swi_command->wait_until_completed(error);
        check(completed, "SwiGLU active-slice GPU completion", error);
        if (completed) {
            const auto* output = reinterpret_cast<const std::uint16_t*>(
                static_cast<const std::uint8_t*>(output_view.contents));
            bool matches = true;
            for (std::size_t i = 0u; i < swi_values; ++i) {
                const double x = static_cast<double>(gate_before[i]);
                const float expected = static_cast<float>(
                    (x / (1.0 + std::exp(-x))) *
                    static_cast<double>(up_before[i]));
                matches &= half_ulp_distance(output[i], expected_half(expected))
                           <= 1u;
            }
            check(matches, "SwiGLU active-slice output matches scalar oracle");
            check(std::equal(gate_before.begin(), gate_before.end(),
                             gate.begin()) &&
                      std::equal(up_before.begin(), up_before.end(),
                                 up.begin()),
                  "SwiGLU active-slice inputs remain unchanged");
            check(tail_is_poison(swi_arena, swi_input_bytes + swi_output_bytes),
                  "SwiGLU active-slice output tail remains untouched");
        }
    }
    passed &= swi_encoded;

    constexpr std::uint32_t kCombineRows = 1u;
    constexpr std::size_t kPartValues = kCombineRows * kTopK * kWidth;
    constexpr std::size_t kWeightValues = kCombineRows * kTopK;
    constexpr std::size_t kSharedValues = kCombineRows * kWidth;
    constexpr std::size_t kGateValues = kCombineRows;
    const std::size_t part_bytes = kPartValues * sizeof(float);
    const std::size_t weight_bytes = kWeightValues * sizeof(float);
    const std::size_t shared_bytes = kSharedValues * sizeof(float);
    const std::size_t gate_bytes = kGateValues * sizeof(float);
    const std::size_t output_bytes = kSharedValues * sizeof(float);
    const std::size_t input_bytes =
        part_bytes + weight_bytes + shared_bytes + gate_bytes;
    twin::GuardedBuffer combine_arena;
    const bool combine_allocated = combine_arena.allocate(
        context, "Prefill MoE combine active-slice arena",
        input_bytes + output_bytes + kTailBytes, error);
    check(combine_allocated, "combine disjoint active-slice allocate", error);
    if (!combine_allocated) return false;

    auto all_values = typed<float>(combine_arena);
    auto parts = all_values.subspan(0u, kPartValues);
    auto weights = all_values.subspan(kPartValues, kWeightValues);
    auto shared =
        all_values.subspan(kPartValues + kWeightValues, kSharedValues);
    auto combine_gate =
        all_values.subspan(kPartValues + kWeightValues + kSharedValues,
                           kGateValues);
    for (std::size_t rank = 0u; rank < kTopK; ++rank) {
        weights[rank] = fixture_weight(0u, rank);
        for (std::size_t column = 0u; column < kWidth; ++column) {
            parts[rank * kWidth + column] =
                fixture_part(0u, rank, column);
        }
    }
    combine_gate[0] = fixture_shared_gate(0u);
    for (std::size_t column = 0u; column < kWidth; ++column) {
        shared[column] = fixture_shared(0u, column);
    }
    std::fill(
        combine_arena.payload().begin() + input_bytes,
        combine_arena.payload().end(), kPoison);
    const std::vector<std::uint8_t> inputs_before(
        combine_arena.payload().begin(),
        combine_arena.payload().begin() + input_bytes);

    const BufferRef combine_root = combine_arena.view();
    std::size_t offset = 0u;
    const BufferRef wide_parts =
        combine_root.slice(offset, combine_root.length - offset);
    offset += part_bytes;
    const BufferRef wide_weights =
        combine_root.slice(offset, combine_root.length - offset);
    offset += weight_bytes;
    const BufferRef wide_shared =
        combine_root.slice(offset, combine_root.length - offset);
    offset += shared_bytes;
    const BufferRef wide_combine_gate =
        combine_root.slice(offset, combine_root.length - offset);
    offset += gate_bytes;
    const BufferRef combine_output =
        combine_root.slice(offset, combine_root.length - offset);
    const PrefillMoeCombineBuffers combine_buffers{
        context.buffers().read_only(wide_parts),
        context.buffers().read_only(wide_weights),
        context.buffers().read_only(wide_shared),
        context.buffers().read_only(wide_combine_gate), combine_output};
    auto combine_command =
        context.make_command_buffer("Prefill MoE combine active slices",
                                    error);
    check(combine_command != nullptr, "combine active-slice command", error);
    if (!combine_command) return false;
    const bool combine_encoded = encode_prefill_moe_combine(
        context, *combine_command, kCombineRows, combine_buffers, error);
    check(combine_encoded,
          "combine accepts overlapping views with disjoint active ranges",
          error);
    if (combine_encoded) {
        const bool completed =
            combine_command->commit(error) &&
            combine_command->wait_until_completed(error);
        check(completed, "combine active-slice GPU completion", error);
        if (completed) {
            const auto* output =
                static_cast<const float*>(combine_output.contents);
            bool matches = true;
            for (std::size_t column = 0u; column < kWidth; ++column) {
                const float expected = expected_combine_value(
                    0u, column, parts, weights, shared, combine_gate);
                const float tolerance =
                    kCombineTolerance * std::max(1.0f, std::abs(expected));
                matches &= std::abs(output[column] - expected) <= tolerance;
            }
            check(matches, "combine active-slice output matches ordered oracle");
            check(std::equal(inputs_before.begin(), inputs_before.end(),
                             combine_arena.payload().begin()),
                  "combine active-slice inputs remain unchanged");
            check(tail_is_poison(combine_arena, input_bytes + output_bytes),
                  "combine active-slice output tail remains untouched");
        }
    }
    passed &= combine_encoded;
    check(guards_intact(swi_arena) && guards_intact(swi_up) &&
              guards_intact(combine_arena),
          "MoE post-op active-slice cases preserve guards");
    return passed;
}

bool test_validation(Context& context, Context& foreign_context) {
    constexpr std::uint32_t kRows = 2u;
    constexpr std::size_t kSwiValues = kRows * kSwiGLUWidth;
    constexpr std::size_t kRowValues = kRows * kWidth;
    constexpr std::size_t kRouteValues = kRows * kTopK;
    std::string error;
    twin::GuardedBuffer gate, up, output;
    twin::GuardedBuffer parts, weights, shared, route_gate, combined;
    twin::GuardedBuffer foreign_gate;
    const bool allocated =
        gate.allocate(context, "post validation SwiGLU gate",
                      kSwiValues * sizeof(float), error) &&
        up.allocate(context, "post validation SwiGLU up",
                    kSwiValues * sizeof(float), error) &&
        output.allocate(context, "post validation SwiGLU output",
                        kSwiValues * sizeof(std::uint16_t), error) &&
        parts.allocate(context, "post validation parts",
                       kRouteValues * kWidth * sizeof(float), error) &&
        weights.allocate(context, "post validation weights",
                         kRouteValues * sizeof(float), error) &&
        shared.allocate(context, "post validation shared",
                        kRowValues * sizeof(float), error) &&
        route_gate.allocate(context, "post validation gate",
                            kRows * sizeof(float), error) &&
        combined.allocate(context, "post validation output",
                          kRowValues * sizeof(float), error) &&
        foreign_gate.allocate(foreign_context, "post foreign gate",
                              kSwiValues * sizeof(float), error);
    check(allocated, "post-op validation allocate buffers", error);
    if (!allocated) return false;

    const PrefillMoeSwiGLUBuffers swiglu{
        context.buffers().read_only(gate.view()),
        context.buffers().read_only(up.view()), output.view()};
    auto swiglu_command =
        context.make_command_buffer("Prefill MoE SwiGLU validation", error);
    check(swiglu_command != nullptr, "SwiGLU validation command", error);
    if (!swiglu_command) return false;
    check(!encode_prefill_moe_swiglu(context, *swiglu_command, 0u, swiglu,
                                     error) &&
              error.find("row count") != std::string::npos,
          "SwiGLU rejects zero rows before dispatch", error);

    const PrefillMoeCombineBuffers combine{
        context.buffers().read_only(parts.view()),
        context.buffers().read_only(weights.view()),
        context.buffers().read_only(shared.view()),
        context.buffers().read_only(route_gate.view()), combined.view()};
    auto combine_command =
        context.make_command_buffer("Prefill MoE combine validation", error);
    check(combine_command != nullptr, "combine validation command", error);
    if (!combine_command) return false;
    check(!encode_prefill_moe_combine(context, *combine_command, 0u, combine,
                                      error) &&
              error.find("row count") != std::string::npos,
          "combine rejects zero rows before dispatch", error);

    auto foreign_command =
        context.make_command_buffer("Prefill MoE foreign view", error);
    check(foreign_command != nullptr, "foreign-view command", error);
    if (!foreign_command) return false;
    const PrefillMoeSwiGLUBuffers foreign_buffers{
        context.buffers().read_only(foreign_gate.view()),
        context.buffers().read_only(up.view()), output.view()};
    check(!encode_prefill_moe_swiglu(context, *foreign_command, kRows,
                                     foreign_buffers, error) &&
              error.find("not owned by this Context") != std::string::npos,
          "SwiGLU rejects a foreign Context input", error);

    auto alias_command =
        context.make_command_buffer("Prefill MoE output alias", error);
    check(alias_command != nullptr, "output-alias command", error);
    if (!alias_command) return false;
    const PrefillMoeSwiGLUBuffers alias_buffers{
        context.buffers().read_only(gate.view()),
        context.buffers().read_only(up.view()), gate.view()};
    check(!encode_prefill_moe_swiglu(context, *alias_command, kRows,
                                     alias_buffers, error) &&
              error.find("overlaps active gate") != std::string::npos,
          "SwiGLU rejects output/input overlap", error);

    auto open_group_command =
        context.make_command_buffer("Prefill MoE open group", error);
    check(open_group_command != nullptr, "open-group command", error);
    if (!open_group_command) return false;
    const bool swiglu_encoded = Encoder::encode(
        context, *open_group_command, "Prefill MoE open SwiGLU group",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_moe_swiglu(context, encoder, kRows, swiglu,
                                              encode_error);
        },
        error);
    const bool concurrent =
        std::getenv("STRATA_METAL_DISPATCH") != nullptr &&
        std::string_view(std::getenv("STRATA_METAL_DISPATCH")) ==
            "concurrent";
    check(concurrent
              ? !swiglu_encoded &&
                    error.find("ordered concurrent-group boundary") !=
                        std::string::npos
              : swiglu_encoded,
          concurrent ? "SwiGLU rejects an open concurrent group"
                     : "SwiGLU permits grouping in serial mode",
          error);

    auto combine_group_command =
        context.make_command_buffer("Prefill MoE open combine group", error);
    check(combine_group_command != nullptr, "open combine-group command",
          error);
    if (!combine_group_command) return false;
    const bool combine_encoded = Encoder::encode(
        context, *combine_group_command, "Prefill MoE open combine group",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_moe_combine(context, encoder, kRows, combine,
                                               encode_error);
        },
        error);
    check(concurrent
              ? !combine_encoded &&
                    error.find("ordered concurrent-group boundary") !=
                        std::string::npos
              : combine_encoded,
          concurrent ? "combine rejects an open concurrent group"
                     : "combine permits grouping in serial mode",
          error);

    return g_failures == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_moe_post_test strata.metallib\n");
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
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal Context", error);
    if (!foreign_context) return 1;

    bool passed = test_validation(*context, *foreign_context);
    passed &= run_disjoint_active_slice_cases(*context);
    const std::array<std::uint32_t, 5u> row_counts{1u, 2u, 3u, 7u, 65u};
    for (const std::uint32_t rows : row_counts) {
        passed &= run_swiglu_case(*context, rows, rows == 3u);
        passed &= run_combine_case(*context, rows, rows == 2u);
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill MoE post-op test(s) failed\n",
                     g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all Prefill MoE post-op tests passed");
    return 0;
}
