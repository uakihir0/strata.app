// Synthetic composition twin for Prefill MoE shared-expert steps 7–10.
#include "metallib_sha256.inc"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_moe_shared_pipeline.hpp"
#include "../../src/metal/kernels/common/types.h"
#include "twin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kHidden = 640u;
constexpr std::size_t kQ2BlocksPerWidth = kWidth / 64u;
constexpr std::size_t kQ2BlocksPerHidden = kHidden / 64u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint8_t kGuardPoison = 0xffu;

static_assert(sizeof(block_q2_0) == 18u);

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

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
    const std::size_t suffix = twin::GuardedBuffer::kGuardBytes +
                               payload.size();
    return allocation.size() >= suffix &&
           std::all_of(allocation.begin(),
                       allocation.begin() +
                           twin::GuardedBuffer::kGuardBytes,
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       }) &&
           std::all_of(allocation.begin() + suffix, allocation.end(),
                       [](std::uint8_t byte) {
                           return byte == kGuardPoison;
                       });
}

bool close_enough(float actual, double expected, double absolute_floor,
                  double relative_tolerance) {
    return std::isfinite(actual) &&
           std::abs(static_cast<double>(actual) - expected) <=
               absolute_floor +
                   relative_tolerance * std::abs(expected);
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

std::uint16_t half_ulp_distance(std::uint16_t left,
                                std::uint16_t right) {
    const std::uint16_t a = ordered_half(left);
    const std::uint16_t b = ordered_half(right);
    return a > b ? static_cast<std::uint16_t>(a - b)
                 : static_cast<std::uint16_t>(b - a);
}

struct CaseBuffers {
    twin::GuardedBuffer mixed_h;
    twin::GuardedBuffer gate_weights;
    twin::GuardedBuffer gate_dequant;
    twin::GuardedBuffer gate_output;
    twin::GuardedBuffer up_weights;
    twin::GuardedBuffer up_dequant;
    twin::GuardedBuffer up_output;
    twin::GuardedBuffer hidden;
    twin::GuardedBuffer down_weights;
    twin::GuardedBuffer down_dequant;
    twin::GuardedBuffer shared_output;
    twin::GuardedBuffer gate_hi;
    twin::GuardedBuffer gate_lo;
    twin::GuardedBuffer gate_router_weights;
    twin::GuardedBuffer gate_logits;

    PrefillMoeSharedFFNBuffers views(Context& context) {
        return {
            context.buffers().read_only(mixed_h.view()),
            context.buffers().read_only(gate_weights.view()),
            gate_dequant.view(),
            gate_output.view(),
            context.buffers().read_only(up_weights.view()),
            up_dequant.view(),
            up_output.view(),
            hidden.view(),
            context.buffers().read_only(down_weights.view()),
            down_dequant.view(),
            shared_output.view(),
            {context.buffers().read_only(gate_hi.view()),
             context.buffers().read_only(gate_lo.view()),
             context.buffers().read_only(gate_router_weights.view()),
             gate_logits.view()},
        };
    }
};

bool allocate_case(Context& context, CaseBuffers& b, std::uint32_t rows,
                   const std::string& label, std::string& error) {
    const std::size_t gate_up_weight_bytes =
        static_cast<std::size_t>(kHidden) * kQ2BlocksPerWidth *
        sizeof(block_q2_0);
    const std::size_t down_weight_bytes =
        static_cast<std::size_t>(kWidth) * kQ2BlocksPerHidden *
        sizeof(block_q2_0);
    const std::size_t gate_up_scratch_bytes =
        static_cast<std::size_t>(kWidth) * kHidden * sizeof(std::uint16_t);
    const std::size_t down_scratch_bytes =
        static_cast<std::size_t>(kHidden) * kWidth * sizeof(std::uint16_t);
    return b.mixed_h.allocate(
               context, label + " mixed H",
               static_cast<std::size_t>(rows) * kWidth * sizeof(std::uint16_t),
               error) &&
           b.gate_weights.allocate(context, label + " gate weights",
                                   gate_up_weight_bytes, error) &&
           b.gate_dequant.allocate(context, label + " gate dequant",
                                   gate_up_scratch_bytes, error) &&
           b.gate_output.allocate(
               context, label + " gate output",
               static_cast<std::size_t>(rows) * kHidden * sizeof(float),
               error) &&
           b.up_weights.allocate(context, label + " up weights",
                                 gate_up_weight_bytes, error) &&
           b.up_dequant.allocate(context, label + " up dequant",
                                 gate_up_scratch_bytes, error) &&
           b.up_output.allocate(
               context, label + " up output",
               static_cast<std::size_t>(rows) * kHidden * sizeof(float),
               error) &&
           b.hidden.allocate(
               context, label + " shared hidden",
               static_cast<std::size_t>(rows) * kHidden *
                   sizeof(std::uint16_t),
               error) &&
           b.down_weights.allocate(context, label + " down weights",
                                   down_weight_bytes, error) &&
           b.down_dequant.allocate(context, label + " down dequant",
                                   down_scratch_bytes, error) &&
           b.shared_output.allocate(
               context, label + " shared down output",
               static_cast<std::size_t>(rows) * kWidth * sizeof(float),
               error) &&
           b.gate_hi.allocate(
               context, label + " gate input high",
               static_cast<std::size_t>(rows) * kWidth *
                   sizeof(std::uint16_t),
               error) &&
           b.gate_lo.allocate(
               context, label + " gate input low",
               static_cast<std::size_t>(rows) * kWidth *
                   sizeof(std::uint16_t),
               error) &&
           b.gate_router_weights.allocate(
               context, label + " gate router weights",
               static_cast<std::size_t>(kWidth) * sizeof(std::uint16_t),
               error) &&
           b.gate_logits.allocate(
               context, label + " shared gate logits",
               static_cast<std::size_t>(rows) * sizeof(float), error);
}

void fill_q2_weights(twin::GuardedBuffer& buffer, std::uint8_t code) {
    const std::uint8_t packed =
        static_cast<std::uint8_t>(code * 0x55u);
    for (block_q2_0& block : typed<block_q2_0>(buffer)) {
        block.d = 0x2800u;  // F16 0.03125.
        std::fill(std::begin(block.qs), std::end(block.qs), packed);
    }
}

void initialize_case(CaseBuffers& b, std::uint32_t rows) {
    fill_q2_weights(b.gate_weights, 2u);  // code 2 -> +1 × d.
    fill_q2_weights(b.up_weights, 3u);    // code 3 -> +2 × d.
    fill_q2_weights(b.down_weights, 0u);  // code 0 -> -1 × d.
    auto mixed = typed<std::uint16_t>(b.mixed_h);
    auto gate_hi = typed<std::uint16_t>(b.gate_hi);
    auto gate_lo = typed<std::uint16_t>(b.gate_lo);
    auto gate_weights = typed<std::uint16_t>(b.gate_router_weights);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const float row_offset = static_cast<float>(row) * 0.0625f;
        for (std::uint32_t k = 0u; k < kWidth; ++k) {
            const float value =
                0.125f + row_offset +
                static_cast<float>(k % 13u) * 0.015625f;
            mixed[static_cast<std::size_t>(row) * kWidth + k] =
                strata::kernels::f16_from_f32(value);
            gate_hi[static_cast<std::size_t>(row) * kWidth + k] =
                strata::kernels::bf16_from_f32(0.25f + row_offset);
            gate_lo[static_cast<std::size_t>(row) * kWidth + k] =
                strata::kernels::bf16_from_f32(0.015625f);
        }
    }
    std::fill(gate_weights.begin(), gate_weights.end(),
              strata::kernels::bf16_from_f32(0.125f));
    for (twin::GuardedBuffer* output :
         {&b.gate_dequant, &b.gate_output, &b.up_dequant, &b.up_output,
          &b.hidden, &b.down_dequant, &b.shared_output, &b.gate_logits}) {
        std::fill(output->payload().begin(), output->payload().end(),
                  kPoison);
    }
}

float q2_value(const block_q2_0& block, std::size_t index) {
    const std::uint8_t packed = block.qs[index / 4u];
    const int code =
        static_cast<int>((packed >> (2u * (index % 4u))) & 3u) - 1;
    const float scaled =
        strata::kernels::f32_from_f16(block.d) * static_cast<float>(code);
    return strata::kernels::f32_from_f16(
        strata::kernels::f16_from_f32(scaled));
}

double q2_dot(const twin::GuardedBuffer& weights, std::size_t output_row,
              std::size_t k_width, std::span<const std::uint16_t> input) {
    const auto blocks = typed<block_q2_0>(weights);
    const std::size_t blocks_per_row = k_width / 64u;
    double result = 0.0;
    for (std::size_t k = 0u; k < k_width; ++k) {
        const block_q2_0& block =
            blocks[output_row * blocks_per_row + k / 64u];
        result +=
            static_cast<double>(strata::kernels::f32_from_f16(input[k])) *
            static_cast<double>(q2_value(block, k % 64u));
    }
    return result;
}

std::uint16_t expected_hidden(double gate, double up) {
    const float gate_f32 = static_cast<float>(gate);
    const float up_f32 = static_cast<float>(up);
    const float silu = gate_f32 / (1.0f + std::exp(-gate_f32));
    return expected_half(silu * up_f32);
}

bool run_case(Context& context, std::uint32_t rows,
              bool caller_owned) {
    const std::string label =
        "Prefill MoE shared FFN T=" + std::to_string(rows) +
        (caller_owned ? " caller-owned" : " standalone");
    CaseBuffers b;
    std::string error;
    const bool allocated = allocate_case(context, b, rows, label, error);
    check(allocated, label + " allocate", error);
    if (!allocated) return false;
    initialize_case(b, rows);
    const PrefillMoeSharedFFNFormats formats{
        PrefillNativeWeightFormat::Q2_0,
        PrefillNativeWeightFormat::Q2_0,
        PrefillNativeWeightFormat::Q2_0};
    constexpr auto projection_mode = PrefillBF16X2Mode::all_projections;
    const auto buffers = b.views(context);

    const std::array<const twin::GuardedBuffer*, 7u> inputs{{
        &b.mixed_h, &b.gate_weights, &b.up_weights, &b.down_weights,
        &b.gate_hi, &b.gate_lo, &b.gate_router_weights}};
    std::array<std::vector<std::uint8_t>, inputs.size()> input_bytes;
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const auto payload = inputs[index]->payload();
        input_bytes[index].assign(payload.begin(), payload.end());
    }

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    bool encoded = false;
    if (caller_owned) {
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_moe_shared_ffn(
                    context, encoder, rows, projection_mode, formats, buffers,
                    encode_error);
            },
            error);
    } else {
        encoded = encode_prefill_moe_shared_ffn(
            context, *command, rows, projection_mode, formats, buffers,
            error);
    }
    check(encoded, label + " encode shared stages 7-10", error);
    const bool completed =
        encoded && command->commit(error) &&
        command->wait_until_completed(error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const auto mixed = typed<std::uint16_t>(b.mixed_h);
    const auto gate_actual = typed<float>(b.gate_output);
    const auto up_actual = typed<float>(b.up_output);
    const auto hidden_actual = typed<std::uint16_t>(b.hidden);
    const auto shared_actual = typed<float>(b.shared_output);
    const auto gate_logits = typed<float>(b.gate_logits);
    bool gate_matches = true;
    bool up_matches = true;
    bool hidden_matches = true;
    bool down_matches = true;
    bool router_gate_matches = true;
    std::array<double, 8u> expected_gate{};
    std::array<double, 8u> expected_up{};
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const auto row_input =
            mixed.subspan(static_cast<std::size_t>(row) * kWidth, kWidth);
        expected_gate[row] = q2_dot(b.gate_weights, 0u, kWidth, row_input);
        expected_up[row] = q2_dot(b.up_weights, 0u, kWidth, row_input);
        for (std::uint32_t column = 0u; column < kHidden; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kHidden + column;
            gate_matches &=
                close_enough(gate_actual[index], expected_gate[row], 1.0e-3,
                             1.0e-5);
            up_matches &=
                close_enough(up_actual[index], expected_up[row], 1.0e-3,
                             1.0e-5);
            const std::uint16_t expected =
                expected_hidden(expected_gate[row], expected_up[row]);
            hidden_matches &=
                half_ulp_distance(hidden_actual[index], expected) <= 2u;
        }
        double expected_down = 0.0;
        for (std::uint32_t k = 0u; k < kHidden; ++k) {
            const std::uint16_t hidden =
                hidden_actual[static_cast<std::size_t>(row) * kHidden + k];
            expected_down +=
                static_cast<double>(
                    strata::kernels::f32_from_f16(hidden)) *
                static_cast<double>(q2_value(
                    typed<block_q2_0>(b.down_weights)
                        [static_cast<std::size_t>(0u) *
                             kQ2BlocksPerHidden +
                         k / 64u],
                    k % 64u));
        }
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kWidth + column;
            down_matches &=
                close_enough(shared_actual[index], expected_down, 1.0e-3,
                             1.0e-5);
        }
        const double hi = strata::kernels::f32_from_bf16(
            typed<std::uint16_t>(b.gate_hi)[
                static_cast<std::size_t>(row) * kWidth]);
        const double lo = strata::kernels::f32_from_bf16(
            typed<std::uint16_t>(b.gate_lo)[
                static_cast<std::size_t>(row) * kWidth]);
        const double router_weight = strata::kernels::f32_from_bf16(
            typed<std::uint16_t>(b.gate_router_weights)[0u]);
        const double expected_logit =
            static_cast<double>(kWidth) * (hi + lo) * router_weight;
        router_gate_matches &=
            close_enough(gate_logits[row], expected_logit, 1.0e-3, 1.0e-5);
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            const float hi_value = strata::kernels::f32_from_bf16(
                typed<std::uint16_t>(b.gate_hi)[
                    static_cast<std::size_t>(row) * kWidth + column]);
            const float lo_value = strata::kernels::f32_from_bf16(
                typed<std::uint16_t>(b.gate_lo)[
                    static_cast<std::size_t>(row) * kWidth + column]);
            const float weight = strata::kernels::f32_from_bf16(
                typed<std::uint16_t>(b.gate_router_weights)[column]);
            const double expected_piece =
                static_cast<double>(hi_value + lo_value) * weight;
            router_gate_matches &=
                close_enough(gate_logits[row], expected_piece *
                                                  static_cast<double>(kWidth),
                             1.0e-3, 1.0e-5);
        }
    }
    check(gate_matches, label + " native gate projection matches Q2_0 oracle");
    check(up_matches, label + " native up projection matches Q2_0 oracle");
    check(hidden_matches, label + " F16 SwiGLU matches scalar oracle");
    check(down_matches, label + " native down projection matches Q2_0 oracle");
    check(router_gate_matches,
          label + " BF16X2 shared gate matches scalar oracle");

    const bool dequant_matches =
        std::all_of(typed<std::uint16_t>(b.gate_dequant).begin(),
                    typed<std::uint16_t>(b.gate_dequant).end(),
                    [](std::uint16_t value) { return value == 0x2800u; }) &&
        std::all_of(typed<std::uint16_t>(b.up_dequant).begin(),
                    typed<std::uint16_t>(b.up_dequant).end(),
                    [](std::uint16_t value) { return value == 0x2c00u; }) &&
        std::all_of(typed<std::uint16_t>(b.down_dequant).begin(),
                    typed<std::uint16_t>(b.down_dequant).end(),
                    [](std::uint16_t value) { return value == 0xa800u; });
    check(dequant_matches, label + " K-major F16 scratch values");

    bool inputs_unchanged = true;
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const auto payload = inputs[index]->payload();
        inputs_unchanged &=
            std::equal(input_bytes[index].begin(), input_bytes[index].end(),
                       payload.begin());
    }
    check(inputs_unchanged, label + " inputs remain unchanged");

    const std::array<const twin::GuardedBuffer*, 15u> all{{
        &b.mixed_h,          &b.gate_weights,       &b.gate_dequant,
        &b.gate_output,      &b.up_weights,         &b.up_dequant,
        &b.up_output,        &b.hidden,             &b.down_weights,
        &b.down_dequant,     &b.shared_output,      &b.gate_hi,
        &b.gate_lo,          &b.gate_router_weights, &b.gate_logits,
    }};
    const bool guards_match = std::all_of(
        all.begin(), all.end(),
        [](const twin::GuardedBuffer* buffer) {
            return guards_intact(*buffer);
        });
    check(guards_match, label + " allocation guards remain intact");

    const std::vector<std::uint8_t> output_before(
        b.shared_output.payload().begin(), b.shared_output.payload().end());
    auto bad_buffers = buffers;
    bad_buffers.gate_f32 = buffers.shared_f32;
    auto rejected_command =
        context.make_command_buffer(label + " alias rejection", error);
    if (!rejected_command) {
        check(false, label + " alias command buffer", error);
        return false;
    }
    const bool alias_rejected = !encode_prefill_moe_shared_ffn(
        context, *rejected_command, rows, projection_mode, formats,
        bad_buffers, error);
    check(alias_rejected, label + " rejects cross-stage writable overlap");
    const bool alias_rollback =
        std::equal(output_before.begin(), output_before.end(),
                   b.shared_output.payload().begin());
    check(alias_rollback,
          label + " alias preflight preserves existing output bytes");

    auto bad_mode_command =
        context.make_command_buffer(label + " mode 0 rejection", error);
    if (!bad_mode_command) {
        check(false, label + " mode rejection command buffer", error);
        return false;
    }
    const bool mode_rejected = !encode_prefill_moe_shared_ffn(
        context, *bad_mode_command, rows, PrefillBF16X2Mode::off, formats,
        buffers, error);
    check(mode_rejected,
          label + " rejects mode without BF16X2 shared gate remainder");
    const bool mode_rollback =
        std::equal(output_before.begin(), output_before.end(),
                   b.shared_output.payload().begin());
    check(mode_rollback,
          label + " mode rejection preserves existing output bytes");

    const std::vector<std::uint8_t> gate_scratch_before(
        b.gate_dequant.payload().begin(), b.gate_dequant.payload().end());
    const std::vector<std::uint8_t> shared_output_before(
        b.shared_output.payload().begin(), b.shared_output.payload().end());
    auto late_failure_command =
        context.make_command_buffer(label + " late GEMM failure", error);
    if (!late_failure_command) {
        check(false, label + " late failure command buffer", error);
        return false;
    }
    bool late_encoded = false;
    {
        ScopedEnvironmentOverride invalid_gemm("STRATA_METAL_GEMM",
                                               "invalid");
        if (!invalid_gemm.ready()) {
            check(false, label + " invalid GEMM selector setup");
            return false;
        }
        if (caller_owned) {
            late_encoded = Encoder::encode(
                context, *late_failure_command, label + " late failure",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_prefill_moe_shared_ffn(
                        context, encoder, rows, projection_mode, formats,
                        buffers, encode_error);
                },
                error);
        } else {
            late_encoded = encode_prefill_moe_shared_ffn(
                context, *late_failure_command, rows, projection_mode,
                formats, buffers, error);
        }
    }
    const bool late_rejected =
        !late_encoded &&
        error == "STRATA_METAL_GEMM must be 'mpp' or 'simd'";
    check(late_rejected,
          label + " late GEMM validation fails after native dequant");
    const bool late_discarded =
        late_failure_command->unsafe_native_handle() == nullptr;
    check(late_discarded,
          label + " late failure discards every earlier projection dispatch");
    const bool late_rollback =
        late_discarded &&
        std::equal(gate_scratch_before.begin(), gate_scratch_before.end(),
                   b.gate_dequant.payload().begin()) &&
        std::equal(shared_output_before.begin(), shared_output_before.end(),
                   b.shared_output.payload().begin());
    check(late_rollback,
          label + " late rollback preserves scratch and output bytes");

    return gate_matches && up_matches && hidden_matches && down_matches &&
           router_gate_matches && dequant_matches && inputs_unchanged &&
           guards_match && alias_rejected && alias_rollback && mode_rejected &&
           mode_rollback && late_rejected && late_discarded && late_rollback;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(
            stderr,
            "usage: metal_prefill_moe_shared_pipeline_test strata.metallib\n");
        return 2;
    }
    std::string error;
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;

    bool passed = true;
    passed &= run_case(*context, 1u, false);
    passed &= run_case(*context, 7u, true);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill MoE shared pipeline test(s) failed\n",
                     g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all Prefill MoE shared pipeline tests passed");
    return 0;
}
